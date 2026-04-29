//! ManagedTranscriber — the runner's public surface. See spec §6.1.

use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossbeam_channel::TrySendError;

use crate::core::{AsrParams, AsrParamsOverride, Command, Event, LanguagePolicy, Transcriber};
use crate::runner::{RunnerError, WhisperPoolConfig};
use crate::runner::whisper_pool::{AsrWorkItem, WhisperPool};
use crate::types::{ChunkId, Transcript, VadSegment, WorkFailure};
use mediatime::Timestamp;

/// Outcome of a single try-send into the work_tx channel.
#[derive(Debug)]
pub(super) enum DispatchOutcome {
    /// Command was sent and consumed.
    Sent,
    /// Channel was full; the command must be re-parked via
    /// `Transcriber::unpoll_command`.
    Backpressure(Command),
    /// All worker channels are disconnected — the pool has shut down.
    Disconnected,
}

/// Public runner: wraps `core::Transcriber` and a `WhisperPool` with
/// the saturation-deadlock-safe dispatch loop from spec §6.4.1.
pub struct ManagedTranscriber {
    core: Transcriber,
    whisper_pool: WhisperPool,
    asr_params_default: AsrParams,
    asr_timeout: Duration,
    drain_timeout: Duration,
    block_on_full_queue: bool,
    dispatch_idle_poll: Duration,
    buffer_cap_samples: usize,
}

impl ManagedTranscriber {
    /// Try to send a Command into the worker pool. Non-blocking.
    fn try_dispatch(
        &self,
        cmd: Command,
        asr_timeout: Duration,
    ) -> DispatchOutcome {
        let item = match cmd {
            Command::RunAsr { chunk_id, samples, params, sample_rate: _ } => {
                let abort_flag = Arc::new(AtomicBool::new(false));
                AsrWorkItem {
                    chunk_id,
                    samples,
                    params,
                    asr_timeout,
                    abort_flag,
                }
            }
            // RunAlignment is Plan C scope; the core only emits it when
            // word_alignment=true was set, which Plan B does not
            // enable. If a Plan B builder somehow ends up with
            // alignment on (e.g., from the `alignment` cargo feature
            // without supplying an AlignmentSet), the runner refuses
            // to dispatch the alignment command and re-parks it.
            cmd @ Command::RunAlignment { .. } => {
                return DispatchOutcome::Backpressure(cmd);
            }
        };
        match self.whisper_pool.work_tx.try_send(item) {
            Ok(()) => DispatchOutcome::Sent,
            Err(TrySendError::Full(item)) => {
                // Reconstruct the original Command so the core can
                // re-park it via unpoll_command.
                let cmd = Command::RunAsr {
                    chunk_id: item.chunk_id,
                    samples: item.samples,
                    sample_rate: crate::time::SAMPLE_RATE_HZ,
                    params: item.params,
                };
                DispatchOutcome::Backpressure(cmd)
            }
            Err(TrySendError::Disconnected(_)) => DispatchOutcome::Disconnected,
        }
    }

    /// One non-blocking step of the inline dispatch loop.
    ///
    /// Returns `Ok(true)` if any of (drain ≥ 1 result | send ≥ 1
    /// command | core surfaced ≥ 1 event); `Ok(false)` if nothing
    /// changed.
    ///
    /// `Err(RunnerError::Backpressure)` is returned only when
    /// `block_on_full_queue=false` and a try_send hit Full. The
    /// command was re-parked via `Transcriber::unpoll_command`; the
    /// core's buffer state has already advanced (samples buffered,
    /// segments merged into possibly-pending chunks). Per spec
    /// §6.4.2 the caller must drain via `poll_*` before pushing again.
    ///
    /// `Err(RunnerError::WhisperPoolShutdown)` is fatal: a worker
    /// channel disconnected.
    pub(super) fn drive_one_step(&mut self) -> Result<bool, RunnerError> {
        let mut progress = false;

        // Phase 1: drain results first.
        loop {
            match self.whisper_pool.result_rx.try_recv() {
                Ok((chunk_id, Ok(asr_result))) => {
                    progress = true;
                    self.core.inject_asr_result(chunk_id, asr_result)?;
                }
                Ok((chunk_id, Err(failure))) => {
                    progress = true;
                    self.core.inject_failure(chunk_id, failure)?;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    return Err(RunnerError::WhisperPoolShutdown);
                }
            }
        }

        // Phase 2: drain core's events. Plan A's Transcriber emits
        // events directly via poll_event, but ManagedTranscriber
        // exposes them via poll_transcript / poll_error (split by
        // Event variant). We pull events into the per-Transcriber
        // emit queue, which lives inside the core itself (no extra
        // channel needed).
        // (No code here — `poll_transcript` calls poll_event inline.)

        // Phase 3: drain commands and try to dispatch each.
        while let Some(cmd) = self.core.poll_command() {
            match self.try_dispatch(cmd, self.asr_timeout) {
                DispatchOutcome::Sent => progress = true,
                DispatchOutcome::Backpressure(parked) => {
                    self.core.unpoll_command(parked);
                    if !self.block_on_full_queue {
                        return Err(RunnerError::Backpressure {
                            buffered: self.core.buffered_samples(),
                            cap: self.buffer_cap_samples,
                        });
                    }
                    return Ok(progress);
                }
                DispatchOutcome::Disconnected => {
                    return Err(RunnerError::WhisperPoolShutdown);
                }
            }
        }

        Ok(progress)
    }

    /// Block (with `dispatch_idle_poll` safety) until at least one
    /// worker channel has data, OR the safety-timeout fires. Does NOT
    /// consume any message — the next `drive_one_step` does that via
    /// `try_recv`. See spec §6.4.1 for why `Select::ready_timeout`
    /// is the correct primitive (consuming variants would silently
    /// drop results: NB-β).
    fn wait_for_progress(&self) -> Result<(), RunnerError> {
        let mut sel = crossbeam_channel::Select::new();
        sel.recv(&self.whisper_pool.result_rx);
        // ready_timeout returns Ok(idx) with idx of the first ready
        // op (including disconnects), or Err(SelectTimeoutError) on
        // timeout. We don't care which arm fired — the next
        // drive_one_step's try_recv handles message vs. disconnect.
        let _ = sel.ready_timeout(self.dispatch_idle_poll);
        Ok(())
    }

    /// Drive the inline dispatch loop in a saturation wait.
    ///
    /// Loops:
    ///   1. drive_one_step — if Ok(true), made progress; loop again.
    ///   2. else if no command is parked, exit (genuine idle).
    ///   3. else wait_for_progress, then loop.
    ///
    /// Used by both `process_packet` (after pushing inputs) and
    /// `drain` (until idle).
    fn pump_until_idle_or_progress(&mut self) -> Result<(), RunnerError> {
        loop {
            if self.drive_one_step()? {
                continue;
            }
            // No progress. Is there a parked command waiting?
            // We can't peek without popping; the only way to detect
            // a parked command is to call poll_command, then re-park
            // if Some.
            match self.core.poll_command() {
                None => return Ok(()),
                Some(cmd) => {
                    self.core.unpoll_command(cmd);
                    self.wait_for_progress()?;
                }
            }
        }
    }
}
