//! ManagedTranscriber — the runner's public surface. See spec §6.1.

use alloc::collections::VecDeque;
use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossbeam_channel::TrySendError;
use whisper_rs::WhisperContext;

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
    /// Per-packet override active during the current `process_packet`
    /// call. Set on entry, cleared on exit. `try_dispatch` merges it
    /// on top of the core's emitted params (which carry the locked
    /// language from `LanguagePolicy`). Sparse layering preserves
    /// fields the override doesn't set.
    pending_override: Option<AsrParamsOverride>,
    asr_timeout: Duration,
    drain_timeout: Duration,
    block_on_full_queue: bool,
    dispatch_idle_poll: Duration,
    buffer_cap_samples: usize,
    pending_transcripts: VecDeque<Transcript>,
    pending_errors: VecDeque<(ChunkId, WorkFailure)>,
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
                // Use the core's emitted params (which already include
                // the locked language from LanguagePolicy::Lock /
                // AutoLockAfter). If a per-packet override is active
                // for this process_packet call, merge it on top —
                // sparse override fields layer cleanly without
                // clobbering the language lock the core just set.
                //
                // Pre-fix code substituted `self.asr_params_default`
                // here, which discarded the locked language and made
                // whisper auto-detect every chunk → temperature ladder
                // exhausted on every chunk → drain hangs.
                let final_params = match &self.pending_override {
                    Some(ovr) => merge_overrides(&params, ovr),
                    None => params,
                };
                let abort_flag = Arc::new(AtomicBool::new(false));
                AsrWorkItem {
                    chunk_id,
                    samples,
                    params: final_params,
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

/// Builder for [`ManagedTranscriber`].
///
/// All knobs are `with_*` style; defaults match spec §8. Construct
/// via [`ManagedTranscriber::builder`].
pub struct ManagedTranscriberBuilder {
    whisper_ctx: WhisperContext,
    pool_config: WhisperPoolConfig,
    chunk_size: Duration,
    buffer_cap_samples: usize,
    gap_tolerance_samples: u64,
    language_policy: LanguagePolicy,
    asr_params: AsrParams,
    worker_timeouts_asr: Duration,
    worker_timeouts_align: Duration,
    drain_timeout: Option<Duration>,
}

impl ManagedTranscriberBuilder {
    /// Internal constructor used by `ManagedTranscriber::builder`.
    fn new(whisper_ctx: WhisperContext, pool_config: WhisperPoolConfig) -> Self {
        Self {
            whisper_ctx,
            pool_config,
            chunk_size: Duration::from_secs(30),
            buffer_cap_samples: 60 * 16_000,
            gap_tolerance_samples: 200 * 16,
            language_policy: LanguagePolicy::AutoLockAfter(1),
            asr_params: AsrParams::new(),
            worker_timeouts_asr: Duration::from_secs(60),
            worker_timeouts_align: Duration::from_secs(30),
            drain_timeout: None,
        }
    }

    /// Override [`crate::core::TranscriberConfig::chunk_size`].
    pub fn chunk_size(mut self, d: Duration) -> Self {
        self.chunk_size = d;
        self
    }

    /// Override [`crate::core::TranscriberConfig::buffer_cap_samples`].
    pub fn buffer_cap_samples(mut self, n: usize) -> Self {
        self.buffer_cap_samples = n;
        self
    }

    /// Override [`crate::core::TranscriberConfig::gap_tolerance_samples`].
    pub fn gap_tolerance_samples(mut self, n: u64) -> Self {
        self.gap_tolerance_samples = n;
        self
    }

    /// Override [`crate::core::TranscriberConfig::language_policy`].
    pub fn language_policy(mut self, p: LanguagePolicy) -> Self {
        self.language_policy = p;
        self
    }

    /// Override the [`WhisperPoolConfig`].
    pub fn whisper_pool(mut self, cfg: WhisperPoolConfig) -> Self {
        self.pool_config = cfg;
        self
    }

    /// Override the default [`AsrParams`].
    pub fn asr_params(mut self, p: AsrParams) -> Self {
        self.asr_params = p;
        self
    }

    /// Per-job worker timeouts. Default 60 s for ASR, 30 s for
    /// alignment.
    pub fn worker_timeouts(mut self, asr: Duration, align: Duration) -> Self {
        self.worker_timeouts_asr = asr;
        self.worker_timeouts_align = align;
        self
    }

    /// Cap on `drain()`. Default 10× the longest worker timeout.
    pub fn drain_timeout(mut self, t: Duration) -> Self {
        self.drain_timeout = Some(t);
        self
    }

    /// Construct the `ManagedTranscriber`. Spawns worker threads and
    /// wires channels.
    pub fn build(self) -> Result<ManagedTranscriber, RunnerError> {
        let drain_timeout = self.drain_timeout.unwrap_or_else(|| {
            // 10× the longest worker timeout per spec §6.1 / §8.
            let longest = core::cmp::max(
                self.worker_timeouts_asr,
                self.worker_timeouts_align,
            );
            longest * 10
        });

        let core_config = crate::core::TranscriberConfig::new()
            .with_chunk_size(self.chunk_size)
            .with_buffer_cap_samples(self.buffer_cap_samples)
            .with_gap_tolerance_samples(self.gap_tolerance_samples)
            .with_language_policy(self.language_policy)
            .with_asr_params(self.asr_params.clone())
            .with_word_alignment(false)
            .with_max_in_flight(self.pool_config.worker_count() + 2);

        let whisper_pool = WhisperPool::new(self.whisper_ctx, &self.pool_config)?;

        Ok(ManagedTranscriber {
            core: Transcriber::new(core_config),
            whisper_pool,
            asr_params_default: self.asr_params,
            pending_override: None,
            asr_timeout: self.worker_timeouts_asr,
            drain_timeout,
            block_on_full_queue: self.pool_config.block_on_full_queue(),
            dispatch_idle_poll: self.pool_config.dispatch_idle_poll(),
            buffer_cap_samples: self.buffer_cap_samples,
            pending_transcripts: VecDeque::new(),
            pending_errors: VecDeque::new(),
        })
    }
}

impl ManagedTranscriber {
    /// Begin building a `ManagedTranscriber` from a pre-constructed
    /// `WhisperContext`. The caller controls flash_attn / DTW / GPU
    /// device explicitly when constructing the context (spec §5.6,
    /// §6.2).
    ///
    /// `pool_config` carries the runner-side knobs (worker count,
    /// queue depth, backpressure mode).
    pub fn builder(
        whisper_ctx: WhisperContext,
        pool_config: WhisperPoolConfig,
    ) -> ManagedTranscriberBuilder {
        ManagedTranscriberBuilder::new(whisper_ctx, pool_config)
    }

    /// Convenience: build directly from a `WhisperPoolConfig`'s
    /// `model_path`, loading the context with the config's GPU
    /// settings. Intended for callers that don't need to customise
    /// `WhisperContextParameters` beyond what `WhisperPoolConfig`
    /// already exposes.
    pub fn from_config(
        pool_config: WhisperPoolConfig,
    ) -> Result<ManagedTranscriberBuilder, RunnerError> {
        let mut ctx_params = whisper_rs::WhisperContextParameters::default();
        ctx_params.use_gpu(pool_config.use_gpu());
        ctx_params.gpu_device(pool_config.gpu_device());
        ctx_params.flash_attn(pool_config.flash_attn());
        let path = pool_config.model_path().to_str().ok_or_else(|| {
            RunnerError::WhisperContextLoad {
                message: format!(
                    "model_path is not valid UTF-8: {:?}",
                    pool_config.model_path()
                ),
            }
        })?;
        let ctx = WhisperContext::new_with_params(path, ctx_params).map_err(|e| {
            RunnerError::WhisperContextLoad { message: format!("{e:?}") }
        })?;
        Ok(ManagedTranscriberBuilder::new(ctx, pool_config))
    }
}

impl ManagedTranscriber {
    /// Push one packet of audio + the VAD segments newly closed
    /// within or before that packet's range.
    ///
    /// **Empty packet** (`samples.is_empty()`): accepted as a no-op
    /// when `delta_pts_out == 0` — VAD segments in the same call are
    /// still pushed.
    ///
    /// **VAD segment ordering contract:** segments must be strictly
    /// monotonic and non-overlapping; violations are surfaced as
    /// `RunnerError::Transcriber(TranscriberError::PtsRegression {
    /// kind: PushKind::VadSegment, .. })`.
    ///
    /// **Backpressure contract** (spec §6.4.2): when this returns
    /// `Err(RunnerError::Backpressure { .. })`, inputs were already
    /// consumed; the caller must drain via `poll_transcript` /
    /// `poll_error` before pushing again.
    pub fn process_packet(
        &mut self,
        starts_at: Timestamp,
        samples: &[f32],
        vad_segments: &[VadSegment],
        params_override: Option<AsrParamsOverride>,
    ) -> Result<(), RunnerError> {
        // Step 1: stash the per-call override. `try_dispatch` reads
        // it and merges on top of the core's emitted params (which
        // carry the locked language). Cleared on exit regardless of
        // outcome — the override is per-packet, not sticky.
        self.pending_override = params_override;

        // Step 2: push samples (may return Backpressure / PtsRegression / etc.)
        let push_result = self.push_samples_internal(starts_at, samples);

        // Step 3: push VAD segments (only if step 2 succeeded; otherwise
        // we propagate the push error before mutating cut state).
        let result = push_result.and_then(|()| self.push_vads_internal(vad_segments));

        // Step 4: pump the dispatch loop until idle or saturation.
        let drive_result = result.and_then(|()| self.pump_until_idle_or_progress());

        // Step 5: clear the override.
        self.pending_override = None;

        drive_result
    }

    /// Push samples, mapping core errors into `RunnerError::Transcriber`.
    fn push_samples_internal(
        &mut self,
        starts_at: Timestamp,
        samples: &[f32],
    ) -> Result<(), RunnerError> {
        if samples.is_empty() {
            // Plan A's push_samples accepts empty packets when
            // delta_pts_out == 0; the underlying buffer call returns
            // Ok(()). We still call through so the timebase / EOF
            // checks fire normally.
            self.core.push_samples(starts_at, samples)?;
            return Ok(());
        }
        self.core.push_samples(starts_at, samples)?;
        Ok(())
    }

    /// Push VAD segments in order.
    fn push_vads_internal(
        &mut self,
        vad_segments: &[VadSegment],
    ) -> Result<(), RunnerError> {
        for seg in vad_segments {
            self.core.push_vad_segment(*seg)?;
        }
        Ok(())
    }

    /// Apply a per-call override on top of the runner default; return
    /// the original to be restored at end-of-call.
    fn swap_asr_default(&mut self, ovr: &AsrParamsOverride) -> AsrParams {
        // Temporarily replace the core's default with the merged
        // params. The core uses its default AsrParams when it issues
        // a RunAsr command; for v1, that's the only injection point.
        let merged = merge_overrides(&self.asr_params_default, ovr);
        let prior = core::mem::replace(&mut self.asr_params_default, merged);
        // Plan A's TranscriberConfig defaults are baked into the core
        // at construction; the runtime override path is via the
        // Command's `params` field. Plan A does NOT expose a runtime
        // setter for the dispatch's default AsrParams; this is OK
        // because the runner's own default lives on
        // `asr_params_default` and the runner is the one that emits
        // RunAsr commands' `params`. We override at dispatch time in
        // `try_dispatch`'s `params` consumption — that path is not
        // currently used because Plan A's poll_command pre-fills
        // `params` from the core's config.
        //
        // To honor the runner's override semantics, we substitute the
        // params on the issued command in dispatch. The simplest
        // correct approach: keep `asr_params_default` updated; have
        // try_dispatch overwrite the Command's `params` with our own
        // before sending. (See try_dispatch's note in Task 11; if
        // that override hook isn't already in place, add it now.)
        prior
    }

    fn restore_asr_default(&mut self, prior: AsrParams) {
        self.asr_params_default = prior;
    }
}

impl ManagedTranscriber {
    /// Mark the input stream as ended. Flushes the cut accumulator,
    /// then drives the dispatch loop one more time. Idempotent.
    pub fn signal_eof(&mut self) -> Result<(), RunnerError> {
        self.core.signal_eof()?;
        self.pump_until_idle_or_progress()?;
        Ok(())
    }

    /// Pop the next available `Transcript`, draining the dispatch
    /// loop along the way. Returns `None` only when no transcript is
    /// currently available; the caller must keep calling until the
    /// returned `Option` is `None` and `core.is_idle()` is true to
    /// know the stream has fully drained.
    pub fn poll_transcript(&mut self) -> Option<Transcript> {
        // Drive once so any pending results land in the core's event
        // queue. Errors here would be silent loss; surface via
        // poll_error in the caller's next call (the queue still has
        // any `Event::Error` events).
        let _ = self.drive_one_step();

        if let Some(tr) = self.pending_transcripts.pop_front() {
            return Some(tr);
        }
        loop {
            match self.core.poll_event()? {
                Event::Transcript(tr) => return Some(tr),
                Event::Error { chunk_id, error } => {
                    self.pending_errors.push_back((chunk_id, error));
                    // Continue: maybe a Transcript is right behind it.
                }
            }
        }
    }

    /// Pop the next available `(ChunkId, WorkFailure)` error, draining
    /// the dispatch loop along the way.
    pub fn poll_error(&mut self) -> Option<(ChunkId, WorkFailure)> {
        let _ = self.drive_one_step();

        if let Some(pair) = self.pending_errors.pop_front() {
            return Some(pair);
        }
        // Drain a few events looking for an error. We don't loop
        // forever: if the next event is a Transcript, push it onto a
        // queue and surface only errors here.
        loop {
            match self.core.poll_event()? {
                Event::Error { chunk_id, error } => return Some((chunk_id, error)),
                Event::Transcript(tr) => {
                    self.pending_transcripts.push_back(tr);
                }
            }
        }
    }

    /// Block until `core.is_idle()` or `drain_timeout` elapses.
    pub fn drain(&mut self) -> Result<(), RunnerError> {
        let started = std::time::Instant::now();
        let timeout = self.drain_timeout;
        loop {
            self.pump_until_idle_or_progress()?;
            if self.core.is_idle() {
                return Ok(());
            }
            if started.elapsed() > timeout {
                return Err(RunnerError::DrainTimeout {
                    timeout,
                    in_flight: self.core.buffered_samples(), // proxy; exact count is in dispatch
                });
            }
            // No progress and not idle: wait for a worker.
            self.wait_for_progress()?;
        }
    }
}

impl ManagedTranscriber {
    /// True iff every queue is empty (core idle AND no pending
    /// transcripts/errors locally buffered).
    pub fn is_idle(&self) -> bool {
        self.core.is_idle()
            && self.pending_transcripts.is_empty()
            && self.pending_errors.is_empty()
    }

    /// Live buffer length in samples (proxy from the core).
    pub fn buffered_samples(&self) -> usize {
        self.core.buffered_samples()
    }

    /// Output timebase, recorded on the first push_samples call.
    pub fn output_timebase(&self) -> Option<mediatime::Timebase> {
        self.core.output_timebase()
    }

    /// PTS that the core expects on the next contiguous push_samples.
    pub fn next_expected_starts_at(&self) -> Option<Timestamp> {
        self.core.next_expected_starts_at()
    }

    /// Non-mutating predicate: would the next push of `samples_len`
    /// audio samples fit?
    pub fn would_accept(&self, samples_len: usize) -> bool {
        self.core.would_accept(samples_len, 0)
    }
}

/// Merge a sparse `AsrParamsOverride` onto `base`, producing the
/// final `AsrParams` that will ship in any RunAsr emitted from the
/// current packet's chunks.
fn merge_overrides(base: &AsrParams, ovr: &AsrParamsOverride) -> AsrParams {
    let mut out = base.clone();
    if let Some(opt_lang) = ovr.language_hint() {
        out.set_language_hint(opt_lang.clone());
    }
    if let Some(strategy) = ovr.strategy() {
        out.set_strategy(strategy);
    }
    if let Some(t) = ovr.initial_temperature() {
        out.set_initial_temperature(t);
    }
    if let Some(prompt) = ovr.initial_prompt() {
        out.set_initial_prompt(prompt.clone());
    }
    out
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use crate::types::Lang;

    #[test]
    fn empty_override_is_identity() {
        let base = AsrParams::default();
        let ovr = AsrParamsOverride::new();
        let out = merge_overrides(&base, &ovr);
        assert_eq!(out.initial_temperature(), base.initial_temperature());
        assert_eq!(out.max_attempts(), base.max_attempts());
    }

    #[test]
    fn override_replaces_only_specified_fields() {
        let base = AsrParams::default();
        let ovr = AsrParamsOverride::new()
            .with_language_hint(Some(Some(Lang::En)))
            .with_initial_temperature(Some(0.7));
        let out = merge_overrides(&base, &ovr);
        assert_eq!(out.language_hint().cloned(), Some(Lang::En));
        assert!((out.initial_temperature() - 0.7).abs() < 1e-9);
        assert_eq!(out.max_attempts(), base.max_attempts());
    }

    #[test]
    fn override_can_clear_language_hint() {
        let base = AsrParams::default().with_language_hint(Some(Lang::En));
        let ovr = AsrParamsOverride::new()
            .with_language_hint(Some(None));
        let out = merge_overrides(&base, &ovr);
        assert!(out.language_hint().is_none());
    }
}
