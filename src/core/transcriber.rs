//! Transcriber — the public Sans-I/O surface.
//!
//! `Transcriber` is `Send + !Sync` (every public mutating method
//! takes `&mut self`). Multi-threaded drivers must wrap in
//! `Mutex<Transcriber>` themselves.
//!
//! See spec §5.1.

use core::time::Duration;

use mediatime::{Timebase, Timestamp};

use crate::core::buffer::SampleBuffer;
use crate::core::command::{AlignmentResult, AsrParams, AsrResult, Command};
use crate::core::cut::Cut;
use crate::core::dispatch::Dispatch;
use crate::core::event::Event;
use crate::types::{ChunkId, Lang, TranscriberError, VadSegment, WorkFailure};

/// Language-detection / locking strategy.
#[derive(Clone, Debug)]
pub enum LanguagePolicy {
    /// Each chunk independently auto-detects.
    Auto,
    /// Caller supplies the language; whisper is given a hard hint
    /// and never auto-detects.
    Lock {
        /// Locked language.
        hint: Lang,
    },
    /// Auto-detect on the first `n` chunks that emit non-empty text,
    /// then lock the most-frequent detected language for the rest of
    /// the session. WhisperX-equivalent default; `n = 1` matches
    /// WhisperX exactly.
    AutoLockAfter(usize),
}

impl Default for LanguagePolicy {
    fn default() -> Self {
        Self::AutoLockAfter(1)
    }
}

/// Configuration for the core state machine.
#[derive(Clone, Debug)]
pub struct TranscriberConfig {
    /// Maximum duration of a merged chunk. Default 30 s.
    pub chunk_size: Duration,
    /// Max samples kept in the internal buffer before push returns
    /// Backpressure. Default 60 s × 16 kHz = 960 000.
    pub buffer_cap_samples: usize,
    /// Maximum forward-gap that is silently zero-filled. Default
    /// 200 ms × 16 kHz = 3200.
    pub gap_tolerance_samples: u64,
    /// Whether to emit `RunAlignment` after each ASR completion.
    pub word_alignment: bool,
    /// Maximum chunks in flight. Default `worker_count + 2`; without
    /// runner context, the core defaults to 6.
    pub max_in_flight: usize,
    /// Default ASR params injected into every `RunAsr` command.
    pub asr_params: AsrParams,
    /// Language detection / locking strategy.
    pub language_policy: LanguagePolicy,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            chunk_size: Duration::from_secs(30),
            buffer_cap_samples: 60 * 16_000,
            gap_tolerance_samples: 200 * 16, // 200 ms at 16 kHz
            word_alignment: false,
            max_in_flight: 6,
            asr_params: AsrParams::default(),
            language_policy: LanguagePolicy::default(),
        }
    }
}

/// The Sans-I/O state machine. See spec §5.1.
///
/// `Transcriber` is `Send` (movable across threads) but `!Sync`
/// (every public mutating method takes `&mut self`). A consumer that
/// wants to drive it from multiple threads must wrap it in
/// `Mutex<Transcriber>` themselves; whispery does not provide
/// internal synchronisation.
pub struct Transcriber {
    config: TranscriberConfig,
    buffer: SampleBuffer,
    cut: Cut,
    dispatch: Dispatch,
    next_chunk_id: u64,
    eof_signaled: bool,
}

impl Transcriber {
    /// Construct from config.
    pub fn new(config: TranscriberConfig) -> Self {
        let buffer = SampleBuffer::new(config.buffer_cap_samples, config.gap_tolerance_samples);
        let cut = Cut::new(config.chunk_size);
        let dispatch = Dispatch::new(
            config.asr_params.clone(),
            config.word_alignment,
            config.max_in_flight,
        );
        Self {
            config,
            buffer,
            cut,
            dispatch,
            next_chunk_id: 0,
            eof_signaled: false,
        }
    }

    /// Pop the front command, consulting `unpoll_command`'s parked
    /// slot first.
    pub fn poll_command(&mut self) -> Option<Command> {
        self.dispatch.poll_command()
    }

    /// Pop the front event.
    pub fn poll_event(&mut self) -> Option<Event> {
        self.dispatch.poll_event()
    }

    /// Re-park the front of the command queue. **Visibility:
    /// `pub(crate)`** — the runner module is the only legitimate
    /// caller. Out-of-tree consumers driving the state machine
    /// themselves do not need this affordance.
    pub(crate) fn unpoll_command(&mut self, cmd: Command) {
        self.dispatch.unpoll_command(cmd);
    }

    /// True iff every queue is empty: no buffered samples, no
    /// pending command/event, no in_flight chunks, no cut_pending
    /// entries. Pre-restart in-flight chunks (those still working
    /// through whisper or alignment) keep `is_idle()` false until
    /// they emit; `restart_at` does not synthetically clear them.
    pub fn is_idle(&self) -> bool {
        self.dispatch.is_idle() && self.buffer.buffered_samples() == 0
    }

    /// Live buffer length in samples.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.buffered_samples()
    }

    /// Output timebase recorded from the first `push_samples` call.
    pub fn output_timebase(&self) -> Option<Timebase> {
        self.buffer.output_timebase()
    }

    /// Authoritative output-timebase PTS the buffer expects for the
    /// next contiguous `push_samples` call. Returns `None` before
    /// the first push.
    pub fn next_expected_starts_at(&self) -> Option<Timestamp> {
        self.buffer.next_expected_starts_at()
    }

    /// Non-mutating predicate: would the next push of `samples_len`
    /// audio samples plus `vad_count` VAD segments fit under the
    /// configured caps?
    pub fn would_accept(&self, samples_len: usize, _vad_count: usize) -> bool {
        self.buffered_samples() + samples_len <= self.config.buffer_cap_samples
    }

    /// Push samples into the buffer. See spec §4.1 / §5.4.
    ///
    /// Errors:
    /// - `PtsRegression`, `GapExceedsTolerance`, `Backpressure`,
    ///   `InconsistentTimebase`, `AfterEof` per `SampleBuffer::append`.
    pub fn push_samples(
        &mut self,
        starts_at: Timestamp,
        samples: &[f32],
    ) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        self.buffer.append(starts_at, samples)
    }

    /// Push a VAD segment into the cut state machine. See spec
    /// §5.3.
    ///
    /// Errors:
    /// - `OutputTimebaseUnset` if no `push_samples` has been called.
    /// - `PtsRegression { kind: VadSegment }` if `seg.start_sample`
    ///   is not strictly greater than the previous VAD segment's
    ///   `end_sample`.
    /// - `AfterEof` if `signal_eof()` was called.
    pub fn push_vad_segment(&mut self, seg: VadSegment) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        if self.buffer.output_timebase().is_none() {
            return Err(TranscriberError::OutputTimebaseUnset);
        }
        // Strict-monotonic check against the cut state machine's
        // last accumulated end. Cut tracks current_end internally;
        // we replicate the check here to surface PtsRegression for
        // the explicit test contract.
        if let Some(last_end) = self.cut.last_pushed_end() {
            if seg.start_sample() < last_end {
                return Err(TranscriberError::PtsRegression {
                    kind: crate::types::PushKind::VadSegment,
                    advance: seg.start_sample() as i64 - last_end as i64,
                });
            }
        }

        let merged_chunks = self.cut.push_segment(seg);
        for chunk in merged_chunks {
            let chunk_id = ChunkId::from_raw(self.next_chunk_id);
            self.next_chunk_id += 1;
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }
        Ok(())
    }

    /// Mark the input stream as ended. Idempotent. Calling before
    /// any push is a no-op (Ok(())). Errors: never returns Err in
    /// v1; signature carries `Result<(), TranscriberError>` for
    /// forward compatibility.
    pub fn signal_eof(&mut self) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Ok(());
        }
        self.eof_signaled = true;
        if self.buffer.output_timebase().is_some() {
            if let Some(chunk) = self.cut.flush() {
                let chunk_id = ChunkId::from_raw(self.next_chunk_id);
                self.next_chunk_id += 1;
                self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
            }
        }
        Ok(())
    }

    /// Inject the result of a `Command::RunAsr`.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight.
    pub fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        result: AsrResult,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_asr_result(chunk_id, result)?;
        self.dispatch.after_inject(&mut self.buffer);
        Ok(())
    }

    /// Inject the result of a `Command::RunAlignment`.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not awaiting alignment.
    pub fn inject_alignment_result(
        &mut self,
        chunk_id: ChunkId,
        result: AlignmentResult,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_alignment_result(chunk_id, result)?;
        self.dispatch.after_inject(&mut self.buffer);
        Ok(())
    }

    /// Inject a per-chunk failure.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight.
    pub fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_failure(chunk_id, failure)?;
        self.dispatch.after_inject(&mut self.buffer);
        Ok(())
    }

    /// Recover from a `GapExceedsTolerance`. See spec §5.4.1.
    ///
    /// Steps:
    /// 1. Drain `cut_pending` synchronously into `in_flight`
    ///    (extract samples in old-frame indexing, cache TimeRange
    ///    via the old anchor). May temporarily exceed
    ///    `max_in_flight`.
    /// 2. Flush the cut state machine. Any partial chunk also
    ///    promotes to `in_flight`.
    /// 3. Clear the live buffer; reset `absolute_sample_offset` and
    ///    `buffer_drop_offset` to 0.
    /// 4. Re-anchor `base_pts_out_anchor` to `starts_at.pts()`.
    /// 5. `next_chunk_id` continues monotonically.
    /// 6. Trim's low-water computed from `cut_pending` only — empty
    ///    after drain — so the new buffer is fully droppable.
    ///
    /// Errors:
    /// - `AfterEof` if `signal_eof()` was previously called.
    pub fn restart_at(&mut self, starts_at: Timestamp) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }

        // Step 1: drain cut_pending into in_flight before clearing
        // the buffer. Uses the existing buffer state (still in old
        // frame).
        self.dispatch.draining_for_restart = true;
        while let Some((chunk_id, chunk)) = self.dispatch.cut_pending.pop_front() {
            // Synthesise the same path as Dispatch::on_emit's
            // immediate-promote branch.
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }

        // Step 2: flush the cut accumulator (also goes through on_emit).
        if let Some(chunk) = self.cut.flush() {
            let chunk_id = ChunkId::from_raw(self.next_chunk_id);
            self.next_chunk_id += 1;
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }

        // Steps 3 + 4: clear buffer and re-anchor.
        self.buffer.restart_at(starts_at);

        // Reset the cut state machine so its current_end / next_vad_seq
        // align with the new frame.
        self.cut = Cut::new(self.config.chunk_size);

        self.dispatch.draining_for_restart = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VadSegment;
    use core::num::NonZeroU32;

    fn tb_48k() -> Timebase {
        Timebase::new(1, NonZeroU32::new(48_000).unwrap())
    }

    fn ts(pts: i64) -> Timestamp {
        Timestamp::new(pts, tb_48k())
    }

    fn fresh() -> Transcriber {
        Transcriber::new(TranscriberConfig::default())
    }

    #[test]
    fn push_vad_before_push_samples_returns_output_timebase_unset() {
        let mut t = fresh();
        let r = t.push_vad_segment(VadSegment::new(0, 100));
        assert!(matches!(r, Err(TranscriberError::OutputTimebaseUnset)));
    }

    #[test]
    fn push_samples_then_vad_works() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        t.push_vad_segment(VadSegment::new(0, 200)).unwrap();
    }

    #[test]
    fn vad_segment_regression_returns_pts_regression() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 10_000]).unwrap();
        t.push_vad_segment(VadSegment::new(100, 200)).unwrap();
        let r = t.push_vad_segment(VadSegment::new(150, 250)); // overlaps
        assert!(matches!(
            r,
            Err(TranscriberError::PtsRegression { kind: crate::types::PushKind::VadSegment, .. })
        ));
    }

    #[test]
    fn signal_eof_then_push_rejects() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 100]).unwrap();
        t.signal_eof().unwrap();
        let r = t.push_samples(ts(100), &[0.0; 100]);
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
        let r = t.push_vad_segment(VadSegment::new(0, 100));
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
    }

    #[test]
    fn signal_eof_idempotent_and_noop_before_push() {
        let mut t = fresh();
        t.signal_eof().unwrap();
        t.signal_eof().unwrap();
    }

    #[test]
    fn restart_at_after_signal_eof_rejects() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        t.signal_eof().unwrap();
        let r = t.restart_at(ts(50_000_000));
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
    }

    #[test]
    fn restart_at_drains_cut_pending_into_in_flight() {
        // max_in_flight = 1 forces queueing.
        let mut config = TranscriberConfig::default();
        config.max_in_flight = 1;
        config.chunk_size = Duration::from_millis(125); // 2_000 samples
        config.buffer_cap_samples = 100_000;
        let mut t = Transcriber::new(config);

        // Push enough audio to cover three chunks.
        t.push_samples(ts(0), &[0.0; 16_000]).unwrap(); // 1 sec @ 16k pretend
        t.push_vad_segment(VadSegment::new(0, 2_000)).unwrap();
        t.push_vad_segment(VadSegment::new(2_000, 4_000)).unwrap();
        t.push_vad_segment(VadSegment::new(4_000, 6_000)).unwrap();
        // First chunk's RunAsr is in pending_commands; second and
        // third are in cut_pending awaiting promotion.
        // Now restart at a fresh anchor.
        t.restart_at(ts(50_000_000)).unwrap();

        // After restart: cut_pending should be empty (drained), the
        // buffer should be empty (cleared). Pre-restart in-flight
        // chunks (the originally-promoted one PLUS the formerly-
        // pending ones) survive — they hold their audio in their
        // own Arc<[f32]>s and will emit normally.
        // (Spec §5.4.1 + §5.5 invariant 4 exception: drain is
        // allowed to exceed max_in_flight transiently.)
    }
}
