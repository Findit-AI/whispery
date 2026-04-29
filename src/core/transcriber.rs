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
    /// Highest sample index that VAD has analyzed — either as the
    /// `end_sample` of a pushed `VadSegment`, or as the explicit
    /// watermark in `signal_no_speech_through`. Future VAD pushes
    /// and no-speech signals must advance this; regressions
    /// surface as `TranscriberError::PtsRegression { kind: VadSegment }`.
    /// Independent from `Cut::last_pushed_end()` because the cut
    /// state machine only tracks pushed segments, while the
    /// watermark also incorporates explicit silence declarations
    /// (Codex round-5 fix).
    vad_watermark: u64,
}

impl Transcriber {
    /// Construct from config.
    ///
    /// # Panics
    ///
    /// Codex round-4 finding [high]: invalid config values are
    /// rejected up-front rather than turning into deadlocks (zero
    /// `max_in_flight`), divide-by-zero panics in cut
    /// (`chunk_size` that rounds to 0 samples), or empty-list panics
    /// in the auto-lock mode helper (`AutoLockAfter(0)`).
    ///
    /// - `max_in_flight == 0` — the dispatch loop would route every
    ///   emitted chunk to `cut_pending` and never issue a `RunAsr`.
    /// - `LanguagePolicy::AutoLockAfter(0)` — a 0-observation lock
    ///   has no defined mode and would call the tiebreak helper on
    ///   an empty list.
    /// - `chunk_size` that rounds to 0 16 kHz samples (e.g.
    ///   `Duration::ZERO`) — Cut's hard-split path divides by it.
    pub fn new(config: TranscriberConfig) -> Self {
        assert!(
            config.max_in_flight > 0,
            "TranscriberConfig::max_in_flight must be > 0 (got 0; would deadlock the dispatch loop)"
        );
        if let LanguagePolicy::AutoLockAfter(n) = config.language_policy {
            assert!(
                n > 0,
                "LanguagePolicy::AutoLockAfter(n) requires n > 0 (got 0)"
            );
        }
        let chunk_size_samples = (config.chunk_size.as_secs_f64()
            * crate::time::SAMPLE_RATE_HZ as f64
            + 0.5) as u64;
        assert!(
            chunk_size_samples > 0,
            "TranscriberConfig::chunk_size must round to at least 1 sample at 16 kHz; got {:?}",
            config.chunk_size
        );

        let buffer = SampleBuffer::new(config.buffer_cap_samples, config.gap_tolerance_samples);
        let cut = Cut::new(config.chunk_size);
        let dispatch = Dispatch::new(
            config.asr_params.clone(),
            config.word_alignment,
            config.max_in_flight,
            config.language_policy.clone(),
        );
        Self {
            config,
            buffer,
            cut,
            dispatch,
            next_chunk_id: 0,
            eof_signaled: false,
            vad_watermark: 0,
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
    ///   overlaps the previous VAD segment — i.e., is strictly
    ///   *less than* its `end_sample`. Touching segments (where
    ///   `new.start == prev.end`) are accepted: silero occasionally
    ///   emits them on silence-edge transitions and rejecting them
    ///   would force callers to add gap-injection logic for the
    ///   same data silero already produced cleanly.
    /// - `VadAheadOfAudio` if `seg.end_sample()` is past the
    ///   buffer's current high-water mark. The cut state machine
    ///   would otherwise accept the segment and emit chunks that
    ///   later panic in `buffer.extract` once they reach
    ///   promotion.
    /// - `AfterEof` if `signal_eof()` was called.
    pub fn push_vad_segment(&mut self, seg: VadSegment) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        if self.buffer.output_timebase().is_none() {
            return Err(TranscriberError::OutputTimebaseUnset);
        }
        // Codex round-2 fix: VAD must not reference audio that has
        // not been buffered. Otherwise the cut state machine would
        // happily accept the segment, accumulate it, and later
        // (on a chunk emit or signal_eof flush) trip a panic in
        // buffer.extract when it tries to slice past the tail.
        let high_water = self.buffer.absolute_sample_offset();
        if seg.end_sample() > high_water {
            return Err(TranscriberError::VadAheadOfAudio {
                vad_end: seg.end_sample(),
                buffered: high_water,
            });
        }
        // Strict-monotonic check against the VAD watermark. The
        // watermark advances with every push_vad_segment AND every
        // signal_no_speech_through, so a VAD push that would
        // contradict an explicit silence declaration (Codex round-5)
        // is also caught here.
        if seg.start_sample() < self.vad_watermark {
            return Err(TranscriberError::PtsRegression {
                kind: crate::types::PushKind::VadSegment,
                advance: seg.start_sample() as i64 - self.vad_watermark as i64,
            });
        }

        let merged_chunks = self.cut.push_segment(seg);
        self.vad_watermark = seg.end_sample();
        for chunk in merged_chunks {
            let chunk_id = ChunkId::from_raw(self.next_chunk_id);
            self.next_chunk_id += 1;
            self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
        }
        Ok(())
    }

    /// Declare that VAD has finished analyzing audio through
    /// `sample_index` and produced no segments past the most
    /// recent `push_vad_segment` call. The core uses this signal
    /// to:
    ///
    /// 1. Trim audio that is no longer referenced by any live
    ///    chunk — without this, a stream with long silences would
    ///    accumulate audio in the buffer until the configured cap
    ///    is hit and `push_samples` returns
    ///    `TranscriberError::Backpressure` with no recovery path
    ///    (chunks emit only on VAD or EOF).
    /// 2. Pre-flush the cut accumulator if a hypothetical future
    ///    segment starting at `sample_index` would force a flush
    ///    (`sample_index - current_start > chunk_size_samples`).
    ///    This handles the speech-followed-by-long-silence case
    ///    where a trailing partial chunk would otherwise sit in
    ///    the cut state until EOF.
    ///
    /// `sample_index` advances the VAD watermark; subsequent
    /// `push_vad_segment` calls with `start_sample < sample_index`
    /// or `signal_no_speech_through` calls with a smaller
    /// `sample_index` return `PtsRegression { kind: VadSegment }`.
    ///
    /// Errors:
    /// - `OutputTimebaseUnset` if no `push_samples` has been called.
    /// - `AfterEof` if `signal_eof()` was called.
    /// - `PtsRegression { kind: VadSegment }` if `sample_index` is
    ///   less than the current VAD watermark.
    pub fn signal_no_speech_through(
        &mut self,
        sample_index: u64,
    ) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        if self.buffer.output_timebase().is_none() {
            return Err(TranscriberError::OutputTimebaseUnset);
        }
        if sample_index < self.vad_watermark {
            return Err(TranscriberError::PtsRegression {
                kind: crate::types::PushKind::VadSegment,
                advance: sample_index as i64 - self.vad_watermark as i64,
            });
        }
        self.vad_watermark = sample_index;

        // Pre-flush the cut accumulator if a hypothetical segment
        // arriving at `sample_index` would have forced a flush
        // (extension would exceed chunk_size_samples). Otherwise
        // the partial chunk sits forever waiting for a segment
        // that the caller has now declared isn't coming.
        if let Some(start) = self.cut.pending_start() {
            if sample_index.saturating_sub(start) > self.cut.chunk_size_samples() {
                if let Some(chunk) = self.cut.flush() {
                    let chunk_id = ChunkId::from_raw(self.next_chunk_id);
                    self.next_chunk_id += 1;
                    self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
                }
            }
        }

        // Run the standard post-mutation drain so trim drops audio
        // unreferenced by any live chunk.
        self.dispatch.after_inject(&mut self.buffer, self.cut.pending_start());
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
            // Run after_inject so trim drops any audio not referenced
            // by either cut_pending or the cut accumulator. Without
            // this, a silent stream (samples pushed but no VAD) would
            // leave the buffer non-empty forever and `is_idle()`
            // would never become true.
            self.dispatch.after_inject(&mut self.buffer, self.cut.pending_start());
        }
        Ok(())
    }

    /// Inject the result of a `Command::RunAsr`.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight or
    ///   is in flight but not awaiting an ASR result.
    pub fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        result: AsrResult,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_asr_result(chunk_id, result)?;
        self.dispatch.after_inject(&mut self.buffer, self.cut.pending_start());
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
        self.dispatch.after_inject(&mut self.buffer, self.cut.pending_start());
        Ok(())
    }

    /// Inject a per-chunk failure.
    ///
    /// Errors:
    /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight or
    ///   is in flight but not awaiting any worker result.
    pub fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError> {
        self.dispatch.inject_failure(chunk_id, failure)?;
        self.dispatch.after_inject(&mut self.buffer, self.cut.pending_start());
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
    /// - `InconsistentTimebase` if the buffer already has an established
    ///   output timebase from a prior `push_samples` and `starts_at`'s
    ///   timebase doesn't match. Codex round-4 fix: pre-fix code
    ///   silently overwrote the timebase, so a 48 kHz stream restarted
    ///   at a millisecond timebase would produce post-restart chunks
    ///   in a different unit from pre-restart ones — corrupting
    ///   ordering and PTS arithmetic with no surfaced error.
    pub fn restart_at(&mut self, starts_at: Timestamp) -> Result<(), TranscriberError> {
        if self.eof_signaled {
            return Err(TranscriberError::AfterEof);
        }
        if let Some(expected_tb) = self.buffer.output_timebase() {
            if starts_at.timebase() != expected_tb {
                return Err(TranscriberError::InconsistentTimebase {
                    expected: expected_tb,
                    got: starts_at.timebase(),
                });
            }
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

        // The VAD watermark lives in absolute-sample space, which
        // restart_at just reset to 0. Reset the watermark too,
        // otherwise post-restart VAD pushes at small sample indices
        // fail the regression check against the pre-restart end
        // (Codex round-5 corollary).
        self.vad_watermark = 0;

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

    /// Codex round-1 finding [high]: trim must NOT drop audio still
    /// referenced by the cut accumulator. Reproduction: chunk 0
    /// emitted, chunk 1 accumulating in Cut without being emitted,
    /// chunk 0 ASR completes → after_inject runs trim. Without the
    /// fix, trim's low-water defaulted to absolute_sample_offset
    /// when cut_pending was empty and dropped the in-buffer audio
    /// that chunk 1 would later need; the next push_vad_segment
    /// that closed chunk 1 would then panic in buffer.extract.
    #[test]
    fn trim_keeps_samples_for_unextracted_cut_accumulator() {
        use crate::core::command::Command;
        use smol_str::SmolStr;

        let mut config = TranscriberConfig::default();
        config.chunk_size = Duration::from_secs(2); // 32_000 samples @ 16k
        config.buffer_cap_samples = 200_000;
        config.max_in_flight = 4;
        let mut t = Transcriber::new(config);

        // 4 seconds of 16 kHz audio = 64_000 samples.
        t.push_samples(ts(0), &vec![0.0_f32; 64_000]).unwrap();

        // Two VAD segments. The second pushes the merge past the
        // 32_000-sample chunk_size, so chunk 0 emits with range
        // [0, 24_000) and chunk 1 starts accumulating at 25_600.
        t.push_vad_segment(VadSegment::new(0, 24_000)).unwrap();
        t.push_vad_segment(VadSegment::new(25_600, 48_000)).unwrap();

        // Resolve chunk 0 — this is where the bug used to fire trim
        // with low_water = absolute_sample_offset = 64_000 because
        // cut_pending was empty (chunk 1 wasn't yet emitted by Cut).
        let cmd = t.poll_command().unwrap();
        let Command::RunAsr { chunk_id, .. } = cmd else { panic!("expected RunAsr") };
        let asr = crate::core::command::AsrResult {
            text: SmolStr::new("c0"),
            language: crate::types::Lang::En,
            avg_logprob: -0.5,
            no_speech_prob: 0.05,
            temperature: 0.0,
        };
        t.inject_asr_result(chunk_id, asr).unwrap();

        // Drain chunk 0's Transcript event.
        let _ = t.poll_event().expect("chunk 0 transcript");

        // Now close chunk 1. With the fix, the buffer still has
        // samples back to 25_600, so this extract succeeds. Without
        // the fix, the buffer was cleared past 64_000 and this
        // panics inside buffer.extract.
        t.push_vad_segment(VadSegment::new(50_000, 60_000)).unwrap();

        // The third VAD push triggered chunk 1's emission with
        // range [25_600, 48_000) — its RunAsr is queued.
        let cmd = t.poll_command().unwrap();
        let Command::RunAsr { chunk_id, .. } = cmd else { panic!("expected RunAsr") };
        assert_eq!(chunk_id.as_u64(), 1, "chunk 1 ran without panic");
    }

    /// Codex round-1 finding [medium]: silent EOF (samples pushed,
    /// no VAD ever pushed) must leave the transcriber idle.
    /// Without the fix, signal_eof returned without trimming the
    /// buffer; is_idle()'s `buffered_samples() == 0` check stayed
    /// false forever.
    #[test]
    fn silent_eof_makes_transcriber_idle() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        assert!(!t.is_idle(), "buffer has 1000 samples; not idle yet");
        t.signal_eof().unwrap();
        assert!(t.is_idle(), "after silent EOF, transcriber should be idle");
    }

    /// Codex round-4 finding [high]: `max_in_flight = 0` deadlocks
    /// the dispatch loop — every emitted chunk goes to cut_pending,
    /// no `RunAsr` command ever fires. Reject at construction.
    #[test]
    #[should_panic(expected = "max_in_flight")]
    fn config_with_zero_max_in_flight_panics() {
        let mut config = TranscriberConfig::default();
        config.max_in_flight = 0;
        let _ = Transcriber::new(config);
    }

    /// Round-4 corollary: `AutoLockAfter(0)` calls
    /// `mode_with_first_occurrence_tiebreak` on a possibly-empty
    /// observation list (when the first chunk lands empty/failed),
    /// which panics. Reject at construction.
    #[test]
    #[should_panic(expected = "AutoLockAfter")]
    fn config_with_zero_auto_lock_after_panics() {
        let mut config = TranscriberConfig::default();
        config.language_policy = LanguagePolicy::AutoLockAfter(0);
        let _ = Transcriber::new(config);
    }

    /// Round-4 corollary: a `chunk_size` that rounds to 0 samples
    /// (e.g. `Duration::ZERO`) makes `Cut::push_segment`'s
    /// `len.div_ceil(self.chunk_size_samples)` divide by zero on
    /// any non-trivial VAD segment. Reject at construction.
    #[test]
    #[should_panic(expected = "chunk_size")]
    fn config_with_zero_chunk_size_panics() {
        let mut config = TranscriberConfig::default();
        config.chunk_size = Duration::ZERO;
        let _ = Transcriber::new(config);
    }

    /// Codex round-4 finding [high]: restart_at must not silently
    /// switch the output timebase. Without the guard, a stream
    /// anchored at 1/48000 could be restarted at 1/1000 and produce
    /// post-restart `TimeRange`s in a different unit from pre-restart
    /// chunks — corrupts ordering and downstream PTS arithmetic with
    /// no error surfaced.
    #[test]
    fn restart_at_with_different_timebase_returns_inconsistent_timebase() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        // Stream is now anchored at 1/48000. Try to restart at 1/1000.
        let other_tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
        let r = t.restart_at(Timestamp::new(0, other_tb));
        assert!(
            matches!(r, Err(TranscriberError::InconsistentTimebase {
                expected,
                got,
            }) if expected == tb_48k() && got == other_tb),
            "expected InconsistentTimebase, got {:?}",
            r
        );
        // Original timebase must still be in effect.
        assert_eq!(t.output_timebase(), Some(tb_48k()));
    }

    /// Round-4 corollary: a restart at the same timebase succeeds.
    #[test]
    fn restart_at_same_timebase_succeeds() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        t.restart_at(ts(50_000_000)).unwrap();
        assert_eq!(t.output_timebase(), Some(tb_48k()));
    }

    /// Round-5 corollary: `restart_at` resets the buffer's
    /// `absolute_sample_offset` to 0, so the VAD watermark — which
    /// is in absolute-sample space — must reset too. Without the
    /// reset, a post-restart VAD push starting near sample 0
    /// fails the watermark regression check against the
    /// pre-restart VAD's end.
    #[test]
    fn restart_at_resets_vad_watermark() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 50_000]).unwrap();
        t.push_vad_segment(VadSegment::new(0, 30_000)).unwrap();
        // Pre-restart watermark is now 30_000.
        t.restart_at(ts(50_000_000)).unwrap();
        // Post-restart, push at sample 0 of the new frame must succeed.
        t.push_samples(ts(50_000_000), &[0.0; 10_000]).unwrap();
        t.push_vad_segment(VadSegment::new(0, 5_000)).unwrap();
    }

    /// Codex round-5 finding [high]: a stream with no VAD activity
    /// for longer than `buffer_cap_samples` would fill the buffer
    /// and trip Backpressure with no recovery path — chunks emit
    /// only on VAD or EOF, so trim never runs. The watermark API
    /// lets the caller explicitly declare "VAD analyzed through
    /// here, no segments" so the core can safely drop the buffered
    /// audio.
    #[test]
    fn signal_no_speech_through_drains_pure_silence_buffer() {
        let mut config = TranscriberConfig::default();
        // Tighten cap so test doesn't have to push 60 s of audio.
        config.buffer_cap_samples = 20_000;
        let mut t = Transcriber::new(config);
        // Push 16 000 samples (close to cap, but under).
        t.push_samples(ts(0), &[0.0; 16_000]).unwrap();
        assert_eq!(t.buffered_samples(), 16_000);
        // Tell whispery VAD has analyzed through sample 16_000 with
        // no segments. Buffer should drop everything; the next push
        // can land cleanly even though a contiguous push without
        // the signal would have hit Backpressure (16 000 + 16 000
        // > 20 000).
        t.signal_no_speech_through(16_000).unwrap();
        assert_eq!(t.buffered_samples(), 0,
            "post-watermark trim must drop unreferenced audio");
        assert!(t.is_idle());
        // Subsequent push at the same anchor (no time gap, since the
        // buffer's anchor was preserved) succeeds.
        let next = t.next_expected_starts_at().unwrap();
        t.push_samples(next, &[0.0; 16_000]).unwrap();
    }

    /// Round-5 corollary: speech followed by long silence must be
    /// flushable. After a partial chunk has been accumulating in the
    /// cut state machine, signal_no_speech_through(sample_index)
    /// past `current_start + chunk_size_samples` pre-flushes the
    /// chunk — any future segment starting >= sample_index would
    /// force the cut to flush anyway.
    #[test]
    fn signal_no_speech_through_flushes_orphaned_partial_chunk() {
        let mut config = TranscriberConfig::default();
        config.chunk_size = Duration::from_secs(2); // 32 000 samples
        let mut t = Transcriber::new(config);
        // Push enough audio to cover speech + lots of silence.
        t.push_samples(ts(0), &[0.0; 200_000]).unwrap();
        // Speech segment: [0, 16 000) — half a chunk_size.
        t.push_vad_segment(VadSegment::new(0, 16_000)).unwrap();
        // No emit yet: the cut accumulator is half-full, still under
        // chunk_size.
        assert!(t.poll_event().is_none());

        // Now signal silence past chunk_size_samples. The hypothetical
        // next segment at sample 100_000 would trigger flush
        // (100_000 - 0 > 32_000); pre-flush instead.
        t.signal_no_speech_through(100_000).unwrap();

        // The cut accumulator should have been flushed — chunk 0 is
        // now in_flight awaiting an ASR result.
        match t.poll_command() {
            Some(crate::core::command::Command::RunAsr { chunk_id, .. })
                if chunk_id.as_u64() == 0 => {}
            other => panic!("expected RunAsr for chunk 0, got {:?}", other),
        }
    }

    /// Round-5 corollary: a no-speech signal advances the VAD
    /// watermark, so subsequent VAD pushes that start before the
    /// watermark must be rejected with PtsRegression — otherwise
    /// the caller could contradict their own no-speech declaration.
    #[test]
    fn vad_segment_before_no_speech_watermark_returns_pts_regression() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 50_000]).unwrap();
        t.signal_no_speech_through(10_000).unwrap();
        let r = t.push_vad_segment(VadSegment::new(5_000, 8_000));
        assert!(matches!(
            r,
            Err(TranscriberError::PtsRegression {
                kind: crate::types::PushKind::VadSegment,
                ..
            })
        ));
    }

    /// Round-5 corollary: a regression on the no-speech watermark
    /// itself (calling it twice with a smaller index second) returns
    /// PtsRegression.
    #[test]
    fn signal_no_speech_through_regression_returns_error() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 50_000]).unwrap();
        t.signal_no_speech_through(20_000).unwrap();
        let r = t.signal_no_speech_through(10_000);
        assert!(matches!(
            r,
            Err(TranscriberError::PtsRegression {
                kind: crate::types::PushKind::VadSegment,
                ..
            })
        ));
    }

    /// Round-5 corollary: signal_no_speech_through before any push
    /// returns OutputTimebaseUnset (consistent with push_vad_segment).
    #[test]
    fn signal_no_speech_through_before_push_samples_returns_unset() {
        let mut t = fresh();
        let r = t.signal_no_speech_through(1000);
        assert!(matches!(r, Err(TranscriberError::OutputTimebaseUnset)));
    }

    /// Round-5 corollary: signal_no_speech_through after signal_eof
    /// returns AfterEof (consistent with push_*).
    #[test]
    fn signal_no_speech_through_after_eof_returns_after_eof() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        t.signal_eof().unwrap();
        let r = t.signal_no_speech_through(2000);
        assert!(matches!(r, Err(TranscriberError::AfterEof)));
    }

    /// Codex round-2 finding [high]: VAD must not reference audio
    /// past the buffer's high-water mark. Without the guard, the
    /// segment is accumulated, signal_eof flushes it, dispatch's
    /// promote calls buffer.extract on a range that doesn't exist,
    /// and the program panics.
    #[test]
    fn vad_segment_past_buffered_audio_returns_typed_error() {
        let mut t = fresh();
        t.push_samples(ts(0), &[0.0; 1000]).unwrap();
        // VAD segment claims audio out to sample 2000, but buffer
        // only has 1000 samples.
        let r = t.push_vad_segment(VadSegment::new(0, 2000));
        assert!(
            matches!(r, Err(TranscriberError::VadAheadOfAudio { vad_end: 2000, buffered: 1000 })),
            "expected VadAheadOfAudio, got {:?}",
            r
        );
        // Critically: signal_eof must now NOT panic — the segment
        // was rejected, the cut accumulator is empty.
        t.signal_eof().unwrap();
        assert!(t.is_idle());
    }
}
