//! Dispatch state machine — per-chunk lifecycle, in-order emission.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::sync::Arc;
use alloc::vec::Vec;

use mediatime::TimeRange;

use crate::core::buffer::SampleBuffer;
use crate::core::command::{AsrParams, AsrResult, Command};
use crate::core::cut::{MergedChunk, SampleRange, SubOrigin, SubRange};
use crate::core::event::Event;
use crate::core::transcriber::LanguagePolicy;
use crate::types::{ChunkId, Lang, Transcript, TranscriberError, WorkFailure};

/// Pick the most-frequent language in `observations`, with
/// first-occurrence tiebreaking among ties.
///
/// O(n²) is fine — `observations.len()` equals the
/// `LanguagePolicy::AutoLockAfter(n)` threshold and is bounded by a
/// small constant (typically 1–10). Avoids pulling in a HashMap on
/// no_std for what's essentially a trivial mode computation.
///
/// Panics if `observations` is empty (caller guards against that).
fn mode_with_first_occurrence_tiebreak(observations: &[Lang]) -> Lang {
    let mut best: Option<(&Lang, usize)> = None;
    for (idx, lang) in observations.iter().enumerate() {
        // Skip if we've already counted this language at an earlier
        // index — first occurrence is the canonical tiebreaker, so
        // we evaluate each unique language exactly once.
        if observations[..idx].iter().any(|l| l == lang) {
            continue;
        }
        let count = observations.iter().filter(|l| *l == lang).count();
        match best {
            None => best = Some((lang, count)),
            Some((_, b_count)) if count > b_count => best = Some((lang, count)),
            _ => {} // count <= b_count: keep the earlier-occurring one
        }
    }
    best.expect("observations must not be empty").0.clone()
}

#[allow(dead_code)] // alignment fields land in Plan C
#[derive(Debug)]
pub(crate) enum ChunkPhase {
    AwaitingAsr,
    AwaitingAlignment,
    Ready { transcript: Transcript },
    FailedReady { failure: WorkFailure },
}

#[derive(Debug)]
pub(crate) struct ChunkRecord {
    pub chunk_id: ChunkId,
    pub range: TimeRange,
    pub samples: Arc<[f32]>,
    pub sample_range: SampleRange,
    pub sub_segments: Vec<TimeRange>,
    #[allow(dead_code)] // used by alignment in Plan C
    pub sub_origins: Vec<SubOrigin>,
    pub phase: ChunkPhase,
    pub asr_result: Option<AsrResult>,
}

pub(crate) struct Dispatch {
    pub cut_pending: VecDeque<(ChunkId, MergedChunk)>,
    pub in_flight: BTreeMap<ChunkId, ChunkRecord>,
    pub next_emit_chunk_id: ChunkId,
    pub pending_commands: VecDeque<Command>,
    pub pending_events: VecDeque<Event>,
    pub word_alignment: bool,
    pub max_in_flight: usize,
    pub asr_params: AsrParams,
    /// Language detection / locking strategy. Applied at promote
    /// time (sets `RunAsr.params.language_hint` based on the policy
    /// + the most recent locked-language detection).
    pub language_policy: LanguagePolicy,
    /// The language to lock subsequent ASR commands to, once a lock
    /// has happened. Independent from `LanguagePolicy::Lock { hint }`,
    /// which is applied directly at promote time without observation.
    /// `None` until either (a) `LanguagePolicy::Lock` is in effect or
    /// (b) `LanguagePolicy::AutoLockAfter(n)` reaches its threshold.
    pub locked_language: Option<Lang>,
    /// First `n` non-empty observations under
    /// `LanguagePolicy::AutoLockAfter(n)`, in ChunkId order. When
    /// this reaches `n` entries, `locked_language` is set to the
    /// most-frequent language in the list (with first-occurrence
    /// tiebreaking among ties — the language that appeared first
    /// in chunk_id order wins).
    ///
    /// Codex round-2 fix: previously this was a `usize` counter
    /// that just stored the last-observed language at threshold.
    /// For `n > 1` that diverged from the spec's "most-frequent"
    /// contract — a noisy `En, En, Zh` sequence would have
    /// locked to Zh.
    pub auto_lock_observations: alloc::vec::Vec<Lang>,
    /// Per-ChunkId resolution status for AutoLockAfter ordering. An
    /// entry's value is `Some(lang)` for a non-empty ASR result and
    /// `None` for either an empty-text result or an ASR-stage
    /// failure. Entries ahead of `auto_lock_cursor` are buffered
    /// here until earlier chunks resolve; the cursor drains them in
    /// chunk_id order via `advance_auto_lock_cursor`.
    ///
    /// Codex round-3 fix: previously observations were appended in
    /// ASR completion order, so out-of-order completion (chunk 1
    /// finishing before chunk 0) race-determined the locked
    /// language. The spec calls for locking on the first non-empty
    /// chunks *in the stream*, not the first to complete on the
    /// runner.
    pub auto_lock_pending: BTreeMap<ChunkId, Option<Lang>>,
    /// Next ChunkId the auto-lock cursor will consider. Advances
    /// monotonically, only moving past a ChunkId once that chunk has
    /// an entry in `auto_lock_pending` (i.e., its ASR stage has
    /// resolved one way or another). Independent from
    /// `next_emit_chunk_id` because the cursor advances on ASR
    /// resolution, not on full chunk readiness — a chunk awaiting
    /// alignment has already produced its language signal.
    pub auto_lock_cursor: ChunkId,
    /// Set true while `restart_at` is draining `cut_pending`. While
    /// true, the promotion guard `in_flight.len() < max_in_flight`
    /// is suspended (per §5.5 invariant 4 exception). Reset to
    /// false at the end of restart_at.
    pub draining_for_restart: bool,
    /// Single-slot undo for the runner's dispatch loop. Set by
    /// `unpoll_command`, consumed by the next `poll_command` (which
    /// returns the parked command first).
    pub parked_command: Option<Command>,
}

impl Dispatch {
    pub(crate) fn new(
        asr_params: AsrParams,
        word_alignment: bool,
        max_in_flight: usize,
        language_policy: LanguagePolicy,
    ) -> Self {
        // For LanguagePolicy::Lock, pre-fill locked_language so the
        // first promotion already applies the hint. Auto and
        // AutoLockAfter both start with no lock; AutoLockAfter
        // populates locked_language after observing n non-empty
        // results in inject_asr_result.
        let locked_language = match &language_policy {
            LanguagePolicy::Lock { hint } => Some(hint.clone()),
            _ => None,
        };
        Self {
            cut_pending: VecDeque::new(),
            in_flight: BTreeMap::new(),
            next_emit_chunk_id: ChunkId::from_raw(0),
            pending_commands: VecDeque::new(),
            pending_events: VecDeque::new(),
            word_alignment,
            max_in_flight,
            asr_params,
            language_policy,
            locked_language,
            auto_lock_observations: alloc::vec::Vec::new(),
            auto_lock_pending: BTreeMap::new(),
            auto_lock_cursor: ChunkId::from_raw(0),
            draining_for_restart: false,
            parked_command: None,
        }
    }

    /// Drain `auto_lock_pending` from `auto_lock_cursor` forward,
    /// appending non-empty observations to `auto_lock_observations`
    /// in ChunkId order. Stops at the first cursor position with no
    /// resolution recorded yet, or as soon as `n` observations have
    /// been collected (then sets `locked_language`).
    fn advance_auto_lock_cursor(&mut self, n: usize) {
        while let Some(entry) = self.auto_lock_pending.remove(&self.auto_lock_cursor) {
            if let Some(lang) = entry {
                self.auto_lock_observations.push(lang);
            }
            self.auto_lock_cursor = ChunkId::from_raw(self.auto_lock_cursor.as_u64() + 1);
            if self.auto_lock_observations.len() >= n {
                self.locked_language = Some(mode_with_first_occurrence_tiebreak(
                    &self.auto_lock_observations,
                ));
                // Drop the buffered tail; nothing past this point
                // contributes to the lock decision.
                self.auto_lock_pending.clear();
                return;
            }
        }
    }

    /// Called by `Transcriber` whenever the cut state machine emits
    /// a `MergedChunk`. Either promotes the chunk to `in_flight`
    /// immediately (and emits a `RunAsr` command) or queues it on
    /// `cut_pending` if `max_in_flight` is saturated.
    pub(crate) fn on_emit(
        &mut self,
        chunk: MergedChunk,
        chunk_id: ChunkId,
        buffer: &SampleBuffer,
    ) {
        if self.draining_for_restart || self.in_flight.len() < self.max_in_flight {
            self.promote(chunk_id, chunk, buffer);
        } else {
            self.cut_pending.push_back((chunk_id, chunk));
        }
    }

    /// Move a chunk from "just produced by Cut" or "pending" to
    /// "in_flight" by extracting its samples and queuing a
    /// `RunAsr` command. Crate-private; the trim path also calls it.
    fn promote(&mut self, chunk_id: ChunkId, chunk: MergedChunk, buffer: &SampleBuffer) {
        let samples = buffer.extract(chunk.range);
        let range = buffer.samples_to_output_range(chunk.range);
        let sub_segments: Vec<TimeRange> = chunk
            .subs
            .iter()
            .map(|s| buffer.samples_to_output_range(s.range))
            .collect();
        let sub_origins: Vec<SubOrigin> = chunk.subs.iter().map(|s| s.origin).collect();

        let record = ChunkRecord {
            chunk_id,
            range,
            samples: samples.clone(),
            sample_range: chunk.range,
            sub_segments,
            sub_origins,
            phase: ChunkPhase::AwaitingAsr,
            asr_result: None,
        };
        self.in_flight.insert(chunk_id, record);

        // Apply LanguagePolicy at promote time: if a language has
        // been locked (via Lock pre-fill or AutoLockAfter
        // observation), set the hint on this chunk's params.
        let mut params = self.asr_params.clone();
        if let Some(locked) = &self.locked_language {
            params.language_hint = Some(locked.clone());
        }

        self.pending_commands.push_back(Command::RunAsr {
            chunk_id,
            samples,
            sample_rate: crate::time::SAMPLE_RATE_HZ,
            params,
        });
    }

    /// Drain pending events to the caller in chunk-id order.
    /// Idempotent / re-entrant: stops when the head of `in_flight`
    /// is not yet `Ready` / `FailedReady`, or when `next_emit_chunk_id`
    /// is past every record in `in_flight`.
    fn flush_in_order_events(&mut self) {
        loop {
            let head_id = self.next_emit_chunk_id;
            let entry = match self.in_flight.get(&head_id) {
                Some(e) => e,
                None => break,
            };
            match &entry.phase {
                ChunkPhase::Ready { .. } | ChunkPhase::FailedReady { .. } => {}
                _ => break,
            }
            let mut record = self.in_flight.remove(&head_id).expect("just got");
            let phase = core::mem::replace(&mut record.phase, ChunkPhase::AwaitingAsr);
            let event = match phase {
                ChunkPhase::Ready { transcript } => Event::Transcript(transcript),
                ChunkPhase::FailedReady { failure } => Event::Error {
                    chunk_id: head_id,
                    error: failure,
                },
                _ => unreachable!("phase guarded above"),
            };
            self.pending_events.push_back(event);
            self.next_emit_chunk_id = ChunkId::from_raw(head_id.as_u64() + 1);
        }
    }

    /// Compute trim's low-water from `cut_pending` only — in-flight
    /// chunks have their audio in their own Arc<[f32]>s and are
    /// decoupled from the live buffer. If `cut_pending` is empty,
    /// the buffer can be trimmed all the way to its high-water
    /// (caller passes `absolute_sample_offset`).
    ///
    /// `cut_accumulator_start` is the start of the chunk currently
    /// being accumulated in the cut state machine, if any. Samples
    /// back to that index are still referenced by an unextracted
    /// partial chunk; trim must not drop them. The caller computes
    /// this via `Cut::pending_start()` and passes it in (Dispatch
    /// has no direct handle on Cut).
    pub(crate) fn low_water_samples(
        &self,
        cut_accumulator_start: Option<u64>,
        fallback_high_water: u64,
    ) -> u64 {
        let cut_pending_min = self.cut_pending.iter().map(|(_, c)| c.range.start).min();
        [cut_pending_min, cut_accumulator_start]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(fallback_high_water)
    }

    /// After an inject_* path, try to land any newly-eligible
    /// in-flight chunks as events, then promote pending chunks if
    /// slots have opened. The caller (`Transcriber`) must invoke
    /// `flush_in_order_events()` then `trim()` in this order on
    /// every inject path (§5.5 invariant 3).
    ///
    /// `cut_accumulator_start` is `Cut::pending_start()` — see
    /// `low_water_samples`.
    pub(crate) fn after_inject(
        &mut self,
        buffer: &mut SampleBuffer,
        cut_accumulator_start: Option<u64>,
    ) {
        self.flush_in_order_events();
        // Trim the buffer to the lowest live-chunk start (the lowest
        // start across cut_pending + the cut accumulator, if any).
        let low = self.low_water_samples(cut_accumulator_start, buffer.absolute_sample_offset());
        buffer.trim_to(low);
        // Promote pending chunks if slots are open.
        while !self.draining_for_restart
            && self.in_flight.len() < self.max_in_flight
            && !self.cut_pending.is_empty()
        {
            let (chunk_id, chunk) = self.cut_pending.pop_front().expect("just checked non-empty");
            self.promote(chunk_id, chunk, buffer);
        }
    }

    /// Inject an ASR result for the given chunk. The dispatch state
    /// machine builds the `Transcript` (with empty `words` if
    /// alignment is off) and either marks the chunk Ready, or — if
    /// alignment is on AND the result has non-empty text —
    /// transitions to AwaitingAlignment and queues a RunAlignment
    /// command. Caller must invoke `after_inject(&mut buffer)` to
    /// flush events and run trim.
    ///
    /// Phase contract: only chunks in `AwaitingAsr` accept an ASR
    /// result. Calling on a chunk in any other phase (e.g., already
    /// `Ready` and waiting in-order behind an earlier chunk, or
    /// `AwaitingAlignment` that should be receiving an alignment
    /// result instead) returns `UnknownChunk` — the in-flight record
    /// is treated as opaque outside its expected phase.
    pub(crate) fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        result: AsrResult,
    ) -> Result<(), TranscriberError> {
        // Phase check via shared borrow first; the borrow drops at
        // the end of this statement so the auto-lock block below
        // can take `&mut self`. Holding a mutable record borrow
        // across `advance_auto_lock_cursor` (a `&mut self` method)
        // is what tripped E0499.
        match self.in_flight.get(&chunk_id) {
            None => return Err(TranscriberError::UnknownChunk(chunk_id)),
            Some(r) if !matches!(r.phase, ChunkPhase::AwaitingAsr) => {
                return Err(TranscriberError::UnknownChunk(chunk_id));
            }
            Some(_) => {}
        }

        // Update LanguagePolicy::AutoLockAfter observations. The
        // cursor advances strictly in ChunkId order so out-of-order
        // ASR completion can't race-determine the locked language —
        // pre-fix code recorded observations on completion, so
        // chunk 5 finishing before chunk 0 could lock against an
        // unrepresentative early sample of the stream. Empty-text
        // results and ASR failures don't add an observation, but
        // they DO advance the cursor so a single empty/failed chunk
        // doesn't block auto-lock forever.
        if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy {
            if self.locked_language.is_none() {
                let entry = if result.text.is_empty() {
                    None
                } else {
                    Some(result.language.clone())
                };
                self.auto_lock_pending.insert(chunk_id, entry);
                let n = *n;
                self.advance_auto_lock_cursor(n);
            }
        }

        let record = self.in_flight.get_mut(&chunk_id).expect("phase-checked above");
        if self.word_alignment && !result.text.is_empty() {
            // Cache only when alignment will consume it. Alignment-off
            // builds the Transcript directly below; caching there
            // would let an unsolicited alignment result later
            // overwrite the Ready transcript.
            record.asr_result = Some(result.clone());
            record.phase = ChunkPhase::AwaitingAlignment;
            self.pending_commands.push_back(Command::RunAlignment {
                chunk_id,
                samples: record.samples.clone(),
                sub_segments: record.sub_segments.clone(),
                text: result.text.clone(),
                language: result.language.clone(),
            });
        } else {
            // Build the Transcript with empty words.
            let transcript = Transcript::new(
                record.range,
                result.language.clone(),
                result.text.clone(),
                Vec::new(),
                result.avg_logprob,
                result.no_speech_prob,
                result.temperature,
                record.sub_segments.clone(),
                chunk_id,
            );
            record.phase = ChunkPhase::Ready { transcript };
        }
        Ok(())
    }

    /// Inject the alignment result for a chunk awaiting alignment.
    /// Consumes the cached `AsrResult` to build the final
    /// `Transcript`.
    ///
    /// Phase contract: only chunks in `AwaitingAlignment` accept an
    /// alignment result. Calling on a chunk in any other phase
    /// returns `UnknownChunk`.
    pub(crate) fn inject_alignment_result(
        &mut self,
        chunk_id: ChunkId,
        result: crate::core::command::AlignmentResult,
    ) -> Result<(), TranscriberError> {
        let record = self.in_flight.get_mut(&chunk_id).ok_or(TranscriberError::UnknownChunk(chunk_id))?;
        if !matches!(record.phase, ChunkPhase::AwaitingAlignment) {
            return Err(TranscriberError::UnknownChunk(chunk_id));
        }
        let asr = record.asr_result.take().ok_or(TranscriberError::UnknownChunk(chunk_id))?;
        let transcript = Transcript::new(
            record.range,
            asr.language.clone(),
            asr.text.clone(),
            result.words,
            asr.avg_logprob,
            asr.no_speech_prob,
            asr.temperature,
            record.sub_segments.clone(),
            chunk_id,
        );
        record.phase = ChunkPhase::Ready { transcript };
        Ok(())
    }

    /// Inject a failure for the given chunk. The chunk transitions
    /// to FailedReady; once `flush_in_order_events` reaches it, an
    /// `Event::Error` is emitted.
    ///
    /// Phase contract: only chunks awaiting a worker (AwaitingAsr or
    /// AwaitingAlignment) accept a failure. Already-resolved chunks
    /// (Ready / FailedReady, blocked behind an earlier chunk's
    /// emission) return `UnknownChunk` rather than letting an
    /// unsolicited failure overwrite their final outcome.
    pub(crate) fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError> {
        // Snapshot the pre-transition phase via a shared borrow so
        // the auto-lock branch below can take `&mut self`.
        let was_awaiting_asr = match self.in_flight.get(&chunk_id) {
            None => return Err(TranscriberError::UnknownChunk(chunk_id)),
            Some(r) => match r.phase {
                ChunkPhase::AwaitingAsr => true,
                ChunkPhase::AwaitingAlignment => false,
                _ => return Err(TranscriberError::UnknownChunk(chunk_id)),
            },
        };

        // An ASR-stage failure produces no language signal but still
        // resolves the chunk, so the auto-lock cursor must advance
        // past it. An alignment-stage failure has already had its
        // language observed at ASR-result time, so we don't touch
        // auto_lock_pending for those.
        if was_awaiting_asr {
            if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy {
                if self.locked_language.is_none() {
                    self.auto_lock_pending.insert(chunk_id, None);
                    let n = *n;
                    self.advance_auto_lock_cursor(n);
                }
            }
        }

        self.in_flight
            .get_mut(&chunk_id)
            .expect("phase-checked above")
            .phase = ChunkPhase::FailedReady { failure };
        Ok(())
    }

    /// Pop the front command for the runner to process. Consults
    /// `parked_command` first (set by `unpoll_command`).
    pub(crate) fn poll_command(&mut self) -> Option<Command> {
        self.parked_command
            .take()
            .or_else(|| self.pending_commands.pop_front())
    }

    /// Park a command at the front of the queue. The next
    /// `poll_command` returns it. Asserts in debug that no command
    /// is already parked (single-slot undo).
    pub(crate) fn unpoll_command(&mut self, cmd: Command) {
        debug_assert!(self.parked_command.is_none(), "unpoll_command called twice without intervening poll_command");
        self.parked_command = Some(cmd);
    }

    /// Pop the front event for the caller.
    pub(crate) fn poll_event(&mut self) -> Option<Event> {
        self.pending_events.pop_front()
    }

    /// True iff every queue is empty: no buffered samples (caller
    /// checks the buffer separately), no pending commands/events,
    /// no in-flight chunks, no cut_pending entries, no parked
    /// command.
    pub(crate) fn is_idle(&self) -> bool {
        self.cut_pending.is_empty()
            && self.in_flight.is_empty()
            && self.pending_commands.is_empty()
            && self.pending_events.is_empty()
            && self.parked_command.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer::SampleBuffer;
    use crate::core::cut::{Cut, MergedChunk, SampleRange, SubOrigin, SubRange};
    use crate::types::{Lang, VadSegment, transcript::for_test as tr};
    use core::num::NonZeroU32;
    use core::time::Duration;
    use mediatime::{Timebase, Timestamp};
    use smol_str::SmolStr;

    fn tb() -> Timebase {
        Timebase::new(1, NonZeroU32::new(48_000).unwrap())
    }

    fn make_buffer_with_samples(n_samples: usize) -> SampleBuffer {
        let mut b = SampleBuffer::new(1_000_000, 3200);
        let samples: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
        b.append(Timestamp::new(0, tb()), &samples).unwrap();
        b
    }

    fn dispatch_default() -> Dispatch {
        Dispatch::new(AsrParams::default(), /* word_alignment = */ false, /* max_in_flight = */ 4, LanguagePolicy::default())
    }

    fn fake_chunk(start: u64, end: u64) -> MergedChunk {
        MergedChunk {
            range: SampleRange::new(start, end),
            subs: alloc::vec![SubRange {
                range: SampleRange::new(start, end),
                origin: SubOrigin::Vad { vad_seq: 0 },
            }],
        }
    }

    fn fake_asr_result(text: &str) -> AsrResult {
        AsrResult {
            text: SmolStr::new(text),
            language: Lang::En,
            avg_logprob: -0.5,
            no_speech_prob: 0.05,
            temperature: 0.0,
        }
    }

    #[test]
    fn out_of_order_completion_emits_in_chunk_id_order() {
        let mut d = dispatch_default();
        let mut b = make_buffer_with_samples(10_000);

        // Issue three chunks: 0, 1, 2.
        d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(2_000, 4_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(4_000, 6_000), ChunkId::from_raw(2), &b);
        // All three issued RunAsr.
        assert_eq!(d.in_flight.len(), 3);
        assert_eq!(d.pending_commands.len(), 3);

        // Resolve out of order: 2, 0, 1.
        d.inject_asr_result(ChunkId::from_raw(2), fake_asr_result("c2")).unwrap();
        d.after_inject(&mut b, None);
        // Chunk 2 is Ready but cannot emit yet (next_emit is 0).
        assert!(d.pending_events.is_empty());

        d.inject_asr_result(ChunkId::from_raw(0), fake_asr_result("c0")).unwrap();
        d.after_inject(&mut b, None);
        // Chunk 0 emitted; chunk 1 still in_flight.
        assert_eq!(d.pending_events.len(), 1);

        d.inject_asr_result(ChunkId::from_raw(1), fake_asr_result("c1")).unwrap();
        d.after_inject(&mut b, None);
        // Chunks 1 and 2 now emit (cascade).
        assert_eq!(d.pending_events.len(), 3);

        // Verify order.
        let ids: Vec<u64> = d.pending_events.iter().map(|e| match e {
            Event::Transcript(t) => t.chunk_id().as_u64(),
            Event::Error { chunk_id, .. } => chunk_id.as_u64(),
        }).collect();
        assert_eq!(ids, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn unknown_chunk_id_returns_error() {
        let mut d = dispatch_default();
        let r = d.inject_asr_result(ChunkId::from_raw(99), fake_asr_result("nope"));
        assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 99));
    }

    #[test]
    fn inject_failure_emits_error_event_in_order() {
        let mut d = dispatch_default();
        let mut b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
        d.inject_failure(
            ChunkId::from_raw(0),
            WorkFailure::AsrFailed {
                kind: crate::types::AsrFailureKind::AllTemperaturesFailed,
                message: "x".into(),
            },
        ).unwrap();
        d.after_inject(&mut b, None);
        assert_eq!(d.pending_events.len(), 1);
        match d.pending_events.front().unwrap() {
            Event::Error { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
            _ => panic!("expected Error event"),
        }
    }

    #[test]
    fn cut_pending_holds_chunks_when_max_in_flight_reached() {
        let mut d = Dispatch::new(AsrParams::default(), false, 2, LanguagePolicy::default());
        let mut b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
        d.on_emit(fake_chunk(3_000, 4_000), ChunkId::from_raw(3), &b);
        assert_eq!(d.in_flight.len(), 2);
        assert_eq!(d.cut_pending.len(), 2);
        assert_eq!(d.pending_commands.len(), 2,
            "only first two chunks issued RunAsr; pending chunks have no commands yet");
    }

    #[test]
    fn unpoll_command_parks_for_next_poll() {
        let mut d = dispatch_default();
        let mut b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        let cmd = d.poll_command().unwrap();
        d.unpoll_command(cmd);
        let cmd_again = d.poll_command().unwrap();
        match cmd_again {
            Command::RunAsr { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
            _ => panic!("expected RunAsr"),
        }
    }

    /// Covers the dequeue half of §5.5 invariant 4: when an in-flight
    /// chunk completes and `after_inject` runs, a chunk that was
    /// queued in `cut_pending` because `max_in_flight` was full
    /// must be promoted (audio extracted, RunAsr command queued)
    /// in the same call.
    #[test]
    fn cut_pending_promotes_on_slot_open() {
        let mut d = Dispatch::new(AsrParams::default(), false, 2, LanguagePolicy::default());
        let mut b = make_buffer_with_samples(10_000);

        // Fill in_flight (cap=2) and queue one in cut_pending.
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
        assert_eq!(d.in_flight.len(), 2);
        assert_eq!(d.cut_pending.len(), 1);
        assert_eq!(d.pending_commands.len(), 2);

        // Resolve chunk 0; after_inject should both flush its event
        // AND promote chunk 2 from cut_pending into in_flight,
        // emitting a third RunAsr command.
        d.inject_asr_result(ChunkId::from_raw(0), fake_asr_result("c0")).unwrap();
        d.after_inject(&mut b, None);

        assert_eq!(d.cut_pending.len(), 0, "cut_pending should be drained");
        assert_eq!(d.in_flight.len(), 2,
            "chunk 0 emitted (out), chunk 2 promoted (in) — net stays at 2");
        assert!(d.in_flight.contains_key(&ChunkId::from_raw(1)));
        assert!(d.in_flight.contains_key(&ChunkId::from_raw(2)));
        assert_eq!(d.pending_commands.len(), 3,
            "third RunAsr was issued for chunk 2 on promotion");
        assert_eq!(d.pending_events.len(), 1, "chunk 0's Transcript emitted");
    }

    /// Codex round-1 finding [high]: `LanguagePolicy::Lock { hint }`
    /// must apply the hint to every emitted RunAsr command.
    #[test]
    fn language_policy_lock_applies_hint_to_first_chunk() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            4,
            LanguagePolicy::Lock { hint: Lang::Zh },
        );
        let b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        let cmd = d.poll_command().unwrap();
        match cmd {
            Command::RunAsr { params, .. } => {
                assert_eq!(params.language_hint, Some(Lang::Zh),
                    "Lock {{ hint: Zh }} must set language_hint on every RunAsr");
            }
            _ => panic!("expected RunAsr"),
        }
    }

    /// Codex round-1 finding [high]: `LanguagePolicy::AutoLockAfter(1)`
    /// must lock the language after observing the first non-empty
    /// ASR result, then apply that hint to all subsequent RunAsr
    /// commands.
    #[test]
    fn language_policy_auto_lock_after_one_locks_on_first_observation() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            4,
            LanguagePolicy::AutoLockAfter(1),
        );
        let mut b = make_buffer_with_samples(10_000);

        // First chunk: no lock yet — hint is None.
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        let cmd = d.poll_command().unwrap();
        match cmd {
            Command::RunAsr { params, .. } => {
                assert_eq!(params.language_hint, None,
                    "first chunk under AutoLockAfter(1) has no hint yet");
            }
            _ => panic!("expected RunAsr"),
        }

        // Inject ASR result with detected language Zh — this is
        // the first non-empty observation.
        d.inject_asr_result(
            ChunkId::from_raw(0),
            AsrResult {
                text: SmolStr::new("你好"),
                language: Lang::Zh,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        // Pretend Cut is still accumulating starting at sample 1_000
        // (the start of the second chunk we're about to emit). This
        // keeps samples 1_000.. alive in the buffer past the
        // post-inject trim, so the next on_emit's extract succeeds.
        d.after_inject(&mut b, Some(1_000));

        // Second chunk: hint should now be locked to Zh.
        d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
        // poll_command pops chunk 0's parked stuff first (none here)
        // then chunk 1's RunAsr.
        let cmd = d.pending_commands.pop_back().unwrap();
        match cmd {
            Command::RunAsr { chunk_id, params, .. } => {
                assert_eq!(chunk_id.as_u64(), 1);
                assert_eq!(params.language_hint, Some(Lang::Zh),
                    "second chunk hint must be locked to first detection");
            }
            _ => panic!("expected RunAsr"),
        }
    }

    /// Codex round-1 finding [medium]: a duplicate `inject_asr_result`
    /// on a chunk that's already `Ready` (waiting in-order) must be
    /// rejected — otherwise the second call could overwrite the
    /// final transcript.
    #[test]
    fn inject_asr_on_ready_phase_returns_unknown_chunk() {
        let mut d = dispatch_default();
        let mut b = make_buffer_with_samples(10_000);
        // Two chunks; resolve the second first so the first stays
        // in_flight as a Ready chunk while the cursor is at 0.
        // Actually for a single-chunk repro we can resolve and then
        // try to re-inject — the chunk is removed from in_flight
        // immediately after flush_in_order_events emits its Transcript,
        // so we need to keep it Ready by leaving an earlier chunk
        // unresolved.
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
        // Resolve chunk 1 first — it transitions to Ready but stays
        // in_flight because the cursor is at 0.
        d.inject_asr_result(ChunkId::from_raw(1), fake_asr_result("c1")).unwrap();
        // Now chunk 1's phase is Ready. Duplicate inject must be rejected.
        let r = d.inject_asr_result(ChunkId::from_raw(1), fake_asr_result("c1-dup"));
        assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 1));
    }

    /// Codex round-1 finding [medium]: alignment results aimed at a
    /// chunk in `AwaitingAsr` (not `AwaitingAlignment`) must be
    /// rejected — otherwise an unsolicited alignment result could
    /// overwrite a still-in-flight chunk.
    #[test]
    fn inject_alignment_on_awaiting_asr_returns_unknown_chunk() {
        let mut d = dispatch_default();
        let b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        // Phase is AwaitingAsr.
        let r = d.inject_alignment_result(
            ChunkId::from_raw(0),
            crate::core::command::AlignmentResult { words: alloc::vec::Vec::new() },
        );
        assert!(matches!(r, Err(TranscriberError::UnknownChunk(_))));
    }

    /// Codex round-1 finding [medium]: a failure aimed at a chunk
    /// already in `Ready` phase must be rejected — it must not
    /// retroactively turn a successful Transcript into an Error.
    #[test]
    fn inject_failure_on_ready_returns_unknown_chunk() {
        let mut d = dispatch_default();
        let b = make_buffer_with_samples(10_000);
        d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
        // Resolve chunk 1 to Ready (waiting on chunk 0 in-order).
        d.inject_asr_result(ChunkId::from_raw(1), fake_asr_result("c1")).unwrap();
        let r = d.inject_failure(
            ChunkId::from_raw(1),
            WorkFailure::AsrFailed {
                kind: crate::types::AsrFailureKind::AllTemperaturesFailed,
                message: alloc::string::String::from("late failure"),
            },
        );
        assert!(matches!(r, Err(TranscriberError::UnknownChunk(_))));
    }

    /// Codex round-2 finding [medium]: `AutoLockAfter(n)` must lock
    /// to the most-frequent observed language, not the last
    /// observation. With n=3 and observations [En, En, Zh], the
    /// pre-fix code locked to Zh (last seen); the spec says En
    /// (most frequent). First-occurrence tiebreaking handles
    /// equally-frequent languages deterministically.
    #[test]
    fn auto_lock_after_three_locks_to_most_frequent() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            8,
            LanguagePolicy::AutoLockAfter(3),
        );
        let mut b = make_buffer_with_samples(20_000);

        // Three chunks, observations: En, En, Zh.
        for (i, lang) in [Lang::En, Lang::En, Lang::Zh].iter().enumerate() {
            let s = (i as u64) * 1_000;
            d.on_emit(fake_chunk(s, s + 500), ChunkId::from_raw(i as u64), &b);
            d.inject_asr_result(
                ChunkId::from_raw(i as u64),
                AsrResult {
                    text: SmolStr::new("text"),
                    language: lang.clone(),
                    avg_logprob: -0.5,
                    no_speech_prob: 0.05,
                    temperature: 0.0,
                },
            ).unwrap();
            // Pretend Cut still has a future chunk accumulating
            // so trim doesn't drop chunk samples we haven't yet
            // emitted.
            // Pass Some(0) to pin the trim low-water at the buffer
            // start, keeping all chunks' samples alive for the
            // duration of the test. This test exercises language
            // policy, not trim behavior.
            d.after_inject(&mut b, Some(0));
        }

        // After 3 observations, locked_language should be En —
        // the mode of [En, En, Zh].
        assert_eq!(
            d.locked_language,
            Some(Lang::En),
            "AutoLockAfter(3) must lock to the most-frequent language (En), not the last (Zh)"
        );

        // Fourth chunk should now have En as its hint.
        d.on_emit(fake_chunk(3_000, 3_500), ChunkId::from_raw(3), &b);
        let cmd = d.pending_commands.pop_back().unwrap();
        match cmd {
            Command::RunAsr { params, chunk_id, .. } => {
                assert_eq!(chunk_id.as_u64(), 3);
                assert_eq!(params.language_hint, Some(Lang::En),
                    "post-lock chunks must carry the locked language");
            }
            _ => panic!("expected RunAsr"),
        }
    }

    /// Codex round-3 finding [high]: AutoLockAfter must order
    /// observations by ChunkId, not by ASR completion order. With
    /// max_in_flight > 1, chunk 1 can finish before chunk 0; the
    /// pre-fix code recorded observations in completion order,
    /// race-determining the lock based on which worker happened to
    /// finish first. Reproduction: chunk 0 = En, chunk 1 = Zh, ASR
    /// for chunk 1 arrives first. With first-occurrence tiebreaking,
    /// chunk_id order [En, Zh] picks En; completion order [Zh, En]
    /// picks Zh — pre-fix would have locked Zh.
    #[test]
    fn auto_lock_after_orders_by_chunk_id_not_completion() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            4,
            LanguagePolicy::AutoLockAfter(2),
        );
        let mut b = make_buffer_with_samples(10_000);

        d.on_emit(fake_chunk(0, 500), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(500, 1_000), ChunkId::from_raw(1), &b);

        // Chunk 1's ASR result arrives FIRST (out of order). Lock
        // must NOT advance — chunk 0 is still in flight.
        d.inject_asr_result(
            ChunkId::from_raw(1),
            AsrResult {
                text: SmolStr::new("zh"),
                language: Lang::Zh,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));
        assert_eq!(d.locked_language, None,
            "auto-lock must not advance until chunk 0 resolves, regardless of completion order");

        // Chunk 0's ASR result arrives — En. Now both have resolved
        // and the cursor can advance through both in chunk_id order:
        // observations = [En, Zh] → mode picks En (first occurrence
        // wins on ties).
        d.inject_asr_result(
            ChunkId::from_raw(0),
            AsrResult {
                text: SmolStr::new("en"),
                language: Lang::En,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));

        assert_eq!(d.locked_language, Some(Lang::En),
            "auto-lock must observe in chunk_id order: chunk 0 = En first, then chunk 1 = Zh");
    }

    /// Round-3 corollary: an ASR failure on AwaitingAsr must advance
    /// the auto-lock cursor without contributing an observation.
    /// Otherwise a single failed chunk would block auto-lock forever.
    /// Reproduction: chunk 0 fails ASR; chunks 1 and 2 succeed in
    /// English. AutoLockAfter(2) must still lock to En.
    #[test]
    fn auto_lock_after_skips_failed_chunks() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            4,
            LanguagePolicy::AutoLockAfter(2),
        );
        let mut b = make_buffer_with_samples(10_000);

        d.on_emit(fake_chunk(0, 500), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(500, 1_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(1_000, 1_500), ChunkId::from_raw(2), &b);

        // Chunk 0 fails ASR.
        d.inject_failure(
            ChunkId::from_raw(0),
            WorkFailure::AsrFailed {
                kind: crate::types::AsrFailureKind::AllTemperaturesFailed,
                message: "fail".into(),
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));
        assert_eq!(d.locked_language, None, "single failed chunk produced no observation yet");

        // Chunks 1 and 2 succeed in English. After both land, cursor
        // advances through 0 (failed, skipped) → 1 (En) → 2 (En) and
        // locks once observations.len() reaches 2.
        d.inject_asr_result(
            ChunkId::from_raw(1),
            AsrResult {
                text: SmolStr::new("hello"),
                language: Lang::En,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));
        d.inject_asr_result(
            ChunkId::from_raw(2),
            AsrResult {
                text: SmolStr::new("world"),
                language: Lang::En,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));

        assert_eq!(d.locked_language, Some(Lang::En),
            "auto-lock must skip failed chunk 0 and lock to En from chunks 1 + 2");
    }

    /// Round-3 corollary: an empty-text ASR result must advance the
    /// cursor without contributing an observation, even when arriving
    /// out of order. Reproduction: chunks 0–2 promoted; chunk 1 = En,
    /// chunk 0 = empty silent chunk, chunk 2 = En. AutoLockAfter(2)
    /// must lock on En after chunk 2 resolves.
    #[test]
    fn auto_lock_after_skips_empty_chunks_in_chunk_id_order() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            4,
            LanguagePolicy::AutoLockAfter(2),
        );
        let mut b = make_buffer_with_samples(10_000);

        d.on_emit(fake_chunk(0, 500), ChunkId::from_raw(0), &b);
        d.on_emit(fake_chunk(500, 1_000), ChunkId::from_raw(1), &b);
        d.on_emit(fake_chunk(1_000, 1_500), ChunkId::from_raw(2), &b);

        // Chunk 1 (En) lands first (out of order).
        d.inject_asr_result(
            ChunkId::from_raw(1),
            AsrResult {
                text: SmolStr::new("hello"),
                language: Lang::En,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));
        assert_eq!(d.locked_language, None);

        // Chunk 0 (empty) — the cursor advances to 1, picks up En.
        d.inject_asr_result(
            ChunkId::from_raw(0),
            AsrResult {
                text: SmolStr::new(""),
                language: Lang::En,
                avg_logprob: -1.0,
                no_speech_prob: 0.95,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));
        assert_eq!(d.locked_language, None,
            "only chunk 1 contributed; need a second non-empty observation");

        // Chunk 2 (En) — second observation lands; lock to En.
        d.inject_asr_result(
            ChunkId::from_raw(2),
            AsrResult {
                text: SmolStr::new("world"),
                language: Lang::En,
                avg_logprob: -0.5,
                no_speech_prob: 0.05,
                temperature: 0.0,
            },
        ).unwrap();
        d.after_inject(&mut b, Some(0));
        assert_eq!(d.locked_language, Some(Lang::En));
    }

    /// Tiebreaking: with n=2 and [En, Zh] (each observed once), the
    /// first-occurrence rule picks En.
    #[test]
    fn auto_lock_after_two_first_occurrence_tiebreak() {
        let mut d = Dispatch::new(
            AsrParams::default(),
            false,
            4,
            LanguagePolicy::AutoLockAfter(2),
        );
        let mut b = make_buffer_with_samples(10_000);

        for (i, lang) in [Lang::En, Lang::Zh].iter().enumerate() {
            let s = (i as u64) * 500;
            d.on_emit(fake_chunk(s, s + 250), ChunkId::from_raw(i as u64), &b);
            d.inject_asr_result(
                ChunkId::from_raw(i as u64),
                AsrResult {
                    text: SmolStr::new("text"),
                    language: lang.clone(),
                    avg_logprob: -0.5,
                    no_speech_prob: 0.05,
                    temperature: 0.0,
                },
            ).unwrap();
            // Pass Some(0) to pin trim at the buffer start.
            d.after_inject(&mut b, Some(0));
        }

        assert_eq!(
            d.locked_language,
            Some(Lang::En),
            "first-occurrence tiebreaking picks En over Zh when each appears once"
        );
    }
}
