//! Dispatch state machine — per-chunk lifecycle, in-order emission.

use std::{
  collections::{BTreeMap, VecDeque},
  sync::Arc,
};

use mediatime::TimeRange;

use crate::{
  core::{
    buffer::SampleBuffer,
    command::{AsrParams, AsrResult, Command},
    cut::{MergedChunk, SampleRange, SubOrigin},
    event::Event,
    transcriber::LanguagePolicy,
  },
  types::{ChunkId, Lang, TranscriberError, Transcript, WorkFailure},
};

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

#[allow(dead_code)] // alignment fields land in alignment feature
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
  /// Sub-VAD-segments in stream-coordinate 16 kHz sample indices,
  /// preserved alongside the output-timebase form so the alignment
  /// worker can build the silence mask in chunk-local space.
  /// Each `(start, end)` is half-open in 16 kHz stream samples.
  #[cfg(feature = "alignment")]
  #[allow(dead_code)] // exposed via Dispatch::chunk_sub_segments_samples
  pub sub_segments_samples: Vec<(u64, u64)>,
  /// Output timebase snapshot, captured at chunk-extract time.
  /// Held alongside [`Self::base_pts_out_anchor`] so the
  /// runner's alignment dispatch can rebuild a
  /// `samples_to_output_range` closure for *this* chunk's
  /// epoch — necessary because `Transcriber::handle_restart` resets
  /// the live buffer's anchor while in-flight chunks survive,
  /// so a fresh closure built post-restart would map this
  /// chunk's pre-restart sample indices through the wrong PTS
  /// origin.
  #[cfg(feature = "alignment")]
  pub output_tb: mediatime::Timebase,
  /// PTS-anchor snapshot at stream-zero in `output_tb`,
  /// captured at chunk-extract time. See
  /// [`Self::output_tb`] for the rationale.
  #[cfg(feature = "alignment")]
  pub base_pts_out_anchor: i64,
  #[allow(dead_code)] // used by alignment feature
  pub sub_origins: Vec<SubOrigin>,
  pub phase: ChunkPhase,
  pub asr_result: Option<AsrResult>,
}

/// A chunk whose audio has been extracted from the live buffer and
/// whose output-timebase ranges have been computed, but which has
/// not yet been promoted to `in_flight` (no `Asr` command issued
/// yet). `cut_pending` entries are stored as `ExtractedChunk` so
/// they survive `handle_restart`'s buffer reset without needing the old
/// `draining_for_restart` bypass — and so the AutoLockAfter
/// observation-window gate is preserved during recovery.
#[derive(Debug)]
pub(crate) struct ExtractedChunk {
  pub chunk_id: ChunkId,
  pub samples: Arc<[f32]>,
  pub sample_range: SampleRange,
  pub range: TimeRange,
  pub sub_segments: Vec<TimeRange>,
  /// Sub-VAD-segments in stream-coordinate 16 kHz sample indices.
  /// Preserved alongside the output-timebase `sub_segments` so the
  /// runner's alignment dispatch can rebuild chunk-local sample
  /// indices for the aligner's silence mask.
  #[cfg(feature = "alignment")]
  pub sub_segments_samples: Vec<(u64, u64)>,
  /// Output timebase snapshot captured at extract time. Promoted
  /// onto [`ChunkRecord::output_tb`] so the runner's alignment
  /// dispatch can rebuild a per-chunk
  /// `samples_to_output_range` closure that survives a later
  /// `handle_restart`.
  #[cfg(feature = "alignment")]
  pub output_tb: mediatime::Timebase,
  /// PTS anchor at stream-zero, captured at extract time. See
  /// [`Self::output_tb`] for the rationale.
  #[cfg(feature = "alignment")]
  pub base_pts_out_anchor: i64,
  pub sub_origins: Vec<SubOrigin>,
  /// Per-packet `AsrParamsOverride` snapshot captured at the
  /// moment this chunk was extracted from the live buffer. The
  /// runner stamps the dispatch's `current_override` here so a
  /// `Asr` command emitted at promote time (which can happen
  /// in a different `process_packet` call than the one that
  /// pushed the audio) carries the override that was active at
  /// chunk-creation time. Without this snapshot the runner used
  /// to merge the *current* override into every dispatched
  /// command, which corrupted parked/deferred commands with the
  /// wrong packet's params.
  pub override_at_creation: Option<crate::core::AsrParamsOverride>,
}

impl ExtractedChunk {
  /// Pull a chunk's audio out of the live buffer and compute its
  /// output-timebase ranges. Crate-private; used by `Dispatch::on_emit`
  /// at the moment a `MergedChunk` is produced.
  ///
  /// `asr_params_override` is the dispatch's `current_override`
  /// snapshot at extract time — see `override_at_creation`.
  pub(crate) fn extract_from(
    chunk_id: ChunkId,
    chunk: MergedChunk,
    buffer: &SampleBuffer,
    asr_params_override: Option<crate::core::AsrParamsOverride>,
  ) -> Self {
    let samples = buffer.extract(chunk.range);
    let range = buffer.samples_to_output_range(chunk.range);
    let sub_segments: Vec<TimeRange> = chunk
      .subs
      .iter()
      .map(|s| buffer.samples_to_output_range(s.range))
      .collect();
    #[cfg(feature = "alignment")]
    let sub_segments_samples: Vec<(u64, u64)> = chunk
      .subs
      .iter()
      .map(|s| (s.range.start, s.range.end))
      .collect();
    let sub_origins: Vec<SubOrigin> = chunk.subs.iter().map(|s| s.origin).collect();
    // Capture the output timebase + PTS anchor *now*, before
    // any later `handle_restart` shifts the buffer onto a new
    // epoch. Promoted to `ChunkRecord` at promote-time and
    // consulted at alignment-dispatch time.
    #[cfg(feature = "alignment")]
    let output_tb = buffer
      .output_timebase()
      .expect("output timebase established by first push (extract_from runs after push)");
    #[cfg(feature = "alignment")]
    let base_pts_out_anchor = buffer.base_pts_out_anchor();
    Self {
      chunk_id,
      samples,
      sample_range: chunk.range,
      range,
      sub_segments,
      #[cfg(feature = "alignment")]
      sub_segments_samples,
      #[cfg(feature = "alignment")]
      output_tb,
      #[cfg(feature = "alignment")]
      base_pts_out_anchor,
      sub_origins,
      override_at_creation: asr_params_override,
    }
  }

  /// Stream-coordinate first 16 kHz sample index of this chunk's
  /// audio. Used by the alignment worker to map wav2vec2 frame
  /// indices back to stream sample positions.
  ///
  /// `SampleRange` is half-open and stream-relative, so
  /// `sample_range.start` is exactly the chunk's first sample
  /// index since stream zero.
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_first_sample_in_stream(&self) -> u64 {
    self.sample_range.start
  }
}

pub(crate) struct Dispatch {
  /// Chunks emitted by Cut that haven't yet been promoted to
  /// `in_flight`. Stored as `ExtractedChunk` (audio already
  /// pulled from the live buffer) so they survive `handle_restart`'s
  /// buffer reset without bypassing the AutoLockAfter gate.
  pub cut_pending: VecDeque<ExtractedChunk>,
  pub in_flight: BTreeMap<ChunkId, ChunkRecord>,
  pub next_emit_chunk_id: ChunkId,
  pub pending_commands: VecDeque<Command>,
  pub pending_events: VecDeque<Event>,
  pub word_alignment: bool,
  pub max_in_flight: usize,
  pub asr_params: AsrParams,
  /// Language detection / locking strategy. Applied at promote
  /// time (sets `Asr.params.language_hint` based on the policy
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
  /// Previously this was a `usize` counter that just stored the
  /// last-observed language at threshold. For `n > 1` that
  /// diverged from the "most-frequent" contract — a noisy
  /// `En, En, Zh` sequence would have locked to Zh.
  pub auto_lock_observations: Vec<Lang>,
  /// Per-ChunkId resolution status for AutoLockAfter ordering. An
  /// entry's value is `Some(lang)` for a non-empty ASR result and
  /// `None` for either an empty-text result or an ASR-stage
  /// failure. Entries ahead of `auto_lock_cursor` are buffered
  /// here until earlier chunks resolve; the cursor drains them in
  /// chunk_id order via `advance_auto_lock_cursor`.
  ///
  /// Previously observations were appended in ASR completion
  /// order, so out-of-order completion (chunk 1 finishing before
  /// chunk 0) race-determined the locked language. The contract
  /// is to lock on the first non-empty chunks *in the stream*,
  /// not the first to complete on the runner.
  pub auto_lock_pending: BTreeMap<ChunkId, Option<Lang>>,
  /// Next ChunkId the auto-lock cursor will consider. Advances
  /// monotonically, only moving past a ChunkId once that chunk has
  /// an entry in `auto_lock_pending` (i.e., its ASR stage has
  /// resolved one way or another). Independent from
  /// `next_emit_chunk_id` because the cursor advances on ASR
  /// resolution, not on full chunk readiness — a chunk awaiting
  /// alignment has already produced its language signal.
  pub auto_lock_cursor: ChunkId,
  /// Single-slot undo for the runner's dispatch loop. Set by
  /// `unpoll_command`, consumed by the next `poll_command` (which
  /// returns the parked command first).
  pub parked_command: Option<Command>,
  /// Per-packet `AsrParamsOverride` the runner has stamped on
  /// the dispatch for the duration of the current
  /// `process_packet` call. `extract_from` reads this and
  /// snapshots it onto each newly-created `ExtractedChunk` —
  /// chunks queued in `cut_pending` therefore remember the
  /// override that was active when their audio was pushed, even
  /// if they are promoted during a later `process_packet`. The
  /// runner sets this before pushing audio and clears it on
  /// exit; the dispatch never reads it after extract_from.
  pub current_override: Option<crate::core::AsrParamsOverride>,
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
    // results in handle_asr.
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
      auto_lock_observations: Vec::new(),
      auto_lock_pending: BTreeMap::new(),
      auto_lock_cursor: ChunkId::from_raw(0),
      parked_command: None,
      current_override: None,
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
  /// a `MergedChunk`. Always pre-extracts the chunk's audio (so it
  /// survives later `handle_restart` buffer resets), then either
  /// promotes the chunk to `in_flight` immediately (and emits a
  /// `Asr` command) or queues it on `cut_pending` if the
  /// effective cap is saturated.
  ///
  /// The chunk arrives with its `override_at_start` already
  /// populated by the cut state machine (snapshotted when this
  /// chunk's accumulation began, NOT now). We forward that
  /// snapshot rather than reading `self.current_override` —
  /// otherwise a chunk whose audio was pushed under packet A's
  /// override but whose VAD-driven close happened in packet B
  /// would silently get B's override (finding).
  pub(crate) fn on_emit(&mut self, chunk: MergedChunk, chunk_id: ChunkId, buffer: &SampleBuffer) {
    let override_at_start = chunk.override_at_start.clone();
    let extracted = ExtractedChunk::extract_from(chunk_id, chunk, buffer, override_at_start);
    if self.can_promote(chunk_id) {
      self.promote_extracted(extracted);
    } else {
      self.cut_pending.push_back(extracted);
    }
  }

  /// Total audio samples currently held in `cut_pending`'s
  /// pre-extracted `Arc<[f32]>`s. Used by `Transcriber` to count
  /// queued audio toward `buffer_cap_samples`. The pre-extraction
  /// design moved cut_pending audio out of the live buffer;
  /// without including this in the Backpressure check, a slow
  /// runner could let cut_pending grow unboundedly every time the
  /// live buffer trimmed and the caller pushed more samples.
  pub(crate) fn cut_pending_audio_samples(&self) -> usize {
    self.cut_pending.iter().map(|c| c.samples.len()).sum()
  }

  /// Decide whether a chunk with id `chunk_id` may be promoted to
  /// `in_flight` right now, given the current `max_in_flight`
  /// budget and (for unlocked AutoLockAfter) the observation
  /// window threshold.
  ///
  /// The gate is a per-ChunkId threshold:
  /// `threshold = auto_lock_cursor + (n - observations.len())`.
  ///
  /// Chunks with id < threshold are observation candidates —
  /// they may run unhinted and contribute to the lock. Chunks
  /// with id >= threshold wait for the lock regardless of
  /// available in-flight slots.
  ///
  /// A simpler in-flight-count cap had a sliding-window bug: when
  /// chunk 0 of an `AutoLockAfter(3)` stream completed and its
  /// in-flight slot freed, chunk 3 was promoted with no hint
  /// even though chunks 1 and 2 might still complete and lock
  /// the language. Tracking the threshold by ChunkId fixes that:
  /// chunk 3 is past the observation window and waits regardless
  /// of slot count.
  ///
  /// Cut_pending entries hold pre-extracted audio, so the gate is
  /// enforced even across `handle_restart`.
  fn can_promote(&self, chunk_id: ChunkId) -> bool {
    if self.in_flight.len() >= self.max_in_flight {
      return false;
    }
    if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy
      && self.locked_language.is_none()
    {
      let slack = n.saturating_sub(self.auto_lock_observations.len());
      let threshold = self.auto_lock_cursor.as_u64() + slack as u64;
      return chunk_id.as_u64() < threshold;
    }
    true
  }

  /// Move a pre-extracted chunk to `in_flight` and queue its
  /// `Asr` command. Applies the locked language hint if one
  /// has been established, then layers the per-packet override
  /// captured on the chunk at extract time. Crate-private; called
  /// by `on_emit` and by `after_inject`'s post-resolve promotion
  /// loop.
  ///
  /// Param precedence (default → locked → override): the runtime
  /// override merges last. The crucial detail is that
  /// `ext.override_at_creation` is the override that was active
  /// when *this chunk* was extracted, so a chunk parked or held
  /// in `cut_pending` always carries its own override — it can't
  /// inherit a later packet's override.
  fn promote_extracted(&mut self, ext: ExtractedChunk) {
    let mut params = self.asr_params.clone();
    if let Some(locked) = &self.locked_language {
      params.set_language_hint(Some(locked.clone()));
    }
    if let Some(ovr) = &ext.override_at_creation {
      params = ovr.apply_to(&params);
    }

    let chunk_id = ext.chunk_id;
    let samples = ext.samples; // moved into command + record (clone for command)
    let record = ChunkRecord {
      chunk_id,
      range: ext.range,
      samples: samples.clone(),
      sample_range: ext.sample_range,
      sub_segments: ext.sub_segments,
      #[cfg(feature = "alignment")]
      sub_segments_samples: ext.sub_segments_samples,
      #[cfg(feature = "alignment")]
      output_tb: ext.output_tb,
      #[cfg(feature = "alignment")]
      base_pts_out_anchor: ext.base_pts_out_anchor,
      sub_origins: ext.sub_origins,
      phase: ChunkPhase::AwaitingAsr,
      asr_result: None,
    };
    self.in_flight.insert(chunk_id, record);

    self.pending_commands.push_back(Command::Asr {
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

  /// Compute trim's low-water. Both `in_flight` chunks and
  /// `cut_pending` chunks hold their own `Arc<[f32]>` audio
  /// (extracted at emit time), so neither pins the live buffer.
  /// The only constraint from the live audio side is the cut
  /// accumulator: samples back to its start are still referenced
  /// by an unextracted partial chunk.
  ///
  /// `cut_accumulator_start` is `Cut::pending_start()`. If it's
  /// `None` (no chunk accumulating), the trim falls back to
  /// `safe_trim_high_water` — usually the caller's VAD analysis
  /// watermark. A previous fallback to the buffer's absolute
  /// high-water mark dropped audio past any unanalyzed VAD tail.
  /// With the watermark as the upper bound, trim respects "VAD
  /// hasn't analyzed past here yet, don't drop the audio".
  pub(crate) fn low_water_samples(
    &self,
    cut_accumulator_start: Option<u64>,
    safe_trim_high_water: u64,
  ) -> u64 {
    cut_accumulator_start.unwrap_or(safe_trim_high_water)
  }

  /// After an inject_* path, try to land any newly-eligible
  /// in-flight chunks as events, then promote pending chunks if
  /// slots have opened. The caller (`Transcriber`) must invoke
  /// `flush_in_order_events()` then `trim()` in this order on
  /// every inject path.
  ///
  /// `cut_accumulator_start` is `Cut::pending_start()` — see
  /// `low_water_samples`.
  ///
  /// `safe_trim_high_water` is the upper bound on trim: usually
  /// the caller's VAD analysis watermark (`vad_watermark`).
  /// Passing `buffer.absolute_sample_offset()` is only safe in
  /// `handle_eof` paths where the stream is ending and audio
  /// past the watermark won't be analyzed.
  pub(crate) fn after_inject(
    &mut self,
    buffer: &mut SampleBuffer,
    cut_accumulator_start: Option<u64>,
    safe_trim_high_water: u64,
  ) {
    self.flush_in_order_events();
    let low = self.low_water_samples(cut_accumulator_start, safe_trim_high_water);
    buffer.trim_to(low);
    // Promote pending chunks while the gate allows them. The gate
    // is per-ChunkId (auto_lock_cursor + n - observations.len());
    // older slots free up when observations land or the lock fires.
    // Peek the front entry to ask `can_promote(its chunk_id)`; if
    // it's gated, stop (cut_pending is in chunk_id order, so later
    // entries are gated too).
    while let Some(front_id) = self.cut_pending.front().map(|e| e.chunk_id) {
      if !self.can_promote(front_id) {
        break;
      }
      let extracted = self.cut_pending.pop_front().expect("just peeked");
      self.promote_extracted(extracted);
    }
  }

  /// Inject an ASR result for the given chunk. The dispatch state
  /// machine builds the `Transcript` (with empty `words` if
  /// alignment is off) and either marks the chunk Ready, or — if
  /// alignment is on AND the result has non-empty text —
  /// transitions to AwaitingAlignment and queues a Alignment
  /// command. Caller must invoke `after_inject(&mut buffer)` to
  /// flush events and run trim.
  ///
  /// Phase contract: only chunks in `AwaitingAsr` accept an ASR
  /// result. Calling on a chunk in any other phase (e.g., already
  /// `Ready` and waiting in-order behind an earlier chunk, or
  /// `AwaitingAlignment` that should be receiving an alignment
  /// result instead) returns `UnknownChunk` — the in-flight record
  /// is treated as opaque outside its expected phase.
  pub(crate) fn handle_asr(
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
    // earlier code recorded observations on completion, so
    // chunk 5 finishing before chunk 0 could lock against an
    // unrepresentative early sample of the stream. Empty-text
    // results and ASR failures don't add an observation, but
    // they DO advance the cursor so a single empty/failed chunk
    // doesn't block auto-lock forever.
    if let LanguagePolicy::AutoLockAfter(n) = &self.language_policy
      && self.locked_language.is_none()
    {
      let entry = if result.text().is_empty() {
        None
      } else {
        Some(result.language().clone())
      };
      self.auto_lock_pending.insert(chunk_id, entry);
      let n = *n;
      self.advance_auto_lock_cursor(n);
    }

    let record = self
      .in_flight
      .get_mut(&chunk_id)
      .expect("phase-checked above");
    if self.word_alignment && !result.text().is_empty() {
      // Cache only when alignment will consume it. Alignment-off
      // builds the Transcript directly below; caching there
      // would let an unsolicited alignment result later
      // overwrite the Ready transcript.
      record.asr_result = Some(result.clone());
      record.phase = ChunkPhase::AwaitingAlignment;
      self.pending_commands.push_back(Command::Alignment {
        chunk_id,
        samples: record.samples.clone(),
        sub_segments: record.sub_segments.clone(),
        text: result.text().clone(),
        language: result.language().clone(),
        runs: result.runs().to_vec(),
      });
    } else {
      // Build the Transcript with empty words.
      let transcript = Transcript::new(
        record.range,
        result.language().clone(),
        result.text().clone(),
        Vec::new(),
        result.avg_logprob(),
        result.no_speech_prob(),
        result.temperature(),
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
  pub(crate) fn handle_alignment(
    &mut self,
    chunk_id: ChunkId,
    result: crate::core::command::AlignmentResult,
  ) -> Result<(), TranscriberError> {
    let record = self
      .in_flight
      .get_mut(&chunk_id)
      .ok_or(TranscriberError::UnknownChunk(chunk_id))?;
    if !matches!(record.phase, ChunkPhase::AwaitingAlignment) {
      return Err(TranscriberError::UnknownChunk(chunk_id));
    }
    let asr = record
      .asr_result
      .take()
      .ok_or(TranscriberError::UnknownChunk(chunk_id))?;
    let transcript = Transcript::new(
      record.range,
      asr.language().clone(),
      asr.text().clone(),
      result.into_words(),
      asr.avg_logprob(),
      asr.no_speech_prob(),
      asr.temperature(),
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
  pub(crate) fn handle_failure(
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
    if was_awaiting_asr
      && let LanguagePolicy::AutoLockAfter(n) = &self.language_policy
      && self.locked_language.is_none()
    {
      self.auto_lock_pending.insert(chunk_id, None);
      let n = *n;
      self.advance_auto_lock_cursor(n);
    }

    self
      .in_flight
      .get_mut(&chunk_id)
      .expect("phase-checked above")
      .phase = ChunkPhase::FailedReady { failure };
    Ok(())
  }

  /// Pop the front command for the runner to process. Consults
  /// `parked_command` first (set by `unpoll_command`).
  pub(crate) fn poll_command(&mut self) -> Option<Command> {
    self
      .parked_command
      .take()
      .or_else(|| self.pending_commands.pop_front())
  }

  /// Park a command at the front of the queue. The next
  /// `poll_command` returns it. Asserts in debug that no command
  /// is already parked (single-slot undo).
  pub(crate) fn unpoll_command(&mut self, cmd: Command) {
    debug_assert!(
      self.parked_command.is_none(),
      "unpoll_command called twice without intervening poll_command"
    );
    self.parked_command = Some(cmd);
  }

  /// Pop the front event for the caller.
  pub(crate) fn poll_event(&mut self) -> Option<Event> {
    self.pending_events.pop_front()
  }

  /// Stream-coordinate first 16 kHz sample index of the chunk
  /// `chunk_id`, or `None` if the chunk is not in flight. Used by
  /// the runner's alignment dispatch to convert stream-sample
  /// sub_segments into chunk-local space before shipping them to
  /// the alignment worker.
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_first_sample(&self, chunk_id: ChunkId) -> Option<u64> {
    let record = self.in_flight.get(&chunk_id)?;
    Some(record.sample_range.start)
  }

  /// Sub-VAD-segments of the chunk `chunk_id` in stream-coordinate
  /// 16 kHz sample indices, as `(start, end)` pairs. Used by the
  /// runner's alignment dispatch to build the chunk-local
  /// sample-indexed sub_segments the alignment worker consumes
  /// for its silence mask.
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_sub_segments_samples(&self, chunk_id: ChunkId) -> Option<Vec<(u64, u64)>> {
    let record = self.in_flight.get(&chunk_id)?;
    Some(record.sub_segments_samples.clone())
  }

  /// Build the `samples_to_output_range` closure for `chunk_id`
  /// using the chunk's *captured-at-extract-time* `(timebase,
  /// base_pts_out_anchor)` pair, so word ranges land in the
  /// chunk's own PTS epoch even after a `handle_restart` has shifted
  /// the live buffer's anchor.
  ///
  /// Returns `None` if `chunk_id` is not in flight (e.g. already
  /// drained as `Transcript`/`Failed`).
  #[cfg(feature = "alignment")]
  pub(crate) fn chunk_samples_to_output_range_fn(
    &self,
    chunk_id: ChunkId,
  ) -> Option<std::sync::Arc<dyn Fn(u64, u64) -> mediatime::TimeRange + Send + Sync>> {
    let record = self.in_flight.get(&chunk_id)?;
    Some(
      crate::core::buffer::SampleBuffer::samples_to_output_range_fn_at(
        record.output_tb,
        record.base_pts_out_anchor,
      ),
    )
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
  use crate::{
    core::{
      AsrParamsOverride,
      buffer::SampleBuffer,
      cut::{MergedChunk, SampleRange, SubOrigin, SubRange},
    },
    types::{AsrError, AsrFailure, Lang},
  };
  use core::num::NonZeroU32;
  use mediatime::{Timebase, Timestamp};
  use smol_str::SmolStr;

  fn tb() -> Timebase {
    Timebase::new(1, NonZeroU32::new(48_000).unwrap())
  }

  fn make_buffer_with_samples(n_samples: usize) -> SampleBuffer {
    let mut b = SampleBuffer::new(1_000_000, 3200);
    let samples: Vec<f32> = (0..n_samples).map(|i| i as f32).collect();
    b.append(Timestamp::new(0, tb()), &samples, 0).unwrap();
    b
  }

  fn dispatch_default() -> Dispatch {
    // Tests using this helper exercise dispatch ordering / phase
    // checks / commands without language-policy involvement;
    // LanguagePolicy::Auto avoids the auto-lock gate that holds
    // back chunks under unlocked AutoLockAfter.
    Dispatch::new(
      AsrParams::default(),
      /* word_alignment = */ false,
      /* max_in_flight = */ 4,
      LanguagePolicy::Auto,
    )
  }

  fn fake_chunk(start: u64, end: u64) -> MergedChunk {
    MergedChunk {
      range: SampleRange::new(start, end),
      subs: vec![SubRange {
        range: SampleRange::new(start, end),
        origin: SubOrigin::Vad { vad_seq: 0 },
      }],
      override_at_start: None,
    }
  }

  fn fake_asr_result(text: &str) -> AsrResult {
    AsrResult::new(SmolStr::new(text), Lang::En, -0.5, 0.05, 0.0)
  }

  #[test]
  fn out_of_order_completion_emits_in_chunk_id_order() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);

    // Issue three chunks: 0, 1, 2.
    d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(2_000, 4_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(4_000, 6_000), ChunkId::from_raw(2), &b);
    // All three issued Asr.
    assert_eq!(d.in_flight.len(), 3);
    assert_eq!(d.pending_commands.len(), 3);

    // Resolve out of order: 2, 0, 1.
    d.handle_asr(ChunkId::from_raw(2), fake_asr_result("c2"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    // Chunk 2 is Ready but cannot emit yet (next_emit is 0).
    assert!(d.pending_events.is_empty());

    d.handle_asr(ChunkId::from_raw(0), fake_asr_result("c0"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    // Chunk 0 emitted; chunk 1 still in_flight.
    assert_eq!(d.pending_events.len(), 1);

    d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    // Chunks 1 and 2 now emit (cascade).
    assert_eq!(d.pending_events.len(), 3);

    // Verify order.
    let ids: Vec<u64> = d
      .pending_events
      .iter()
      .map(|e| match e {
        Event::Transcript(t) => t.chunk_id().as_u64(),
        Event::Error { chunk_id, .. } => chunk_id.as_u64(),
      })
      .collect();
    assert_eq!(ids, vec![0, 1, 2]);
  }

  /// Adversarial regression for the per-packet override binding
  /// fix: a chunk emitted by the cut state machine under
  /// override O1 (snapshotted on `MergedChunk.override_at_start`),
  /// but promoted in a later "process_packet" with override O2
  /// set, must still emit Asr with O1's params.
  ///
  /// expanded this contract from "override at
  /// emit time" to "override at chunk-accumulation-start time" —
  /// the chunk reaches `on_emit` already carrying its origin
  /// override, and dispatch reads from there rather than its
  /// own `current_override`.
  #[test]
  fn extracted_chunk_keeps_override_through_deferred_promote() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      // max_in_flight = 1 forces chunk 1 to wait in cut_pending.
      /* max_in_flight = */
      1,
      LanguagePolicy::Auto,
    );
    let b = make_buffer_with_samples(20_000);

    // "Packet 1" — override O1 sets initial_temperature = 0.7.
    // Both chunks were accumulated while O1 was active so the
    // cut state machine stamped O1 on each `MergedChunk`.
    let o1 = AsrParamsOverride::new().with_initial_temperature(Some(0.7));
    let mut chunk0 = fake_chunk(0, 4_000);
    chunk0.override_at_start = Some(o1.clone());
    let mut chunk1 = fake_chunk(4_000, 8_000);
    chunk1.override_at_start = Some(o1.clone());
    // `current_override` here represents what the runner has
    // stamped on the dispatch for THIS packet — same as O1.
    d.current_override = Some(o1.clone());
    d.on_emit(chunk0, ChunkId::from_raw(0), &b);
    d.on_emit(chunk1, ChunkId::from_raw(1), &b);

    // Chunk 0 promoted (max_in_flight=1); chunk 1 in cut_pending.
    assert_eq!(d.in_flight.len(), 1);
    assert_eq!(d.cut_pending.len(), 1);
    // Chunk 1's snapshot must record O1, not whatever override
    // is current at promote time later.
    let snap = d
      .cut_pending
      .front()
      .unwrap()
      .override_at_creation
      .as_ref()
      .expect("chunk 1 must carry an override snapshot");
    assert_eq!(snap.initial_temperature(), Some(0.7));
    let _ = o1; // captured semantically via initial_temperature() above

    // Drain chunk 0's command, free the slot.
    let _cmd0 = d.pending_commands.pop_front().unwrap();

    // "Packet 2" — different override, O2 sets temperature = 0.3.
    // Chunk 1 will be promoted from cut_pending below; it must
    // *not* pick up O2 (its `override_at_creation` is already O1).
    let o2 = AsrParamsOverride::new().with_initial_temperature(Some(0.3));
    d.current_override = Some(o2);
    let mut buf_mut = make_buffer_with_samples(20_000);
    d.handle_asr(ChunkId::from_raw(0), fake_asr_result("ok"))
      .unwrap();
    d.after_inject(&mut buf_mut, None, u64::MAX);

    // Chunk 1's Asr should have temperature = 0.7 (O1), not 0.3 (O2).
    let cmd1 = d.pending_commands.pop_front().expect("chunk 1 Asr");
    let Command::Asr { params, .. } = &cmd1 else {
      panic!("expected Asr; got {cmd1:?}");
    };
    assert!(
      (params.initial_temperature() - 0.7).abs() < 1e-6,
      "chunk 1 must keep packet 1's override; got temp={}",
      params.initial_temperature()
    );
  }

  #[test]
  fn unknown_chunk_id_returns_error() {
    let mut d = dispatch_default();
    let r = d.handle_asr(ChunkId::from_raw(99), fake_asr_result("nope"));
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 99));
  }

  #[test]
  fn handle_failure_emits_error_event_in_order() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 2_000), ChunkId::from_raw(0), &b);
    d.handle_failure(
      ChunkId::from_raw(0),
      WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
        "x".into(),
      ))),
    )
    .unwrap();
    d.after_inject(&mut b, None, u64::MAX);
    assert_eq!(d.pending_events.len(), 1);
    match d.pending_events.front().unwrap() {
      Event::Error { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
      _ => panic!("expected Error event"),
    }
  }

  #[test]
  fn cut_pending_holds_chunks_when_max_in_flight_reached() {
    // Auto policy: tests pure max_in_flight gating without the
    // unlocked-AutoLockAfter restriction.
    let mut d = Dispatch::new(AsrParams::default(), false, 2, LanguagePolicy::Auto);
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    d.on_emit(fake_chunk(3_000, 4_000), ChunkId::from_raw(3), &b);
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 2);
    assert_eq!(
      d.pending_commands.len(),
      2,
      "only first two chunks issued Asr; pending chunks have no commands yet"
    );
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
      Command::Asr { chunk_id, .. } => assert_eq!(chunk_id.as_u64(), 0),
      _ => panic!("expected Asr"),
    }
  }

  /// When an in-flight chunk completes and `after_inject` runs,
  /// a chunk that was queued in `cut_pending` because
  /// `max_in_flight` was full must be promoted (audio extracted,
  /// Asr command queued) in the same call.
  #[test]
  fn cut_pending_promotes_on_slot_open() {
    // Auto policy: tests pure max_in_flight gating without the
    // unlocked-AutoLockAfter restriction.
    let mut d = Dispatch::new(AsrParams::default(), false, 2, LanguagePolicy::Auto);
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
    // emitting a third Asr command.
    d.handle_asr(ChunkId::from_raw(0), fake_asr_result("c0"))
      .unwrap();
    d.after_inject(&mut b, None, u64::MAX);

    assert_eq!(d.cut_pending.len(), 0, "cut_pending should be drained");
    assert_eq!(
      d.in_flight.len(),
      2,
      "chunk 0 emitted (out), chunk 2 promoted (in) — net stays at 2"
    );
    assert!(d.in_flight.contains_key(&ChunkId::from_raw(1)));
    assert!(d.in_flight.contains_key(&ChunkId::from_raw(2)));
    assert_eq!(
      d.pending_commands.len(),
      3,
      "third Asr was issued for chunk 2 on promotion"
    );
    assert_eq!(d.pending_events.len(), 1, "chunk 0's Transcript emitted");
  }

  /// `LanguagePolicy::Lock { hint }` must apply the hint to
  /// every emitted Asr command.
  #[test]
  fn language_policy_lock_applies_hint_to_first_chunk() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::Lock { hint: Lang::Zh },
    );
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    let cmd = d.poll_command().unwrap();
    match cmd {
      Command::Asr { params, .. } => {
        assert_eq!(
          params.language_hint(),
          Some(&Lang::Zh),
          "Lock {{ hint: Zh }} must set language_hint on every Asr"
        );
      }
      _ => panic!("expected Asr"),
    }
  }

  /// `LanguagePolicy::AutoLockAfter(1)` must lock the language
  /// after observing the first non-empty ASR result, then apply
  /// that hint to all subsequent Asr commands.
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
      Command::Asr { params, .. } => {
        assert_eq!(
          params.language_hint(),
          None,
          "first chunk under AutoLockAfter(1) has no hint yet"
        );
      }
      _ => panic!("expected Asr"),
    }

    // Inject ASR result with detected language Zh — this is
    // the first non-empty observation.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("你好"), Lang::Zh, -0.5, 0.05, 0.0),
    )
    .unwrap();
    // Pretend Cut is still accumulating starting at sample 1_000
    // (the start of the second chunk we're about to emit). This
    // keeps samples 1_000.. alive in the buffer past the
    // post-inject trim, so the next on_emit's extract succeeds.
    d.after_inject(&mut b, Some(1_000), u64::MAX);

    // Second chunk: hint should now be locked to Zh.
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    // poll_command pops chunk 0's parked stuff first (none here)
    // then chunk 1's Asr.
    let cmd = d.pending_commands.pop_back().unwrap();
    match cmd {
      Command::Asr {
        chunk_id, params, ..
      } => {
        assert_eq!(chunk_id.as_u64(), 1);
        assert_eq!(
          params.language_hint(),
          Some(&Lang::Zh),
          "second chunk hint must be locked to first detection"
        );
      }
      _ => panic!("expected Asr"),
    }
  }

  /// A duplicate `handle_asr` on a chunk that's already
  /// `Ready` (waiting in-order) must be rejected — otherwise the
  /// second call could overwrite the final transcript.
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
    d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1"))
      .unwrap();
    // Now chunk 1's phase is Ready. Duplicate inject must be rejected.
    let r = d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1-dup"));
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(c)) if c.as_u64() == 1));
  }

  /// Alignment results aimed at a chunk in `AwaitingAsr` (not
  /// `AwaitingAlignment`) must be rejected — otherwise an
  /// unsolicited alignment result could overwrite a
  /// still-in-flight chunk.
  #[test]
  fn inject_alignment_on_awaiting_asr_returns_unknown_chunk() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    // Phase is AwaitingAsr.
    let r = d.handle_alignment(
      ChunkId::from_raw(0),
      crate::core::command::AlignmentResult::new(Vec::new()),
    );
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(_))));
  }

  /// A failure aimed at a chunk already in `Ready` phase must
  /// be rejected — it must not retroactively turn a successful
  /// Transcript into an Error.
  #[test]
  fn handle_failure_on_ready_returns_unknown_chunk() {
    let mut d = dispatch_default();
    let mut b = make_buffer_with_samples(10_000);
    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    // Resolve chunk 1 to Ready (waiting on chunk 0 in-order).
    d.handle_asr(ChunkId::from_raw(1), fake_asr_result("c1"))
      .unwrap();
    let r = d.handle_failure(
      ChunkId::from_raw(1),
      WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
        SmolStr::from("late failure"),
      ))),
    );
    assert!(matches!(r, Err(TranscriberError::UnknownChunk(_))));
  }

  /// `AutoLockAfter(n)` must lock to the most-frequent observed
  /// language, not the last observation. With n=3 and
  /// observations [En, En, Zh], the earlier code locked to Zh
  /// (last seen); the contract is En (most frequent).
  /// First-occurrence tiebreaking handles equally-frequent
  /// languages deterministically.
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
      d.handle_asr(
        ChunkId::from_raw(i as u64),
        AsrResult::new(SmolStr::new("text"), lang.clone(), -0.5, 0.05, 0.0),
      )
      .unwrap();
      // Pretend Cut still has a future chunk accumulating
      // so trim doesn't drop chunk samples we haven't yet
      // emitted.
      // Pass Some(0) to pin the trim low-water at the buffer
      // start, keeping all chunks' samples alive for the
      // duration of the test. This test exercises language
      // policy, not trim behavior.
      d.after_inject(&mut b, Some(0), u64::MAX);
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
      Command::Asr {
        params, chunk_id, ..
      } => {
        assert_eq!(chunk_id.as_u64(), 3);
        assert_eq!(
          params.language_hint(),
          Some(&Lang::En),
          "post-lock chunks must carry the locked language"
        );
      }
      _ => panic!("expected Asr"),
    }
  }

  /// AutoLockAfter must order observations by ChunkId, not by
  /// ASR completion order. With max_in_flight > 1, chunk 1 can
  /// finish before chunk 0; earlier code recorded observations
  /// in completion order, race-determining the lock based on
  /// which worker happened to finish first. Reproduction: chunk
  /// 0 = En, chunk 1 = Zh, ASR for chunk 1 arrives first. With
  /// first-occurrence tiebreaking, chunk_id order [En, Zh] picks
  /// En; completion order [Zh, En] picks Zh — would have
  /// locked Zh.
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
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("zh"), Lang::Zh, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(
      d.locked_language, None,
      "auto-lock must not advance until chunk 0 resolves, regardless of completion order"
    );

    // Chunk 0's ASR result arrives — En. Now both have resolved
    // and the cursor can advance through both in chunk_id order:
    // observations = [En, Zh] → mode picks En (first occurrence
    // wins on ties).
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("en"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "auto-lock must observe in chunk_id order: chunk 0 = En first, then chunk 1 = Zh"
    );
  }

  /// An ASR failure on AwaitingAsr must advance the auto-lock
  /// cursor without contributing an observation. Otherwise a
  /// single failed chunk would block auto-lock forever.
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
    d.handle_failure(
      ChunkId::from_raw(0),
      WorkFailure::Asr(AsrError::AllTemperaturesExhausted(AsrFailure::new(
        "fail".into(),
      ))),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(
      d.locked_language, None,
      "single failed chunk produced no observation yet"
    );

    // Chunks 1 and 2 succeed in English. After both land, cursor
    // advances through 0 (failed, skipped) → 1 (En) → 2 (En) and
    // locks once observations.len() reaches 2.
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("hello"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    d.handle_asr(
      ChunkId::from_raw(2),
      AsrResult::new(SmolStr::new("world"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "auto-lock must skip failed chunk 0 and lock to En from chunks 1 + 2"
    );
  }

  /// An empty-text ASR result must advance the cursor without
  /// contributing an observation, even when arriving out of
  /// order. Reproduction: chunks 0–2 promoted; chunk 1 = En,
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
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("hello"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, None);

    // Chunk 0 (empty) — the cursor advances to 1, picks up En.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new(""), Lang::En, -1.0, 0.95, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(
      d.locked_language, None,
      "only chunk 1 contributed; need a second non-empty observation"
    );

    // Chunk 2 (En) — second observation lands; lock to En.
    d.handle_asr(
      ChunkId::from_raw(2),
      AsrResult::new(SmolStr::new("world"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, Some(Lang::En));
  }

  /// Under unlocked `AutoLockAfter(n)`, dispatch must hold back
  /// chunks past the observation window — otherwise chunks 1..N
  /// get Asr with `language_hint = None` and may auto-detect
  /// different languages, defeating the lock contract.
  /// Reproduction: `AutoLockAfter(1)` + `max_in_flight = 4`.
  /// Emit 3 chunks without injecting. Earlier code promoted all
  /// three with no hint. Post-fix code keeps only 1 in flight;
  /// the rest wait.
  #[test]
  fn unlocked_auto_lock_after_caps_in_flight_to_observation_window() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(1),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);

    assert_eq!(
      d.in_flight.len(),
      1,
      "under unlocked AutoLockAfter(1), only n=1 chunk runs in parallel"
    );
    assert_eq!(
      d.cut_pending.len(),
      2,
      "chunks beyond the observation window wait in cut_pending"
    );
    assert_eq!(
      d.pending_commands.len(),
      1,
      "only chunk 0 issued a Asr — chunks 1, 2 wait for the lock"
    );
  }

  /// Once the lock is established, the gate lifts and the
  /// held-back chunks promote with the locked hint.
  #[test]
  fn unlocked_auto_lock_after_releases_pending_with_hint_after_lock() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      4,
      LanguagePolicy::AutoLockAfter(1),
    );
    let mut b = make_buffer_with_samples(10_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    // Drain chunk 0's Asr from pending_commands so we can see
    // chunks 1 and 2's commands when they get emitted post-lock.
    let _ = d.pending_commands.pop_front();

    // Inject chunk 0's Zh — the lock fires.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("zh"), Lang::Zh, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    assert_eq!(d.locked_language, Some(Lang::Zh));
    // Chunks 1 and 2 must now be in flight (cap reverted to 4).
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 0);
    // Their Asr commands must carry the locked hint.
    assert_eq!(d.pending_commands.len(), 2);
    for cmd in d.pending_commands.iter() {
      match cmd {
        Command::Asr { params, .. } => {
          assert_eq!(
            params.language_hint(),
            Some(&Lang::Zh),
            "post-lock chunks must carry the locked hint"
          );
        }
        _ => panic!("expected Asr"),
      }
    }
  }

  /// AutoLockAfter(n>1) must hold back chunks past the
  /// observation window even after earlier observation chunks
  /// complete. A simpler in-flight count cap of `n` slid: when
  /// chunk 0 of an AutoLockAfter(3) stream completed and freed
  /// a slot, chunk 3 was promoted with `language_hint = None`
  /// even though the lock hadn't fired (chunks 1 and 2 still
  /// pending). Chunk 3 would run ASR without the locked
  /// language, defeating the AutoLockAfter contract for n>1.
  ///
  /// The fix gates by ChunkId threshold = auto_lock_cursor +
  /// (n - observations.len()). Chunks past that threshold wait
  /// for the lock regardless of in_flight occupancy.
  #[test]
  fn auto_lock_after_n_holds_back_post_window_chunks_until_lock() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      8,
      LanguagePolicy::AutoLockAfter(3),
    );
    let mut b = make_buffer_with_samples(20_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);
    d.on_emit(fake_chunk(3_000, 4_000), ChunkId::from_raw(3), &b);

    // First three chunks form the observation window — promoted.
    // Chunk 3 is past the window, must wait.
    assert_eq!(d.in_flight.len(), 3);
    assert_eq!(
      d.cut_pending.len(),
      1,
      "chunk 3 must wait past the observation window"
    );

    // Chunk 0 returns En. Only 1/3 observations; lock not set.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new("a"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, None);
    // Chunk 3 must remain pending — observation cursor hasn't advanced.
    assert_eq!(
      d.cut_pending.len(),
      1,
      "chunk 3 must NOT be promoted just because chunk 0 freed a slot"
    );

    // Chunk 1 returns En. 2/3 observations; lock not set.
    d.handle_asr(
      ChunkId::from_raw(1),
      AsrResult::new(SmolStr::new("b"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, None);
    assert_eq!(d.cut_pending.len(), 1, "chunk 3 still must not be promoted");

    // Chunk 2 returns En. 3/3 observations; lock fires.
    d.handle_asr(
      ChunkId::from_raw(2),
      AsrResult::new(SmolStr::new("c"), Lang::En, -0.5, 0.05, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);
    assert_eq!(d.locked_language, Some(Lang::En));

    // Chunk 3 must now be promoted, with the locked hint applied.
    assert_eq!(d.cut_pending.len(), 0, "chunk 3 promoted after lock");
    let mut found_chunk_3 = false;
    for cmd in d.pending_commands.iter() {
      if let Command::Asr {
        chunk_id, params, ..
      } = cmd
      {
        if chunk_id.as_u64() == 3 {
          assert_eq!(
            params.language_hint(),
            Some(&Lang::En),
            "chunk 3 (post-lock) must carry the locked hint"
          );
          found_chunk_3 = true;
        }
      }
    }
    assert!(
      found_chunk_3,
      "chunk 3's Asr command must be queued post-lock"
    );
  }

  /// If an early observation chunk resolves empty/failed, the
  /// threshold slides forward by one and the next chunk becomes
  /// a candidate (still without the lock). Reproduction:
  /// AutoLockAfter(2). Chunk 0 returns empty. The threshold was
  /// 0+2=2 (chunks 0, 1 in window); after chunk 0's empty result
  /// advances cursor to 1, threshold = 1+2 = 3, so chunk 2 is
  /// now a candidate.
  #[test]
  fn auto_lock_after_threshold_slides_on_empty() {
    let mut d = Dispatch::new(
      AsrParams::default(),
      false,
      8,
      LanguagePolicy::AutoLockAfter(2),
    );
    let mut b = make_buffer_with_samples(20_000);

    d.on_emit(fake_chunk(0, 1_000), ChunkId::from_raw(0), &b);
    d.on_emit(fake_chunk(1_000, 2_000), ChunkId::from_raw(1), &b);
    d.on_emit(fake_chunk(2_000, 3_000), ChunkId::from_raw(2), &b);

    // Initial threshold = 0 + 2 = 2. Chunks 0, 1 in flight; 2 waits.
    assert_eq!(d.in_flight.len(), 2);
    assert_eq!(d.cut_pending.len(), 1);

    // Chunk 0 returns empty — cursor advances, observations stays 0.
    d.handle_asr(
      ChunkId::from_raw(0),
      AsrResult::new(SmolStr::new(""), Lang::En, -1.0, 0.95, 0.0),
    )
    .unwrap();
    d.after_inject(&mut b, Some(0), u64::MAX);

    // Threshold = 1 + 2 = 3. Chunk 2 (id=2) is now a candidate
    // and gets promoted (id < 3).
    assert_eq!(d.locked_language, None);
    assert_eq!(
      d.cut_pending.len(),
      0,
      "chunk 2 promoted after empty chunk 0 advanced threshold"
    );
    assert_eq!(
      d.in_flight.len(),
      2,
      "chunk 1 still in flight + chunk 2 just promoted"
    );
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
      d.handle_asr(
        ChunkId::from_raw(i as u64),
        AsrResult::new(SmolStr::new("text"), lang.clone(), -0.5, 0.05, 0.0),
      )
      .unwrap();
      // Pass Some(0) to pin trim at the buffer start.
      d.after_inject(&mut b, Some(0), u64::MAX);
    }

    assert_eq!(
      d.locked_language,
      Some(Lang::En),
      "first-occurrence tiebreaking picks En over Zh when each appears once"
    );
  }
}
