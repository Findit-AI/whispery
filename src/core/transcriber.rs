//! Transcriber — the public Sans-I/O surface.
//!
//! `Transcriber` is `Send + !Sync` (every public mutating method
//! takes `&mut self`). Multi-threaded drivers must wrap in
//! `Mutex<Transcriber>` themselves.

use core::time::Duration;

use mediatime::{Timebase, Timestamp};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
  core::{
    buffer::SampleBuffer,
    command::{AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command},
    cut::Cut,
    dispatch::Dispatch,
    event::Event,
  },
  types::{ChunkId, Lang, TranscriberError, VadSegment, WorkFailure},
};

/// Language-detection / locking strategy.
///
/// Serialised in `snake_case` external representation when the
/// `serde` feature is on:
/// - `"auto"` for [`Self::Auto`]
/// - `{ "lock": { "hint": "en" } }` for [`Self::Lock`]
/// - `{ "auto_lock_after": 3 }` for [`Self::AutoLockAfter`]
///
/// The same shape silero uses for its `SampleRate` enum (silero's
/// options module is the reference design).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
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
///
/// Fields are private; use [`TranscriberOptions::new`] (or
/// [`Default::default`]) and the `set_*` / `with_*` accessors to
/// construct and tweak. The `with_*` methods are consuming
/// builder-style; `set_*` mutate in place. Most accessors are
/// `const fn` and can run in const contexts.
///
/// **Serde encoding** (when `feature = "serde"` is on): every
/// field has a `serde(default = ...)` so deserialising from a
/// partial config (e.g. `{}` or `{"chunk_size": "10s"}`) fills
/// the rest with `Self::new()`'s defaults. `Duration` fields use
/// `humantime_serde` so config files can write `"30s"` /
/// `"100ms"` instead of `{ secs, nanos }`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TranscriberOptions {
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_chunk_size", with = "humantime_serde")
  )]
  chunk_size: Duration,
  #[cfg_attr(feature = "serde", serde(default = "default_buffer_cap_samples"))]
  buffer_cap_samples: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_gap_tolerance_samples"))]
  gap_tolerance_samples: u64,
  #[cfg_attr(feature = "serde", serde(default))]
  word_alignment: bool,
  #[cfg_attr(feature = "serde", serde(default = "default_max_in_flight"))]
  max_in_flight: usize,
  #[cfg_attr(feature = "serde", serde(default))]
  asr_params: AsrParams,
  #[cfg_attr(feature = "serde", serde(default))]
  language_policy: LanguagePolicy,
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      with = "humantime_serde::option",
      skip_serializing_if = "Option::is_none"
    )
  )]
  flush_on_silence_gap: Option<Duration>,
}

#[cfg(feature = "serde")]
const fn default_chunk_size() -> Duration {
  Duration::from_secs(30)
}
#[cfg(feature = "serde")]
const fn default_buffer_cap_samples() -> usize {
  60 * 16_000
}
#[cfg(feature = "serde")]
const fn default_gap_tolerance_samples() -> u64 {
  200 * 16
}
#[cfg(feature = "serde")]
const fn default_max_in_flight() -> usize {
  6
}

// Codex round-37 round-25 [medium]: hard caps on the public
// chunk / buffer / in-flight knobs. Pre-fix the constructor
// only rejected zero values; a tenant-controlled or
// accidentally-large value could push the runner into
// process-scale per-chunk allocations (a 1-hour `chunk_size`
// at 16 kHz f32 ≈ 230 MB per chunk before any copies). Caps
// trade flexibility for a defended OOM boundary; 600 s
// (~38 MB per chunk) and 1 hour (~230 MB total buffer) are
// well past whisper.cpp's practical sweet spot of 30 s.

/// Maximum permitted [`TranscriberOptions::chunk_size`]. At
/// 16 kHz f32 mono this is roughly 38 MB per emitted chunk
/// before any downstream copy.
pub const MAX_CHUNK_SIZE: Duration = Duration::from_secs(600);

/// Maximum permitted [`TranscriberOptions::buffer_cap_samples`].
/// 1 hour at 16 kHz ≈ 230 MB f32 mono — past this the buffer
/// is no longer "input backpressure" and is closer to "in-RAM
/// archive".
pub const MAX_BUFFER_CAP_SAMPLES: usize = 60 * 60 * 16_000;

/// Maximum permitted [`TranscriberOptions::max_in_flight`]. The
/// dispatcher uses this to bound concurrent ASR/alignment jobs;
/// caller-side worker pools dictate the practical ceiling, but
/// a sanity cap protects against accidental e.g. `usize::MAX`.
pub const MAX_IN_FLIGHT: usize = 4_096;

/// Maximum permitted [`TranscriberOptions::gap_tolerance_samples`].
/// Bounded to [`MAX_BUFFER_CAP_SAMPLES`] so a tolerance value
/// can never exceed the buffer it gates: the buffer's overflow-
/// safe capacity arithmetic in
/// [`crate::core::buffer::SampleBuffer::append`] additionally
/// rejects any `delta_samples` whose addition would wrap, but
/// capping the public knob keeps the configuration intent
/// honest. (Codex round-37 round-26 [high].)
pub const MAX_GAP_TOLERANCE_SAMPLES: u64 = MAX_BUFFER_CAP_SAMPLES as u64;

impl TranscriberOptions {
  /// Construct a config with all default values. Equivalent to
  /// [`Default::default`] but `const fn`.
  pub const fn new() -> Self {
    Self {
      chunk_size: Duration::from_secs(30),
      buffer_cap_samples: 60 * 16_000,
      gap_tolerance_samples: 200 * 16, // 200 ms at 16 kHz
      word_alignment: false,
      max_in_flight: 6,
      asr_params: AsrParams::new(),
      language_policy: LanguagePolicy::AutoLockAfter(1),
      flush_on_silence_gap: None,
    }
  }

  /// Maximum duration of a merged chunk. Default 30 s.
  pub const fn chunk_size(&self) -> Duration {
    self.chunk_size
  }

  /// Max samples kept in the internal buffer before push returns
  /// `Backpressure`. Default 60 s × 16 kHz = 960 000.
  ///
  /// **Caller contract:** `buffer_cap_samples` must be at least
  /// as large as the longest single VAD segment the caller
  /// expects to push. `handle_vad_segment` requires all of the
  /// segment's audio to be buffered (the `VadAheadOfAudio`
  /// guard) before it accepts the segment, so a single VAD
  /// segment longer than `buffer_cap_samples` cannot be
  /// processed: `handle_samples` would return `Backpressure`
  /// before enough audio is buffered, and the only recovery is
  /// `handle_restart` (which drops the partial segment). Hard-split
  /// in `Cut::push_segment` only runs after the segment is
  /// accepted, so it cannot rescue this case.
  ///
  /// Practical guidance: with default 60 s cap, a single VAD
  /// segment must fit in 60 s of speech. Configure your upstream
  /// VAD to emit at silence boundaries below that, OR raise
  /// `buffer_cap_samples` to cover your worst-case segment
  /// length.
  pub const fn buffer_cap_samples(&self) -> usize {
    self.buffer_cap_samples
  }

  /// Maximum forward-gap that is silently zero-filled. Default
  /// 200 ms × 16 kHz = 3200.
  pub const fn gap_tolerance_samples(&self) -> u64 {
    self.gap_tolerance_samples
  }

  /// Whether to emit `Alignment` after each ASR completion.
  pub const fn word_alignment(&self) -> bool {
    self.word_alignment
  }

  /// Maximum chunks in flight. Default 6 (worker_count + 2 for a
  /// 4-worker runner; whispery's core has no runner context, so
  /// the default is fixed).
  pub const fn max_in_flight(&self) -> usize {
    self.max_in_flight
  }

  /// Default ASR params injected into every `Asr` command.
  pub const fn asr_params(&self) -> &AsrParams {
    &self.asr_params
  }

  /// Language detection / locking strategy.
  pub const fn language_policy(&self) -> &LanguagePolicy {
    &self.language_policy
  }

  /// If `Some(threshold)`, the cut state machine flushes the
  /// accumulating chunk whenever a new VAD segment arrives after
  /// a silence gap larger than `threshold`. `None` keeps the
  /// WhisperX-style continuous batching where small silences are
  /// merged into a chunk for whisper context. Default `None`.
  pub const fn flush_on_silence_gap(&self) -> Option<Duration> {
    self.flush_on_silence_gap
  }

  // --- Mutating setters ----------------------------------------

  /// Set [`Self::chunk_size`]. Panics if `value` exceeds
  /// [`MAX_CHUNK_SIZE`] (Codex round-37 round-25 [medium]:
  /// per-chunk RAM is unbounded otherwise).
  pub fn set_chunk_size(&mut self, value: Duration) {
    assert!(
      value <= MAX_CHUNK_SIZE,
      "TranscriberOptions::chunk_size {value:?} exceeds MAX_CHUNK_SIZE {MAX_CHUNK_SIZE:?}",
    );
    self.chunk_size = value;
  }

  /// Set [`Self::buffer_cap_samples`]. Panics if `value`
  /// exceeds [`MAX_BUFFER_CAP_SAMPLES`].
  pub fn set_buffer_cap_samples(&mut self, value: usize) {
    assert!(
      value <= MAX_BUFFER_CAP_SAMPLES,
      "TranscriberOptions::buffer_cap_samples {value} exceeds MAX_BUFFER_CAP_SAMPLES {MAX_BUFFER_CAP_SAMPLES}",
    );
    self.buffer_cap_samples = value;
  }

  /// Set [`Self::gap_tolerance_samples`]. Panics if `value`
  /// exceeds [`MAX_GAP_TOLERANCE_SAMPLES`].
  pub fn set_gap_tolerance_samples(&mut self, value: u64) {
    assert!(
      value <= MAX_GAP_TOLERANCE_SAMPLES,
      "TranscriberOptions::gap_tolerance_samples {value} exceeds MAX_GAP_TOLERANCE_SAMPLES {MAX_GAP_TOLERANCE_SAMPLES}",
    );
    self.gap_tolerance_samples = value;
  }

  /// Set [`Self::word_alignment`].
  pub const fn set_word_alignment(&mut self, value: bool) {
    self.word_alignment = value;
  }

  /// Set [`Self::max_in_flight`]. Panics if `value` exceeds
  /// [`MAX_IN_FLIGHT`].
  pub fn set_max_in_flight(&mut self, value: usize) {
    assert!(
      value <= MAX_IN_FLIGHT,
      "TranscriberOptions::max_in_flight {value} exceeds MAX_IN_FLIGHT {MAX_IN_FLIGHT}",
    );
    self.max_in_flight = value;
  }

  /// Set [`Self::asr_params`].
  pub fn set_asr_params(&mut self, value: AsrParams) {
    self.asr_params = value;
  }

  /// Set [`Self::language_policy`].
  pub fn set_language_policy(&mut self, value: LanguagePolicy) {
    self.language_policy = value;
  }

  /// Set [`Self::flush_on_silence_gap`].
  pub const fn set_flush_on_silence_gap(&mut self, value: Option<Duration>) {
    self.flush_on_silence_gap = value;
  }

  // --- Builder-style (consuming) -------------------------------

  /// Builder-style override for [`Self::chunk_size`]. Panics
  /// if `value` exceeds [`MAX_CHUNK_SIZE`] (Codex round-37
  /// round-25 [medium]).
  pub fn with_chunk_size(mut self, value: Duration) -> Self {
    assert!(
      value <= MAX_CHUNK_SIZE,
      "TranscriberOptions::chunk_size {value:?} exceeds MAX_CHUNK_SIZE {MAX_CHUNK_SIZE:?}",
    );
    self.chunk_size = value;
    self
  }

  /// Builder-style override for [`Self::buffer_cap_samples`].
  /// Panics if `value` exceeds [`MAX_BUFFER_CAP_SAMPLES`].
  pub fn with_buffer_cap_samples(mut self, value: usize) -> Self {
    assert!(
      value <= MAX_BUFFER_CAP_SAMPLES,
      "TranscriberOptions::buffer_cap_samples {value} exceeds MAX_BUFFER_CAP_SAMPLES {MAX_BUFFER_CAP_SAMPLES}",
    );
    self.buffer_cap_samples = value;
    self
  }

  /// Builder-style override for [`Self::gap_tolerance_samples`].
  /// Panics if `value` exceeds [`MAX_GAP_TOLERANCE_SAMPLES`]
  /// (Codex round-37 round-26 [high]).
  pub fn with_gap_tolerance_samples(mut self, value: u64) -> Self {
    assert!(
      value <= MAX_GAP_TOLERANCE_SAMPLES,
      "TranscriberOptions::gap_tolerance_samples {value} exceeds MAX_GAP_TOLERANCE_SAMPLES {MAX_GAP_TOLERANCE_SAMPLES}",
    );
    self.gap_tolerance_samples = value;
    self
  }

  /// Builder-style override for [`Self::word_alignment`].
  pub const fn with_word_alignment(mut self, value: bool) -> Self {
    self.word_alignment = value;
    self
  }

  /// Builder-style override for [`Self::max_in_flight`].
  /// Panics if `value` exceeds [`MAX_IN_FLIGHT`] (Codex
  /// round-37 round-25 [medium]).
  pub fn with_max_in_flight(mut self, value: usize) -> Self {
    assert!(
      value <= MAX_IN_FLIGHT,
      "TranscriberOptions::max_in_flight {value} exceeds MAX_IN_FLIGHT {MAX_IN_FLIGHT}",
    );
    self.max_in_flight = value;
    self
  }

  /// Builder-style override for [`Self::asr_params`].
  pub fn with_asr_params(mut self, value: AsrParams) -> Self {
    self.asr_params = value;
    self
  }

  /// Builder-style override for [`Self::language_policy`].
  pub fn with_language_policy(mut self, value: LanguagePolicy) -> Self {
    self.language_policy = value;
    self
  }

  /// Builder-style override for [`Self::flush_on_silence_gap`].
  pub const fn with_flush_on_silence_gap(mut self, value: Option<Duration>) -> Self {
    self.flush_on_silence_gap = value;
    self
  }
}

impl Default for TranscriberOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// The Sans-I/O state machine.
///
/// `Transcriber` is `Send` (movable across threads) but `!Sync`
/// (every public mutating method takes `&mut self`). A consumer that
/// wants to drive it from multiple threads must wrap it in
/// `Mutex<Transcriber>` themselves; whispery does not provide
/// internal synchronisation.
pub struct Transcriber {
  config: TranscriberOptions,
  buffer: SampleBuffer,
  cut: Cut,
  dispatch: Dispatch,
  next_chunk_id: u64,
  eof_signaled: bool,
  /// Highest sample index that VAD has analyzed — either as the
  /// `end_sample` of a pushed `VadSegment`, or as the explicit
  /// watermark in `handle_no_speech_through`. Future VAD pushes
  /// and no-speech signals must advance this; regressions
  /// surface as `TranscriberError::PtsRegression { kind: VadSegment }`.
  /// Independent from `Cut::last_pushed_end()` because the cut
  /// state machine only tracks pushed segments, while the
  /// watermark also incorporates explicit silence declarations.
  vad_watermark: u64,
}

impl Transcriber {
  /// Construct from config.
  ///
  /// # Panics
  ///
  /// Invalid config values are rejected up-front rather than
  /// turning into deadlocks (zero `max_in_flight`),
  /// divide-by-zero panics in cut (`chunk_size` that rounds to 0
  /// samples), or empty-list panics in the auto-lock mode helper
  /// (`AutoLockAfter(0)`).
  ///
  /// - `max_in_flight == 0` — the dispatch loop would route every
  ///   emitted chunk to `cut_pending` and never issue a `Asr`.
  /// - `LanguagePolicy::AutoLockAfter(0)` — a 0-observation lock
  ///   has no defined mode and would call the tiebreak helper on
  ///   an empty list.
  /// - `chunk_size` that rounds to 0 16 kHz samples (e.g.
  ///   `Duration::ZERO`) — Cut's hard-split path divides by it.
  pub fn new(config: TranscriberOptions) -> Self {
    assert!(
      config.max_in_flight > 0,
      "TranscriberOptions::max_in_flight must be > 0 (got 0; would deadlock the dispatch loop)"
    );
    // Borrow during validation: matching by value would partially
    // move `config.language_policy`, but since the inner `n: usize`
    // is Copy, Rust's pattern-matching elides the move and the
    // code compiles. Borrowing makes the intent explicit and stops
    // mechanical reviewers from flagging a false-positive
    // partial-move every round.
    if let LanguagePolicy::AutoLockAfter(n) = &config.language_policy {
      assert!(
        *n > 0,
        "LanguagePolicy::AutoLockAfter(n) requires n > 0 (got 0)"
      );
    }
    let chunk_size_samples =
      (config.chunk_size.as_secs_f64() * crate::time::SAMPLE_RATE_HZ as f64 + 0.5) as u64;
    assert!(
      chunk_size_samples > 0,
      "TranscriberOptions::chunk_size must round to at least 1 sample at 16 kHz; got {:?}",
      config.chunk_size
    );
    // Codex round-37 round-25 [medium]: enforce documented
    // ceilings even on direct `Transcriber::new(config)` calls
    // — `with_*` and `set_*` already guard their inputs, but
    // a serde-deserialised or hand-built `TranscriberOptions`
    // skips those paths.
    assert!(
      config.chunk_size <= MAX_CHUNK_SIZE,
      "TranscriberOptions::chunk_size {:?} exceeds MAX_CHUNK_SIZE {MAX_CHUNK_SIZE:?}",
      config.chunk_size,
    );
    assert!(
      config.buffer_cap_samples <= MAX_BUFFER_CAP_SAMPLES,
      "TranscriberOptions::buffer_cap_samples {} exceeds MAX_BUFFER_CAP_SAMPLES {MAX_BUFFER_CAP_SAMPLES}",
      config.buffer_cap_samples,
    );
    assert!(
      config.max_in_flight <= MAX_IN_FLIGHT,
      "TranscriberOptions::max_in_flight {} exceeds MAX_IN_FLIGHT {MAX_IN_FLIGHT}",
      config.max_in_flight,
    );
    assert!(
      config.gap_tolerance_samples <= MAX_GAP_TOLERANCE_SAMPLES,
      "TranscriberOptions::gap_tolerance_samples {} exceeds MAX_GAP_TOLERANCE_SAMPLES {MAX_GAP_TOLERANCE_SAMPLES}",
      config.gap_tolerance_samples,
    );

    let buffer = SampleBuffer::new(config.buffer_cap_samples, config.gap_tolerance_samples);
    let cut = Cut::new(config.chunk_size, config.flush_on_silence_gap);
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

  /// Stamp a per-packet `AsrParamsOverride` on the dispatch.
  /// Chunks extracted from the live buffer while this is `Some`
  /// snapshot the override into their `ExtractedChunk`; promote
  /// time then layers the snapshot on top of `asr_params` /
  /// `locked_language` to produce the final `Asr.params`.
  /// **Visibility: `pub(crate)`** — the runner sets and clears
  /// this around `process_packet`; out-of-tree consumers don't
  /// touch it.
  pub(crate) fn set_runtime_override(&mut self, ovr: Option<AsrParamsOverride>) {
    self.dispatch.current_override = ovr;
  }

  /// True iff every queue is empty: no buffered samples, no
  /// pending command/event, no in_flight chunks, no cut_pending
  /// entries. Pre-restart in-flight chunks (those still working
  /// through whisper or alignment) keep `is_idle()` false until
  /// they emit; `handle_restart` does not synthetically clear them.
  pub fn is_idle(&self) -> bool {
    self.dispatch.is_idle() && self.buffer.buffered_samples() == 0
  }

  /// Live buffer length in samples.
  pub fn buffered_samples(&self) -> usize {
    self.buffer.buffered_samples()
  }

  /// Total chunks waiting on a worker result: those in
  /// `cut_pending` (extracted, not yet promoted) plus those
  /// `in_flight` (promoted, awaiting whisper / alignment
  /// result). Excludes chunks whose results have arrived but
  /// not yet been observed by `poll_event` (those are in
  /// `pending_events`).
  ///
  /// Codex round-35: the runner's `DrainTimeout.in_flight`
  /// previously reported `buffered_samples()` (a sample count)
  /// despite the field being documented as "chunks awaiting
  /// results". A trimmed buffer can read `0` while real chunks
  /// are still in-flight; this method is the honest count.
  pub fn in_flight_chunk_count(&self) -> usize {
    self.dispatch.cut_pending.len() + self.dispatch.in_flight.len()
  }

  /// Output timebase recorded from the first `handle_samples` call.
  pub fn output_timebase(&self) -> Option<Timebase> {
    self.buffer.output_timebase()
  }

  /// Authoritative output-timebase PTS the buffer expects for the
  /// next contiguous `handle_samples` call. Returns `None` before
  /// the first push.
  pub fn next_expected_starts_at(&self) -> Option<Timestamp> {
    self.buffer.next_expected_starts_at()
  }

  /// Pre-bind a closure mapping `(start_sample, end_sample)` (in
  /// stream coordinates) to an output-timebase `TimeRange`.
  ///
  /// Used by the alignment worker to convert wav2vec2 frame
  /// indices back into the caller's output timebase. The
  /// closure captures snapshots of the buffer's timebase and
  /// pts-anchor so it can outlive borrows of `Transcriber`; the
  /// alignment worker keeps it alive across thread boundaries via
  /// `Arc<dyn Fn>`.
  ///
  /// Returns `None` if `chunk_id` is not in flight (e.g. already
  /// drained as `Transcript`/`Failed`).
  ///
  /// The closure operates in *stream coordinates*: the sample
  /// indices it accepts are absolute positions in the input audio
  /// stream **as they were at chunk-extract time**. The aligner
  /// has the chunk's `chunk_first_sample_in_stream` offset and
  /// adds `frame * hop` to land in the same coordinate system.
  ///
  /// A previous live-buffer-based variant captured the buffer's
  /// *current* PTS anchor, which a `handle_restart` between
  /// chunk-extract and ASR-finish would have re-anchored —
  /// mapping the chunk's pre-restart sample indices through the
  /// post-restart PTS origin and emitting word ranges far
  /// outside the transcript's own range. The per-chunk form
  /// snapshots the anchor pair at extract time (see
  /// [`super::dispatch::ChunkRecord::output_tb`]) so the rebuilt
  /// closure stays in the chunk's own epoch.
  #[cfg(feature = "alignment")]
  pub fn chunk_samples_to_output_range_fn(
    &self,
    chunk_id: ChunkId,
  ) -> Option<alloc::sync::Arc<dyn Fn(u64, u64) -> mediatime::TimeRange + Send + Sync>> {
    self.dispatch.chunk_samples_to_output_range_fn(chunk_id)
  }

  /// Stream-coordinate first 16 kHz sample index of the chunk
  /// `chunk_id`, or `None` if the chunk is not in flight.
  ///
  /// Used by alignment-driving callers to convert
  /// output-timebase `sub_segments` (received via
  /// [`crate::core::Command::Alignment`]) into chunk-local
  /// 16 kHz sample indices before passing them to
  /// [`crate::Aligner::align_chunk`].
  ///
  /// Codex round-37 round-10 [high]: this was `pub(crate)`
  /// when `ManagedTranscriber` was the only caller; the
  /// Sans-I/O pivot exposes it so external pump-driver code
  /// can construct alignment inputs.
  #[cfg(feature = "alignment")]
  pub fn chunk_first_sample(&self, chunk_id: ChunkId) -> Option<u64> {
    self.dispatch.chunk_first_sample(chunk_id)
  }

  /// Sub-VAD-segments of the chunk `chunk_id`, in stream-coordinate
  /// 16 kHz sample indices, as `(start, end)` pairs. The pairs
  /// must be offset by [`Self::chunk_first_sample`] before being
  /// wrapped into chunk-local `TimeRange`s for the aligner.
  /// Returns `None` if the chunk is not in flight.
  ///
  /// Codex round-37 round-10 [high]: bumped from `pub(crate)`
  /// alongside the rest of the alignment-context helpers.
  #[cfg(feature = "alignment")]
  pub fn chunk_sub_segments_samples(
    &self,
    chunk_id: ChunkId,
  ) -> Option<alloc::vec::Vec<(u64, u64)>> {
    self.dispatch.chunk_sub_segments_samples(chunk_id)
  }

  /// Non-mutating predicate: would the next push of `samples_len`
  /// audio samples plus `vad_count` VAD segments fit under the
  /// configured caps? Counts cut_pending's pre-extracted audio
  /// alongside the live buffer.
  ///
  /// Codex round-37 round-34 [medium]: arithmetic is overflow-
  /// safe. Pre-fix the `usize` chain wrapped on adversarial
  /// `samples_len = usize::MAX` inputs (panic in debug, false
  /// "yes" in release once the wrap dropped below the cap).
  /// Now any overflow returns `false` — the request cannot fit
  /// by any measure, so the predicate's safe answer is "do not
  /// push".
  pub fn would_accept(&self, samples_len: usize, _vad_count: usize) -> bool {
    let queued = self.dispatch.cut_pending_audio_samples();
    let Some(total) = self
      .buffered_samples()
      .checked_add(samples_len)
      .and_then(|v| v.checked_add(queued))
    else {
      return false;
    };
    total <= self.config.buffer_cap_samples
  }

  /// Push samples into the buffer.
  ///
  /// Errors:
  /// - `PtsRegression`, `GapExceedsTolerance`, `Backpressure`,
  ///   `InconsistentTimebase`, `AfterEof` per `SampleBuffer::append`.
  pub fn handle_samples(
    &mut self,
    starts_at: Timestamp,
    samples: &[f32],
  ) -> Result<(), TranscriberError> {
    if self.eof_signaled {
      return Err(TranscriberError::AfterEof);
    }
    // Reject malformed timebases at the API boundary.
    // mediatime::Timebase::new(0, _) is constructible and would
    // panic later in rescale_pts.
    let tb = starts_at.timebase();
    if tb.num() == 0 {
      return Err(TranscriberError::InvalidTimebase {
        numerator: tb.num(),
      });
    }
    // Count cut_pending audio (pre-extracted Arcs) toward the
    // buffer cap so a slow runner can't accumulate queued audio
    // outside any cap.
    let queued = self.dispatch.cut_pending_audio_samples();
    self.buffer.append(starts_at, samples, queued)
  }

  /// Push a VAD segment into the cut state machine.
  ///
  /// **Caller contract:** the entire segment's audio must be
  /// buffered first (`seg.end_sample() <= absolute_sample_offset()`).
  /// In particular, a segment longer than `buffer_cap_samples`
  /// cannot be pushed at all — `handle_samples` would return
  /// `Backpressure` before enough audio accumulates. Either
  /// configure upstream VAD to emit shorter segments at silence
  /// boundaries, or raise `buffer_cap_samples`. See
  /// [`TranscriberOptions::buffer_cap_samples`] for details.
  ///
  /// Errors:
  /// - `OutputTimebaseUnset` if no `handle_samples` has been called.
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
  /// - `AfterEof` if `handle_eof()` was called.
  pub fn handle_vad_segment(&mut self, seg: VadSegment) -> Result<(), TranscriberError> {
    if self.eof_signaled {
      return Err(TranscriberError::AfterEof);
    }
    if self.buffer.output_timebase().is_none() {
      return Err(TranscriberError::OutputTimebaseUnset);
    }
    // VAD must not reference audio that has not been buffered.
    // Otherwise the cut state machine would happily accept the
    // segment, accumulate it, and later (on a chunk emit or
    // handle_eof flush) trip a panic in buffer.extract when it
    // tries to slice past the tail.
    let high_water = self.buffer.absolute_sample_offset();
    if seg.end_sample() > high_water {
      return Err(TranscriberError::VadAheadOfAudio {
        vad_end: seg.end_sample(),
        buffered: high_water,
      });
    }
    // Strict-monotonic check against the VAD watermark. The
    // watermark advances with every handle_vad_segment AND every
    // handle_no_speech_through, so a VAD push that would
    // contradict an explicit silence declaration is also caught
    // here.
    if seg.start_sample() < self.vad_watermark {
      return Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::VadSegment,
        advance: seg.start_sample() as i64 - self.vad_watermark as i64,
      });
    }

    let merged_chunks = self
      .cut
      .push_segment(seg, self.dispatch.current_override.as_ref());
    self.vad_watermark = seg.end_sample();
    let emitted_any = !merged_chunks.is_empty();
    for chunk in merged_chunks {
      let chunk_id = ChunkId::from_raw(self.next_chunk_id);
      self.next_chunk_id += 1;
      self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
    }
    // Each on_emit pre-extracts a chunk's audio into an Arc but
    // the same audio is still live in the buffer. Without trim,
    // total queued = live + cut_pending briefly exceeds the cap.
    // Run after_inject so trim drops the duplicated live audio
    // (low-water = cut.pending_start or vad_watermark).
    // flush_in_order_events is a no-op (no injects happened);
    // the promote loop is also typically a no-op (cut_pending
    // was just added; gate state hasn't changed).
    if emitted_any {
      self.dispatch.after_inject(
        &mut self.buffer,
        self.cut.pending_start(),
        self.vad_watermark,
      );
    }
    Ok(())
  }

  /// Pre-flight validation for a packet of VAD segments.
  ///
  /// Mirrors the per-segment checks that [`Self::handle_vad_segment`]
  /// performs (eof, timebase, high-water, strict-monotonic
  /// ordering) WITHOUT mutating any state. Designed for callers
  /// (the runner's `process_packet`) that want all-or-nothing
  /// atomicity: if `precheck_vad_segments` returns `Ok`, then
  /// `handle_vad_segment(seg)` for each segment in `vad_segments`
  /// in order is guaranteed to succeed at the validation level.
  ///
  /// `samples_to_be_pushed` is the length of the audio packet the
  /// caller is about to push BEFORE the VADs. The high-water
  /// projection is conservative (floor): we assume `handle_samples`
  /// will append exactly `samples.len()` samples without zero-fill.
  /// Real `append` may zero-fill up to `gap_tolerance_samples`
  /// (giving a higher actual high-water), so this precheck can
  /// false-reject a VAD whose `end_sample` lies inside the
  /// to-be-zero-filled gap region — extremely rare, since silero
  /// only emits VAD segments over real audio. False-accept is
  /// impossible: anything the precheck approves will also pass
  /// the runtime check.
  ///
  /// Codex round-35: `process_packet` previously committed
  /// samples first, then pushed VADs one at a time. A failing
  /// VAD #N left samples + VADs 0..N-1 committed, so the caller
  /// could not safely retry the packet. Pre-flighting fixes the
  /// atomicity gap.
  pub(crate) fn precheck_vad_segments(
    &self,
    vad_segments: &[VadSegment],
    samples_to_be_pushed: usize,
  ) -> Result<(), TranscriberError> {
    if vad_segments.is_empty() {
      return Ok(());
    }
    if self.eof_signaled {
      return Err(TranscriberError::AfterEof);
    }
    // After a successful handle_samples, output_timebase() is
    // guaranteed to be Some. It's None now only if (a) no
    // handle_samples has ever happened AND (b) `samples` is empty
    // (so the upcoming handle_samples won't establish the timebase
    // either).
    if self.buffer.output_timebase().is_none() && samples_to_be_pushed == 0 {
      return Err(TranscriberError::OutputTimebaseUnset);
    }

    // Floor projection: assume zero gap-fill. See type doc.
    let projected_high_water = self.buffer.absolute_sample_offset() + samples_to_be_pushed as u64;
    let mut running_watermark = self.vad_watermark;

    for seg in vad_segments {
      if seg.end_sample() > projected_high_water {
        return Err(TranscriberError::VadAheadOfAudio {
          vad_end: seg.end_sample(),
          buffered: projected_high_water,
        });
      }
      if seg.start_sample() < running_watermark {
        return Err(TranscriberError::PtsRegression {
          kind: crate::types::PushKind::VadSegment,
          advance: seg.start_sample() as i64 - running_watermark as i64,
        });
      }
      running_watermark = seg.end_sample();
    }
    Ok(())
  }

  /// Declare that VAD has finished analyzing audio through
  /// `sample_index` and produced no segments past the most
  /// recent `handle_vad_segment` call. The core uses this signal
  /// to:
  ///
  /// 1. Trim audio that is no longer referenced by any live
  ///    chunk — without this, a stream with long silences would
  ///    accumulate audio in the buffer until the configured cap
  ///    is hit and `handle_samples` returns
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
  /// `handle_vad_segment` calls with `start_sample < sample_index`
  /// or `handle_no_speech_through` calls with a smaller
  /// `sample_index` return `PtsRegression { kind: VadSegment }`.
  ///
  /// Errors:
  /// - `OutputTimebaseUnset` if no `handle_samples` has been called.
  /// - `AfterEof` if `handle_eof()` was called.
  /// - `PtsRegression { kind: VadSegment }` if `sample_index` is
  ///   less than the current VAD watermark.
  pub fn handle_no_speech_through(&mut self, sample_index: u64) -> Result<(), TranscriberError> {
    if self.eof_signaled {
      return Err(TranscriberError::AfterEof);
    }
    if self.buffer.output_timebase().is_none() {
      return Err(TranscriberError::OutputTimebaseUnset);
    }
    // Like handle_vad_segment, signal_no_speech must not advance
    // the watermark past audio the buffer hasn't seen. Without
    // the guard, a future-sample signal would poison the
    // watermark and later valid VAD inside the
    // (eventually-buffered) interval would be rejected as
    // PtsRegression. Atomic rejection: vad_watermark stays put.
    let high_water = self.buffer.absolute_sample_offset();
    if sample_index > high_water {
      return Err(TranscriberError::VadAheadOfAudio {
        vad_end: sample_index,
        buffered: high_water,
      });
    }
    if sample_index < self.vad_watermark {
      return Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::VadSegment,
        advance: sample_index as i64 - self.vad_watermark as i64,
      });
    }
    self.vad_watermark = sample_index;

    // Pre-flush the cut accumulator if a hypothetical segment
    // arriving at `sample_index` would have forced a flush —
    // either by exceeding `chunk_size_samples` or by exceeding
    // the configured `flush_on_silence_gap` threshold. Without
    // this, a partial chunk would sit until chunk_size or EOF,
    // defeating utterance-boundary mode.
    if self.cut.would_flush_at(sample_index) {
      if let Some(chunk) = self.cut.flush() {
        let chunk_id = ChunkId::from_raw(self.next_chunk_id);
        self.next_chunk_id += 1;
        self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
      }
    }

    // Run the standard post-mutation drain so trim drops audio
    // unreferenced by any live chunk. Trim's upper bound is the
    // VAD watermark we just advanced — never past audio that
    // hasn't been declared analyzed.
    self.dispatch.after_inject(
      &mut self.buffer,
      self.cut.pending_start(),
      self.vad_watermark,
    );
    Ok(())
  }

  /// Mark the input stream as ended. Idempotent. Calling before
  /// any push is a no-op (Ok(())). Errors: never returns Err in
  /// v1; signature carries `Result<(), TranscriberError>` for
  /// forward compatibility.
  pub fn handle_eof(&mut self) -> Result<(), TranscriberError> {
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
      // by either cut_pending or the cut accumulator. handle_eof
      // is the one path where trimming past the VAD watermark is
      // safe — the stream is ended; no further VAD will arrive.
      // Without this, a silent stream (samples pushed but no VAD)
      // would leave the buffer non-empty forever and `is_idle()`
      // would never become true.
      let high_water = self.buffer.absolute_sample_offset();
      self
        .dispatch
        .after_inject(&mut self.buffer, self.cut.pending_start(), high_water);
    }
    Ok(())
  }

  /// Inject the result of a `Command::Asr`.
  ///
  /// Errors:
  /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight or
  ///   is in flight but not awaiting an ASR result.
  pub fn handle_asr(
    &mut self,
    chunk_id: ChunkId,
    result: AsrResult,
  ) -> Result<(), TranscriberError> {
    self.dispatch.handle_asr(chunk_id, result)?;
    // Trim cap: VAD watermark, not high-water. inject_* doesn't
    // change what's been analyzed, so audio past the watermark
    // must survive trim until VAD catches up.
    self.dispatch.after_inject(
      &mut self.buffer,
      self.cut.pending_start(),
      self.vad_watermark,
    );
    Ok(())
  }

  /// Inject the result of a `Command::Alignment`.
  ///
  /// Errors:
  /// - `UnknownChunk(chunk_id)` if `chunk_id` is not awaiting alignment.
  pub fn handle_alignment(
    &mut self,
    chunk_id: ChunkId,
    result: AlignmentResult,
  ) -> Result<(), TranscriberError> {
    self.dispatch.handle_alignment(chunk_id, result)?;
    self.dispatch.after_inject(
      &mut self.buffer,
      self.cut.pending_start(),
      self.vad_watermark,
    );
    Ok(())
  }

  /// Inject a per-chunk failure.
  ///
  /// Errors:
  /// - `UnknownChunk(chunk_id)` if `chunk_id` is not in flight or
  ///   is in flight but not awaiting any worker result.
  pub fn handle_failure(
    &mut self,
    chunk_id: ChunkId,
    failure: WorkFailure,
  ) -> Result<(), TranscriberError> {
    self.dispatch.handle_failure(chunk_id, failure)?;
    self.dispatch.after_inject(
      &mut self.buffer,
      self.cut.pending_start(),
      self.vad_watermark,
    );
    Ok(())
  }

  /// Recover from a `GapExceedsTolerance`.
  ///
  /// Steps:
  /// 1. Flush the cut state machine. Any partial chunk goes through
  ///    `on_emit`, which pre-extracts its audio and either promotes
  ///    or queues per the AutoLockAfter gate.
  /// 2. Clear the live buffer; reset `absolute_sample_offset` and
  ///    `buffer_drop_offset` to 0.
  /// 3. Re-anchor `base_pts_out_anchor` to `starts_at.pts()`.
  /// 4. `next_chunk_id` continues monotonically.
  /// 5. Pre-existing `cut_pending` entries already hold their audio
  ///    in their own `Arc<[f32]>`s — they survive the buffer reset
  ///    without a special drain pass.
  ///
  /// Previously this method drained `cut_pending` into `in_flight`
  /// via a `draining_for_restart` bypass that ignored the
  /// AutoLockAfter observation-window cap. With the cap suspended,
  /// chunks 1..N could be issued as `Asr` without the language
  /// hint, defeating the lock contract during recovery. The
  /// refactored `cut_pending` (stores `ExtractedChunk` rather than
  /// a sample range into the live buffer) makes the drain
  /// unnecessary; the gate is preserved.
  ///
  /// Errors:
  /// - `AfterEof` if `handle_eof()` was previously called.
  /// - `InconsistentTimebase` if the buffer already has an established
  ///   output timebase from a prior `handle_samples` and `starts_at`'s
  ///   timebase doesn't match. Pre-fix code silently overwrote the
  ///   timebase, so a 48 kHz stream restarted at a millisecond
  ///   timebase would produce post-restart chunks in a different
  ///   unit from pre-restart ones — corrupting ordering and PTS
  ///   arithmetic with no surfaced error.
  pub fn handle_restart(&mut self, starts_at: Timestamp) -> Result<(), TranscriberError> {
    if self.eof_signaled {
      return Err(TranscriberError::AfterEof);
    }
    // Reject malformed (zero-numerator) timebases at the API
    // boundary; rescale_pts later would panic.
    let tb = starts_at.timebase();
    if tb.num() == 0 {
      return Err(TranscriberError::InvalidTimebase {
        numerator: tb.num(),
      });
    }
    if let Some(expected_tb) = self.buffer.output_timebase() {
      if starts_at.timebase() != expected_tb {
        return Err(TranscriberError::InconsistentTimebase {
          expected: expected_tb,
          got: starts_at.timebase(),
        });
      }
    }

    // Step 1: flush the cut accumulator's partial chunk. on_emit
    // pre-extracts its audio (so it survives the buffer reset
    // below) and either promotes it (if the gate allows) or
    // queues it on cut_pending.
    if let Some(chunk) = self.cut.flush() {
      let chunk_id = ChunkId::from_raw(self.next_chunk_id);
      self.next_chunk_id += 1;
      self.dispatch.on_emit(chunk, chunk_id, &self.buffer);
    }

    // Steps 2 + 3: clear buffer and re-anchor.
    self.buffer.handle_restart(starts_at);

    // Reset the cut state machine so its current_end / next_vad_seq
    // align with the new frame.
    self.cut = Cut::new(self.config.chunk_size, self.config.flush_on_silence_gap);

    // The VAD watermark lives in absolute-sample space, which
    // handle_restart just reset to 0. Reset the watermark too,
    // otherwise post-restart VAD pushes at small sample indices
    // fail the regression check against the pre-restart end.
    self.vad_watermark = 0;

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
    Transcriber::new(TranscriberOptions::default())
  }

  #[test]
  fn push_vad_before_handle_samples_returns_output_timebase_unset() {
    let mut t = fresh();
    let r = t.handle_vad_segment(VadSegment::new(0, 100));
    assert!(matches!(r, Err(TranscriberError::OutputTimebaseUnset)));
  }

  #[test]
  fn handle_samples_then_vad_works() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 200)).unwrap();
  }

  #[test]
  fn vad_segment_regression_returns_pts_regression() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 10_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(100, 200)).unwrap();
    let r = t.handle_vad_segment(VadSegment::new(150, 250)); // overlaps
    assert!(matches!(
      r,
      Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::VadSegment,
        ..
      })
    ));
  }

  #[test]
  fn handle_eof_then_push_rejects() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 100]).unwrap();
    t.handle_eof().unwrap();
    let r = t.handle_samples(ts(100), &[0.0; 100]);
    assert!(matches!(r, Err(TranscriberError::AfterEof)));
    let r = t.handle_vad_segment(VadSegment::new(0, 100));
    assert!(matches!(r, Err(TranscriberError::AfterEof)));
  }

  #[test]
  fn handle_eof_idempotent_and_noop_before_push() {
    let mut t = fresh();
    t.handle_eof().unwrap();
    t.handle_eof().unwrap();
  }

  #[test]
  fn handle_restart_after_handle_eof_rejects() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    t.handle_eof().unwrap();
    let r = t.handle_restart(ts(50_000_000));
    assert!(matches!(r, Err(TranscriberError::AfterEof)));
  }

  #[test]
  fn handle_restart_drains_cut_pending_into_in_flight() {
    // max_in_flight = 1 forces queueing.
    let config = TranscriberOptions::default()
            .with_max_in_flight(1)
            .with_chunk_size(Duration::from_millis(125)) // 2_000 samples
            .with_buffer_cap_samples(100_000);
    let mut t = Transcriber::new(config);

    // Push enough audio to cover three chunks.
    t.handle_samples(ts(0), &[0.0; 16_000]).unwrap(); // 1 sec @ 16k pretend
    t.handle_vad_segment(VadSegment::new(0, 2_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(2_000, 4_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(4_000, 6_000)).unwrap();
    // First chunk's Asr is in pending_commands; second and
    // third are in cut_pending awaiting promotion.
    // Now restart at a fresh anchor.
    t.handle_restart(ts(50_000_000)).unwrap();

    // After restart: cut_pending should be empty (drained), the
    // buffer should be empty (cleared). Pre-restart in-flight
    // chunks (the originally-promoted one PLUS the formerly-
    // pending ones) survive — they hold their audio in their
    // own Arc<[f32]>s and will emit normally.
    // (Drain is allowed to exceed max_in_flight transiently.)
  }

  /// A `handle_restart` between a chunk's extraction and its
  /// ASR-finish callback used to silently corrupt that chunk's
  /// word timestamps. The alignment dispatch built
  /// `samples_to_output_range` from the *live* buffer, which
  /// `handle_restart` had just re-anchored onto a new PTS epoch —
  /// but the chunk's `chunk_first_sample_in_stream` was still
  /// in the pre-restart epoch. Result: words landed at
  /// `new_anchor + (pre_restart_sample * scale)`, often far
  /// outside the surviving chunk's own `Transcript::range`.
  /// Fix: snapshot `(output_tb, base_pts_out_anchor)` onto
  /// `ChunkRecord` at extract time, and rebuild the closure
  /// from those captured fields at alignment-dispatch time.
  #[cfg(feature = "alignment")]
  #[test]
  fn chunk_samples_to_output_range_fn_uses_extract_time_anchor_after_restart() {
    use crate::core::command::Command;

    let config = TranscriberOptions::default()
      .with_chunk_size(Duration::from_millis(125)) // 2_000 samples
      .with_max_in_flight(4)
      .with_buffer_cap_samples(100_000);
    let mut t = Transcriber::new(config);

    // Pre-restart epoch: anchor at PTS 0 in the 1/48000 timebase.
    // Two consecutive 2 000-sample VAD segments — Cut emits chunk
    // 0 once a second segment forces the merged length past
    // `chunk_size_samples` (`len > chunk_size_samples` is the
    // emission predicate).
    t.handle_samples(ts(0), &[0.0; 8_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 2_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(2_000, 4_000)).unwrap();

    // Drain Asr — chunk 0 is now in_flight, ASR not yet
    // resolved. This is the surviving pre-restart chunk.
    let cmd = t.poll_command().expect("Asr for chunk 0");
    let chunk_id = match cmd {
      Command::Asr { chunk_id, .. } => chunk_id,
      other => panic!("expected Asr, got {other:?}"),
    };

    // Sanity: closure built *before* any restart anchors at PTS 0.
    let closure_pre = t
      .chunk_samples_to_output_range_fn(chunk_id)
      .expect("chunk in flight");
    let pre = closure_pre(0, 16_000);
    assert_eq!(pre.start_pts(), 0, "pre-restart closure starts at PTS 0");
    // 16_000 samples at 1/16000 → 1 second → 48_000 in 1/48000.
    assert_eq!(
      pre.end_pts(),
      48_000,
      "pre-restart closure ends at PTS 48_000"
    );

    // Restart far ahead in PTS-space — 5e9 in 1/48000 ticks,
    // ~28.9 hours, so any leakage of the new anchor is loud.
    let new_anchor_pts: i64 = 5_000_000_000;
    t.handle_restart(ts(new_anchor_pts)).unwrap();

    // The pre-restart chunk is still in_flight (it owns its
    // audio in an `Arc<[f32]>`). Its alignment-dispatch
    // closure must STILL anchor at PTS 0, not at the post-
    // restart anchor — otherwise word ranges land 28.9 hours
    // past the chunk's own `Transcript::range`.
    let closure_post = t
      .chunk_samples_to_output_range_fn(chunk_id)
      .expect("pre-restart chunk survives restart");
    let post = closure_post(0, 16_000);
    assert_eq!(
      post.start_pts(),
      0,
      "post-restart closure for pre-restart chunk must use the chunk's extract-time anchor (got {})",
      post.start_pts()
    );
    assert_eq!(
      post.end_pts(),
      48_000,
      "post-restart closure end_pts drifted (got {})",
      post.end_pts()
    );
    assert_eq!(post.start().timebase(), tb_48k());
  }

  /// Trim must NOT drop audio still referenced by the cut
  /// accumulator. Reproduction: chunk 0 emitted, chunk 1
  /// accumulating in Cut without being emitted, chunk 0 ASR
  /// completes → after_inject runs trim. Without the fix,
  /// trim's low-water defaulted to absolute_sample_offset when
  /// cut_pending was empty and dropped the in-buffer audio that
  /// chunk 1 would later need; the next handle_vad_segment that
  /// closed chunk 1 would then panic in buffer.extract.
  #[test]
  fn trim_keeps_samples_for_unextracted_cut_accumulator() {
    use crate::core::command::Command;
    use smol_str::SmolStr;

    let config = TranscriberOptions::default()
            .with_chunk_size(Duration::from_secs(2)) // 32_000 samples @ 16k
            .with_buffer_cap_samples(200_000)
            .with_max_in_flight(4);
    let mut t = Transcriber::new(config);

    // 4 seconds of 16 kHz audio = 64_000 samples.
    t.handle_samples(ts(0), &vec![0.0_f32; 64_000]).unwrap();

    // Two VAD segments. The second pushes the merge past the
    // 32_000-sample chunk_size, so chunk 0 emits with range
    // [0, 24_000) and chunk 1 starts accumulating at 25_600.
    t.handle_vad_segment(VadSegment::new(0, 24_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(25_600, 48_000))
      .unwrap();

    // Resolve chunk 0 — this is where the bug used to fire trim
    // with low_water = absolute_sample_offset = 64_000 because
    // cut_pending was empty (chunk 1 wasn't yet emitted by Cut).
    let cmd = t.poll_command().unwrap();
    let Command::Asr { chunk_id, .. } = cmd else {
      panic!("expected Asr")
    };
    let asr = crate::core::command::AsrResult::new(
      SmolStr::new("c0"),
      crate::types::Lang::En,
      -0.5,
      0.05,
      0.0,
    );
    t.handle_asr(chunk_id, asr).unwrap();

    // Drain chunk 0's Transcript event.
    let _ = t.poll_event().expect("chunk 0 transcript");

    // Now close chunk 1. With the fix, the buffer still has
    // samples back to 25_600, so this extract succeeds. Without
    // the fix, the buffer was cleared past 64_000 and this
    // panics inside buffer.extract.
    t.handle_vad_segment(VadSegment::new(50_000, 60_000))
      .unwrap();

    // The third VAD push triggered chunk 1's emission with
    // range [25_600, 48_000) — its Asr is queued.
    let cmd = t.poll_command().unwrap();
    let Command::Asr { chunk_id, .. } = cmd else {
      panic!("expected Asr")
    };
    assert_eq!(chunk_id.as_u64(), 1, "chunk 1 ran without panic");
  }

  /// Silent EOF (samples pushed, no VAD ever pushed) must leave
  /// the transcriber idle. Without the fix, handle_eof returned
  /// without trimming the buffer; is_idle()'s
  /// `buffered_samples() == 0` check stayed false forever.
  #[test]
  fn silent_eof_makes_transcriber_idle() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    assert!(!t.is_idle(), "buffer has 1000 samples; not idle yet");
    t.handle_eof().unwrap();
    assert!(t.is_idle(), "after silent EOF, transcriber should be idle");
  }

  /// `max_in_flight = 0` deadlocks the dispatch loop — every
  /// emitted chunk goes to cut_pending, no `Asr` command ever
  /// fires. Reject at construction.
  #[test]
  #[should_panic(expected = "max_in_flight")]
  fn config_with_zero_max_in_flight_panics() {
    let config = TranscriberOptions::default().with_max_in_flight(0);
    let _ = Transcriber::new(config);
  }

  /// `AutoLockAfter(0)` calls `mode_with_first_occurrence_tiebreak`
  /// on a possibly-empty observation list (when the first chunk
  /// lands empty/failed), which panics. Reject at construction.
  #[test]
  #[should_panic(expected = "AutoLockAfter")]
  fn config_with_zero_auto_lock_after_panics() {
    let config =
      TranscriberOptions::default().with_language_policy(LanguagePolicy::AutoLockAfter(0));
    let _ = Transcriber::new(config);
  }

  /// A `chunk_size` that rounds to 0 samples (e.g.
  /// `Duration::ZERO`) makes `Cut::push_segment`'s
  /// `len.div_ceil(self.chunk_size_samples)` divide by zero on
  /// any non-trivial VAD segment. Reject at construction.
  #[test]
  #[should_panic(expected = "chunk_size")]
  fn config_with_zero_chunk_size_panics() {
    let config = TranscriberOptions::default().with_chunk_size(Duration::ZERO);
    let _ = Transcriber::new(config);
  }

  /// Codex round-37 round-25 [medium]: chunk_size beyond
  /// MAX_CHUNK_SIZE is rejected at the builder boundary so a
  /// caller-supplied or tenant-controlled value can't push the
  /// runner into per-chunk RAM scaling with input.
  #[test]
  #[should_panic(expected = "MAX_CHUNK_SIZE")]
  fn with_chunk_size_above_cap_panics() {
    let _ = TranscriberOptions::default().with_chunk_size(MAX_CHUNK_SIZE + Duration::from_secs(1));
  }

  #[test]
  #[should_panic(expected = "MAX_BUFFER_CAP_SAMPLES")]
  fn with_buffer_cap_samples_above_cap_panics() {
    let _ = TranscriberOptions::default().with_buffer_cap_samples(MAX_BUFFER_CAP_SAMPLES + 1);
  }

  #[test]
  #[should_panic(expected = "MAX_IN_FLIGHT")]
  fn with_max_in_flight_above_cap_panics() {
    let _ = TranscriberOptions::default().with_max_in_flight(MAX_IN_FLIGHT + 1);
  }

  /// Cap also enforced when `Transcriber::new` is called with
  /// a hand-built / serde-deserialised `TranscriberOptions`
  /// that bypassed the builder.
  #[test]
  #[should_panic(expected = "MAX_CHUNK_SIZE")]
  fn config_with_chunk_size_above_cap_panics_in_new() {
    let mut config = TranscriberOptions::default();
    config.chunk_size = MAX_CHUNK_SIZE + Duration::from_secs(1);
    let _ = Transcriber::new(config);
  }

  /// Codex round-37 round-34 [medium]: `would_accept` must be
  /// safe against adversarial `samples_len = usize::MAX`. Pre-
  /// fix the unchecked `usize` add wrapped, returning `true`
  /// for an enormous request. Post-fix the overflow path
  /// returns `false` — the request cannot fit by any measure.
  #[test]
  fn would_accept_overflow_returns_false() {
    let t = fresh();
    assert!(
      !t.would_accept(usize::MAX, 0),
      "near-MAX samples_len must NOT report acceptance via wrap"
    );
  }

  #[test]
  fn would_accept_normal_value_returns_true() {
    let t = fresh();
    // Tiny request well under the default cap.
    assert!(t.would_accept(100, 0));
  }

  /// handle_restart must not silently switch the output timebase.
  /// Without the guard, a stream anchored at 1/48000 could be
  /// restarted at 1/1000 and produce post-restart `TimeRange`s
  /// in a different unit from pre-restart chunks — corrupts
  /// ordering and downstream PTS arithmetic with no error
  /// surfaced.
  #[test]
  fn handle_restart_with_different_timebase_returns_inconsistent_timebase() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    // Stream is now anchored at 1/48000. Try to restart at 1/1000.
    let other_tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
    let r = t.handle_restart(Timestamp::new(0, other_tb));
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

  /// A restart at the same timebase succeeds.
  #[test]
  fn handle_restart_same_timebase_succeeds() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    t.handle_restart(ts(50_000_000)).unwrap();
    assert_eq!(t.output_timebase(), Some(tb_48k()));
  }

  /// VAD-emission used to grow cut_pending (pre-extracted Arcs)
  /// without dropping the duplicated live-buffer audio. After
  /// several handle_vad_segment calls, total queued = live +
  /// cut_pending could exceed the configured cap, even though
  /// handle_samples's cap check would later reject. Fix:
  /// handle_vad_segment runs after_inject at the end so trim drops
  /// the live audio that's now in cut_pending Arcs.
  /// `would_accept(0, 0)` is the predicate for "cap is currently
  /// respected" — if that returns false right after a sequence of
  /// handle_vad_segment calls, the cap was transiently violated.
  #[test]
  fn vad_emission_keeps_total_memory_within_cap() {
    let config = TranscriberOptions::default()
      .with_buffer_cap_samples(12_000)
      .with_max_in_flight(1)
      .with_language_policy(LanguagePolicy::Auto)
      .with_chunk_size(Duration::from_millis(250)); // 4 000 samples
    let mut t = Transcriber::new(config);

    // Fill buffer to cap.
    t.handle_samples(ts(0), &[0.0; 12_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 4_000)).unwrap();
    // Emit chunk 0 (cap=1 → in_flight) and start a new accumulator.
    t.handle_vad_segment(VadSegment::new(4_000, 8_000)).unwrap();
    // Emit chunk 1 (gated by cap=1 → cut_pending). Pre-fix: live
    // stayed at 12 000, cut_pending now adds 4 000 — total 16 000
    // > cap. Post-fix: trim ran, live dropped, total bounded.
    t.handle_vad_segment(VadSegment::new(8_000, 12_000))
      .unwrap();

    // After VAD-only emission, the cap must still hold:
    // would_accept(0, 0) = (live + queued ≤ cap). False would
    // mean we're already over.
    assert!(
      t.would_accept(0, 0),
      "after VAD-only emission, buffered + cut_pending audio must stay within cap; \
             live={}, cap=12_000",
      t.buffered_samples()
    );
  }

  /// `mediatime::Timebase::new(0, _)` is constructible (the type
  /// only enforces non-zero denominator). Using such a timebase
  /// as the target of `Timebase::rescale_pts` panics. handle_samples
  /// and handle_restart reject it explicitly with `InvalidTimebase`
  /// so a malformed caller timebase surfaces as `Err(_)` instead
  /// of an abort.
  #[test]
  fn handle_samples_rejects_zero_numerator_timebase() {
    let mut t = fresh();
    let bad_tb = Timebase::new(0, NonZeroU32::new(48_000).unwrap());
    let r = t.handle_samples(Timestamp::new(0, bad_tb), &[0.0; 100]);
    assert!(
      matches!(r, Err(TranscriberError::InvalidTimebase { numerator: 0 })),
      "expected InvalidTimebase, got {:?}",
      r
    );
  }

  /// handle_restart also validates the timebase.
  #[test]
  fn handle_restart_rejects_zero_numerator_timebase() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 100]).unwrap();
    let bad_tb = Timebase::new(0, NonZeroU32::new(48_000).unwrap());
    let r = t.handle_restart(Timestamp::new(0, bad_tb));
    assert!(
      matches!(r, Err(TranscriberError::InvalidTimebase { numerator: 0 })),
      "expected InvalidTimebase, got {:?}",
      r
    );
  }

  /// handle_no_speech_through must not let trim drop audio past
  /// the declared `sample_index`. Pre-fix code passed
  /// `cut.pending_start()` to `after_inject`; when no cut
  /// accumulator existed, after_inject's fallback was
  /// `buffer.absolute_sample_offset()` (drop everything). With
  /// audio buffered ahead of the no-speech declaration, that
  /// dropped the still-unanalyzed tail. A subsequent VAD push
  /// inside the dropped range would call buffer.extract on
  /// already-trimmed indices and panic.
  #[test]
  fn handle_no_speech_through_does_not_trim_past_declared_index() {
    use crate::core::command::Command;

    let config = TranscriberOptions::default()
      .with_buffer_cap_samples(20_000)
      .with_language_policy(LanguagePolicy::Auto);
    let mut t = Transcriber::new(config);

    // Push 1 000 samples of audio.
    t.handle_samples(ts(0), &[0.0; 1_000]).unwrap();

    // Declare silence only through sample 500. Audio at [500,
    // 1 000) is still unanalyzed and must NOT be trimmed.
    t.handle_no_speech_through(500).unwrap();
    assert_eq!(
      t.buffered_samples(),
      500,
      "trim must drop [0..500) but keep [500..1 000) — VAD hasn't analyzed past 500 yet"
    );

    // Push a VAD segment in the unanalyzed tail. Must succeed,
    // and the audio must still be available for extraction.
    t.handle_vad_segment(VadSegment::new(600, 800)).unwrap();
    t.handle_eof().unwrap();

    let mut got_chunk = false;
    while let Some(cmd) = t.poll_command() {
      if let Command::Asr { samples, .. } = cmd {
        assert_eq!(
          samples.len(),
          200,
          "extracted audio should be 200 samples for VAD [600, 800)"
        );
        got_chunk = true;
      }
    }
    assert!(got_chunk, "chunk for VAD [600, 800) should have emitted");
  }

  /// cut_pending audio (the pre-extracted ExtractedChunk Arcs)
  /// must count toward the buffer cap for Backpressure. Pre-fix
  /// code only counted the live buffer's samples, so a runner
  /// slower than ingest could build up cut_pending arbitrarily —
  /// every trim emptied the live buffer and let the caller push
  /// more, but cut_pending retained the audio. Net effect:
  /// unbounded memory growth on long indexing jobs.
  ///
  /// Reproduction: cap=12 000, max_in_flight=1, chunk_size=0.25 s
  /// (4 000 samples). Push 12 000 samples + 3 VAD segments: chunks
  /// 0 and 1 emit. With cap=1, chunk 0 → in_flight, chunk 1 →
  /// cut_pending (4 000 audio samples in Arc). handle_no_speech_through
  /// trims the live buffer to 4 000 samples (cut accumulator's
  /// start is at 8 000, high-water is 12 000). Then push 6 000
  /// more samples. Pre-fix: live(4 000) + new(6 000) = 10 000 ≤
  /// cap(12 000), accepted — even though cut_pending adds another
  /// 4 000 for a real total of 14 000. Post-fix: 14 000 > 12 000
  /// → Backpressure.
  #[test]
  fn cut_pending_audio_counts_against_buffer_cap() {
    let config = TranscriberOptions::default()
      .with_buffer_cap_samples(12_000)
      .with_max_in_flight(1)
      .with_language_policy(LanguagePolicy::Auto)
      .with_chunk_size(Duration::from_millis(250)); // 4 000 samples
    let mut t = Transcriber::new(config);

    t.handle_samples(ts(0), &[0.0; 12_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 4_000)).unwrap(); // accumulates
    t.handle_vad_segment(VadSegment::new(4_000, 8_000)).unwrap(); // chunk 0 emits
    t.handle_vad_segment(VadSegment::new(8_000, 12_000))
      .unwrap(); // chunk 1 emits

    // After: chunk 0 in_flight (cap=1); chunk 1 in cut_pending
    // (4 000 audio samples in Arc); cut accumulator = [8 000, 12 000).
    // Live buffer still holds 12 000 samples (no trim ran yet).

    // handle_no_speech_through(12 000) trims via after_inject.
    // low_water = cut.pending_start() = 8 000 → buffer drops
    // [0, 8 000), now holds [8 000, 12 000) = 4 000 samples.
    t.handle_no_speech_through(12_000).unwrap();
    assert_eq!(
      t.buffered_samples(),
      4_000,
      "trim drops up to cut accumulator's start"
    );

    // Push 6 000 more samples. Without the cut_pending-aware
    // cap check, buffer.append's check is live(4 000) +
    // new(6 000) = 10 000 ≤ cap(12 000) — accepted. With the
    // check, queued(4 000) is included → 14 000 > 12 000 →
    // Backpressure.
    let next = t.next_expected_starts_at().unwrap();
    let r = t.handle_samples(next, &[0.0; 6_000]);
    assert!(
      matches!(r, Err(TranscriberError::Backpressure { buffered, cap })
                if buffered == 14_000 && cap == 12_000),
      "expected Backpressure {{ buffered: 14_000, cap: 12_000 }}, got {:?}",
      r
    );
  }

  /// handle_restart must NOT bypass the AutoLockAfter
  /// observation-window gate. Pre-fix code routed every drained
  /// chunk through on_emit with `draining_for_restart` flag set,
  /// which forced promotion regardless of the effective-cap-of-n
  /// rule. Net effect: a recovery happening before the lock
  /// fires would dispatch chunks 1..N as Asr without the
  /// language hint, defeating the lock contract.
  ///
  /// Reproduction: AutoLockAfter(1) + max_in_flight = 4. Push
  /// audio + 4 VAD segments. The gate caps in_flight at 1, so
  /// chunks 1, 2, 3 sit in cut_pending. Trigger handle_restart
  /// without injecting chunk 0's result. Pre-fix would have
  /// promoted chunks 1, 2, 3 (issuing Asr without lock).
  /// Post-fix: chunks 1, 2, 3 stay in cut_pending; only after the
  /// lock fires (chunk 0 returns) do they promote with the hint.
  #[test]
  fn handle_restart_preserves_auto_lock_gate() {
    use crate::core::command::Command;

    let config = TranscriberOptions::default()
            .with_chunk_size(Duration::from_millis(125)) // 2_000 samples
            .with_max_in_flight(4)
            .with_buffer_cap_samples(100_000)
            .with_language_policy(LanguagePolicy::AutoLockAfter(1));
    let mut t = Transcriber::new(config);

    // 4 VAD segments emitted; with gate cap = 1, chunk 0 in
    // flight + chunks 1, 2, 3 in cut_pending.
    t.handle_samples(ts(0), &[0.0; 16_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 2_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(2_000, 4_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(4_000, 6_000)).unwrap();
    t.handle_vad_segment(VadSegment::new(6_000, 8_000)).unwrap();

    // Restart without injecting chunk 0.
    t.handle_restart(ts(50_000_000)).unwrap();

    // Drain commands. There must be exactly ONE Asr — chunk 0's
    // — issued before the restart. Pre-fix would have produced
    // additional Asr commands (without lock) for the drained
    // chunks 1, 2, 3.
    let mut run_asr_count = 0;
    let mut hints = alloc::vec::Vec::new();
    while let Some(cmd) = t.poll_command() {
      if let Command::Asr {
        params, chunk_id, ..
      } = cmd
      {
        run_asr_count += 1;
        hints.push((chunk_id.as_u64(), params.language_hint().cloned()));
      }
    }
    assert_eq!(
      run_asr_count, 1,
      "only chunk 0 issued Asr before lock; got hints {:?}",
      hints
    );
    assert_eq!(hints[0].0, 0);
    assert_eq!(
      hints[0].1, None,
      "chunk 0 dispatched without hint — that's expected, it's the observation chunk"
    );
  }

  /// `handle_restart` resets the buffer's `absolute_sample_offset`
  /// to 0, so the VAD watermark — which is in absolute-sample
  /// space — must reset too. Without the reset, a post-restart
  /// VAD push starting near sample 0 fails the watermark
  /// regression check against the pre-restart VAD's end.
  #[test]
  fn handle_restart_resets_vad_watermark() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 50_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 30_000)).unwrap();
    // Pre-restart watermark is now 30_000.
    t.handle_restart(ts(50_000_000)).unwrap();
    // Post-restart, push at sample 0 of the new frame must succeed.
    t.handle_samples(ts(50_000_000), &[0.0; 10_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 5_000)).unwrap();
  }

  /// User feature: `flush_on_silence_gap = Some(threshold)` makes
  /// the cut state machine yield a chunk whenever a silence gap
  /// between consecutive VAD segments exceeds the threshold —
  /// utterance-boundary chunking instead of WhisperX-style
  /// continuous batching. With the threshold set, two short VAD
  /// segments separated by silence longer than the threshold
  /// produce two chunks instead of one merged chunk.
  #[test]
  fn flush_on_silence_gap_yields_at_utterance_boundary() {
    // Auto policy: this test exercises silence-flush behavior;
    // default AutoLockAfter(1) would gate the second chunk
    // until chunk 0's ASR result lands, hiding the silence-flush
    // effect we're trying to verify.
    let config = TranscriberOptions::default()
      .with_chunk_size(Duration::from_secs(30))
      .with_flush_on_silence_gap(Some(Duration::from_millis(500)))
      .with_language_policy(LanguagePolicy::Auto);
    let mut t = Transcriber::new(config);

    t.handle_samples(ts(0), &[0.0; 100_000]).unwrap();
    // Speech utterance 1: [0, 16 000) — half a second.
    t.handle_vad_segment(VadSegment::new(0, 16_000)).unwrap();
    // 1 s gap (16 000 samples) > 500 ms threshold (8 000 samples).
    // Speech utterance 2: [32 000, 48 000).
    t.handle_vad_segment(VadSegment::new(32_000, 48_000))
      .unwrap();
    t.handle_eof().unwrap();

    let mut chunk_starts = alloc::vec::Vec::new();
    while let Some(cmd) = t.poll_command() {
      if let crate::core::command::Command::Asr { chunk_id, .. } = cmd {
        chunk_starts.push(chunk_id.as_u64());
      }
    }
    assert_eq!(
      chunk_starts.len(),
      2,
      "silence-flush yielded one chunk per utterance"
    );
  }

  /// A stream with no VAD activity for longer than
  /// `buffer_cap_samples` would fill the buffer and trip
  /// Backpressure with no recovery path — chunks emit only on
  /// VAD or EOF, so trim never runs. The watermark API lets the
  /// caller explicitly declare "VAD analyzed through here, no
  /// segments" so the core can safely drop the buffered audio.
  #[test]
  fn handle_no_speech_through_drains_pure_silence_buffer() {
    // Tighten cap so test doesn't have to push 60 s of audio.
    let config = TranscriberOptions::default().with_buffer_cap_samples(20_000);
    let mut t = Transcriber::new(config);
    // Push 16 000 samples (close to cap, but under).
    t.handle_samples(ts(0), &[0.0; 16_000]).unwrap();
    assert_eq!(t.buffered_samples(), 16_000);
    // Tell whispery VAD has analyzed through sample 16_000 with
    // no segments. Buffer should drop everything; the next push
    // can land cleanly even though a contiguous push without
    // the signal would have hit Backpressure (16 000 + 16 000
    // > 20 000).
    t.handle_no_speech_through(16_000).unwrap();
    assert_eq!(
      t.buffered_samples(),
      0,
      "post-watermark trim must drop unreferenced audio"
    );
    assert!(t.is_idle());
    // Subsequent push at the same anchor (no time gap, since the
    // buffer's anchor was preserved) succeeds.
    let next = t.next_expected_starts_at().unwrap();
    t.handle_samples(next, &[0.0; 16_000]).unwrap();
  }

  /// Speech followed by long silence must be flushable. After a
  /// partial chunk has been accumulating in the cut state
  /// machine, handle_no_speech_through(sample_index) past
  /// `current_start + chunk_size_samples` pre-flushes the chunk
  /// — any future segment starting >= sample_index would force
  /// the cut to flush anyway.
  #[test]
  fn handle_no_speech_through_flushes_orphaned_partial_chunk() {
    let config = TranscriberOptions::default().with_chunk_size(Duration::from_secs(2)); // 32 000 samples
    let mut t = Transcriber::new(config);
    // Push enough audio to cover speech + lots of silence.
    t.handle_samples(ts(0), &[0.0; 200_000]).unwrap();
    // Speech segment: [0, 16 000) — half a chunk_size.
    t.handle_vad_segment(VadSegment::new(0, 16_000)).unwrap();
    // No emit yet: the cut accumulator is half-full, still under
    // chunk_size.
    assert!(t.poll_event().is_none());

    // Now signal silence past chunk_size_samples. The hypothetical
    // next segment at sample 100_000 would trigger flush
    // (100_000 - 0 > 32_000); pre-flush instead.
    t.handle_no_speech_through(100_000).unwrap();

    // The cut accumulator should have been flushed — chunk 0 is
    // now in_flight awaiting an ASR result.
    match t.poll_command() {
      Some(crate::core::command::Command::Asr { chunk_id, .. }) if chunk_id.as_u64() == 0 => {}
      other => panic!("expected Asr for chunk 0, got {:?}", other),
    }
  }

  /// handle_no_speech_through must pre-flush when the silence
  /// gap exceeds `flush_on_silence_gap` even if the chunk is far
  /// from `chunk_size`. Otherwise the utterance-boundary mode is
  /// defeated for trailing silence — the partial chunk sits in
  /// the cut state until chunk_size or EOF.
  #[test]
  fn handle_no_speech_through_flushes_on_silence_gap_below_chunk_size() {
    // chunk_size = 30 s (480 000 samples); silence threshold = 500 ms (8 000 samples).
    let config = TranscriberOptions::default()
      .with_chunk_size(Duration::from_secs(30))
      .with_flush_on_silence_gap(Some(Duration::from_millis(500)))
      .with_language_policy(LanguagePolicy::Auto);
    let mut t = Transcriber::new(config);

    t.handle_samples(ts(0), &[0.0; 200_000]).unwrap();
    // One short utterance: [0, 16 000) — half a chunk_size.
    t.handle_vad_segment(VadSegment::new(0, 16_000)).unwrap();
    // No emit yet (chunk_size not crossed).
    assert!(t.poll_command().is_none());

    // Signal silence through sample 30 000 — gap from the
    // segment's end (16 000) is 14 000 samples (~875 ms),
    // greater than the 500 ms (8 000-sample) silence threshold,
    // but FAR below chunk_size (480 000). Pre-fix code only
    // pre-flushed on chunk_size; the chunk stayed pending.
    t.handle_no_speech_through(30_000).unwrap();

    // Post-fix: the silence gap triggers the pre-flush, the
    // partial chunk emits as a Asr command immediately.
    match t.poll_command() {
      Some(crate::core::command::Command::Asr { chunk_id, .. }) if chunk_id.as_u64() == 0 => {}
      other => panic!("expected Asr for chunk 0, got {:?}", other),
    }
  }

  /// A no-speech signal advances the VAD watermark, so
  /// subsequent VAD pushes that start before the watermark must
  /// be rejected with PtsRegression — otherwise the caller could
  /// contradict their own no-speech declaration.
  #[test]
  fn vad_segment_before_no_speech_watermark_returns_pts_regression() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 50_000]).unwrap();
    t.handle_no_speech_through(10_000).unwrap();
    let r = t.handle_vad_segment(VadSegment::new(5_000, 8_000));
    assert!(matches!(
      r,
      Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::VadSegment,
        ..
      })
    ));
  }

  /// A regression on the no-speech watermark itself (calling it
  /// twice with a smaller index second) returns PtsRegression.
  #[test]
  fn handle_no_speech_through_regression_returns_error() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 50_000]).unwrap();
    t.handle_no_speech_through(20_000).unwrap();
    let r = t.handle_no_speech_through(10_000);
    assert!(matches!(
      r,
      Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::VadSegment,
        ..
      })
    ));
  }

  /// handle_no_speech_through before any push returns
  /// OutputTimebaseUnset (consistent with handle_vad_segment).
  #[test]
  fn handle_no_speech_through_before_handle_samples_returns_unset() {
    let mut t = fresh();
    let r = t.handle_no_speech_through(1000);
    assert!(matches!(r, Err(TranscriberError::OutputTimebaseUnset)));
  }

  /// handle_no_speech_through must not advance the watermark
  /// past audio the buffer hasn't seen yet. Without the guard, a
  /// caller that mistakenly signals a future sample index would
  /// poison the watermark — later valid VAD inside the
  /// not-yet-buffered interval would get rejected as
  /// PtsRegression even though the audio eventually arrives.
  /// Symmetric with `handle_vad_segment`'s VadAheadOfAudio guard.
  #[test]
  fn handle_no_speech_through_past_buffered_audio_returns_typed_error() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    // Signal silence through sample 5_000, but only 1_000 samples
    // are buffered. Must reject; watermark must stay unchanged.
    let r = t.handle_no_speech_through(5_000);
    assert!(
      matches!(
        r,
        Err(TranscriberError::VadAheadOfAudio {
          vad_end: 5_000,
          buffered: 1_000
        })
      ),
      "expected VadAheadOfAudio, got {:?}",
      r
    );
    // Subsequent VAD inside the rejected interval (which we
    // didn't actually push) succeeds — i.e., watermark wasn't
    // poisoned.
    t.handle_samples(ts(0).clone(), &[]).ok(); // no-op
    // Push more audio so VAD in the original interval is buffered.
    let next = t.next_expected_starts_at().unwrap();
    t.handle_samples(next, &[0.0; 5_000]).unwrap();
    // VAD start = 2_000 < the rejected sample_index = 5_000 must
    // STILL succeed, proving the watermark wasn't moved.
    t.handle_vad_segment(VadSegment::new(2_000, 4_000)).unwrap();
  }

  /// handle_no_speech_through after handle_eof returns AfterEof
  /// (consistent with push_*).
  #[test]
  fn handle_no_speech_through_after_eof_returns_after_eof() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    t.handle_eof().unwrap();
    let r = t.handle_no_speech_through(2000);
    assert!(matches!(r, Err(TranscriberError::AfterEof)));
  }

  /// VAD must not reference audio past the buffer's high-water
  /// mark. Without the guard, the segment is accumulated,
  /// handle_eof flushes it, dispatch's promote calls
  /// buffer.extract on a range that doesn't exist, and the
  /// program panics.
  #[test]
  fn vad_segment_past_buffered_audio_returns_typed_error() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1000]).unwrap();
    // VAD segment claims audio out to sample 2000, but buffer
    // only has 1000 samples.
    let r = t.handle_vad_segment(VadSegment::new(0, 2000));
    assert!(
      matches!(
        r,
        Err(TranscriberError::VadAheadOfAudio {
          vad_end: 2000,
          buffered: 1000
        })
      ),
      "expected VadAheadOfAudio, got {:?}",
      r
    );
    // Critically: handle_eof must now NOT panic — the segment
    // was rejected, the cut accumulator is empty.
    t.handle_eof().unwrap();
    assert!(t.is_idle());
  }

  /// Codex round-30 regression: a chunk whose accumulation begins
  /// in packet A (under override O_A) but whose VAD-driven close
  /// only happens in packet B (under override O_B or none) must
  /// emit Asr with O_A's params, not O_B's. The cut state
  /// machine snapshots the override on accumulation start, and
  /// the snapshot rides on `MergedChunk.override_at_start` until
  /// the dispatch consumes it.
  ///
  /// Pre-fix: dispatch.on_emit read `current_override` directly,
  /// so the override active at the *closing* packet won —
  /// language hints / prompts / strategy overrides got attached
  /// to the wrong audio.
  #[test]
  fn override_binds_to_packet_that_started_chunk_not_packet_that_closed_it() {
    use crate::core::{AsrParamsOverride, command::Command};

    // chunk_size = 2 s (32 000 samples). One short VAD seg won't
    // close the chunk on its own; only EOF (or a chunk_size-
    // crossing follow-up segment) will.
    let config = TranscriberOptions::default()
      .with_chunk_size(Duration::from_secs(2))
      .with_max_in_flight(4);
    let mut t = Transcriber::new(config);

    // Packet A: stamp override O_A, push audio + a half-chunk
    // VAD segment. Cut accumulator now holds chunk 0 with
    // `override_at_start = O_A`.
    let o_a = AsrParamsOverride::new().with_initial_temperature(Some(0.7));
    t.set_runtime_override(Some(o_a.clone()));
    t.handle_samples(ts(0), &[0.0; 50_000]).unwrap();
    t.handle_vad_segment(VadSegment::new(0, 16_000)).unwrap();
    // Clear override at the end of "packet A" the way the runner
    // would.
    t.set_runtime_override(None);
    // No emit yet — chunk is still accumulating below chunk_size.
    assert!(
      t.poll_command().is_none(),
      "chunk should not emit during packet A — only half a chunk"
    );

    // Packet B: stamp a *different* override O_B. Pre-fix code
    // would let O_B leak onto chunk 0 when EOF closes it.
    let o_b = AsrParamsOverride::new().with_initial_temperature(Some(0.3));
    t.set_runtime_override(Some(o_b));
    t.handle_eof().unwrap();
    t.set_runtime_override(None);

    // Drain the Asr for chunk 0; its params must reflect O_A.
    let cmd = t
      .poll_command()
      .expect("EOF flush should emit chunk 0's Asr");
    let Command::Asr {
      params, chunk_id, ..
    } = &cmd
    else {
      panic!("expected Asr; got {cmd:?}");
    };
    assert_eq!(chunk_id.as_u64(), 0);
    assert!(
      (params.initial_temperature() - 0.7).abs() < 1e-6,
      "chunk 0 must keep packet A's override (temp=0.7); got temp={}",
      params.initial_temperature()
    );
    let _ = o_a; // captured semantically via initial_temperature
  }

  // --- Codex round-35: precheck_vad_segments ---

  #[test]
  fn precheck_vad_segments_passes_for_valid_packet() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 10_000]).unwrap();
    let segs = [VadSegment::new(0, 1000), VadSegment::new(2000, 3000)];
    let r = t.precheck_vad_segments(&segs, 0);
    assert!(r.is_ok());
  }

  /// VAD ahead-of-projected-high-water rejects without committing.
  #[test]
  fn precheck_vad_segments_rejects_ahead_of_audio() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1_000]).unwrap();
    // Segment ends at 5_000, but only 1_000 samples buffered + 0
    // about-to-push.
    let segs = [VadSegment::new(0, 5_000)];
    match t.precheck_vad_segments(&segs, 0) {
      Err(TranscriberError::VadAheadOfAudio {
        vad_end: 5_000,
        buffered: 1_000,
      }) => {}
      other => panic!("expected VadAheadOfAudio; got {other:?}"),
    }
  }

  /// VAD ordering regression caught by precheck.
  #[test]
  fn precheck_vad_segments_rejects_out_of_order() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 10_000]).unwrap();
    let segs = [
      VadSegment::new(2_000, 3_000),
      VadSegment::new(1_000, 1_500), // start < running watermark (3_000)
    ];
    match t.precheck_vad_segments(&segs, 0) {
      Err(TranscriberError::PtsRegression {
        kind: crate::types::PushKind::VadSegment,
        ..
      }) => {}
      other => panic!("expected PtsRegression; got {other:?}"),
    }
  }

  /// Precheck projects the post-push high water using
  /// `samples_to_be_pushed`, so a VAD targeting samples in the
  /// upcoming push is accepted.
  #[test]
  fn precheck_vad_segments_uses_projected_high_water() {
    let t = fresh();
    // Buffer empty, output timebase unset. samples_to_be_pushed=
    // 1000 means after push, high_water = 1000.
    let segs = [VadSegment::new(0, 800)];
    let r = t.precheck_vad_segments(&segs, 1_000);
    assert!(r.is_ok(), "VAD inside projected high water must pass");

    // VAD past the projected high water is rejected.
    let segs2 = [VadSegment::new(0, 2_000)];
    let r2 = t.precheck_vad_segments(&segs2, 1_000);
    assert!(matches!(
      r2,
      Err(TranscriberError::VadAheadOfAudio {
        buffered: 1_000,
        ..
      })
    ));
  }

  /// Empty `vad_segments` is always Ok.
  #[test]
  fn precheck_vad_segments_empty_is_ok() {
    let t = fresh();
    assert!(t.precheck_vad_segments(&[], 0).is_ok());
    assert!(t.precheck_vad_segments(&[], 1_000).is_ok());
  }

  /// AfterEof is caught by precheck without mutating state.
  #[test]
  fn precheck_vad_segments_rejects_after_eof() {
    let mut t = fresh();
    t.handle_samples(ts(0), &[0.0; 1_000]).unwrap();
    t.handle_eof().unwrap();
    let segs = [VadSegment::new(0, 500)];
    match t.precheck_vad_segments(&segs, 0) {
      Err(TranscriberError::AfterEof) => {}
      other => panic!("expected AfterEof; got {other:?}"),
    }
  }

  /// OutputTimebaseUnset caught by precheck when no samples will
  /// be pushed AND no prior push has set the timebase.
  #[test]
  fn precheck_vad_segments_rejects_no_timebase_no_samples() {
    let t = fresh();
    let segs = [VadSegment::new(0, 100)];
    match t.precheck_vad_segments(&segs, 0) {
      Err(TranscriberError::OutputTimebaseUnset) => {}
      other => panic!("expected OutputTimebaseUnset; got {other:?}"),
    }
  }
}
