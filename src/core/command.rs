//! `Command` enum and its result-side companions.
//!
//! These types are deliberately backend-agnostic — they don't name
//! `whisper-rs` types and don't include whisper.cpp-specific fields.
//! The runner's `whisper_pool` translates `AsrParams` into
//! `FullParams`; a future swap to candle-whisper or a CTranslate2
//! binding would change only the runner.

use alloc::{sync::Arc, vec::Vec};

use mediatime::TimeRange;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang};

/// Universal ASR knobs. Each field corresponds to either a knob
/// exposed by whisper-rs's `FullParams` or a parameter the runner's
/// own temperature retry loop consumes; nothing aspirational lives
/// here.
///
/// Fields are private; use [`AsrParams::new`] (or
/// [`Default::default`]) and the `set_*` / `with_*` accessors.
///
/// **Serde encoding** (when `feature = "serde"` is on): every
/// field carries a `serde(default = ...)` matching the value
/// `Self::new()` produces, so partial config files round-trip
/// without forcing every knob to be present. `Option<T>` fields
/// use `skip_serializing_if = "Option::is_none"` to keep
/// serialised configs compact.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AsrParams {
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  language_hint: Option<Lang>,
  #[cfg_attr(feature = "serde", serde(default))]
  strategy: SamplingStrategy,
  #[cfg_attr(feature = "serde", serde(default = "default_initial_temperature"))]
  initial_temperature: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_temperature_increment"))]
  temperature_increment: f32,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_max_attempts",
      deserialize_with = "deserialize_nonzero_max_attempts"
    )
  )]
  max_attempts: u8,
  #[cfg_attr(feature = "serde", serde(default = "default_log_prob_threshold"))]
  log_prob_threshold: f32,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_compression_ratio_threshold")
  )]
  compression_ratio_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_no_speech_threshold"))]
  no_speech_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_no_context"))]
  no_context: bool,
  #[cfg_attr(feature = "serde", serde(default = "default_suppress_blank"))]
  suppress_blank: bool,
  #[cfg_attr(feature = "serde", serde(default))]
  suppress_non_speech_tokens: bool,
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  initial_prompt: Option<SmolStr>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_n_threads",
      deserialize_with = "deserialize_positive_n_threads"
    )
  )]
  n_threads: i32,
}

#[cfg(feature = "serde")]
const fn default_initial_temperature() -> f32 {
  0.0
}
#[cfg(feature = "serde")]
const fn default_temperature_increment() -> f32 {
  0.2
}
#[cfg(feature = "serde")]
const fn default_max_attempts() -> u8 {
  6
}
/// Validate `max_attempts` at the serde boundary. The setters
/// already panic on `0`, but a deserialized config bypasses
/// them. Surface the violation as a typed deserialization
/// error instead of letting a misconfigured serde value silently
/// drop every ASR result. Codex round-33.
#[cfg(feature = "serde")]
fn deserialize_nonzero_max_attempts<'de, D>(deserializer: D) -> Result<u8, D::Error>
where
  D: serde::Deserializer<'de>,
{
  use serde::de::Error as _;
  let v = u8::deserialize(deserializer)?;
  if v == 0 {
    return Err(D::Error::custom(
      "max_attempts must be > 0; use 1 for a single attempt with no retries",
    ));
  }
  Ok(v)
}
#[cfg(feature = "serde")]
const fn default_log_prob_threshold() -> f32 {
  -1.0
}
#[cfg(feature = "serde")]
const fn default_compression_ratio_threshold() -> f32 {
  2.4
}
#[cfg(feature = "serde")]
const fn default_no_speech_threshold() -> f32 {
  0.6
}
#[cfg(feature = "serde")]
const fn default_no_context() -> bool {
  true
}
#[cfg(feature = "serde")]
const fn default_suppress_blank() -> bool {
  true
}
#[cfg(feature = "serde")]
const fn default_n_threads() -> i32 {
  1
}
/// Validate `n_threads >= 1` at the serde boundary. The setters
/// already panic on `<= 0`, but a deserialized config bypasses
/// them. Codex round-34: whisper.cpp's decoder loop allocates
/// `std::vector<std::thread>(n_threads - 1)` when not exactly 1,
/// so `n_threads = 0` underflows to a huge allocation request and
/// `n_threads < 0` aborts. Surface the violation as a typed
/// deserialize error.
#[cfg(feature = "serde")]
fn deserialize_positive_n_threads<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
  D: serde::Deserializer<'de>,
{
  use serde::de::Error as _;
  let v = i32::deserialize(deserializer)?;
  if v < 1 {
    return Err(D::Error::custom(alloc::format!(
      "n_threads must be >= 1 (got {v}); whisper.cpp would underflow / abort otherwise"
    )));
  }
  Ok(v)
}

impl AsrParams {
  /// Construct with all default values. Equivalent to
  /// [`Default::default`] but `const fn`.
  pub const fn new() -> Self {
    Self {
      language_hint: None,
      strategy: SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
      },
      initial_temperature: 0.0,
      temperature_increment: 0.2,
      max_attempts: 6,
      log_prob_threshold: -1.0,
      compression_ratio_threshold: 2.4,
      no_speech_threshold: 0.6,
      no_context: true,
      suppress_blank: true,
      suppress_non_speech_tokens: false,
      initial_prompt: None,
      n_threads: 1,
    }
  }

  /// Language hint passed to `FullParams::set_language`. `None`
  /// means auto-detect.
  pub const fn language_hint(&self) -> Option<&Lang> {
    self.language_hint.as_ref()
  }

  /// Sampling strategy. The runner constructs a fresh `FullParams`
  /// per chunk via `FullParams::new(strategy.into_whisper_rs())`.
  pub const fn strategy(&self) -> SamplingStrategy {
    self.strategy
  }

  /// Initial decoding temperature; first attempt of the runner's
  /// retry ladder.
  pub const fn initial_temperature(&self) -> f32 {
    self.initial_temperature
  }

  /// Increment applied to temperature on each retry attempt.
  pub const fn temperature_increment(&self) -> f32 {
    self.temperature_increment
  }

  /// Maximum total attempts (initial + retries). Default 6.
  pub const fn max_attempts(&self) -> u8 {
    self.max_attempts
  }

  /// Triggers temperature retry when avg_logprob falls below this.
  pub const fn log_prob_threshold(&self) -> f32 {
    self.log_prob_threshold
  }

  /// Triggers temperature retry when output compression ratio
  /// exceeds this.
  pub const fn compression_ratio_threshold(&self) -> f32 {
    self.compression_ratio_threshold
  }

  /// Threshold above which a chunk is reported as silence
  /// (`Transcript.no_speech_prob`).
  pub const fn no_speech_threshold(&self) -> f32 {
    self.no_speech_threshold
  }

  /// Forwarded to `FullParams::set_no_context`. **Polarity matches
  /// whisper-rs**: `true` = do not use past transcription.
  pub const fn no_context(&self) -> bool {
    self.no_context
  }

  /// Forwarded to `FullParams::set_suppress_blank`.
  pub const fn suppress_blank(&self) -> bool {
    self.suppress_blank
  }

  /// Forwarded to `FullParams::set_suppress_nst`.
  pub const fn suppress_non_speech_tokens(&self) -> bool {
    self.suppress_non_speech_tokens
  }

  /// Forwarded to `FullParams::set_initial_prompt`.
  pub const fn initial_prompt(&self) -> Option<&SmolStr> {
    self.initial_prompt.as_ref()
  }

  /// Forwarded to `FullParams::set_n_threads`.
  pub const fn n_threads(&self) -> i32 {
    self.n_threads
  }

  // --- Mutating setters ----------------------------------------

  /// Set [`Self::language_hint`].
  pub fn set_language_hint(&mut self, value: Option<Lang>) {
    self.language_hint = value;
  }

  /// Set [`Self::strategy`].
  pub const fn set_strategy(&mut self, value: SamplingStrategy) {
    self.strategy = value;
  }

  /// Set [`Self::initial_temperature`].
  pub const fn set_initial_temperature(&mut self, value: f32) {
    self.initial_temperature = value;
  }

  /// Set [`Self::temperature_increment`].
  pub const fn set_temperature_increment(&mut self, value: f32) {
    self.temperature_increment = value;
  }

  /// Set [`Self::max_attempts`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. The retry ladder iterates
  /// `for _attempt in 0..max_attempts`, so `0` would skip
  /// `state.full(...)` entirely and return
  /// [`AsrFailureKind::AllTemperaturesFailed`](crate::types::AsrFailureKind::AllTemperaturesFailed)
  /// for every chunk — total ASR data loss with no model
  /// inference attempted (Codex round-33). Use `1` for
  /// "single attempt, no temperature retries"; the temperature
  /// ladder needs at least one pass.
  pub const fn set_max_attempts(&mut self, value: u8) {
    assert!(
      value > 0,
      "max_attempts must be > 0 (got 0); use 1 for a single attempt with no retries"
    );
    self.max_attempts = value;
  }

  /// Set [`Self::log_prob_threshold`].
  pub const fn set_log_prob_threshold(&mut self, value: f32) {
    self.log_prob_threshold = value;
  }

  /// Set [`Self::compression_ratio_threshold`].
  pub const fn set_compression_ratio_threshold(&mut self, value: f32) {
    self.compression_ratio_threshold = value;
  }

  /// Set [`Self::no_speech_threshold`].
  pub const fn set_no_speech_threshold(&mut self, value: f32) {
    self.no_speech_threshold = value;
  }

  /// Set [`Self::no_context`].
  pub const fn set_no_context(&mut self, value: bool) {
    self.no_context = value;
  }

  /// Set [`Self::suppress_blank`].
  pub const fn set_suppress_blank(&mut self, value: bool) {
    self.suppress_blank = value;
  }

  /// Set [`Self::suppress_non_speech_tokens`].
  pub const fn set_suppress_non_speech_tokens(&mut self, value: bool) {
    self.suppress_non_speech_tokens = value;
  }

  /// Set [`Self::initial_prompt`].
  pub fn set_initial_prompt(&mut self, value: Option<SmolStr>) {
    self.initial_prompt = value;
  }

  /// Set [`Self::n_threads`].
  ///
  /// # Panics
  ///
  /// Panics if `value < 1`. whisper.cpp's decoder loops allocate
  /// `std::vector<std::thread>(n_threads - 1)` when `n_threads`
  /// isn't exactly `1`, so `0` underflows to a huge allocation
  /// request and any negative value aborts inside the worker.
  /// Codex round-34 flagged this as a high-severity FFI footgun.
  pub const fn set_n_threads(&mut self, value: i32) {
    assert!(
      value >= 1,
      "n_threads must be >= 1; whisper.cpp would underflow / abort otherwise"
    );
    self.n_threads = value;
  }

  // --- Builder-style (consuming) -------------------------------

  /// Builder-style override for [`Self::language_hint`].
  pub fn with_language_hint(mut self, value: Option<Lang>) -> Self {
    self.language_hint = value;
    self
  }

  /// Builder-style override for [`Self::strategy`].
  pub const fn with_strategy(mut self, value: SamplingStrategy) -> Self {
    self.strategy = value;
    self
  }

  /// Builder-style override for [`Self::initial_temperature`].
  pub const fn with_initial_temperature(mut self, value: f32) -> Self {
    self.initial_temperature = value;
    self
  }

  /// Builder-style override for [`Self::temperature_increment`].
  pub const fn with_temperature_increment(mut self, value: f32) -> Self {
    self.temperature_increment = value;
    self
  }

  /// Builder-style override for [`Self::max_attempts`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See
  /// [`Self::set_max_attempts`].
  pub const fn with_max_attempts(mut self, value: u8) -> Self {
    assert!(
      value > 0,
      "max_attempts must be > 0 (got 0); use 1 for a single attempt with no retries"
    );
    self.max_attempts = value;
    self
  }

  /// Builder-style override for [`Self::log_prob_threshold`].
  pub const fn with_log_prob_threshold(mut self, value: f32) -> Self {
    self.log_prob_threshold = value;
    self
  }

  /// Builder-style override for [`Self::compression_ratio_threshold`].
  pub const fn with_compression_ratio_threshold(mut self, value: f32) -> Self {
    self.compression_ratio_threshold = value;
    self
  }

  /// Builder-style override for [`Self::no_speech_threshold`].
  pub const fn with_no_speech_threshold(mut self, value: f32) -> Self {
    self.no_speech_threshold = value;
    self
  }

  /// Builder-style override for [`Self::no_context`].
  pub const fn with_no_context(mut self, value: bool) -> Self {
    self.no_context = value;
    self
  }

  /// Builder-style override for [`Self::suppress_blank`].
  pub const fn with_suppress_blank(mut self, value: bool) -> Self {
    self.suppress_blank = value;
    self
  }

  /// Builder-style override for [`Self::suppress_non_speech_tokens`].
  pub const fn with_suppress_non_speech_tokens(mut self, value: bool) -> Self {
    self.suppress_non_speech_tokens = value;
    self
  }

  /// Builder-style override for [`Self::initial_prompt`].
  pub fn with_initial_prompt(mut self, value: Option<SmolStr>) -> Self {
    self.initial_prompt = value;
    self
  }

  /// Builder-style override for [`Self::n_threads`].
  ///
  /// # Panics
  ///
  /// Panics if `value < 1`. See [`Self::set_n_threads`].
  pub const fn with_n_threads(mut self, value: i32) -> Self {
    assert!(
      value >= 1,
      "n_threads must be >= 1; whisper.cpp would underflow / abort otherwise"
    );
    self.n_threads = value;
    self
  }
}

impl Default for AsrParams {
  fn default() -> Self {
    Self::new()
  }
}

/// Decoder sampling strategy.
///
/// `snake_case` external representation when `serde` is on,
/// matching the silero options pattern (`{ "greedy": { "best_of": 1 } }`
/// / `{ "beam_search": { "beam_size": 5, "patience": -1.0 } }`).
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SamplingStrategy {
  /// Greedy decoding: pick the token with highest probability
  /// after considering `best_of` candidates.
  Greedy {
    /// Candidates considered per token.
    best_of: i32,
  },
  /// Beam search.
  BeamSearch {
    /// Maximum beam width.
    beam_size: i32,
    /// Patience factor (whisper.cpp ignores this as of v1.7.6;
    /// keep `-1.0` to match whisper-rs default).
    patience: f32,
  },
}

impl Default for SamplingStrategy {
  fn default() -> Self {
    Self::BeamSearch {
      beam_size: 5,
      patience: -1.0,
    }
  }
}

/// Result of one chunk's ASR inference. Fields are private; use
/// [`AsrResult::new`] and accessors.
///
/// The `runs` field carries the script-dispatcher's per-language
/// breakdown of the transcript (see
/// [`crate::align::dispatch_segments`]). Whisper workers that have
/// access to the live `whisper.cpp` segment list populate it; the
/// alignment worker uses it to dispatch each language slice to the
/// matching [`crate::runner::Aligner`]. An empty `runs` is valid —
/// it falls back to whole-chunk alignment using
/// [`AsrResult::language`], identical to the pre-script-dispatch
/// behaviour.
#[derive(Clone, Debug)]
pub struct AsrResult {
  text: SmolStr,
  language: Lang,
  avg_logprob: f32,
  no_speech_prob: f32,
  temperature: f32,
  runs: Vec<crate::align::Run>,
}

impl AsrResult {
  /// Construct from all fields except [`Self::runs`], which defaults
  /// to empty. Callers that have access to the whisper segment list
  /// (e.g. the runner's whisper worker) should populate it via
  /// [`Self::with_runs`] / [`Self::set_runs`] so the alignment stage
  /// can dispatch per-language.
  pub fn new(
    text: SmolStr,
    language: Lang,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
  ) -> Self {
    Self {
      text,
      language,
      avg_logprob,
      no_speech_prob,
      temperature,
      runs: Vec::new(),
    }
  }

  /// Transcribed text, verbatim from whisper.
  pub fn text(&self) -> &SmolStr {
    &self.text
  }

  /// Detected (or hint-confirmed) language.
  pub fn language(&self) -> &Lang {
    &self.language
  }

  /// Mean log-probability over emitted tokens.
  pub const fn avg_logprob(&self) -> f32 {
    self.avg_logprob
  }

  /// No-speech probability.
  pub const fn no_speech_prob(&self) -> f32 {
    self.no_speech_prob
  }

  /// Final temperature used after fallback retries.
  pub const fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Per-language script-dispatcher runs over the transcript.
  /// Empty when no dispatcher was run (or when the chunk produced
  /// no segments).
  pub fn runs(&self) -> &[crate::align::Run] {
    &self.runs
  }

  /// Builder-style: replace the runs vector.
  #[must_use]
  pub fn with_runs(mut self, runs: Vec<crate::align::Run>) -> Self {
    self.runs = runs;
    self
  }

  /// In-place: replace the runs vector.
  pub fn set_runs(&mut self, runs: Vec<crate::align::Run>) {
    self.runs = runs;
  }
}

/// Result of one chunk's word-level alignment. Empty `words` is a
/// valid result (e.g., when whisper text was empty or normalisation
/// produced an empty string). Fields are private; use
/// [`AlignmentResult::new`] and accessors.
#[derive(Clone, Debug)]
#[cfg(feature = "alignment")]
pub struct AlignmentResult {
  words: Vec<crate::types::Word>,
}

#[cfg(feature = "alignment")]
impl AlignmentResult {
  /// Construct from a list of per-word alignment entries.
  pub fn new(words: Vec<crate::types::Word>) -> Self {
    Self { words }
  }

  /// Per-word alignment entries.
  pub fn words(&self) -> &[crate::types::Word] {
    &self.words
  }

  /// Consume the result, returning ownership of the words vector.
  pub fn into_words(self) -> Vec<crate::types::Word> {
    self.words
  }
}

/// Stub when alignment feature is off so other code paths can refer
/// to the type without a feature gate.
#[derive(Clone, Debug)]
#[cfg(not(feature = "alignment"))]
pub struct AlignmentResult {
  words: Vec<crate::types::Word>,
}

#[cfg(not(feature = "alignment"))]
impl AlignmentResult {
  /// Construct from a list of per-word alignment entries (always
  /// empty without the `alignment` feature).
  pub fn new(words: Vec<crate::types::Word>) -> Self {
    Self { words }
  }

  /// Per-word alignment entries (always empty without the
  /// `alignment` feature).
  pub fn words(&self) -> &[crate::types::Word] {
    &self.words
  }

  /// Consume the result, returning ownership of the words vector.
  pub fn into_words(self) -> Vec<crate::types::Word> {
    self.words
  }
}

/// A directive the runner consumes.
#[derive(Debug)]
pub enum Command {
  /// Run ASR on the chunk's audio. The runner ships the result
  /// back via `Transcriber::handle_asr`.
  Asr {
    /// Chunk identity.
    chunk_id: ChunkId,
    /// Chunk audio (16 kHz f32 mono).
    samples: Arc<[f32]>,
    /// Sample rate of the audio. Always
    /// [`crate::time::SAMPLE_RATE_HZ`] in v1; the field exists
    /// for forward compatibility.
    sample_rate: u32,
    /// ASR knobs for this chunk.
    params: AsrParams,
  },

  /// Run word-level alignment on the chunk's audio + transcribed
  /// text. Only emitted when the caller is configured to dispatch
  /// alignment work. Result returns via
  /// [`super::Transcriber::handle_alignment`].
  ///
  /// **Coordinate-space contract.** Codex round-37 round-18
  /// [high]: `sub_segments` here is in the caller's **output
  /// timebase** (the timebase of the first `handle_samples`
  /// `Timestamp`). For human readers / downstream consumers
  /// this is the "natural" form. **The aligner does NOT accept
  /// it directly** — `Aligner::align_chunk` requires
  /// chunk-local 1/16000 sample ranges and the dispatcher
  /// hard-errors on any other timebase. Convert via
  /// [`super::Transcriber::chunk_sub_segments_samples`] (which
  /// returns stream-coordinate samples) plus
  /// [`super::Transcriber::chunk_first_sample`] before
  /// constructing an [`crate::AlignWorkItem`]:
  ///
  /// ```ignore
  /// let chunk_first = transcriber.chunk_first_sample(chunk_id).unwrap();
  /// let raw_subs = transcriber.chunk_sub_segments_samples(chunk_id).unwrap();
  /// let tb_16k = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
  /// let aligner_subs: Vec<TimeRange> = raw_subs.iter()
  ///   .map(|(s, e)| TimeRange::new(
  ///     (*s as i64) - (chunk_first as i64),
  ///     (*e as i64) - (chunk_first as i64),
  ///     tb_16k))
  ///   .collect();
  /// ```
  ///
  /// The `sub_segments` field stays in output timebase here so
  /// callers that DON'T drive alignment (e.g. they want raw
  /// VAD ranges for bookkeeping) get the natural form without
  /// a back-conversion. The runner module's worked example
  /// shows the full pump shape including the coordinate flip.
  Alignment {
    /// Chunk identity.
    chunk_id: ChunkId,
    /// Chunk audio (16 kHz f32 mono).
    samples: Arc<[f32]>,
    /// Sub-VAD-segments in the caller's **output** timebase
    /// (not the aligner's chunk-local 1/16000 form). See the
    /// variant doc above for the coordinate-flip helper.
    sub_segments: Vec<TimeRange>,
    /// Whisper's transcribed text.
    text: SmolStr,
    /// Detected language.
    language: Lang,
    /// Script-dispatcher per-language runs derived from the
    /// whisper segments. Empty when the runner did not populate
    /// it (legacy single-language path); the alignment worker
    /// then falls back to whole-chunk alignment keyed on
    /// `language`.
    runs: Vec<crate::align::Run>,
  },
}

/// Compact override applied per-packet. Each `Some` field replaces
/// the corresponding default from the runner's `AsrParams` for chunks
/// produced from the packet. Fields are private; use the builder-style
/// `with_*` accessors or the `set_*` mutators.
///
/// Serde encoding: every field is `serde(default)` and skips
/// serialisation when `None`, so the empty override round-trips
/// as `{}`. The double-`Option` on `language_hint` /
/// `initial_prompt` is the override-vs-clear distinction:
/// `Some(None)` clears the field on the underlying `AsrParams`,
/// `Some(Some(_))` sets it, and `None` leaves the field
/// untouched.
///
/// **Serde wire form for `Option<Option<T>>` fields.** Codex
/// round-37: the derived `Option<Option<T>>` impl collapses
/// "field absent" and "field present with null" into the same
/// outer `None` — so `{"language_hint": null}` would be
/// indistinguishable from omitting the field, defeating the
/// "clear this override" intent. Both `language_hint` and
/// `initial_prompt` carry a custom `deserialize_with` that
/// preserves the distinction:
///
/// - **field absent** → outer `None` (serde uses `default`)
/// - **field set to JSON `null`** → `Some(None)` (clear)
/// - **field set to value** → `Some(Some(value))` (set)
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AsrParamsOverride {
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      deserialize_with = "deserialize_double_option_lang"
    )
  )]
  language_hint: Option<Option<Lang>>,
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  strategy: Option<SamplingStrategy>,
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  initial_temperature: Option<f32>,
  #[cfg_attr(
    feature = "serde",
    serde(
      default,
      skip_serializing_if = "Option::is_none",
      deserialize_with = "deserialize_double_option_smolstr"
    )
  )]
  initial_prompt: Option<Option<SmolStr>>,
}

/// Double-option deserializer for `language_hint`. See the
/// type-level doc on [`AsrParamsOverride`] for the absent / null /
/// value contract that this helper implements. Codex round-37.
#[cfg(feature = "serde")]
fn deserialize_double_option_lang<'de, D>(d: D) -> Result<Option<Option<Lang>>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  // serde's `default` triggers when the FIELD is absent; when
  // the field is present with `null`, this function is called
  // and `Option::deserialize` sees the null and returns
  // `Ok(None)` — which we wrap in `Some(None)` to mean "clear".
  // Any other value goes through `Lang`'s `Deserialize` impl
  // and lands in `Some(Some(value))`.
  Ok(Some(Option::<Lang>::deserialize(d)?))
}

/// Double-option deserializer for `initial_prompt`. See
/// [`deserialize_double_option_lang`].
#[cfg(feature = "serde")]
fn deserialize_double_option_smolstr<'de, D>(d: D) -> Result<Option<Option<SmolStr>>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  Ok(Some(Option::<SmolStr>::deserialize(d)?))
}

impl AsrParamsOverride {
  /// Construct an empty override (every field `None`). Equivalent
  /// to [`Default::default`] but `const fn`.
  pub const fn new() -> Self {
    Self {
      language_hint: None,
      strategy: None,
      initial_temperature: None,
      initial_prompt: None,
    }
  }

  /// Apply this sparse override on top of `base`, returning the
  /// merged `AsrParams`. `Some` fields on the override replace
  /// `base`'s; `None` fields leave `base` unchanged. Used at
  /// promote time by the dispatch (with the per-chunk
  /// `override_at_creation`) and by the runner's tests.
  pub fn apply_to(&self, base: &AsrParams) -> AsrParams {
    let mut out = base.clone();
    if let Some(opt_lang) = &self.language_hint {
      out.set_language_hint(opt_lang.clone());
    }
    if let Some(strategy) = self.strategy {
      out.set_strategy(strategy);
    }
    if let Some(t) = self.initial_temperature {
      out.set_initial_temperature(t);
    }
    if let Some(prompt) = &self.initial_prompt {
      out.set_initial_prompt(prompt.clone());
    }
    out
  }

  /// Override for [`AsrParams::language_hint`].
  pub const fn language_hint(&self) -> Option<&Option<Lang>> {
    self.language_hint.as_ref()
  }

  /// Override for [`AsrParams::strategy`].
  pub const fn strategy(&self) -> Option<SamplingStrategy> {
    self.strategy
  }

  /// Override for [`AsrParams::initial_temperature`].
  pub const fn initial_temperature(&self) -> Option<f32> {
    self.initial_temperature
  }

  /// Override for [`AsrParams::initial_prompt`].
  pub const fn initial_prompt(&self) -> Option<&Option<SmolStr>> {
    self.initial_prompt.as_ref()
  }

  /// Set [`Self::language_hint`].
  pub fn set_language_hint(&mut self, value: Option<Option<Lang>>) {
    self.language_hint = value;
  }

  /// Set [`Self::strategy`].
  pub const fn set_strategy(&mut self, value: Option<SamplingStrategy>) {
    self.strategy = value;
  }

  /// Set [`Self::initial_temperature`].
  pub const fn set_initial_temperature(&mut self, value: Option<f32>) {
    self.initial_temperature = value;
  }

  /// Set [`Self::initial_prompt`].
  pub fn set_initial_prompt(&mut self, value: Option<Option<SmolStr>>) {
    self.initial_prompt = value;
  }

  /// Builder-style override for [`Self::language_hint`].
  pub fn with_language_hint(mut self, value: Option<Option<Lang>>) -> Self {
    self.language_hint = value;
    self
  }

  /// Builder-style override for [`Self::strategy`].
  pub const fn with_strategy(mut self, value: Option<SamplingStrategy>) -> Self {
    self.strategy = value;
    self
  }

  /// Builder-style override for [`Self::initial_temperature`].
  pub const fn with_initial_temperature(mut self, value: Option<f32>) -> Self {
    self.initial_temperature = value;
    self
  }

  /// Builder-style override for [`Self::initial_prompt`].
  pub fn with_initial_prompt(mut self, value: Option<Option<SmolStr>>) -> Self {
    self.initial_prompt = value;
    self
  }
}

/// Used by the dispatch state machine to refer to a chunk's audio
/// + sub-segments without copying.
#[allow(dead_code)] // consumed by the dispatch state machine
pub(crate) type ChunkAudio = Arc<[f32]>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn asr_params_defaults_match_spec() {
    let p = AsrParams::default();
    match p.strategy {
      SamplingStrategy::BeamSearch {
        beam_size,
        patience,
      } => {
        assert_eq!(beam_size, 5);
        assert!((patience - -1.0).abs() < 1e-9);
      }
      _ => panic!("default should be BeamSearch"),
    }
    assert!((p.initial_temperature - 0.0).abs() < 1e-9);
    assert!((p.temperature_increment - 0.2).abs() < 1e-9);
    assert_eq!(p.max_attempts, 6);
    assert!(p.no_context);
  }

  /// Serde round-trip + partial-config contract for `AsrParams`.
  /// Mirrors silero's `test_serde` shape: tweak a non-default
  /// field, encode, decode, assert equal; then deserialise from
  /// `{}` and assert every field matches `Default::default`.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_serde_round_trip() {
    let mut p = AsrParams::default();
    p.set_initial_temperature(0.7);
    p.set_max_attempts(3);
    let json = serde_json::to_string(&p).expect("serialize");
    let back: AsrParams = serde_json::from_str(&json).expect("deserialize");
    assert!((back.initial_temperature() - 0.7).abs() < 1e-9);
    assert_eq!(back.max_attempts(), 3);
  }

  /// Codex round-33: `max_attempts = 0` would silently drop
  /// every chunk's ASR (the retry ladder iterates `0..0` and
  /// returns `AllTemperaturesFailed`). The setters panic; the
  /// `with_*` builder panics symmetrically.
  #[test]
  #[should_panic(expected = "max_attempts must be > 0")]
  fn set_max_attempts_zero_panics() {
    let mut p = AsrParams::default();
    p.set_max_attempts(0);
  }

  #[test]
  #[should_panic(expected = "max_attempts must be > 0")]
  fn with_max_attempts_zero_panics() {
    let _ = AsrParams::default().with_max_attempts(0);
  }

  // Codex round-34: n_threads validation.

  #[test]
  #[should_panic(expected = "n_threads must be >= 1")]
  fn set_n_threads_zero_panics() {
    let mut p = AsrParams::default();
    p.set_n_threads(0);
  }

  #[test]
  #[should_panic(expected = "n_threads must be >= 1")]
  fn set_n_threads_negative_panics() {
    let mut p = AsrParams::default();
    p.set_n_threads(-3);
  }

  #[test]
  #[should_panic(expected = "n_threads must be >= 1")]
  fn with_n_threads_zero_panics() {
    let _ = AsrParams::default().with_n_threads(0);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_zero_n_threads() {
    let json = r#"{"n_threads": 0}"#;
    let res: Result<AsrParams, _> = serde_json::from_str(json);
    assert!(res.is_err(), "n_threads=0 must be rejected");
    let err = res.err().unwrap().to_string();
    assert!(err.contains("n_threads must be >= 1"), "got {err:?}");
  }

  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_negative_n_threads() {
    let json = r#"{"n_threads": -2}"#;
    let res: Result<AsrParams, _> = serde_json::from_str(json);
    assert!(res.is_err(), "n_threads=-2 must be rejected");
  }

  /// Codex round-33: deserialized config must fail loudly on
  /// `max_attempts: 0` rather than producing a runner that
  /// silently drops every chunk.
  #[cfg(feature = "serde")]
  #[test]
  fn deserialize_rejects_zero_max_attempts() {
    let json = r#"{"max_attempts": 0}"#;
    let res: Result<AsrParams, _> = serde_json::from_str(json);
    assert!(res.is_err(), "max_attempts=0 must be rejected");
    let err = res.err().unwrap().to_string();
    assert!(
      err.contains("max_attempts must be > 0"),
      "expected diagnostic, got {err:?}"
    );
  }

  /// Partial config — `{}` deserialises to defaults thanks to
  /// per-field `serde(default = "...")`. Exercises the
  /// silero-shape contract that human-edited configs only need
  /// to mention the fields they want to change.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_serde_empty_yields_defaults() {
    let p: AsrParams = serde_json::from_str("{}").expect("deserialize empty");
    assert_eq!(
      p.initial_temperature(),
      AsrParams::default().initial_temperature()
    );
    assert_eq!(p.max_attempts(), AsrParams::default().max_attempts());
    assert_eq!(p.no_context(), AsrParams::default().no_context());
    // `language_hint` / `initial_prompt` round-trip as absent.
    assert!(p.language_hint().is_none());
    assert!(p.initial_prompt().is_none());
  }

  // --- Codex round-37: AsrParamsOverride double-option serde ---

  /// Field absent → outer `None` (no override on this field).
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_absent_means_no_override() {
    let ovr: AsrParamsOverride = serde_json::from_str("{}").expect("deserialize empty");
    assert!(
      ovr.language_hint().is_none(),
      "absent field must mean None (no override)"
    );
    assert!(
      ovr.initial_prompt().is_none(),
      "absent field must mean None (no override)"
    );
  }

  /// Field set to JSON `null` → `Some(None)` (clear the override).
  /// Pre-fix this was indistinguishable from "absent" because
  /// the derived `Option<Option<T>>` impl collapsed both to
  /// outer `None`.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_null_means_clear() {
    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"language_hint": null}"#).expect("deserialize null");
    match ovr.language_hint() {
      Some(None) => {}
      other => panic!("JSON null on language_hint must produce Some(None) (clear); got {other:?}"),
    }

    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"initial_prompt": null}"#).expect("deserialize null");
    match ovr.initial_prompt() {
      Some(None) => {}
      other => panic!("JSON null on initial_prompt must produce Some(None) (clear); got {other:?}"),
    }
  }

  /// Field set to a real value → `Some(Some(value))` (set the
  /// override). Lang's case-insensitive ISO deserializer is
  /// preserved through the double-option helper.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_value_means_set() {
    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"language_hint": "EN"}"#).expect("deserialize value");
    match ovr.language_hint() {
      Some(Some(Lang::En)) => {}
      other => panic!("expected Some(Some(Lang::En)); got {other:?}"),
    }

    let ovr: AsrParamsOverride =
      serde_json::from_str(r#"{"initial_prompt": "hint"}"#).expect("deserialize value");
    match ovr.initial_prompt() {
      Some(Some(s)) if s.as_str() == "hint" => {}
      other => panic!("expected Some(Some(\"hint\")); got {other:?}"),
    }
  }

  /// Round-trip the three states through serialize → deserialize:
  /// absent must stay absent; Some(None) must round-trip via null;
  /// Some(Some(v)) must round-trip via the value form.
  #[cfg(feature = "serde")]
  #[test]
  fn asr_params_override_serde_round_trips_three_states() {
    // Absent.
    let mut ovr_absent = AsrParamsOverride::new();
    ovr_absent.set_initial_temperature(Some(0.7)); // unrelated field set so JSON isn't empty
    let json = serde_json::to_string(&ovr_absent).unwrap();
    assert!(
      !json.contains("language_hint") && !json.contains("initial_prompt"),
      "absent fields must skip-serialize; got {json}"
    );
    let back: AsrParamsOverride = serde_json::from_str(&json).unwrap();
    assert!(back.language_hint().is_none());
    assert!(back.initial_prompt().is_none());

    // Some(None) — clear.
    let ovr_clear = AsrParamsOverride::new()
      .with_language_hint(Some(None))
      .with_initial_prompt(Some(None));
    let json = serde_json::to_string(&ovr_clear).unwrap();
    assert!(json.contains("\"language_hint\":null"), "got {json}");
    assert!(json.contains("\"initial_prompt\":null"), "got {json}");
    let back: AsrParamsOverride = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.language_hint(), Some(None)));
    assert!(matches!(back.initial_prompt(), Some(None)));

    // Some(Some(_)) — set.
    let ovr_set = AsrParamsOverride::new()
      .with_language_hint(Some(Some(Lang::En)))
      .with_initial_prompt(Some(Some(SmolStr::new("hint"))));
    let json = serde_json::to_string(&ovr_set).unwrap();
    let back: AsrParamsOverride = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.language_hint(), Some(Some(Lang::En))));
    assert!(
      matches!(back.initial_prompt(), Some(Some(s)) if s.as_str() == "hint"),
      "got {:?}",
      back.initial_prompt()
    );
  }

  /// `SamplingStrategy` snake_case external representation,
  /// matching the silero `SampleRate` precedent.
  #[cfg(feature = "serde")]
  #[test]
  fn sampling_strategy_serde_uses_snake_case() {
    let strat = SamplingStrategy::Greedy { best_of: 1 };
    let json = serde_json::to_string(&strat).expect("serialize");
    assert!(
      json.contains("greedy"),
      "external rep must be snake_case; got {json}"
    );
    let back: SamplingStrategy = serde_json::from_str(&json).expect("deserialize");
    match back {
      SamplingStrategy::Greedy { best_of } => assert_eq!(best_of, 1),
      _ => panic!("expected Greedy"),
    }
  }
}
