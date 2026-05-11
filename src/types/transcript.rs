//! `Transcript` and `Word` — the per-chunk emission unit and its
//! word-level alignment entries.

use std::vec::Vec;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang};

/// Per-chunk transcription result.
///
/// One emitted `MergedChunk` produces exactly one `Transcript`.
/// Fields are private; access is via getters per the findit-studio
/// convention.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transcript {
  range: TimeRange,
  language: Lang,
  text: SmolStr,
  words: Vec<Word>,
  avg_logprob: f32,
  no_speech_prob: f32,
  temperature: f32,
  vad_segments: Vec<TimeRange>,
  chunk_id: ChunkId,
}

impl Transcript {
  /// Crate-private constructor used by the dispatch state machine.
  /// Tests in this crate use it directly via the `for_test`
  /// helper below.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    range: TimeRange,
    language: Lang,
    text: SmolStr,
    words: Vec<Word>,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
    vad_segments: Vec<TimeRange>,
    chunk_id: ChunkId,
  ) -> Self {
    Self {
      range,
      language,
      text,
      words,
      avg_logprob,
      no_speech_prob,
      temperature,
      vad_segments,
      chunk_id,
    }
  }

  /// Bounds of the merged chunk in the caller's output timebase
  /// (the timebase of the first `handle_samples` Timestamp).
  pub fn range(&self) -> TimeRange {
    self.range
  }

  /// Detected (or hint-supplied) language for this chunk.
  pub fn language(&self) -> &Lang {
    &self.language
  }

  /// Verbatim Whisper output for this chunk: includes punctuation,
  /// casing, and any model-emitted special characters. The
  /// word-level `words[].text()` values are matching original
  /// surface forms with punctuation and casing preserved.
  pub fn text(&self) -> &str {
    self.text.as_str()
  }

  /// Word-level alignment results, in time order. Empty when
  /// alignment was disabled, the chunk's language has no
  /// registered aligner with `AlignmentFallback::SkipChunk`, or
  /// some words landed in silence-masked regions and were dropped.
  pub fn words(&self) -> &[Word] {
    &self.words
  }

  /// Whisper's mean log-probability over emitted tokens.
  pub fn avg_logprob(&self) -> f32 {
    self.avg_logprob
  }

  /// Whisper's no-speech probability for this chunk.
  pub fn no_speech_prob(&self) -> f32 {
    self.no_speech_prob
  }

  /// Final decoding temperature after fallback retries.
  pub fn temperature(&self) -> f32 {
    self.temperature
  }

  /// Sub-VAD-segments that composed this merged chunk, in the
  /// caller's output timebase.
  pub fn vad_segments(&self) -> &[TimeRange] {
    &self.vad_segments
  }

  /// Monotonic chunk identity within a single `Transcriber`
  /// lifetime.
  pub fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }
}

/// One word in a [`Transcript`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Word {
  text: SmolStr,
  range: TimeRange,
  score: f32,
  /// Per-word language tag — `Some(lang)` when the script-dispatch
  /// pass split this word out of a code-switched chunk into a
  /// language-tagged run; `None` when the word came from a
  /// non-dispatched path (e.g. the alignment-disabled placeholder
  /// or pre-dispatch test fixtures). [medium]
  /// recommended preserving run language so a downstream consumer
  /// can route per-word output by language without reverse-mapping
  /// from text or timing.
  #[cfg_attr(
    feature = "serde",
    serde(default, skip_serializing_if = "Option::is_none")
  )]
  language: Option<Lang>,
}

impl Word {
  /// Crate-private constructor used by the alignment pipeline.
  /// `language` is `None` for the parent-language path (no
  /// script-dispatch); set via [`Self::with_language`] when the
  /// word originated from a language-tagged run.
  pub(crate) fn new(text: SmolStr, range: TimeRange, score: f32) -> Self {
    Self {
      text,
      range,
      score,
      language: None,
    }
  }

  /// Builder-style override for [`Self::language`]. Used by the
  /// alignment dispatcher to tag a word with its run's language
  /// after `Aligner::align` returns.
  #[must_use]
  pub fn with_language(mut self, language: Option<Lang>) -> Self {
    self.language = language;
    self
  }

  /// Set [`Self::language`] in place.
  pub fn set_language(&mut self, language: Option<Lang>) {
    self.language = language;
  }

  /// Original surface form of the word, preserving casing and
  /// punctuation as Whisper emitted them. Recovered after CTC
  /// alignment via the normalisation map; the word that wav2vec2
  /// actually aligned was the lowercased, punctuation-stripped
  /// form.
  pub fn text(&self) -> &str {
    self.text.as_str()
  }

  /// Sample-accurate range of the word in the caller's output
  /// timebase. Half-open. When silence-aware alignment drops words
  /// inside zero-masked regions, this range covers only the frames
  /// the Viterbi path attributed to the word — never frames inside
  /// masked regions, never adjacent words' frames.
  pub fn range(&self) -> TimeRange {
    self.range
  }

  /// Alignment confidence in `[0, 1]`, NaN-free. Defined as
  /// `exp(mean(log_p_t))` where `log_p_t` is the per-frame
  /// log-probability of the chosen vocab item along the Viterbi
  /// path for the frames spanning this word.
  pub fn score(&self) -> f32 {
    self.score
  }

  /// Per-word language, when the script-dispatch pass tagged this
  /// word's run. `None` when alignment was driven without
  /// dispatch (e.g. the legacy single-language path or the
  /// placeholder builder used in tests). Whispery's runner
  /// always populates this for words emitted via
  /// `crate::align::dispatch`.
  pub fn language(&self) -> Option<&Lang> {
    self.language.as_ref()
  }
}

#[cfg(test)]
pub(crate) mod for_test {
  //! Test-only constructors. Crate-private to avoid leaking into
  //! the public API while keeping the dispatch and alignment
  //! tests concise.

  use super::*;
  use core::num::NonZeroU32;

  pub(crate) fn ms_timebase() -> mediatime::Timebase {
    mediatime::Timebase::new(1, NonZeroU32::new(1000).unwrap())
  }

  pub(crate) fn transcript(chunk_id: u64, text: &str, words: Vec<Word>) -> Transcript {
    let tb = ms_timebase();
    let range = TimeRange::new(0, 1000, tb);
    Transcript::new(
      range,
      Lang::En,
      SmolStr::new(text),
      words,
      -0.5,
      0.05,
      0.0,
      vec![range],
      ChunkId::from_raw(chunk_id),
    )
  }

  pub(crate) fn word(text: &str, start_ms: i64, end_ms: i64, score: f32) -> Word {
    Word::new(
      SmolStr::new(text),
      TimeRange::new(start_ms, end_ms, ms_timebase()),
      score,
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn transcript_round_trip() {
    let t = for_test::transcript(
      7,
      "hello world",
      vec![
        for_test::word("hello", 0, 500, 0.95),
        for_test::word("world", 500, 1000, 0.92),
      ],
    );
    assert_eq!(t.text(), "hello world");
    assert_eq!(t.chunk_id().as_u64(), 7);
    assert_eq!(t.words().len(), 2);
    assert_eq!(t.words()[0].text(), "hello");
    assert_eq!(t.words()[1].score(), 0.92);
  }
}
