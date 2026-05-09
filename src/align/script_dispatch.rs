//! Script-dispatch — split a Whisper segment into language-tagged
//! [`Run`]s using per-character script classification.
//!
//! A Whisper segment is one block of decoded text plus its tokens.
//! `dispatch` walks each segment character-by-character, classifies
//! each character against the [`crate::align::script`] rules, and
//! groups consecutive same-language characters into a [`Run`].
//! Each run carries its own audio time bounds, derived from
//! whichever timing source is available (DTW preferred, segment
//! envelope as fallback, whole-clip sentinel as last resort —
//! recorded on [`BoundsSource`]).
//!
//! The dispatcher is generic over [`SegmentLike`] so unit tests can
//! exercise it without constructing a real [`whispercpp::Segment`]
//! (which is an FFI projection with a private constructor). The
//! `runner`-feature [`dispatch`] entry point wraps real whispercpp
//! segments through the trait; tests build mock segments directly.

use alloc::vec::Vec;

use smol_str::SmolStr;

use crate::align::script::{CharClass, SegmentContext, script_to_lang};
use crate::types::Lang;

/// One language-tagged slice of a Whisper segment, with its own
/// audio time bounds and a record of how those bounds were
/// derived.
///
/// Fields are private — accessors mirror the project's
/// [`crate::types::Word`] convention. `with_*` consumes `self`
/// builder-style; `set_*` mutates in place.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Run {
  language: Lang,
  text: SmolStr,
  audio_t0_ms: i64,
  audio_t1_ms: i64,
  source_segment_idx: i32,
  bounds_source: BoundsSource,
}

impl Run {
  /// Crate-private constructor used by [`dispatch_segments`] and
  /// the runner-feature [`dispatch`] wrapper. External callers
  /// build runs by going through `dispatch`.
  pub(crate) const fn new(
    language: Lang,
    text: SmolStr,
    audio_t0_ms: i64,
    audio_t1_ms: i64,
    source_segment_idx: i32,
    bounds_source: BoundsSource,
  ) -> Self {
    Self {
      language,
      text,
      audio_t0_ms,
      audio_t1_ms,
      source_segment_idx,
      bounds_source,
    }
  }

  /// Detected language of this run.
  #[must_use]
  pub fn language(&self) -> &Lang {
    &self.language
  }

  /// Verbatim text of this run, preserving casing, punctuation,
  /// and any leading/trailing whitespace that was carried into the
  /// run from neighbouring concrete characters.
  #[must_use]
  pub fn text(&self) -> &str {
    self.text.as_str()
  }

  /// Run start time, in milliseconds. Origin matches the
  /// underlying segment's timing source (DTW-derived or segment
  /// envelope); see [`Self::bounds_source`].
  #[must_use]
  pub const fn audio_t0_ms(&self) -> i64 {
    self.audio_t0_ms
  }

  /// Run end time, in milliseconds. Half-open with [`Self::audio_t0_ms`].
  #[must_use]
  pub const fn audio_t1_ms(&self) -> i64 {
    self.audio_t1_ms
  }

  /// Index of the parent Whisper segment this run was carved out
  /// of. Useful for telemetry: a single segment producing multiple
  /// runs is the code-switch case.
  #[must_use]
  pub const fn source_segment_idx(&self) -> i32 {
    self.source_segment_idx
  }

  /// Which timing source produced [`Self::audio_t0_ms`] /
  /// [`Self::audio_t1_ms`]. Drives downstream telemetry that
  /// counts DTW-vs-fallback usage per run.
  #[must_use]
  pub const fn bounds_source(&self) -> BoundsSource {
    self.bounds_source
  }

  /// Builder-style: replace the run's language. Consumes `self`
  /// to allow chaining without intermediate bindings. Not
  /// `const fn` because [`Lang`] is non-`Copy` (the
  /// `Lang::Other(SmolStr)` variant); replacing it must drop the
  /// previous value, which `const fn` forbids.
  #[must_use]
  pub fn with_language(mut self, language: Lang) -> Self {
    self.language = language;
    self
  }

  /// In-place: replace the run's language.
  pub fn set_language(&mut self, language: Lang) {
    self.language = language;
  }

  /// Builder-style: replace the run's text.
  #[must_use]
  pub fn with_text(mut self, text: SmolStr) -> Self {
    self.text = text;
    self
  }

  /// In-place: replace the run's text.
  pub fn set_text(&mut self, text: SmolStr) {
    self.text = text;
  }

  /// Builder-style: replace the run's start time.
  #[must_use]
  pub const fn with_audio_t0_ms(mut self, audio_t0_ms: i64) -> Self {
    self.audio_t0_ms = audio_t0_ms;
    self
  }

  /// In-place: replace the run's start time.
  pub const fn set_audio_t0_ms(&mut self, audio_t0_ms: i64) {
    self.audio_t0_ms = audio_t0_ms;
  }

  /// Builder-style: replace the run's end time.
  #[must_use]
  pub const fn with_audio_t1_ms(mut self, audio_t1_ms: i64) -> Self {
    self.audio_t1_ms = audio_t1_ms;
    self
  }

  /// In-place: replace the run's end time.
  pub const fn set_audio_t1_ms(&mut self, audio_t1_ms: i64) {
    self.audio_t1_ms = audio_t1_ms;
  }

  /// Builder-style: replace the source segment index.
  #[must_use]
  pub const fn with_source_segment_idx(mut self, source_segment_idx: i32) -> Self {
    self.source_segment_idx = source_segment_idx;
    self
  }

  /// In-place: replace the source segment index.
  pub const fn set_source_segment_idx(&mut self, source_segment_idx: i32) {
    self.source_segment_idx = source_segment_idx;
  }

  /// Builder-style: replace the bounds-source tag.
  #[must_use]
  pub const fn with_bounds_source(mut self, bounds_source: BoundsSource) -> Self {
    self.bounds_source = bounds_source;
    self
  }

  /// In-place: replace the bounds-source tag.
  pub const fn set_bounds_source(&mut self, bounds_source: BoundsSource) {
    self.bounds_source = bounds_source;
  }
}

/// Origin of a [`Run`]'s `audio_t0_ms` / `audio_t1_ms` bounds.
///
/// The dispatcher prefers DTW (most accurate, derived from the
/// per-token cross-attention backtrace), falls back to the
/// segment envelope (whisper.cpp's standard timestamp-token path)
/// when DTW is not fully populated, and falls back further to
/// whole-clip sentinels when even the segment envelope is missing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundsSource {
  /// Bounds are min/max of [`SegmentLike::token_dtw_timestamps`]
  /// across the run's tokens. Every token in the run had a
  /// concrete DTW timestamp.
  Dtw,
  /// At least one token in the run had `t_dtw == None`; the run
  /// inherited the parent segment's envelope ([`SegmentLike::t0`]
  /// / [`SegmentLike::t1`]).
  Segment,
  /// Neither DTW nor segment envelope was usable. Bounds are
  /// [`i64::MIN`] / [`i64::MAX`] sentinels — caller must treat
  /// them as "unknown" and fall back to whole-clip timing
  /// downstream. Should not occur on real whisper.cpp output;
  /// guarded defensively.
  Wholeclip,
}

/// Trait abstraction over a Whisper segment.
///
/// The runner-feature [`dispatch`] function takes
/// `&[whispercpp::Segment<'_>]`; tests construct mock segments
/// implementing this trait directly. The trait keeps the
/// dispatch core decoupled from the FFI surface so script
/// classification can be exercised without spinning up a real
/// model context.
///
/// Timing units are whatever the implementor provides — the
/// dispatcher passes them through unchanged. Real whispercpp
/// segments report centiseconds; the dispatcher's DTW timestamps
/// also come in centiseconds; the runner-feature [`dispatch`]
/// wrapper converts to milliseconds before constructing [`Run`]s.
pub trait SegmentLike {
  /// Decoded text of the segment. May be empty.
  fn text(&self) -> &str;

  /// Segment start time, in centiseconds (matches whisper.cpp's
  /// native unit). [`i64::MIN`] signals "unavailable" — the
  /// dispatcher then escalates to [`BoundsSource::Wholeclip`].
  fn t0(&self) -> i64;

  /// Segment end time, in centiseconds.
  fn t1(&self) -> i64;

  /// Per-token DTW timestamps, in centiseconds, one per text
  /// token in decode order. `None` for tokens without DTW timing
  /// (DTW disabled, special token, segment skipped). Returning
  /// any `None` causes the run carved from this segment to fall
  /// back to [`BoundsSource::Segment`] for its bounds.
  ///
  /// The boxed iterator exists because trait methods cannot
  /// return `impl Trait` in stable Rust 2024 edition without
  /// generic associated types; a `Vec<Option<i64>>` is fine
  /// because the dispatcher consumes it once per segment.
  fn token_dtw_timestamps(&self) -> Vec<Option<i64>>;
}

/// Centiseconds → milliseconds. whisper.cpp's native time unit
/// is 10 ms; whispery's API is milliseconds. Multiplication is
/// exact for valid centisecond values (no rounding).
const fn cs_to_ms(cs: i64) -> i64 {
  cs.saturating_mul(10)
}

/// Core script-dispatch loop, generic over [`SegmentLike`].
///
/// Walks each segment, classifies each character with
/// [`script_to_lang`], and emits a [`Run`] every time the active
/// language changes. `Carry` characters (digits, punctuation,
/// whitespace, ambiguous scripts without a hint) extend the
/// current run; leading carries before any concrete classification
/// fold into the first concrete run that follows. Segments with
/// only carries (e.g. pure punctuation) are skipped — they have no
/// language to attach to.
///
/// `state_lang` is the transcriber's current language hint, used
/// for Latin disambiguation and as a fallback for ambiguous
/// scripts. See [`script_to_lang`] for the per-script rules.
///
/// Time bounds for every run carved from the same segment are
/// computed once per segment and shared across its runs: granular
/// per-run timing requires a per-token offset map the script
/// classifier alone cannot produce, so all runs from a segment
/// inherit the segment's bounds. Downstream alignment refines this
/// further.
#[must_use]
pub fn dispatch_segments<S: SegmentLike>(segments: &[S], state_lang: Option<Lang>) -> Vec<Run> {
  let mut runs = Vec::new();
  let state_lang_ref = state_lang.as_ref();

  for (idx, seg) in segments.iter().enumerate() {
    let text = seg.text();
    if text.is_empty() {
      continue;
    }
    let ctx = SegmentContext::from_text(text);

    let dtw = seg.token_dtw_timestamps();
    let (t0_ms, t1_ms, bounds_source) = compute_bounds(seg.t0(), seg.t1(), &dtw);

    // i32 cast: segment indices in whisper.cpp's API are i32; we
    // accept up to i32::MAX segments per state. Saturate on
    // overflow rather than truncate — wraparound would silently
    // alias telemetry across far-apart segments.
    let source_idx = i32::try_from(idx).unwrap_or(i32::MAX);

    // Single pass: collect runs by tracking the current language
    // and the byte range of the active run within `text`. Carry
    // characters are absorbed (their bytes extend the current
    // run); concrete characters either continue the run or flush
    // it and start a new one.
    //
    // `run_start` always tracks the byte offset where the current
    // run will begin when flushed. It starts at 0 — leading carry
    // bytes (whitespace, punctuation) sit "before" any concrete
    // language signal and naturally attach to the first concrete
    // run that follows. After a flush, `run_start` advances to
    // the boundary character's byte offset so the new run
    // contains the boundary char (and any carry that follows
    // before the next concrete char).
    let mut current_lang: Option<Lang> = None;
    let mut run_start: usize = 0;

    for (byte_idx, ch) in text.char_indices() {
      let class = script_to_lang(ch, ctx, state_lang_ref);
      match class {
        // Extend whatever run is active. If no run yet, the
        // carry sits as leading whitespace/punctuation that
        // will fold into the first concrete run when it
        // arrives — `run_start` is still 0, so those bytes
        // end up inside the first emitted run automatically.
        CharClass::Carry => {}
        CharClass::Lang(lang) => match &current_lang {
          None => {
            // First concrete classification — adopt this
            // language. `run_start` already sits at byte 0,
            // so leading carry bytes attach automatically.
            current_lang = Some(lang);
          }
          Some(active) if *active == lang => {
            // Same language — keep extending.
          }
          Some(_) => {
            // Language change — flush the active run up to
            // (but not including) this character, then start a
            // new run beginning here. Punctuation/whitespace
            // immediately preceding the boundary stays with
            // the earlier run (it was carry-extended into it).
            let end = byte_idx;
            let active = current_lang.take().expect("checked Some above");
            push_run(
              &mut runs,
              active,
              &text[run_start..end],
              t0_ms,
              t1_ms,
              source_idx,
              bounds_source,
            );
            run_start = end;
            current_lang = Some(lang);
          }
        },
      }
    }

    // Flush the trailing run, if any.
    if let Some(active) = current_lang {
      push_run(
        &mut runs,
        active,
        &text[run_start..],
        t0_ms,
        t1_ms,
        source_idx,
        bounds_source,
      );
    }
    // Segments containing only carry characters produce no runs —
    // there is no language to label them with.
  }

  runs
}

/// Append a single [`Run`] to `runs`, skipping empty text slices
/// (defensive — the dispatcher's flush points always have at
/// least one concrete character, but a future refactor that
/// flushes zero-length runs would silently corrupt downstream
/// alignment without this guard).
#[allow(clippy::too_many_arguments)]
fn push_run(
  runs: &mut Vec<Run>,
  language: Lang,
  text: &str,
  audio_t0_ms: i64,
  audio_t1_ms: i64,
  source_segment_idx: i32,
  bounds_source: BoundsSource,
) {
  if text.is_empty() {
    return;
  }
  runs.push(Run::new(
    language,
    SmolStr::new(text),
    audio_t0_ms,
    audio_t1_ms,
    source_segment_idx,
    bounds_source,
  ));
}

/// Resolve `(t0, t1, BoundsSource)` for one segment.
///
/// Preference order:
///
/// 1. **DTW**: every token in `dtw_cs` is `Some(_)`, and the
///    derived min/max range is non-empty. Both endpoints are
///    converted from centiseconds to milliseconds.
/// 2. **Segment**: `seg_t0_cs` and `seg_t1_cs` are both
///    non-sentinel ([`i64::MIN`] indicates "unavailable" per the
///    [`SegmentLike`] contract). Whisper.cpp's normal output
///    falls into this branch when DTW is disabled or partially
///    populated.
/// 3. **Wholeclip**: neither DTW nor segment timing is usable.
///    Returns `(i64::MIN, i64::MAX)` as sentinels — the caller
///    must treat them as "unknown" rather than literal times,
///    and downstream code should fall back to whole-clip timing
///    (the audio's full duration). This branch is defensive;
///    real whisper.cpp output should always populate `t0` /
///    `t1` on emitted segments.
fn compute_bounds(seg_t0_cs: i64, seg_t1_cs: i64, dtw_cs: &[Option<i64>]) -> (i64, i64, BoundsSource) {
  // Case 1: DTW. All tokens must have Some(_); the min/max must
  // form a non-empty range.
  if !dtw_cs.is_empty() && dtw_cs.iter().all(Option::is_some) {
    let mut iter = dtw_cs.iter().filter_map(|v| *v);
    if let Some(first) = iter.next() {
      let mut lo = first;
      let mut hi = first;
      for v in iter {
        if v < lo {
          lo = v;
        }
        if v > hi {
          hi = v;
        }
      }
      return (cs_to_ms(lo), cs_to_ms(hi), BoundsSource::Dtw);
    }
  }

  // Case 2: segment envelope. `i64::MIN` is the documented
  // sentinel for "missing."
  if seg_t0_cs != i64::MIN && seg_t1_cs != i64::MIN {
    return (cs_to_ms(seg_t0_cs), cs_to_ms(seg_t1_cs), BoundsSource::Segment);
  }

  // Case 3: defensive wholeclip fallback. The caller treats
  // these as "unknown" sentinels. We deliberately do NOT
  // saturate here — using the literal i64 extremes makes the
  // sentinel detectable downstream (any finite caller-supplied
  // bound will sit inside `[i64::MIN, i64::MAX]`).
  (i64::MIN, i64::MAX, BoundsSource::Wholeclip)
}

#[cfg(feature = "runner")]
mod runner_glue {
  //! Bridge `whispercpp::Segment<'_>` onto [`super::SegmentLike`].

  use alloc::vec::Vec;

  use super::SegmentLike;

  /// Newtype wrapper so we can implement the local trait for the
  /// foreign `whispercpp::Segment<'_>` without orphan-rule
  /// complications.
  pub(super) struct SegmentRef<'a, 'seg>(pub &'a whispercpp::Segment<'seg>);

  impl<'a, 'seg> SegmentLike for SegmentRef<'a, 'seg> {
    fn text(&self) -> &str {
      // Segment::text returns Result<&str>; a UTF-8 error means
      // the model emitted invalid UTF-8 (extremely unusual).
      // Treating it as empty text drops the segment from
      // dispatch — safer than panicking inside an alignment
      // pipeline. Real production paths log this upstream.
      self.0.text().unwrap_or("")
    }

    fn t0(&self) -> i64 {
      self.0.t0()
    }

    fn t1(&self) -> i64 {
      self.0.t1()
    }

    fn token_dtw_timestamps(&self) -> Vec<Option<i64>> {
      self.0.tokens_iter().map(|tok| tok.t_dtw()).collect()
    }
  }
}

/// Public entry point: dispatch real whisper.cpp segments into
/// language-tagged [`Run`]s.
///
/// Available with `feature = "runner"` (which pulls in the
/// `whispercpp` dependency that defines [`whispercpp::Segment`]).
/// Wraps each segment in a thin [`SegmentLike`] adapter and
/// delegates to [`dispatch_segments`].
///
/// `state_lang` is the transcriber's current language hint,
/// passed through unchanged for Latin / ambiguous-script
/// disambiguation.
#[cfg(feature = "runner")]
#[must_use]
pub fn dispatch(segments: &[whispercpp::Segment<'_>], state_lang: Option<Lang>) -> Vec<Run> {
  let wrapped: Vec<runner_glue::SegmentRef<'_, '_>> =
    segments.iter().map(runner_glue::SegmentRef).collect();
  dispatch_segments(&wrapped, state_lang)
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::string::String;
  use alloc::vec;

  /// Minimal mock implementing [`SegmentLike`] for unit tests.
  /// Times are in centiseconds (matches whisper.cpp's native unit
  /// and the dispatcher's own internal contract).
  struct MockSeg {
    text: String,
    t0_cs: i64,
    t1_cs: i64,
    dtw_cs: Vec<Option<i64>>,
  }

  impl SegmentLike for MockSeg {
    fn text(&self) -> &str {
      &self.text
    }
    fn t0(&self) -> i64 {
      self.t0_cs
    }
    fn t1(&self) -> i64 {
      self.t1_cs
    }
    fn token_dtw_timestamps(&self) -> Vec<Option<i64>> {
      self.dtw_cs.clone()
    }
  }

  fn seg(text: &str, t0_cs: i64, t1_cs: i64, dtw_cs: Vec<Option<i64>>) -> MockSeg {
    MockSeg {
      text: String::from(text),
      t0_cs,
      t1_cs,
      dtw_cs,
    }
  }

  #[test]
  fn empty_segments_produce_no_runs() {
    let runs = dispatch_segments::<MockSeg>(&[], None);
    assert!(runs.is_empty());
  }

  #[test]
  fn empty_text_segment_produces_no_runs() {
    let segs = vec![seg("", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert!(runs.is_empty());
  }

  #[test]
  fn pure_english_one_run() {
    let segs = vec![seg("hello world", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[0].text(), "hello world");
    assert_eq!(runs[0].source_segment_idx(), 0);
  }

  #[test]
  fn pure_chinese_one_run() {
    let segs = vec![seg("你好世界", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::Zh);
    assert_eq!(runs[0].text(), "你好世界");
  }

  #[test]
  fn pure_japanese_with_kana() {
    let segs = vec![seg("これは日本語です", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    // Single run because every char is Ja (kana → Ja, Han → Ja
    // by segment context).
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::Ja);
    assert_eq!(runs[0].text(), "これは日本語です");
  }

  #[test]
  fn pure_korean_with_hangul() {
    let segs = vec![seg("안녕하세요", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::Ko);
    assert_eq!(runs[0].text(), "안녕하세요");
  }

  #[test]
  fn english_chinese_codeswitch() {
    let segs = vec![seg("hello 你好", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[1].language(), &Lang::Zh);
    // Trailing space on the En run lands in the En run because
    // the space (carry) extends the active En run before the
    // first concrete Zh char flushes it.
    assert_eq!(runs[0].text(), "hello ");
    assert_eq!(runs[1].text(), "你好");
  }

  #[test]
  fn ja_zh_kana_precedence_makes_all_han_ja() {
    // Even if one Han char appears in a kana-flagged segment,
    // every Han char in that segment must read as Ja.
    let segs = vec![seg("漢字あ漢字", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::Ja);
  }

  #[test]
  fn hangul_makes_han_ko_in_segment() {
    let segs = vec![seg("漢字한국", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    // All Han chars in this segment fall through to Ko due to
    // the Hangul context, so the entire segment is one Ko run.
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::Ko);
    assert_eq!(runs[0].text(), "漢字한국");
  }

  #[test]
  fn punctuation_does_not_split_runs() {
    let segs = vec![seg("hello, world.", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[0].text(), "hello, world.");
  }

  #[test]
  fn digits_carry_into_active_run() {
    let segs = vec![seg("test 123", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[0].text(), "test 123");
  }

  #[test]
  fn leading_punctuation_attaches_to_first_run() {
    let segs = vec![seg("  hello", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].text(), "  hello");
  }

  #[test]
  fn pure_punctuation_produces_no_run() {
    let segs = vec![seg("...!?", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert!(runs.is_empty());
  }

  #[test]
  fn dtw_available_uses_dtw_bounds() {
    let segs = vec![seg(
      "hello",
      50,
      200,
      vec![Some(70), Some(90), Some(120), Some(180), Some(190)],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    // min/max in cs → ms.
    assert_eq!(runs[0].audio_t0_ms(), 700);
    assert_eq!(runs[0].audio_t1_ms(), 1900);
  }

  #[test]
  fn dtw_partial_falls_back_to_segment() {
    let segs = vec![seg(
      "hello",
      50,
      200,
      vec![Some(70), None, Some(120)],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Segment);
    assert_eq!(runs[0].audio_t0_ms(), 500);
    assert_eq!(runs[0].audio_t1_ms(), 2000);
  }

  #[test]
  fn dtw_absent_uses_segment() {
    let segs = vec![seg("hello", 50, 200, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Segment);
    assert_eq!(runs[0].audio_t0_ms(), 500);
    assert_eq!(runs[0].audio_t1_ms(), 2000);
  }

  #[test]
  fn segment_unavailable_falls_back_to_wholeclip() {
    let segs = vec![seg("hello", i64::MIN, i64::MIN, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Wholeclip);
    assert_eq!(runs[0].audio_t0_ms(), i64::MIN);
    assert_eq!(runs[0].audio_t1_ms(), i64::MAX);
  }

  #[test]
  fn state_lang_disambiguates_latin_to_es() {
    let segs = vec![seg("hola", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, Some(Lang::Es));
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::Es);
  }

  #[test]
  fn non_latin_state_lang_does_not_bleed_into_latin() {
    // state_lang = Zh; Latin chars must NOT become Zh — they
    // fall back to En per the spec's defensive rule.
    let segs = vec![seg("hello", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, Some(Lang::Zh));
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::En);
  }

  #[test]
  fn multiple_segments_preserve_indices() {
    let segs = vec![
      seg("hello", 0, 50, vec![]),
      seg("你好", 50, 100, vec![]),
      seg("world", 100, 150, vec![]),
    ];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].source_segment_idx(), 0);
    assert_eq!(runs[1].source_segment_idx(), 1);
    assert_eq!(runs[2].source_segment_idx(), 2);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[1].language(), &Lang::Zh);
    assert_eq!(runs[2].language(), &Lang::En);
  }

  #[test]
  fn run_accessors_round_trip() {
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      100,
      200,
      3,
      BoundsSource::Dtw,
    );
    assert_eq!(r.language(), &Lang::En);
    assert_eq!(r.text(), "hi");
    assert_eq!(r.audio_t0_ms(), 100);
    assert_eq!(r.audio_t1_ms(), 200);
    assert_eq!(r.source_segment_idx(), 3);
    assert_eq!(r.bounds_source(), BoundsSource::Dtw);
  }

  #[test]
  fn run_with_setters_builder_style() {
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      0,
      100,
      0,
      BoundsSource::Segment,
    )
    .with_language(Lang::Es)
    .with_text(SmolStr::new("hola"))
    .with_audio_t0_ms(50)
    .with_audio_t1_ms(150)
    .with_source_segment_idx(7)
    .with_bounds_source(BoundsSource::Dtw);

    assert_eq!(r.language(), &Lang::Es);
    assert_eq!(r.text(), "hola");
    assert_eq!(r.audio_t0_ms(), 50);
    assert_eq!(r.audio_t1_ms(), 150);
    assert_eq!(r.source_segment_idx(), 7);
    assert_eq!(r.bounds_source(), BoundsSource::Dtw);
  }

  #[test]
  fn run_set_inplace() {
    let mut r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      0,
      100,
      0,
      BoundsSource::Segment,
    );
    r.set_language(Lang::Es);
    r.set_text(SmolStr::new("hola"));
    r.set_audio_t0_ms(50);
    r.set_audio_t1_ms(150);
    r.set_source_segment_idx(7);
    r.set_bounds_source(BoundsSource::Dtw);

    assert_eq!(r.language(), &Lang::Es);
    assert_eq!(r.text(), "hola");
    assert_eq!(r.audio_t0_ms(), 50);
    assert_eq!(r.audio_t1_ms(), 150);
    assert_eq!(r.source_segment_idx(), 7);
    assert_eq!(r.bounds_source(), BoundsSource::Dtw);
  }
}
