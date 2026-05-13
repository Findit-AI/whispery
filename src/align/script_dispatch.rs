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

use smol_str::SmolStr;

use crate::{
  align::script::{CharClass, SegmentContext, script_to_lang},
  types::Lang,
};

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
  ///
  /// **Coordinate contract** ([medium]):
  /// values are **chunk-local** — origin at the start of the
  /// chunk's audio, NOT stream-absolute. The runner's alignment
  /// path
  /// ([`crate::run_one_alignment`]) maps `audio_t0_ms * 16` into
  /// chunk-local sample indices; passing stream-absolute times
  /// silently produces zero-word per-run alignment (a stderr
  /// warning fires when bounds land outside the chunk window).
  /// Pluggable [`crate::AsrSource`] implementations populating
  /// [`crate::types::AsrResult::runs`] must respect this
  /// contract.
  #[must_use]
  pub const fn audio_t0_ms(&self) -> i64 {
    self.audio_t0_ms
  }

  /// Run end time, in milliseconds. Half-open with
  /// [`Self::audio_t0_ms`]; same chunk-local origin contract.
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

  /// Per-token info for every token in this segment, in decode
  /// order. Each [`TokenInfo`] carries the token's byte position
  /// inside [`Self::text`] and its DTW timestamp (`None` when
  /// DTW is disabled or unavailable for the token).
  ///
  /// this replaced an earlier
  /// `token_dtw_timestamps() -> Vec<Option<i64>>` whose lack of
  /// byte-position info meant every run carved from a segment
  /// inherited the parent segment's full DTW list. With per-
  /// token byte offsets, [`dispatch_segments`] can filter the DTW
  /// list to only the tokens belonging to each run, producing
  /// run-scoped bounds for code-switched segments.
  ///
  /// The implementor must guarantee `byte_offset + byte_len`
  /// is in-range for [`Self::text`] (`text().is_char_boundary(...)`
  /// is not required — token byte positions can land mid-codepoint
  /// for partial-byte tokens, which whisper.cpp emits for
  /// surrogate-pair / combining-mark sequences; the dispatcher
  /// uses the offsets only for inclusion testing, not for
  /// indexing).
  fn tokens(&self) -> Vec<TokenInfo>;
}

/// One token's byte position within a [`SegmentLike::text`] and
/// its DTW timestamp (centiseconds, `None` if unavailable).
///
/// Crate-public so [`dispatch_segments`] can slice DTW per run
/// using the byte ranges. `byte_offset + byte_len` must fit
/// inside [`SegmentLike::text`]'s byte length.
#[derive(Clone, Copy, Debug)]
pub struct TokenInfo {
  /// Byte offset where this token's text starts inside
  /// [`SegmentLike::text`].
  byte_offset: usize,
  /// Length in bytes of this token's text inside
  /// [`SegmentLike::text`]. Zero is valid (special / non-text
  /// tokens), in which case the token is dropped from per-run
  /// slicing.
  byte_len: usize,
  /// DTW timestamp in centiseconds, or `None` when DTW is
  /// disabled / unavailable for this token.
  t_dtw_cs: Option<i64>,
}

impl TokenInfo {
  /// Construct from the three positional fields.
  #[must_use]
  pub const fn new(byte_offset: usize, byte_len: usize, t_dtw_cs: Option<i64>) -> Self {
    Self {
      byte_offset,
      byte_len,
      t_dtw_cs,
    }
  }

  /// Byte offset of this token's text in [`SegmentLike::text`].
  #[must_use]
  pub const fn byte_offset(&self) -> usize {
    self.byte_offset
  }

  /// Length in bytes of this token's text.
  #[must_use]
  pub const fn byte_len(&self) -> usize {
    self.byte_len
  }

  /// DTW timestamp in centiseconds, or `None` when unavailable.
  #[must_use]
  pub const fn t_dtw_cs(&self) -> Option<i64> {
    self.t_dtw_cs
  }

  /// Half-open byte range `[byte_offset, byte_offset + byte_len)`.
  pub const fn byte_range(&self) -> core::ops::Range<usize> {
    self.byte_offset..self.byte_offset + self.byte_len
  }
}

/// Centiseconds → milliseconds. whisper.cpp's native time unit
/// is 10 ms; asry's API is milliseconds. Multiplication is
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
/// Time bounds are computed **per run** by slicing the segment's
/// DTW token list to those tokens whose byte range falls inside
/// the run's byte range, then post-processing the per-run bounds
/// across the segment so adjacent runs are monotonic and
/// non-overlapping. Runs without full DTW fall back to character-
/// ratio interpolation against the parent segment envelope (NOT
/// byte-ratio — UTF-8 multi-byte CJK/Hangul would otherwise get
/// disproportionate timing weight).
///
/// : original implementation shared the parent
/// segment's `(t0_ms, t1_ms)` across every run, causing
/// overlapping pseudo-words. Round 2 added run-scoped DTW
/// slicing. Round 3 fixed two follow-up issues: byte-based
/// fallback interpolation skewed toward CJK runs (this commit),
/// and DTW-backed runs each widening to `seg_t1_cs` so earlier
/// runs replayed the parent's tail audio (this commit).
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
    let seg_t0_cs = seg.t0();
    let seg_t1_cs = seg.t1();
    let tokens = seg.tokens();

    // i32 cast: segment indices in whisper.cpp's API are i32; we
    // accept up to i32::MAX segments per state. Saturate on
    // overflow rather than truncate — wraparound would silently
    // alias telemetry across far-apart segments.
    let source_idx = i32::try_from(idx).unwrap_or(i32::MAX);

    // First pass: carve runs by language, tracking BOTH byte and
    // character ranges. Bytes are needed for `&text[..]` slicing
    // and DTW-token overlap math (whisper tokens carry byte
    // offsets); chars are needed for non-DTW interpolation so
    // mixed-script segments (e.g. "hello你好") split timing
    // proportionally to glyphs, not UTF-8 byte length.
    let mut carved: Vec<CarvedRun> = Vec::new();
    let mut current_lang: Option<Lang> = None;
    let mut run_byte_start: usize = 0;
    let mut run_char_start: usize = 0;
    let mut char_idx: usize = 0;

    for (byte_idx, ch) in text.char_indices() {
      let class = script_to_lang(ch, ctx, state_lang_ref);
      match class {
        CharClass::Carry => {}
        CharClass::Lang(lang) => match &current_lang {
          None => {
            current_lang = Some(lang);
          }
          Some(active) if *active == lang => {}
          Some(_) => {
            let active = current_lang.take().expect("checked Some above");
            carved.push(CarvedRun {
              lang: active,
              byte_range: run_byte_start..byte_idx,
              char_range: run_char_start..char_idx,
            });
            run_byte_start = byte_idx;
            run_char_start = char_idx;
            current_lang = Some(lang);
          }
        },
      }
      char_idx += 1;
    }
    let text_char_count = char_idx;
    if let Some(active) = current_lang {
      carved.push(CarvedRun {
        lang: active,
        byte_range: run_byte_start..text.len(),
        char_range: run_char_start..text_char_count,
      });
    }

    // Second pass: per-run DTW slice + raw bounds info. We
    // collect this for ALL runs first, then post-process across
    // the segment to ensure monotonic non-overlapping bounds.
    let mut raw_bounds: Vec<RawBounds> = Vec::with_capacity(carved.len());
    for run in &carved {
      let run_dtw = collect_run_dtw(&tokens, run.byte_range.start, run.byte_range.end);
      let mut bounds = extract_raw_bounds(&run_dtw.dtw);
      // when a boundary-
      // spanning token would have contributed an ambiguous DTW
      // point (now dropped from `run_dtw.dtw`), the slice is
      // partial — prevent the run from being treated as
      // DTW-authoritative so it falls through to segment
      // envelope interpolation instead of emitting bounds that
      // overlap a peer run claiming the same point.
      if run_dtw.had_boundary_token {
        bounds.all_some = false;
      }
      raw_bounds.push(bounds);
    }

    // Third pass: emit final bounds, capping each DTW-backed
    // run's exclusive end at the **immediate** next run's
    // resolved start.  // this used `find_map` to look ahead for the next DTW run,
    // skipping intervening Segment-fallback runs — so a
    // `Dtw → Segment → Dtw` carving could extend the first
    // DTW run past the middle run's character-interpolated
    // start, leaving the middle run aligning audio that
    // overlaps the first run's window. The cap now considers
    // the next run's preferred lo (DTW point when available,
    // character-interpolated lo otherwise) so adjacent runs
    // remain monotonic and non-overlapping regardless of
    // bound source.
    let multi_run = carved.len() > 1;
    // defer span subtraction
    // until AFTER sentinel/overflow validation. this
    // computed `seg_t1_cs - seg_t0_cs` unconditionally; with
    // `seg_t0_cs == i64::MIN` the subtraction would panic in
    // debug and wrap in release, feeding bogus interpolation
    // into run bounds before the wholeclip fallback kicked in.
    // `checked_sub` returns `None` on overflow; the
    // interpolation branch is then bypassed and the run picks
    // its own bounds (Wholeclip / DTW).
    let span_cs_opt: Option<i64> = if seg_t0_cs == i64::MIN || seg_t1_cs == i64::MIN {
      None
    } else {
      seg_t1_cs.checked_sub(seg_t0_cs)
    };
    for (i, run) in carved.iter().enumerate() {
      let next_lo_cs = if i + 1 < carved.len() {
        let next_raw = &raw_bounds[i + 1];
        // ONLY use the next
        // run's `lo_dtw_cs` when that run is actually
        // DTW-authoritative (`all_some == true`).  // `lo_dtw_cs` could be populated by a partial DTW
        // slice (one missing token, or a boundary-spanning
        // token forced `all_some=false`); the next run would
        // then resolve to character-interpolated Segment
        // bounds while THIS run got capped at the unrelated
        // partial DTW point — extending past the middle run's
        // actual start and duplicating audio. Match the
        // condition `compute_run_bounds` uses to decide DTW
        // vs interpolation, so the cap and the resolution
        // agree on which value the next run actually uses.
        if next_raw.all_some
          && let Some(lo) = next_raw.lo_dtw_cs
        {
          Some(lo)
        } else if let (Some(span), true) = (span_cs_opt, text_char_count > 0) {
          // Character-interpolated lo for the next run.
          let frac = carved[i + 1].char_range.start as f64 / text_char_count as f64;
          Some((seg_t0_cs as f64 + frac * span as f64).round() as i64)
        } else {
          // Sentinel / overflow / empty text — wholeclip path
          // handles the next run directly.
          None
        }
      } else {
        None
      };
      let (t0_ms, t1_ms, bounds_source) = compute_run_bounds(
        seg_t0_cs,
        seg_t1_cs,
        &raw_bounds[i],
        run.char_range.start,
        run.char_range.end,
        text_char_count,
        multi_run,
        next_lo_cs,
      );
      push_run(
        &mut runs,
        run.lang.clone(),
        &text[run.byte_range.clone()],
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

/// One language run carved out of a segment. Carries both byte
/// and character ranges so we can slice text/tokens by byte and
/// interpolate audio by character.
struct CarvedRun {
  lang: Lang,
  byte_range: core::ops::Range<usize>,
  char_range: core::ops::Range<usize>,
}

/// Per-run DTW slice summary. `lo_dtw_cs` / `hi_dtw_cs` are the
/// min/max DTW point timestamps across the run's tokens;
/// `all_some` is `true` only when every token in the slice
/// carried a valid DTW timestamp (the dispatcher requires
/// all-or-nothing DTW per run, mirroring the original
/// `compute_bounds` contract).
struct RawBounds {
  lo_dtw_cs: Option<i64>,
  hi_dtw_cs: Option<i64>,
  all_some: bool,
}

/// Per-run DTW slice result. `dtw` lists the timestamps of
/// tokens **fully contained** in the run; `had_boundary_token`
/// is true when at least one token's byte range straddled the
/// run boundary (its DTW point would have been ambiguous if
/// counted, so it's dropped).
struct RunDtw {
  dtw: Vec<Option<i64>>,
  had_boundary_token: bool,
}

/// Filter the segment's tokens to those **fully contained** in
/// a run's byte range, collecting their DTW timestamps in
/// decode order. Boundary-spanning tokens (a BPE token whose
/// byte range straddles two runs) are dropped, and the caller
/// is informed via `had_boundary_token` so it can fall back to
/// segment-envelope interpolation rather than treating partial
/// DTW data as authoritative.
///
/// this used a
/// non-trivial-overlap predicate, so a single boundary BPE
/// token could be counted in BOTH adjacent language runs —
/// each run then reported `BoundsSource::Dtw` with overlapping
/// audio windows. The fully-contained predicate makes DTW
/// run-exclusive; partial-coverage runs surface as
/// `BoundsSource::Segment` with character-ratio interpolation.
fn collect_run_dtw(tokens: &[TokenInfo], run_start: usize, run_end: usize) -> RunDtw {
  let mut dtw = Vec::new();
  let mut had_boundary_token = false;
  for tok in tokens {
    if tok.byte_len == 0 {
      continue;
    }
    let tok_start = tok.byte_offset;
    let tok_end = tok.byte_offset.saturating_add(tok.byte_len);
    // Wholly contained in the run.
    if tok_start >= run_start && tok_end <= run_end {
      dtw.push(tok.t_dtw_cs);
      continue;
    }
    // Overlaps but not contained → boundary-spanning.
    let overlap_start = tok_start.max(run_start);
    let overlap_end = tok_end.min(run_end);
    if overlap_end > overlap_start {
      had_boundary_token = true;
    }
  }
  RunDtw {
    dtw,
    had_boundary_token,
  }
}

/// Reduce a DTW slice to `(lo, hi, all_some)` for downstream
/// per-run bounds resolution.
fn extract_raw_bounds(dtw_slice: &[Option<i64>]) -> RawBounds {
  if dtw_slice.is_empty() {
    return RawBounds {
      lo_dtw_cs: None,
      hi_dtw_cs: None,
      all_some: false,
    };
  }
  let all_some = dtw_slice.iter().all(Option::is_some);
  let mut lo: Option<i64> = None;
  let mut hi: Option<i64> = None;
  for v in dtw_slice.iter().filter_map(|v| *v) {
    lo = Some(match lo {
      None => v,
      Some(l) => l.min(v),
    });
    hi = Some(match hi {
      None => v,
      Some(h) => h.max(v),
    });
  }
  RawBounds {
    lo_dtw_cs: lo,
    hi_dtw_cs: hi,
    all_some,
  }
}

/// Per-run bounds resolution. Cooperative across runs: when DTW
/// is available, the exclusive end caps at the **next run's
/// first DTW point** rather than always widening to `seg_t1_cs`,
/// so adjacent DTW runs are monotonic and non-overlapping. When
/// DTW is unavailable, falls back to character-ratio interpolation
/// (: this used byte-ratio, which
/// over-weighted CJK/Hangul runs because each glyph is 3 bytes).
fn compute_run_bounds(
  seg_t0_cs: i64,
  seg_t1_cs: i64,
  raw: &RawBounds,
  run_char_start: usize,
  run_char_end: usize,
  text_char_count: usize,
  multi_run: bool,
  next_run_lo_dtw_cs: Option<i64>,
) -> (i64, i64, BoundsSource) {
  // DTW path: every token in this run's slice has a valid
  // timestamp.
  if raw.all_some
    && let (Some(lo), Some(hi_point)) = (raw.lo_dtw_cs, raw.hi_dtw_cs)
  {
    // Exclusive end policy:
    // 1. If a later run also has DTW (or interpolated bounds),
    // cap at its preferred lo so adjacent runs remain
    // monotone and non-overlapping in audio order.
    // this clamped
    // `next_lo.max(lo + 1)` whenever the cap would have
    // collapsed (`next_lo <= lo`). That kept the current
    // run as DTW spanning `[lo, lo+1)` while the *next* run
    // resolved to its earlier interpolated lo, putting the
    // two runs out of audio order. Detect that case and
    // fall through to segment interpolation: char-fraction
    // bounds are monotone in run index by construction, so
    // the resulting Segment-Segment pair stays ordered.
    // 2. Otherwise (last DTW run, or solo run), prefer the
    // segment's `seg_t1_cs` envelope; if that's not after
    // the DTW max, widen by one centisecond quantum.
    let exclusive_hi_opt: Option<i64> = if let Some(next_lo) = next_run_lo_dtw_cs {
      if next_lo <= lo {
        None
      } else if seg_t1_cs != i64::MIN {
        Some(next_lo.min(seg_t1_cs))
      } else {
        Some(next_lo)
      }
    } else if seg_t1_cs != i64::MIN && seg_t1_cs > hi_point {
      Some(seg_t1_cs)
    } else {
      Some(hi_point.saturating_add(1))
    };
    if let Some(exclusive_hi) = exclusive_hi_opt
      && exclusive_hi > lo
    {
      return (cs_to_ms(lo), cs_to_ms(exclusive_hi), BoundsSource::Dtw);
    }
    // Fall through to Segment interpolation: either the next
    // run's preferred lo conflicts with our DTW lo (handled
    // above), or the cap collapsed for some other defensive
    // reason.
  }

  // Segment fallback. For single-run segments, use the envelope
  // verbatim — keeps single-language behaviour byte-identical to
  // pre-script-dispatch. For multi-run segments, linearly
  // interpolate by *character* ratio.
  if seg_t0_cs == i64::MIN || seg_t1_cs == i64::MIN {
    return (i64::MIN, i64::MAX, BoundsSource::Wholeclip);
  }
  if !multi_run || text_char_count == 0 {
    return (
      cs_to_ms(seg_t0_cs),
      cs_to_ms(seg_t1_cs),
      BoundsSource::Segment,
    );
  }
  let lo_frac = run_char_start as f64 / text_char_count as f64;
  let hi_frac = run_char_end as f64 / text_char_count as f64;
  // this used
  // `seg_t1_cs - seg_t0_cs` directly; non-sentinel extreme
  // bounds (e.g. `t0 = i64::MIN + 1`, `t1 = i64::MAX`) panic in
  // debug and wrap in release before the wholeclip fallback
  // can fire. `checked_sub` returns `None` on overflow; we
  // route to wholeclip in that case to match the doc-stated
  // overflow contract.
  let span = match seg_t1_cs.checked_sub(seg_t0_cs) {
    Some(s) => s as f64,
    None => return (i64::MIN, i64::MAX, BoundsSource::Wholeclip),
  };
  let interp_lo_cs = (seg_t0_cs as f64 + lo_frac * span).round() as i64;
  let mut interp_hi_cs = (seg_t0_cs as f64 + hi_frac * span).round() as i64;
  // when `lo_frac` and `hi_frac`
  // round to the same centisecond (very short multi-run segments,
  // e.g. a 1-char run inside a 50ms parent envelope), the
  // interpolated bounds collapse to `t0 == t1`. Downstream
  // `run_audio_slice` would then see `t1 <= t0` and re-expand to
  // `(0, samples_len)` — aligning a 1-char run against the
  // entire chunk, polluting word output with unrelated audio.
  // Force a 1cs (10ms) minimum width so degenerate runs stay
  // narrow rather than degrade to wholeclip; the resulting
  // 10ms slice is small enough that `min_speech_coverage`
  // typically drops the word, surfacing a recoverable miss.
  if interp_hi_cs <= interp_lo_cs {
    interp_hi_cs = interp_lo_cs.saturating_add(1);
  }
  (
    cs_to_ms(interp_lo_cs),
    cs_to_ms(interp_hi_cs),
    BoundsSource::Segment,
  )
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
/// derived min/max range is non-empty. Both endpoints are
/// converted from centiseconds to milliseconds.
/// 2. **Segment**: `seg_t0_cs` and `seg_t1_cs` are both
/// non-sentinel ([`i64::MIN`] indicates "unavailable" per the
/// [`SegmentLike`] contract). Whisper.cpp's normal output
/// falls into this branch when DTW is disabled or partially
/// populated.
/// 3. **Wholeclip**: neither DTW nor segment timing is usable.
/// Returns `(i64::MIN, i64::MAX)` as sentinels — the caller
/// must treat them as "unknown" rather than literal times,
/// and downstream code should fall back to whole-clip timing
/// (the audio's full duration). This branch is defensive;
/// real whisper.cpp output should always populate `t0` /
/// `t1` on emitted segments.
fn compute_bounds(
  seg_t0_cs: i64,
  seg_t1_cs: i64,
  dtw_cs: &[Option<i64>],
) -> (i64, i64, BoundsSource) {
  // Case 1: DTW. All tokens must have `Some(_)`. DTW emits per-
  // token *point* timestamps; downstream `run_audio_slice`
  // interprets the returned pair as a half-open audio range, so a
  // single-token run (or any run whose tokens collapse to one
  // timestamp) yields `lo == hi_point` and the exclusive end must
  // be widened past `lo`. the previous
  // implementation returned `(min, max, Dtw)` directly, which (a)
  // truncated the last token by treating its start as the
  // exclusive end and (b) degraded zero-width spans to wholeclip
  // via `run_audio_slice`'s `t1 <= t0` guard, replaying the full
  // chunk for short runs.
  //
  // Widening policy: prefer the segment's `seg_t1_cs` envelope
  // (canonical run end emitted by whisper.cpp); else extend by a
  // single centisecond quantum past the last DTW point. Fall
  // through to the segment branch if neither produces a valid
  // half-open span (defensive — whisper.cpp populates segment
  // timing on real output).
  if !dtw_cs.is_empty() && dtw_cs.iter().all(Option::is_some) {
    let mut iter = dtw_cs.iter().filter_map(|v| *v);
    if let Some(first) = iter.next() {
      let mut lo = first;
      let mut hi_point = first;
      for v in iter {
        if v < lo {
          lo = v;
        }
        if v > hi_point {
          hi_point = v;
        }
      }
      let exclusive_hi = if seg_t1_cs != i64::MIN && seg_t1_cs > hi_point {
        seg_t1_cs
      } else {
        hi_point.saturating_add(1)
      };
      if exclusive_hi > lo {
        return (cs_to_ms(lo), cs_to_ms(exclusive_hi), BoundsSource::Dtw);
      }
      // exclusive_hi <= lo means hi_point.saturating_add(1) saturated
      // (only possible at i64::MAX) — fall through to segment.
    }
  }

  // Case 2: segment envelope. `i64::MIN` is the documented
  // sentinel for "missing."
  if seg_t0_cs != i64::MIN && seg_t1_cs != i64::MIN {
    return (
      cs_to_ms(seg_t0_cs),
      cs_to_ms(seg_t1_cs),
      BoundsSource::Segment,
    );
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

  use super::{SegmentLike, TokenInfo};

  /// Newtype wrapper so we can implement the local trait for the
  /// foreign `whispercpp::Segment<'_>` without orphan-rule
  /// complications. Carries a borrow of the parent
  /// [`whispercpp::Context`] so [`Self::tokens`] can resolve each
  /// token id to its raw text bytes (required to compute the
  /// per-token byte offsets the dispatcher uses for run-scoped
  /// DTW slicing).
  pub(super) struct SegmentRef<'a, 'seg> {
    pub seg: &'a whispercpp::Segment<'seg>,
    pub ctx: &'a whispercpp::Context,
  }

  impl<'a, 'seg> SegmentLike for SegmentRef<'a, 'seg> {
    fn text(&self) -> &str {
      // Segment::text returns Result<&str>; a UTF-8 error means
      // the model emitted invalid UTF-8 (extremely unusual).
      // Treating it as empty text drops the segment from
      // dispatch — safer than panicking inside an alignment
      // pipeline. Real production paths log this upstream.
      self.seg.text().unwrap_or("")
    }

    fn t0(&self) -> i64 {
      self.seg.t0()
    }

    fn t1(&self) -> i64 {
      self.seg.t1()
    }

    fn tokens(&self) -> Vec<TokenInfo> {
      // Walk tokens in decode order; for each, look up its raw
      // bytes via `Context::token_to_bytes`, accumulate offsets,
      // and validate that the assembled byte stream equals
      // `self.seg.text()` ( // this assumption was unchecked, so any divergence
      // — leading-space handling, special-token rendering,
      // tokeniser-side normalisation, etc. — would silently
      // mis-attribute DTW timestamps to the wrong language run
      // for code-switched segments).
      //
      // On mismatch, we fall back to a degraded view: every
      // token reports `byte_len=0`, which the dispatcher's
      // `collect_run_dtw` filters out, forcing the segment-
      // envelope (or wholeclip) bounds path. This is the same
      // shape `dispatch_segments` takes when DTW is disabled at
      // the model context, so the recovery path is well-tested.
      // We log once-per-segment to stderr so the divergence is
      // observable in operator logs.
      let segment_text = self.seg.text().unwrap_or("");
      let mut out = Vec::new();
      let mut offset: usize = 0;
      let mut accumulated: Vec<u8> = Vec::with_capacity(segment_text.len());
      let mut tokens_iter = self.seg.tokens_iter();
      let mut any_token_seen = false;
      for tok in tokens_iter.by_ref() {
        any_token_seen = true;
        let bytes = self.ctx.token_to_bytes(tok.id()).unwrap_or(&[]);
        let byte_len = bytes.len();
        accumulated.extend_from_slice(bytes);
        out.push(TokenInfo {
          byte_offset: offset,
          byte_len,
          t_dtw_cs: tok.t_dtw(),
        });
        offset = offset.saturating_add(byte_len);
      }
      // Validate: accumulated == segment.text() bytewise. If
      // not, neutralise the token byte ranges so the dispatcher
      // falls through to segment / wholeclip bounds.
      if any_token_seen && accumulated.as_slice() != segment_text.as_bytes() {
        std::eprintln!(
          "[asry] script_dispatch: token-byte stream diverges from segment text \
 (tokens={} bytes={} segment={} bytes); falling back to segment bounds for this \
 segment to avoid mis-attributing DTW timestamps. Likely cause: model tokenisation \
 normalises bytes (leading-space stripping, special-token rendering) the dispatcher \
 cannot reconstruct from token ids.",
          out.len(),
          accumulated.len(),
          segment_text.len(),
        );
        for tok in &mut out {
          tok.byte_len = 0;
        }
      }
      out
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
/// `ctx` resolves each token id to its raw text bytes (used by
/// the dispatcher for run-scoped DTW slicing per );
/// pass the same `Context` the segments were decoded against.
/// `state_lang` is the transcriber's current language hint,
/// passed through unchanged for Latin / ambiguous-script
/// disambiguation.
#[cfg(feature = "runner")]
#[must_use]
pub fn dispatch(
  ctx: &whispercpp::Context,
  segments: &[whispercpp::Segment<'_>],
  state_lang: Option<Lang>,
) -> Vec<Run> {
  let wrapped: Vec<runner_glue::SegmentRef<'_, '_>> = segments
    .iter()
    .map(|seg| runner_glue::SegmentRef { seg, ctx })
    .collect();
  dispatch_segments(&wrapped, state_lang)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Minimal mock implementing [`SegmentLike`] for unit tests.
  /// Times are in centiseconds (matches whisper.cpp's native unit
  /// and the dispatcher's own internal contract).
  struct MockSeg {
    text: String,
    t0_cs: i64,
    t1_cs: i64,
    tokens: Vec<TokenInfo>,
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
    fn tokens(&self) -> Vec<TokenInfo> {
      self.tokens.clone()
    }
  }

  /// Build a `MockSeg` with `dtw_cs.len()` byte-uniform tokens
  /// spanning the entire text. Each entry of `dtw_cs` becomes one
  /// token whose byte range covers a contiguous chunk of `text`.
  /// The default test layout: every token belongs to the single
  /// (only) run carved from `text`.
  fn seg(text: &str, t0_cs: i64, t1_cs: i64, dtw_cs: Vec<Option<i64>>) -> MockSeg {
    let n = dtw_cs.len();
    let text_byte_len = text.len();
    let tokens = if n == 0 {
      Vec::new()
    } else {
      // Ceiling division so tokens cover the whole text even when
      // `text_byte_len` doesn't divide evenly by `n`.
      let chunk = text_byte_len.div_ceil(n).max(1);
      dtw_cs
        .iter()
        .enumerate()
        .map(|(i, &t)| {
          let off = (i * chunk).min(text_byte_len);
          let len = chunk.min(text_byte_len.saturating_sub(off));
          TokenInfo {
            byte_offset: off,
            byte_len: len,
            t_dtw_cs: t,
          }
        })
        .collect()
    };
    MockSeg {
      text: String::from(text),
      t0_cs,
      t1_cs,
      tokens,
    }
  }

  /// Build a `MockSeg` with explicit per-token `(byte_offset,
  /// byte_len, dtw_cs)` triples. Used by code-switch tests that
  /// need to exercise per-run DTW slicing.
  fn seg_with_tokens(
    text: &str,
    t0_cs: i64,
    t1_cs: i64,
    tokens: Vec<(usize, usize, Option<i64>)>,
  ) -> MockSeg {
    MockSeg {
      text: String::from(text),
      t0_cs,
      t1_cs,
      tokens: tokens
        .into_iter()
        .map(|(byte_offset, byte_len, t_dtw_cs)| TokenInfo {
          byte_offset,
          byte_len,
          t_dtw_cs,
        })
        .collect(),
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
    let segs = vec![seg(" hello", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].text(), " hello");
  }

  #[test]
  fn pure_punctuation_produces_no_run() {
    let segs = vec![seg("...!?", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert!(runs.is_empty());
  }

  #[test]
  fn dtw_available_uses_dtw_bounds_widened_to_segment_end() {
    // DTW points are start timestamps for each token; the audio
    // slice's exclusive end widens to `seg_t1_cs` (200 cs → 2000 ms)
    // so the last token's audio isn't truncated. this
    // returned (700, 1900), which clipped the last token.
    let segs = vec![seg(
      "hello",
      50,
      200,
      vec![Some(70), Some(90), Some(120), Some(180), Some(190)],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    assert_eq!(runs[0].audio_t0_ms(), 700);
    assert_eq!(runs[0].audio_t1_ms(), 2000);
  }

  #[test]
  fn dtw_single_token_widens_to_segment_end() {
    // Single token; lo == hi_point. Widens to seg_t1_cs (=200 cs →
    // 2000 ms). this would return (700, 700), and
    // `run_audio_slice` would degrade it to the full chunk.
    let segs = vec![seg("hi", 50, 200, vec![Some(70)])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    assert_eq!(runs[0].audio_t0_ms(), 700);
    assert_eq!(runs[0].audio_t1_ms(), 2000);
  }

  #[test]
  fn dtw_collapsed_timestamps_widen_to_segment_end() {
    // Multi-token but all share the same DTW timestamp. lo == hi_point;
    // widens to seg_t1_cs.
    let segs = vec![seg("hi", 50, 200, vec![Some(80), Some(80), Some(80)])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    assert_eq!(runs[0].audio_t0_ms(), 800);
    assert_eq!(runs[0].audio_t1_ms(), 2000);
  }

  #[test]
  fn codeswitch_segment_dtw_runs_are_monotonic_non_overlapping() {
    // DTW-backed runs in a multi-
    // run segment must NOT each widen to `seg_t1_cs`; that would
    // make every earlier run replay the parent segment's tail.
    // The cap-at-next-run policy keeps adjacent DTW runs
    // monotonic and non-overlapping.
    //
    // Layout: "hello你好world" — bytes [0..5)=En, [5..11)=Zh,
    // [11..16)=En. One DTW token per run.
    let segs = vec![seg_with_tokens(
      "hello你好world",
      50,
      300,
      vec![
        (0, 5, Some(70)),   // En run #1 token
        (5, 6, Some(140)),  // Zh run token
        (11, 5, Some(210)), // En run #2 token
      ],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 3);
    // Run 0 (En): DTW lo=70cs; cap at next run's lo=140cs.
    // → (700, 1400) ms — does NOT replay through 3000ms.
    assert_eq!(runs[0].audio_t0_ms(), 700);
    assert_eq!(runs[0].audio_t1_ms(), 1400);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    // Run 1 (Zh): DTW lo=140cs; cap at next run's lo=210cs.
    // → (1400, 2100) ms.
    assert_eq!(runs[1].audio_t0_ms(), 1400);
    assert_eq!(runs[1].audio_t1_ms(), 2100);
    // Run 2 (En last): no next run; widen to seg_t1=300cs.
    // → (2100, 3000) ms.
    assert_eq!(runs[2].audio_t0_ms(), 2100);
    assert_eq!(runs[2].audio_t1_ms(), 3000);
    // Monotonic non-overlap invariant.
    assert_eq!(runs[0].audio_t1_ms(), runs[1].audio_t0_ms());
    assert_eq!(runs[1].audio_t1_ms(), runs[2].audio_t0_ms());
  }

  #[test]
  fn codeswitch_segment_without_dtw_uses_character_ratio_not_byte_ratio() {
    // the fallback split
    // by *byte* offsets, giving multi-byte CJK runs a 3× share
    // of the audio. With character-ratio interpolation, runs are
    // weighted proportionally to glyph count, matching whisperX's
    // codepoint-based fallback.
    //
    // Text: "hello你好world", char count = 12, span = 250cs:
    // En1 [chars 0..5) → 0/12 .. 5/12 → 0..104cs of span
    // Zh [chars 5..7) → 5/12 .. 7/12 → 104..146cs of span
    // En2 [chars 7..12) → 7/12 .. 12/12 → 146..250cs of span
    // Plus seg_t0=50cs → (500, 1542), (1542, 1958), (1958, 3000).
    // (Byte-ratio would have given Zh ~6/16 = 37.5% of the audio
    // for only 2/12 = 17% of the glyphs.)
    let segs = vec![seg("hello你好world", 50, 300, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 3);
    assert!(
      runs
        .iter()
        .all(|r| r.bounds_source() == BoundsSource::Segment)
    );
    let r0_t0 = runs[0].audio_t0_ms();
    let r0_t1 = runs[0].audio_t1_ms();
    let r1_t0 = runs[1].audio_t0_ms();
    let r1_t1 = runs[1].audio_t1_ms();
    let r2_t0 = runs[2].audio_t0_ms();
    let r2_t1 = runs[2].audio_t1_ms();
    // Run 0 (En, 5/12 of glyphs): roughly 5/12 * 2500ms ≈ 1042ms
    // span. Allow ±10ms slack for cs-rounding.
    assert_eq!(r0_t0, 500);
    assert!((r0_t1 - 1542).abs() <= 10, "En1 hi ≈ 1542ms, got {r0_t1}");
    // Run 1 (Zh, 2/12): span ≈ 417ms. Centred around the
    // 5/12..7/12 fraction of the envelope.
    assert!((r1_t1 - r1_t0 - 417).abs() <= 10, "Zh span ≈ 417ms");
    // Boundary continuity.
    assert_eq!(r0_t1, r1_t0);
    assert_eq!(r1_t1, r2_t0);
    assert_eq!(r2_t1, 3000);
    // The Zh run gets fewer ms than either En run (2 chars vs 5).
    assert!(
      r1_t1 - r1_t0 < r0_t1 - r0_t0,
      "Zh (2 glyphs) must get less audio than En1 (5 glyphs); got {r1_t0}..{r1_t1} vs {r0_t0}..{r0_t1}"
    );
    assert!(
      r1_t1 - r1_t0 < r2_t1 - r2_t0,
      "Zh (2 glyphs) must get less audio than En2 (5 glyphs); got {r1_t0}..{r1_t1} vs {r2_t0}..{r2_t1}"
    );
  }

  /// regression: when the
  /// runner-glue layer detects that the token byte stream
  /// diverges from the segment text (a tokeniser-side
  /// normalisation we can't reconstruct), it neutralises every
  /// `TokenInfo` by setting `byte_len = 0`. The dispatcher's
  /// `collect_run_dtw` then drops every token (zero-length
  /// tokens skip the overlap test), so DTW becomes "all None"
  /// for every run and bounds resolution falls through to the
  /// segment envelope. This test simulates the recovered shape
  /// directly via `seg_with_tokens` to lock in the fallback.
  /// a BPE / sub-word token
  /// whose byte range straddles a script boundary used to be
  /// counted in BOTH adjacent language runs (overlap predicate
  /// without containment check). Both runs would then claim
  /// `BoundsSource::Dtw` with overlapping audio windows.
  /// Post-fix, boundary-spanning tokens are dropped from the
  /// DTW slice and the affected runs fall back to segment-
  /// envelope interpolation.
  #[test]
  fn boundary_spanning_dtw_token_does_not_double_count() {
    // Layout: "hello你好" — bytes [0..5)=En, [5..11)=Zh.
    // Three tokens:
    // - "hello" wholly in En run (byte 0..5, dtw=70cs).
    // - boundary token straddles En/Zh (byte 4..6, dtw=120cs)
    // — its DTW point belongs to NEITHER run cleanly.
    // - "你好" wholly in Zh run (byte 5..11, dtw=180cs).
    let segs = vec![seg_with_tokens(
      "hello你好",
      50,
      300,
      vec![(0, 5, Some(70)), (4, 2, Some(120)), (5, 6, Some(180))],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 2);
    // Both runs have at least one wholly-contained DTW token,
    // BUT each also has the boundary token overlapping. The
    // `had_boundary_token` flag forces fall-through to segment
    // interpolation — so neither run is `Dtw`-bounded.
    assert_eq!(
      runs[0].bounds_source(),
      BoundsSource::Segment,
      "En run with a boundary-spanning token must fall back to Segment, not Dtw"
    );
    assert_eq!(runs[1].bounds_source(), BoundsSource::Segment);
    // Crucially: the runs do not share a t0_ms (which would be
    // the case if both adopted DTW point 120cs).
    assert!(runs[0].audio_t0_ms() < runs[1].audio_t0_ms());
    // And the runs don't overlap.
    assert!(runs[0].audio_t1_ms() <= runs[1].audio_t0_ms());
  }

  /// when a multi-run segment
  /// carves runs with mixed bound sources (e.g. Dtw → Segment
  /// → Dtw), the first DTW run must NOT extend past the
  /// intervening fallback run's character-interpolated start.
  /// `find_map` over later runs with DTW skipped the
  /// middle Segment run and capped the first run at the third
  /// run's DTW point — so the first run's window overlapped
  /// the middle run's window, sending the same audio to two
  /// language aligners.
  #[test]
  fn dtw_run_capped_at_immediate_next_segment_run_not_skipping_ahead() {
    // Layout: 6 chars total. We use a 4-byte ASCII boundary
    // and assemble carved runs En → mixed → En. To get
    // Dtw → Segment → Dtw, force the middle run to fall back
    // to Segment by giving it a partial DTW slice (only one of
    // its two tokens has DTW) — extract_raw_bounds sets
    // all_some=false on the Vec<Option<i64>> mismatch.
    //
    // Layout: "abc 你 def" — but easier to control: use
    // explicit per-character runs.
    //
    // Let's use a synthetic but easier layout. Spans:
    // Run 0 (En "abc", chars 0..3, bytes 0..3): 1 token at
    // byte 0, DTW=Some(50). all_some=true → Dtw.
    // Run 1 (Zh "你", chars 3..4, bytes 3..6): 1 token at
    // byte 3, DTW=None. partial → Segment.
    // Run 2 (En "def", chars 4..7, bytes 6..9): 1 token at
    // byte 6, DTW=Some(180). all_some=true → Dtw.
    //
    // Segment envelope: 0..200cs.
    // Char-interp for Run 1: chars 3..4 of 7 = 3/7..4/7 of
    // 2000ms = 857..1143ms.
    //
    // Run 0 cap = Run 2's lo_dtw = 180cs = 1800ms,
    // so Run 0 = (500, 1800ms) — extends past Run 1's
    // interpolated start (857ms). Post-fix Run 0 cap = Run 1's
    // interpolated lo = 857ms.
    //
    // We can't easily simulate this exact char layout via the
    // existing `seg_with_tokens` helper because the script
    // dispatcher determines runs from the text. Use Latin +
    // Han to get the En→Zh→En carving. "abc你def" gives
    // 3+1+3 = 7 chars and 3+3+3 = 9 bytes.
    let segs = vec![seg_with_tokens(
      "abc你def",
      0,
      200,
      vec![
        (0, 3, Some(50)),  // En run #1 token
        (3, 3, None),      // Zh run token (no DTW)
        (6, 3, Some(180)), // En run #2 token
      ],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[1].language(), &Lang::Zh);
    assert_eq!(runs[2].language(), &Lang::En);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    assert_eq!(runs[1].bounds_source(), BoundsSource::Segment);
    assert_eq!(runs[2].bounds_source(), BoundsSource::Dtw);
    // The hard non-overlap invariant: each run's t1 <= next
    // run's t0.
    assert!(
      runs[0].audio_t1_ms() <= runs[1].audio_t0_ms(),
      "Run 0 (Dtw) end {} must not extend past Run 1 (Segment) start {}",
      runs[0].audio_t1_ms(),
      runs[1].audio_t0_ms()
    );
    assert!(
      runs[1].audio_t1_ms() <= runs[2].audio_t0_ms(),
      "Run 1 (Segment) end {} must not extend past Run 2 (Dtw) start {}",
      runs[1].audio_t1_ms(),
      runs[2].audio_t0_ms()
    );
  }

  /// a `SegmentLike` with
  /// `seg_t0() == i64::MIN` and a finite `seg_t1()` previously
  /// caused `seg_t1_cs - seg_t0_cs` to panic in debug and
  /// wrap in release, feeding bogus interpolation into run
  /// bounds before the wholeclip fallback kicked in. The fix
  /// defers the subtraction past the sentinel check.
  #[test]
  fn segment_with_min_t0_does_not_panic_or_wrap() {
    // Multi-run carving + sentinel t0. this would
    // attempt to subtract i64::MIN from a finite seg_t1_cs
    // for the span_cs computation in dispatch_segments.
    let segs = vec![seg("hello你好", i64::MIN, 200, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 2);
    // With sentinel t0, neither run can interpolate safely;
    // both fall through to wholeclip.
    for r in &runs {
      assert_eq!(
        r.bounds_source(),
        BoundsSource::Wholeclip,
        "sentinel t0 should route to Wholeclip; got {:?}",
        r.bounds_source()
      );
    }
  }

  /// when the middle run has
  /// a partial DTW slice (mixed Some/None), the previous DTW
  /// run must NOT cap at the middle run's stale `lo_dtw_cs` —
  /// the middle run will resolve to character-interpolated
  /// Segment bounds, and capping at the DTW point would
  /// extend the previous run past the middle's actual start.
  /// The cap now requires `next_raw.all_some` before adopting
  /// `lo_dtw_cs`, matching the condition `compute_run_bounds`
  /// uses to decide DTW vs interpolation.
  #[test]
  fn dtw_cap_skips_partial_dtw_neighbour() {
    // Layout: "abc你好def"
    // - Run 0 (En "abc", bytes 0..3): 1 token, DTW=Some(50). all_some=true → Dtw.
    // - Run 1 (Zh "你好", bytes 3..9): 2 tokens, MIXED — token at bytes
    // 3..6 has DTW=Some(120), token at bytes 6..3 has DTW=None.
    // all_some=false → Segment.
    // - Run 2 (En "def", bytes 9..12): 1 token, DTW=Some(180).
    //
    // Run 0 must be capped against Run 1's char-interpolated
    // `lo` (3/8 * 2000ms = 750ms), not Run 1's raw `lo_dtw_cs`
    // (= 1200ms), so Run 0.t1 <= Run 1.t0. Capping against the
    // raw DTW value would let Run 0's window extend past
    // Run 1's interpolated start.
    let segs = vec![seg_with_tokens(
      "abc你好def",
      0,
      200,
      vec![
        (0, 3, Some(50)),  // Run 0 (En)
        (3, 3, Some(120)), // Run 1 token 1 (first Han)
        (6, 3, None),      // Run 1 token 2 (second Han) — partial!
        (9, 3, Some(180)), // Run 2 (En)
      ],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[1].language(), &Lang::Zh);
    assert_eq!(runs[2].language(), &Lang::En);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    assert_eq!(
      runs[1].bounds_source(),
      BoundsSource::Segment,
      "partial DTW slice → Segment fallback"
    );
    assert_eq!(runs[2].bounds_source(), BoundsSource::Dtw);
    // Hard non-overlap: previous run's t1 must be <= next run's t0.
    assert!(
      runs[0].audio_t1_ms() <= runs[1].audio_t0_ms(),
      "Run 0 (Dtw) end {} must not extend past Run 1 (Segment) start {}",
      runs[0].audio_t1_ms(),
      runs[1].audio_t0_ms()
    );
    assert!(
      runs[1].audio_t1_ms() <= runs[2].audio_t0_ms(),
      "Run 1 (Segment) end {} must not extend past Run 2 (Dtw) start {}",
      runs[1].audio_t1_ms(),
      runs[2].audio_t0_ms()
    );
  }

  /// when a DTW-backed run's
  /// `lo` lands *after* the next (non-DTW) run's interpolated
  /// lo, the previous fix kept the DTW run as
  /// `[lo, lo+1)` (saturating cap) while the next run resolved
  /// to its earlier interpolated start. The two runs were
  /// then emitted in text order with audio times out of
  /// audio order. Post-fix: the DTW run falls through to
  /// segment interpolation so adjacent Segment-Segment bounds
  /// stay monotone by construction.
  #[test]
  fn dtw_lo_after_next_interp_lo_demotes_to_segment() {
    // Layout: "abc你好" (5 chars, env [0, 200] cs).
    // - Run 0 (En "abc", chars 0..3): single token DTW=Some(180).
    // all_some=true → would prefer Dtw with `lo = 180cs`.
    // - Run 1 (Zh "你好", chars 3..5): partial DTW → Segment.
    // Interpolated lo = (3/5)*200 = 120cs.
    //
    // 180cs > 120cs: emitting Run 0 as Dtw [180, 181) and
    // Run 1 as Segment [120, 200) puts the runs out of audio
    // order. The fix demotes Run 0 to Segment so both runs
    // come from the same monotone char-fraction grid:
    // Run 0 [0, 120), Run 1 [120, 200).
    let segs = vec![seg_with_tokens(
      "abc你好",
      0,
      200,
      vec![
        (0, 3, Some(180)), // Run 0 (En) — DTW lo lands late.
        (3, 3, Some(150)), // Run 1 token 1 — partial slice forces Segment.
        (6, 3, None),
      ],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[1].language(), &Lang::Zh);
    assert_eq!(
      runs[0].bounds_source(),
      BoundsSource::Segment,
      "DTW lo > next interp lo must demote to Segment, got {:?}",
      runs[0].bounds_source()
    );
    assert_eq!(runs[1].bounds_source(), BoundsSource::Segment);
    assert!(
      runs[0].audio_t1_ms() <= runs[1].audio_t0_ms(),
      "post-fix runs must be monotone: run0 ends at {}, run1 starts at {}",
      runs[0].audio_t1_ms(),
      runs[1].audio_t0_ms()
    );
    // Char-interp boundary: 3/5 of [0, 2000ms] = 1200ms.
    assert_eq!(runs[0].audio_t0_ms(), 0);
    assert_eq!(runs[0].audio_t1_ms(), 1200);
    assert_eq!(runs[1].audio_t0_ms(), 1200);
    assert_eq!(runs[1].audio_t1_ms(), 2000);
  }

  #[test]
  fn neutralised_tokens_fall_back_to_segment_bounds() {
    let segs = vec![seg_with_tokens(
      "hello",
      50,
      200,
      // Every token has byte_len=0 → simulates the post-validation
      // recovery shape. DTW timestamps are present but unused.
      vec![(0, 0, Some(70)), (0, 0, Some(140))],
    )];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Segment);
    assert_eq!(runs[0].audio_t0_ms(), 500);
    assert_eq!(runs[0].audio_t1_ms(), 2000);
  }

  #[test]
  fn dtw_single_token_without_segment_envelope_widens_one_quantum() {
    // Single token + missing segment envelope (i64::MIN sentinel).
    // The 1-cs (10 ms) widening produces a non-degenerate range
    // anchored at the DTW point, leaning on the aligner's internal
    // clamping for the last token's audio rather than degrading to
    // wholeclip.
    let segs = vec![seg("hi", i64::MIN, i64::MIN, vec![Some(70)])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].bounds_source(), BoundsSource::Dtw);
    assert_eq!(runs[0].audio_t0_ms(), 700);
    assert_eq!(runs[0].audio_t1_ms(), 710);
  }

  #[test]
  fn dtw_partial_falls_back_to_segment() {
    let segs = vec![seg("hello", 50, 200, vec![Some(70), None, Some(120)])];
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

  /// a Latin span under a
  /// CJK `state_lang` must dispatch to `Lang::En` so genuine
  /// code-switches (e.g. `"hello 你好"` detected as Zh) split
  /// into separate per-language runs instead of collapsing
  /// into one CJK run that routes the Latin through the wrong
  /// aligner / normalizer. Round 9 had the opposite policy
  /// (Latin folded into the active CJK run for loanword
  /// handling); the global rule suppressed code-switch
  /// alignment everywhere it fired.
  #[test]
  fn latin_with_cjk_state_lang_routes_to_en() {
    let segs = vec![seg("hello", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, Some(Lang::Zh));
    assert_eq!(runs.len(), 1);
    assert_eq!(
      runs[0].language(),
      &Lang::En,
      "Latin under Zh state_lang must route to En, not stay as Zh",
    );
  }

  /// regression: mixed
  /// `"hello 你好"` chunk detected as Zh must produce TWO
  /// runs (En "hello" + Zh "你好"), not one collapsed Zh run.
  /// This is the case the README's per-language code-switch
  /// alignment claim explicitly covers.
  #[test]
  fn mixed_latin_han_under_zh_hint_emits_two_runs() {
    let segs = vec![seg("hello 你好", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, Some(Lang::Zh));
    assert_eq!(
      runs.len(),
      2,
      "expected separate En + Zh runs for code-switched chunk; got {runs:?}",
    );
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[0].text(), "hello ");
    assert_eq!(runs[1].language(), &Lang::Zh);
    assert_eq!(runs[1].text(), "你好");
  }

  #[test]
  fn latin_with_no_hint_still_falls_back_to_en() {
    // Without a state_lang hint Latin defaults to En. The
    // round-9 CJK-keep-Latin rule kicks in ONLY when
    // state_lang is set to Ja/Zh/Yue/Ko.
    let segs = vec![seg("hello", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, None);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].language(), &Lang::En);
  }

  /// embedded Latin loanwords
  /// like `"USAで"` under a Ja hint split into TWO runs (En
  /// "USA" + Ja "で"). Folding Latin into the CJK run so the
  /// JapaneseNormalizer's per-char Latin handling could run
  /// would suppress legitimate code-switches everywhere it
  /// applied. Callers who prefer the loanword behaviour can
  /// register `Lang::En` against their Ja aligner (or use
  /// `AlignmentFallback::Any`).
  #[test]
  fn mixed_cjk_latin_segment_splits_under_cjk_hint() {
    let segs = vec![seg("USAで", 0, 100, vec![])];
    let runs = dispatch_segments(&segs, Some(Lang::Ja));
    assert_eq!(runs.len(), 2, "expected En + Ja runs; got {runs:?}");
    assert_eq!(runs[0].language(), &Lang::En);
    assert_eq!(runs[0].text(), "USA");
    assert_eq!(runs[1].language(), &Lang::Ja);
    assert_eq!(runs[1].text(), "で");
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
    let r = Run::new(Lang::En, SmolStr::new("hi"), 100, 200, 3, BoundsSource::Dtw);
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
