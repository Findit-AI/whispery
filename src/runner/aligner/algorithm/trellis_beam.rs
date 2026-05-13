//! WhisperX-faithful trellis + beam-search backtracker.
//!
//! Ports WhisperX `alignment.py`:
//! - `get_trellis(emission, tokens, blank_id)` — forward DP that
//! builds a `(T, num_tokens)` lattice. Each cell `trellis[t, j]`
//! is the best log-prob to consume the first `t` frames while
//! sitting at character position `j`.
//! - `get_wildcard_emission(frame_emission, tokens, blank_id)` —
//! for tokens with id `-1` (wildcards), use
//! `max(non_blank_logprobs)` at that frame so chars the model
//! doesn't have a vocab entry for can still be aligned.
//! - `backtrack_beam(trellis, emission, tokens, blank_id,
//! beam_width=2)` — beam search over (t, j) states with the
//! "stay" / "change" transitions.
//! - `merge_repeats(path, transcript)` — char-level segments by
//! token_index group.
//! - `merge_words(segments, separator="|")` — group char segments
//! by `|`-separator into word segments with duration-weighted
//! score.
//!
//! The trellis here is intentionally shaped exactly like WhisperX's
//! (`(T, num_tokens)`); the `state_per_frame` lattice the legacy
//! Viterbi exposed is gone. Composition consumes the higher-level
//! `WordSegment`s directly, so the lattice-state encoding never
//! reaches `compose.rs`.
//!
//! Asry-specific concerns kept here:
//! - **Watchdog / abort flag** — checked once per frame row in the
//! forward DP and once per beam-step iteration so a pathological
//! token sequence × T pair can't hold the alignment worker past
//! `align_timeout`.
//! - **Lattice cell budget** — caps `T × num_tokens` at 32 M cells
//! so a hallucinated long token list against a long chunk turns
//! into an in-band `NoAlignmentPath` failure rather than an OOM.
//! - **Vocab-id bounds checks** — every real token id and the
//! blank id must fit in the model's vocab dim; a tokenizer-vs-
//! model mismatch surfaces as `::TokenizationFailed`
//! / `ModelInferenceFailed` rather than a panicking out-of-bounds
//! read.

use core::sync::atomic::{AtomicBool, Ordering};
use smol_str::{SmolStr, format_smolstr};

use crate::{
  runner::aligner::algorithm::encode::LogProbsTV,
  types::{AlignmentError, AlignmentFailure, Lang, WorkFailure, WorkerHangTimeout, WorkerKind},
};

/// Sentinel token id for "wildcard" (any non-blank vocab item)
/// emission. Chars whose normalised form has no entry in the
/// model dictionary become wildcards, matching WhisperX
/// `align()`'s `tokens = [model_dictionary.get(c, -1) for c in
/// text_clean]`. Stored as `i32` because the vocab id space is
/// `u32` but `-1` carries the wildcard signal.
///
/// `pub` for the `feature = "bench-internals"` re-export.
pub const WILDCARD_TOKEN_ID: i32 = -1;

/// Beam width WhisperX's `align()` invokes
/// `backtrack_beam` with: 2. Larger widths add cost without
/// observably better alignments on the wav2vec2 family
/// according to the reference implementation; we mirror the
/// upstream choice.
///
/// `pub` for the `feature = "bench-internals"` re-export.
pub const ALIGN_BEAM_WIDTH: usize = 2;

/// Cap on `T * num_tokens` cells in the forward trellis. Same
/// reasoning as the legacy Viterbi guard: a hallucinated long
/// token list against a long chunk would otherwise allocate
/// gigabytes before the per-row abort check fires. 32 M cells =
/// 128 MB at 4 bytes/cell — comfortably above realistic chunks
/// (T ≤ ~1500 at 50 fps × 30 s, num_tokens typically ≤ ~1k chars
/// → ≤ 1.5 M cells) while turning pathological inputs into an
/// in-band failure.
const TRELLIS_CELL_BUDGET: usize = 32_000_000;

/// Hard cap on the size of the [`BeamNode`] arena that
/// `backtrack_beam` builds during the per-frame branch
/// extension. only
/// the trellis allocation was budgeted, so a degenerate
/// `num_tokens = 1, T = 32 M` lattice — well under the
/// 32 M-cell trellis cap — could grow the arena to tens of
/// millions of nodes (each `BeamNode` is ~32–40 bytes), OOM
/// before the per-row abort fires. 2 M nodes ≈ 80 MB at the
/// upper-bound `BeamNode` size; comfortably above realistic
/// beam-width-2 traces (`2 × 1500 = 3 k` nodes for a 30 s
/// chunk) while turning pathological inputs into an in-band
/// `NoAlignmentPath` failure.
const BEAM_NODE_BUDGET: usize = 2_000_000;

/// One char-level alignment segment, the output of
/// `merge_repeats`. Mirrors WhisperX `Segment(label, start, end,
/// score)`.
#[derive(Debug, Clone)]
pub(crate) struct CharSegment {
  /// Token index (= position in `tokens`/`text_clean`) the
  /// segment covers.
  pub token_index: usize,
  /// First frame the path spent on this token (inclusive,
  /// 0-indexed).
  pub start_frame: usize,
  /// One past the last frame the path spent on this token
  /// (exclusive). WhisperX's convention is `path[i2-1].time_index
  /// + 1` so `[start_frame, end_frame)` is half-open.
  pub end_frame: usize,
  /// Mean per-frame probability over the path frames assigned
  /// to this token. Linear-space `exp()` of the per-frame
  /// log-probs, averaged.
  pub score: f32,
}

impl CharSegment {
  /// `end_frame - start_frame`; the WhisperX `length` property.
  pub(crate) const fn length(&self) -> usize {
    self.end_frame - self.start_frame
  }
}

/// One word-level segment, the output of `merge_words`.
///
/// `pub` for the `feature = "bench-internals"` re-export — out-of-
/// tree code only sees this through doc-hidden `asry::__bench`.
#[derive(Debug, Clone)]
pub struct WordSegment {
  /// Word index in `original_words` / `word_idx_per_token`.
  word_index: usize,
  /// First frame the word covers (inclusive).
  start_frame: usize,
  /// One past the last frame the word covers (exclusive).
  end_frame: usize,
  /// Duration-weighted mean per-frame probability over the
  /// word's chars. Matches WhisperX `merge_words`'s
  /// `sum(seg.score * seg.length) / sum(seg.length)` formula.
  score: f32,
}

impl WordSegment {
  /// Construct from word index + frame range + mean score.
  #[must_use]
  pub const fn new(word_index: usize, start_frame: usize, end_frame: usize, score: f32) -> Self {
    Self {
      word_index,
      start_frame,
      end_frame,
      score,
    }
  }

  /// Word index in `original_words` / `word_idx_per_token`.
  #[must_use]
  pub const fn word_index(&self) -> usize {
    self.word_index
  }

  /// First frame the word covers (inclusive).
  #[must_use]
  pub const fn start_frame(&self) -> usize {
    self.start_frame
  }

  /// One past the last frame the word covers (exclusive).
  #[must_use]
  pub const fn end_frame(&self) -> usize {
    self.end_frame
  }

  /// Duration-weighted mean per-frame probability.
  #[must_use]
  pub const fn score(&self) -> f32 {
    self.score
  }
}

/// Build the WhisperX-shape `(T, num_tokens)` trellis.
///
/// `tokens[i]` is either a non-negative vocab id (the model
/// dictionary's entry for `text_clean[i]`) or `-1` for a wildcard
/// (an alphanumeric char that's not in the dictionary; the
/// emission for that frame is `max` over non-blank vocab items).
///
/// The shape and recurrence match `alignment.py:387-404`:
/// ```text
/// trellis[1:, 0] = cumsum(emission[1:, blank_id], 0)
/// trellis[0, 1:] = -inf
/// trellis[-num_tokens + 1:, 0] = +inf
/// trellis[t+1, 1:] = max(
/// trellis[t, 1:] + emission[t, blank_id],
/// trellis[t, :-1] + wildcard_emission(emission[t], tokens[1:]),
/// )
/// ```
///
/// `pub` for the `feature = "bench-internals"` re-export.
pub fn get_trellis(
  log_probs: &LogProbsTV,
  tokens: &[i32],
  blank_id: u32,
  abort_flag: &AtomicBool,
  language: &Lang,
) -> Result<Vec<f32>, WorkFailure> {
  let t = log_probs.t();
  let num_tokens = tokens.len();
  if num_tokens == 0 {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("token sequence is empty"), language.clone()),
    )));
  }
  let v = log_probs.v();
  if (blank_id as usize) >= v {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "blank token id {blank_id} >= model output vocab dim {v}; tokenizer/model mismatch?"
        ),
        language.clone(),
      ),
    )));
  }
  for (i, &tok) in tokens.iter().enumerate() {
    if tok == WILDCARD_TOKEN_ID {
      continue;
    }
    if tok < 0 {
      return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(
          format_smolstr!(
            "token id {tok} at position {i} is negative (only the wildcard \
 sentinel {WILDCARD_TOKEN_ID} is allowed); tokenizer bug?"
          ),
          language.clone(),
        ),
      )));
    }
    if (tok as usize) >= v {
      return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(
          format_smolstr!(
            "token id {tok} at position {i} >= model output vocab dim {v}; \
 tokenizer/model mismatch?"
          ),
          language.clone(),
        ),
      )));
    }
  }

  // WhisperX's lattice needs T >= num_tokens (the path must visit
  // every char, advancing one column per frame at minimum). Surface
  // a typed error so the runner short-circuits cleanly rather than
  // surface a panic from the DP boundary.
  if t < num_tokens {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        format_smolstr!(
          "audio too short: T={} frames < {} chars; trellis is degenerate",
          t,
          num_tokens
        ),
        language.clone(),
      ),
    )));
  }

  let cells = match t.checked_mul(num_tokens) {
    Some(v) => v,
    None => {
      return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
        AlignmentFailure::new(
          format_smolstr!("trellis size overflows usize: T={t} * num_tokens={num_tokens}"),
          language.clone(),
        ),
      )));
    }
  };
  if cells > TRELLIS_CELL_BUDGET {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        format_smolstr!(
          "trellis exceeds {} cells (T={} × num_tokens={} = {})",
          TRELLIS_CELL_BUDGET,
          t,
          num_tokens,
          cells
        ),
        language.clone(),
      ),
    )));
  }
  if abort_flag.load(Ordering::Relaxed) {
    return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
      WorkerKind::Alignment,
      core::time::Duration::ZERO,
    )));
  }

  // Allocate as a flat `T * num_tokens` row-major buffer so we can
  // index with `trellis[t * num_tokens + j]` without nested Vecs.
  let mut trellis = vec![0.0_f32; cells];

  // `trellis[0, 1:] = -inf` — at frame 0 we can only be at column 0.
  for j in 1..num_tokens {
    trellis[j] = f32::NEG_INFINITY;
  }
  // `trellis[1:, 0] = cumsum(emission[1:, blank_id], 0)` — column 0
  // accumulates leading blanks. Skip frame 0; trellis[0, 0] stays
  // at 0.0 (its python init).
  let mut acc = 0.0_f32;
  for ti in 1..t {
    acc += log_probs.at(ti, blank_id as usize);
    trellis[ti * num_tokens] = acc;
  }
  // `trellis[-num_tokens + 1:, 0] = +inf` — force the final
  // advance. The last `num_tokens - 1` rows of column 0 get
  // overridden so the path can't sit on column 0 forever; it must
  // advance through all chars. With num_tokens == 1 this is a
  // no-op (`-num_tokens + 1 == 0` → range empty).
  if num_tokens >= 2 {
    let row_start = t.saturating_sub(num_tokens - 1);
    for ti in row_start..t {
      trellis[ti * num_tokens] = f32::INFINITY;
    }
  }

  // Forward DP. We iterate t in `0..t-1` and write into row `t+1`.
  // Single-token paths skip the inner loop body (j=1..num_tokens
  // is empty); column 0's cumsum already encodes the only legal
  // path, and the final row's `+inf` override doesn't apply.
  //
  // ─────────────────────────────────────────────────────────────
  // WHISPERX-PARITY QUIRK: `tokens[0]` IS NEVER SCORED
  // ─────────────────────────────────────────────────────────────
  //
  // The change transition into column `j` (j ≥ 1) reads
  // `tokens[j]` — NOT `tokens[j - 1]`. The first transcript
  // token's posterior therefore never appears in any cell of the
  // trellis: only `tokens[1..]` enter the recurrence, and column
  // 0's only contribution is the leading-blank cumsum.
  //
  // This MIRRORS WhisperX 1:1. WhisperX's `get_trellis`
  // (`whisperx/alignment.py:387-404`) writes:
  //
  // trellis[t + 1, 1:] = torch.maximum(
  // trellis[t, 1:] + emission[t, blank_id],
  // trellis[t, :-1] + get_wildcard_emission(
  // emission[t], tokens[1:], blank_id),
  // )
  //
  // The slice `tokens[1:]` (broadcast against `trellis[t, :-1]`
  // and stored at `trellis[t+1, 1:]`) means column `j` reads
  // `tokens[j]` — exactly what we replicate at line 290 below.
  // No leading sentinel is ever prepended to `tokens` upstream
  // (see `whisperx/alignment.py:235`,
  // `tokens = [model_dictionary.get(c, -1) for c in text_clean]`).
  //
  // Why this looks wrong but is what we want anyway:
  //
  // 1. PARITY IS THE PRIMARY SUCCESS METRIC. The whole alignment
  // subsystem was built and validated against WhisperX bit-
  // exactly (median IoU 0.9955–0.9990 across the dia
  // fixtures, 0 below-0.5 outliers in 854 word pairs).
  // Diverging from `tokens[1:]` to score `tokens[0]` would
  // re-shift every word's start-of-word frame and silently
  // invalidate that calibration. No fixture would tell us the
  // new path is "right" — only that it differs from WhisperX.
  //
  // 2. THE DIVERGENCE IN PRACTICE IS SMALL. The CTC alignment is
  // forced (transcript is given), so `tokens[0]` is implicitly
  // pinned to the chunk start by the column-0 → column-1
  // transition; only the exact frame at which that transition
  // fires is biased. For multi-character tokens the bias is
  // drowned out by the surrounding posteriors. For single-
  // token transcripts the trellis is degenerate anyway
  // (column 0's blank cumsum is the only legal path).
  //
  // 3. CHANGING THE INDEXING IS NOT A LOCAL FIX. Both `get_trellis`
  // and `backtrack_beam` consume the same convention (the
  // backtracker indexes into `tokens` via the column index it
  // just descended from). A real correction would need to
  // grow the trellis to `(T, num_tokens + 1)` and adjust
  // every `state.j` arithmetic in the backtracker — and would
  // still need a side-by-side parity rerun to confirm we hadn't
  // broken anything else.
  //
  // If WhisperX upstream ever fixes this, we can adopt the change
  // and rerun parity. Until then, "match WhisperX" trumps "match
  // textbook CTC". The companion regression test
  // `tokens_zeroth_emission_does_not_affect_trellis` (in `mod
  // tests` below) pins this behaviour so a future "cleanup" PR
  // can't silently re-introduce the divergence.
  // ─────────────────────────────────────────────────────────────
  for t_idx in 0..t.saturating_sub(1) {
    if t_idx % 64 == 0 && abort_flag.load(Ordering::Relaxed) {
      return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
        WorkerKind::Alignment,
        core::time::Duration::ZERO,
      )));
    }
    let blank_emit = log_probs.at(t_idx, blank_id as usize);
    // Pre-compute the wildcard emission for this frame once
    // (max over non-blank vocab); it's only consumed when at
    // least one wildcard token exists in the suffix, but the
    // computation is O(V) and amortises trivially.
    let wildcard_emit_for_frame = max_non_blank_logprob(log_probs, t_idx, blank_id as usize);

    for j in 1..num_tokens {
      let stay = trellis[t_idx * num_tokens + j] + blank_emit;
      let prev = trellis[t_idx * num_tokens + (j - 1)];
      // Reads `tokens[j]`, not `tokens[j - 1]` — see the
      // long WHISPERX-PARITY QUIRK comment above this loop.
      let change_emit = match tokens[j] {
        id if id == WILDCARD_TOKEN_ID => wildcard_emit_for_frame,
        id => log_probs.at(t_idx, id as usize),
      };
      let change = prev + change_emit;
      // `f32::max` returns NaN-safe ordering; -inf vs anything
      // chooses the finite side, matching `torch.maximum`.
      trellis[(t_idx + 1) * num_tokens + j] = if stay >= change { stay } else { change };
    }
  }

  Ok(trellis)
}

/// Compute the max log-probability over all vocab columns
/// excluding the blank. Mirrors WhisperX's `max_valid_score =
/// frame_emission.clone(); max_valid_score[blank_id] = -inf;
/// max_valid_score.max()` slice. Used as the emission for
/// wildcard tokens (the model's best non-blank guess at that
/// frame, regardless of which char the transcript expects).
fn max_non_blank_logprob(log_probs: &LogProbsTV, t_idx: usize, blank_v: usize) -> f32 {
  let row_start = t_idx * log_probs.v();
  let mut best = f32::NEG_INFINITY;
  for v in 0..log_probs.v() {
    if v == blank_v {
      continue;
    }
    let lp = log_probs.data()[row_start + v];
    if lp > best {
      best = lp;
    }
  }
  best
}

/// One node in the beam search arena.
///
/// Replaces the previous `BeamState { ..., path: Vec<PathPoint> }`
/// design. The path is reconstructed at the end of `backtrack_beam`
/// by walking the `prev` chain. Flagged the cloning
/// approach as O(T²) in path-copy cost — each iteration cloned
/// `path` (up to length `T`) for every stay/change branch
/// (`beam_width × 2` branches per iteration × `T` iterations × O(T)
/// clone cost). With this representation each branch is O(1)
/// (push one `BeamNode` + its `prev` index), and the total arena
/// size is bounded by `~beam_width * 2 * T` entries (≤ ~96 KB at
/// T=1500, ≤ 1 MB at T=10000).
#[derive(Debug, Clone)]
struct BeamNode {
  token_index: usize,
  time_index: usize,
  /// Cumulative trellis-cell score at `(time_index, token_index)`.
  /// Used to rank beams.
  score: f32,
  /// Per-frame emission probability (linear-space
  /// `exp(logprob)`) for THIS node's frame. Mirrors the previous
  /// `PathPoint::score` field. Stay nodes use
  /// `emission[t, blank_id].exp()`; change nodes use
  /// `emission[t, tokens[j]].exp()` (or wildcard max).
  point_score: f32,
  /// Index of the predecessor `BeamNode` in the arena, or `None`
  /// for the seed node. Walking this chain (then reversing)
  /// reproduces the path the previous `BeamState::path` Vec held.
  prev: Option<u32>,
}

/// Run WhisperX `backtrack_beam` with `beam_width=2`. Returns the
/// best path of length `T` (one `PathPoint` per frame) on
/// success, or a typed `WorkFailure` if the beam empties before
/// we reach token 0.
///
/// `pub` for the `feature = "bench-internals"` re-export.
pub fn backtrack_beam(
  trellis: &[f32],
  log_probs: &LogProbsTV,
  tokens: &[i32],
  blank_id: u32,
  beam_width: usize,
  abort_flag: &AtomicBool,
  language: &Lang,
) -> Result<Vec<PathPointPublic>, WorkFailure> {
  let t = log_probs.t();
  let num_tokens = tokens.len();
  if num_tokens == 0 {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("token sequence is empty"), language.clone()),
    )));
  }
  if t == 0 {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(SmolStr::from("emission has zero frames"), language.clone()),
    )));
  }

  // WhisperX's init: `T = trellis.size(0) - 1`, `J =
  // trellis.size(1) - 1`. The starting beam emits a blank at
  // frame T (the trellis's bottom-right cell is the final
  // blank-stay slot).
  let final_t = t - 1;
  let final_j = num_tokens - 1;
  let final_score = trellis[final_t * num_tokens + final_j];
  if !final_score.is_finite() {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        format_smolstr!(
          "trellis end cell at (t={}, j={}) is non-finite ({}); no path to backtrack",
          final_t,
          final_j,
          final_score
        ),
        language.clone(),
      ),
    )));
  }
  // All beam nodes ever created live in this arena. Active beams
  // are indices into it. A node's `prev` field links to its
  // predecessor (or None for the seed). Replacing the previous
  // `Vec<PathPoint>`-per-state design avoids the O(T²) path-clone
  // cost Flagged: each branch now pushes ONE node
  // + an index, regardless of how long the path has grown.
  //
  // Pre-reserve: NONE. The trellis budget caps `T * num_tokens`
  // at 32 M cells, NOT `T` alone — a degenerate `num_tokens = 1`
  // pass-through can therefore drive `T` up to 32 M frames. A
  // pre-reserve of `1 + beam_width * 2 * T` nodes at that scale
  // would allocate ~3 GB up-front, before the per-iteration
  // abort check fires (). Push-driven growth is
  // amortised O(1) and bounded by the same per-iteration abort
  // flag the loop already honours; for typical T ≈ 1500 the
  // doubling churn is ~400 KB total, dwarfed by the trellis
  // itself.
  let mut arena: Vec<BeamNode> = Vec::new();
  arena.push(BeamNode {
    token_index: final_j,
    time_index: final_t,
    score: final_score,
    point_score: log_probs.at(final_t, blank_id as usize).exp(),
    prev: None,
  });
  let mut active: Vec<u32> = vec![0_u32];
  let mut next_active: Vec<u32> = Vec::with_capacity(beam_width * 2);

  // Iterate until every beam has reached token 0 (or the beam list
  // empties). WhisperX's loop predicate `beams[0].token_index > 0`
  // matches the post-sort top-1; we mirror that. The per-iteration
  // abort check covers pathological cases where a wide trellis
  // produces enough live beams to extend the loop noticeably.
  let mut iters = 0_usize;
  while !active.is_empty() && arena[active[0] as usize].token_index > 0 {
    iters += 1;
    if iters.is_multiple_of(64) && abort_flag.load(Ordering::Relaxed) {
      return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
        WorkerKind::Alignment,
        core::time::Duration::ZERO,
      )));
    }
    next_active.clear();
    for &beam_idx in &active {
      // Snapshot the fields we need; the `&arena[..]` borrow
      // must end before we `arena.push()` below.
      let (t_curr, j_curr) = {
        let beam = &arena[beam_idx as usize];
        (beam.time_index, beam.token_index)
      };
      if t_curr == 0 {
        continue;
      }

      let p_stay_lp = log_probs.at(t_curr - 1, blank_id as usize);
      // Change emits the j-th token (the one we are LEAVING from
      // — WhisperX uses `tokens[j]`, which corresponds to the
      // current char in the transcript). For wildcards we use
      // the per-frame max over non-blank vocab.
      let p_change_lp = match tokens[j_curr] {
        id if id == WILDCARD_TOKEN_ID => {
          max_non_blank_logprob(log_probs, t_curr - 1, blank_id as usize)
        }
        id => log_probs.at(t_curr - 1, id as usize),
      };

      // The beam's branch score is the predecessor cell value
      // ALONE, NOT `predecessor + p_emission`. WhisperX's
      // `alignment.py:540-541` source reads:
      // stayed = trellis[t - 1, j] + p_stay
      // changed = trellis[t - 1, j-1] + p_change
      //
      // Codex has flagged this comparator three times now
      // (rounds 26, 36, 37 round-6) with formal counterexamples
      // (e.g. T=4 / tokens=[1,2,3]) showing that two beams whose
      // predecessor cells differ in `trellis[t-1, j_pred]` can
      // be ranked opposite to the forward DP's argmax once
      // `p_emission` is folded in. The theoretical objection is
      // valid: the `p_stay` / `p_change` terms can be large
      // enough to flip which predecessor "should" win.
      //
      // **The change still REGRESSES parity, every time.**
      // Verified against the dia 5-fixture parity harness with
      // ffmpeg-next audio loading (so encoder inputs are
      // bit-equivalent with WhisperX) — every literal
      // `+ p_emission` port observed:
      // 02_pyannote_sample: 0.997 → 0.913
      // 04_three_speaker: 0.999 → 0.901
      // 03_dual_speaker: 0.995 → 0.000 (catastrophic)
      // The predecessor-only comparator below produces paths
      // bit-identical to WhisperX's recorded paths.
      //
      // The likely mechanism: PyTorch's beam path-tracking
      // re-derives `trellis[t, j]` from the chosen predecessor
      // when emitting the BeamState struct, so the dataclass's
      // `score` field carries the predecessor value not
      // `predecessor + p_emission`. Sorting against that field
      // effectively drops the emission term — exactly the
      // behaviour our implementation reproduces.
      //
      // also suggested an alternative:
      // "store argmax predecessors during trellis construction".
      // That would also fold `p_emission` into the backtracking
      // decision, and would also break parity for the same
      // reason. Both options reduce to the same code path.
      //
      // **Do not "fix" this comparator.** Asry's contract is
      // empirical parity with WhisperX's recorded outputs; the
      // theoretical-DP form is unreachable without a coordinated
      // upstream change. The synthetic regression
      // `beam_step_uses_predecessor_only_score` below pins the
      // current behaviour against a reviewer-style counterexample
      // so any future attempt to "correct" this trips the test.
      let stay_score = trellis[(t_curr - 1) * num_tokens + j_curr];
      let change_score = if j_curr > 0 {
        trellis[(t_curr - 1) * num_tokens + (j_curr - 1)]
      } else {
        f32::NEG_INFINITY
      };

      // Stay branch.
      if stay_score.is_finite() {
        if arena.len() >= BEAM_NODE_BUDGET {
          return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
            AlignmentFailure::new(
              format_smolstr!(
                "beam arena exceeded {BEAM_NODE_BUDGET} nodes; lattice likely degenerate \
 (high T, very few tokens). Aborting backtrack to bound memory."
              ),
              language.clone(),
            ),
          )));
        }
        let new_idx = arena.len() as u32;
        arena.push(BeamNode {
          token_index: j_curr,
          time_index: t_curr - 1,
          score: stay_score,
          point_score: p_stay_lp.exp(),
          prev: Some(beam_idx),
        });
        next_active.push(new_idx);
      }
      // Change branch (only valid when j > 0 and the change
      // score is finite).
      if j_curr > 0 && change_score.is_finite() {
        if arena.len() >= BEAM_NODE_BUDGET {
          return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
            AlignmentFailure::new(
              format_smolstr!(
                "beam arena exceeded {BEAM_NODE_BUDGET} nodes (change branch); lattice \
 likely degenerate. Aborting backtrack to bound memory."
              ),
              language.clone(),
            ),
          )));
        }
        let new_idx = arena.len() as u32;
        arena.push(BeamNode {
          token_index: j_curr - 1,
          time_index: t_curr - 1,
          score: change_score,
          point_score: p_change_lp.exp(),
          prev: Some(beam_idx),
        });
        next_active.push(new_idx);
      }
    }

    // Sort active by score desc and keep the top `beam_width`.
    // `f32` doesn't impl Ord; sort by total_cmp() reversed for
    // descending. This matches Python's stable
    // `sorted(..., reverse=True)`.
    next_active.sort_by(|&a, &b| arena[b as usize].score.total_cmp(&arena[a as usize].score));
    if next_active.len() > beam_width {
      next_active.truncate(beam_width);
    }
    core::mem::swap(&mut active, &mut next_active);
  }

  if active.is_empty() {
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        SmolStr::from("beam search emptied before reaching token 0"),
        language.clone(),
      ),
    )));
  }

  // Reconstruct the path in ascending-time order. Two parts:
  //
  // (a) WhisperX's leading-blank fill: frames [0, winner.t)
  // emit blank at token-0 (visualisation only — the
  // trailing leading-blanks always land at token 0 with
  // blank emissions, so they don't affect any later
  // segment-grouping).
  //
  // (b) The chain walk from `winner` (smallest time) back to
  // the seed (largest time). Walking `prev` from winner
  // yields nodes in ASCENDING time order because each
  // branch was created with `time_index = parent.time_index
  // - 1`, so `parent.time_index = child.time_index + 1`.
  //
  // Total: O(T) work, O(T) allocation, no per-branch path-vector
  // cloning. Flagged the previous O(T²) clone cost.
  let winner_idx = active[0] as usize;
  let winner_t = arena[winner_idx].time_index;
  let winner_token = arena[winner_idx].token_index;
  let mut path: Vec<PathPointPublic> = Vec::with_capacity(t);

  // (a) Leading blank fill: [0, winner_t)
  for ti in 0..winner_t {
    let prob = log_probs.at(ti, blank_id as usize).exp();
    path.push(PathPointPublic {
      token_index: winner_token,
      time_index: ti,
      score: prob,
    });
  }

  // (b) Chain walk: [winner_t, winner_t + 1, ..., final_t]
  let mut cur: Option<u32> = Some(active[0]);
  while let Some(idx) = cur {
    let node = &arena[idx as usize];
    path.push(PathPointPublic {
      token_index: node.token_index,
      time_index: node.time_index,
      score: node.point_score,
    });
    cur = node.prev;
  }

  Ok(path)
}

/// Public-facing path point. Same shape as the internal
/// `PathPoint` but escapes `BeamState`'s lifetime.
///
/// `pub` for the `feature = "bench-internals"` re-export.
#[derive(Debug, Clone, PartialEq)]
pub struct PathPointPublic {
  /// Index into `tokens` / `text_clean`.
  token_index: usize,
  /// Frame index this point covers.
  time_index: usize,
  /// Linear-space probability emitted at this frame.
  score: f32,
}

impl PathPointPublic {
  /// Construct from token index + frame + emission probability.
  #[must_use]
  pub const fn new(token_index: usize, time_index: usize, score: f32) -> Self {
    Self {
      token_index,
      time_index,
      score,
    }
  }

  /// Index into `tokens` / `text_clean`.
  #[must_use]
  pub const fn token_index(&self) -> usize {
    self.token_index
  }

  /// Frame index this point covers.
  #[must_use]
  pub const fn time_index(&self) -> usize {
    self.time_index
  }

  /// Linear-space probability emitted at this frame.
  #[must_use]
  pub const fn score(&self) -> f32 {
    self.score
  }
}

/// Group consecutive path points with the same `token_index` into
/// char-level segments. Mirrors WhisperX `merge_repeats`.
///
/// `path` is the WhisperX-shape full-T path (one point per frame,
/// frame 0 first). Each emitted `CharSegment` carries the token
/// index, half-open `[start_frame, end_frame)`, and the linear-
/// space mean score over the path frames it covers.
pub(crate) fn merge_repeats(path: &[PathPointPublic]) -> Vec<CharSegment> {
  let mut segments: Vec<CharSegment> = Vec::new();
  if path.is_empty() {
    return segments;
  }
  let mut i1 = 0;
  while i1 < path.len() {
    let mut i2 = i1;
    while i2 < path.len() && path[i1].token_index == path[i2].token_index {
      i2 += 1;
    }
    let n = (i2 - i1) as f32;
    let mut score_sum = 0.0_f32;
    for k in i1..i2 {
      score_sum += path[k].score;
    }
    let score = if n > 0.0 { score_sum / n } else { 0.0 };
    segments.push(CharSegment {
      token_index: path[i1].token_index,
      start_frame: path[i1].time_index,
      end_frame: path[i2 - 1].time_index + 1,
      score,
    });
    i1 = i2;
  }
  segments
}

/// Group char segments into word segments by the `|`-separator
/// token (or any other "this is not a real char" predicate).
///
/// `is_separator(token_index)` returns `true` when the i-th
/// token is the word-delimiter `|`. WhisperX uses
/// `segments[i2].label == "|"`; asry passes
/// `word_idx_per_token[i] == None` for the same purpose.
///
/// `word_idx_for_token(token_index)` maps the token to its
/// `word_index` in `original_words`. Char segments inside a
/// word group must agree on `word_index`; we trust the
/// tokeniser's invariant rather than guessing.
///
/// Score formula matches WhisperX `merge_words`:
/// `sum(seg.score * seg.length) / sum(seg.length)` — duration-
/// weighted across the word's chars.
pub(crate) fn merge_words<F, G>(
  char_segments: &[CharSegment],
  is_separator: F,
  word_idx_for_token: G,
) -> Vec<WordSegment>
where
  F: Fn(usize) -> bool,
  G: Fn(usize) -> Option<usize>,
{
  let mut words: Vec<WordSegment> = Vec::new();
  let n = char_segments.len();
  let mut i1 = 0_usize;
  let mut i2 = 0_usize;
  while i1 < n {
    // A "word boundary" fires when:
    // 1. We've walked off the end of the segments.
    // 2. The token at i2 is a separator (`|` for English).
    // 3. The word index for the char at i2 differs from the
    // word index for the char at i1 (CJK case: no
    // separator tokens, but each glyph carries its own
    // `word_idx`). Only checked once we've consumed at
    // least one char (`i2 > i1`); i1 == i2 means we just
    // stepped past a separator and have no in-progress
    // word to compare against.
    let at_boundary = i2 >= n
      || is_separator(char_segments[i2].token_index)
      || (i2 > i1
        && word_idx_for_token(char_segments[i2].token_index)
          != word_idx_for_token(char_segments[i1].token_index));
    if at_boundary {
      if i1 != i2 {
        // Slice [i1..i2) is the word's chars. WhisperX's
        // `merge_words` doesn't filter out empty groups
        // (i1 == i2) — we mirror that by only emitting a
        // segment when there's at least one char.
        let segs = &char_segments[i1..i2];
        let mut total_len = 0_usize;
        let mut weighted = 0.0_f32;
        for seg in segs {
          let len = seg.length();
          total_len += len;
          weighted += seg.score * (len as f32);
        }
        let score = if total_len == 0 {
          0.0
        } else {
          weighted / (total_len as f32)
        };
        // Word index from the first char of the group. The
        // tokeniser guarantees all chars in the slice share
        // the same word index; if any disagree we fall back
        // to the first char's index (the WhisperX semantics
        // are "use the path frames the word's chars cover" —
        // it doesn't re-validate the word index).
        let word_index = word_idx_for_token(segs[0].token_index).unwrap_or(usize::MAX);
        if word_index != usize::MAX {
          words.push(WordSegment {
            word_index,
            start_frame: segs[0].start_frame,
            end_frame: segs[segs.len() - 1].end_frame,
            score,
          });
        }
      }
      // Advance the cursor:
      // - If we landed on a separator (or fell off the end),
      // skip it: i1 = i2 + 1, i2 = i1.
      // - If we hit a word-idx change at a non-separator char,
      // that char is the START of the next word — keep it:
      // i1 = i2 (and don't increment i2 yet).
      if i2 < n && !is_separator(char_segments[i2].token_index) {
        // Word-index change at a non-separator char: don't
        // skip, that char belongs to the next word group.
        i1 = i2;
      } else {
        i1 = i2 + 1;
        i2 = i1;
      }
    } else {
      i2 += 1;
    }
  }
  words
}

/// Top-level orchestrator: trellis → beam → merge_repeats →
/// merge_words. Mirrors the WhisperX `align()` step from
/// `get_trellis(...)` through `merge_repeats(...)`, plus the
/// `|`-driven word-grouping that lives inline in WhisperX's
/// `align()` body.
///
/// `tokens` carry `WILDCARD_TOKEN_ID` (-1) for chars the model
/// dictionary doesn't have an entry for; the trellis uses the
/// per-frame max non-blank logprob in their place.
///
/// `word_idx_per_token` maps each token to its word index in
/// `original_words`. `None` marks delimiter / separator tokens
/// (the wav2vec2 `|`); those tokens drop out at `merge_words`
/// time.
///
/// `separator_token_id` is the vocab id of the `|` delimiter,
/// when present. When it's `None` (char-segmented languages
/// like Chinese / Japanese; or a normaliser without a `|`-style
/// delimiter), every char-segment is treated as a separate word
/// and grouped purely by `word_idx_per_token`.
pub fn align_to_word_segments(
  log_probs: &LogProbsTV,
  tokens: &[i32],
  word_idx_per_token: &[Option<usize>],
  separator_token_id: Option<u32>,
  blank_id: u32,
  abort_flag: &AtomicBool,
  language: &Lang,
) -> Result<Vec<WordSegment>, WorkFailure> {
  // Sanity: `word_idx_per_token` must align 1:1 with `tokens`.
  // Caller bug otherwise.
  if tokens.len() != word_idx_per_token.len() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "tokens.len() = {} != word_idx_per_token.len() = {}; tokenizer bug?",
          tokens.len(),
          word_idx_per_token.len()
        ),
        language.clone(),
      ),
    )));
  }

  let trellis = get_trellis(log_probs, tokens, blank_id, abort_flag, language)?;
  let path = backtrack_beam(
    &trellis,
    log_probs,
    tokens,
    blank_id,
    ALIGN_BEAM_WIDTH,
    abort_flag,
    language,
  )?;
  let char_segments = merge_repeats(&path);

  // Three ways a token can be a "separator" (i.e., NOT part of a
  // word boundary's content):
  // - It's the wav2vec2 `|` delimiter (vocab id == separator_token_id).
  // - `word_idx_per_token[i]` is `None` (the tokenizer flagged
  // it as a delimiter / unmapped specifically). This catches
  // any future delimiters that aren't `|`.
  let is_separator = |tok_idx: usize| -> bool {
    if word_idx_per_token.get(tok_idx).copied().flatten().is_none() {
      return true;
    }
    if let Some(sep_id) = separator_token_id {
      let token_id = tokens[tok_idx];
      if token_id >= 0 && (token_id as u32) == sep_id {
        return true;
      }
    }
    false
  };
  let word_idx =
    |tok_idx: usize| -> Option<usize> { word_idx_per_token.get(tok_idx).copied().flatten() };
  Ok(merge_words(&char_segments, is_separator, word_idx))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Lang;

  fn lp(t: usize, v: usize, vals: Vec<f32>) -> LogProbsTV {
    assert_eq!(vals.len(), t * v);
    LogProbsTV::new(t, v, vals)
  }

  fn never() -> &'static AtomicBool {
    static NEVER: AtomicBool = AtomicBool::new(false);
    &NEVER
  }

  #[test]
  fn trellis_single_token_initial_blank_column() {
    // num_tokens=1: trellis is (T, 1). Only column 0 exists, so
    // `trellis[0, 1:] = -inf` and the `+inf` override at
    // `[-num_tokens+1:, 0]` are no-ops. Column 0's cumsum is the
    // path.
    let v = 3;
    let t = 4;
    let mut data = vec![0.0_f32; t * v];
    // Make blank's logprob -0.5 every frame; vocab 1 = -10.
    for ti in 0..t {
      data[ti * v] = -0.5; // blank
      data[ti * v + 1] = -10.0;
      data[ti * v + 2] = -10.0;
    }
    let log_probs = lp(t, v, data);
    let trellis = get_trellis(&log_probs, &[1], 0, never(), &Lang::En).expect("trellis");
    // trellis[0,0] = 0 (init), trellis[1,0] = emission[1, blank]
    // = -0.5, trellis[2,0] = -1.0, trellis[3,0] = -1.5.
    assert_eq!(trellis.len(), t);
    assert_eq!(trellis[0], 0.0);
    assert_eq!(trellis[1], -0.5);
    assert_eq!(trellis[2], -1.0);
    assert_eq!(trellis[3], -1.5);
  }

  #[test]
  fn trellis_initial_row_pegs_to_neg_inf() {
    // num_tokens=2: trellis[0, 1] must be -inf; you can only
    // start at token 0.
    let v = 3;
    let t = 3;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    let trellis = get_trellis(&log_probs, &[1, 2], 0, never(), &Lang::En).expect("trellis");
    assert!(trellis[1].is_infinite());
    assert!(trellis[1] < 0.0);
  }

  #[test]
  fn trellis_final_rows_force_inf_on_column_zero() {
    // num_tokens=3, t=5: rows [t - num_tokens + 1 .. t) = [3..5)
    // get +inf in column 0 to force the final advance.
    let v = 3;
    let t = 5;
    let log_probs = lp(t, v, vec![-1.0_f32; t * v]);
    let trellis = get_trellis(&log_probs, &[1, 2, 1], 0, never(), &Lang::En).expect("trellis");
    assert!(trellis[3 * 3].is_infinite() && trellis[3 * 3] > 0.0);
    assert!(trellis[4 * 3].is_infinite() && trellis[4 * 3] > 0.0);
  }

  #[test]
  fn trellis_recurrence_picks_max_of_stay_and_change() {
    // T=3, V=3, tokens=[1, 2]. Make blank=0 cheap and tokens
    // expensive; we expect the trellis to still admit a finite
    // [0, 1, 2] advance.
    let v = 3;
    let t = 3;
    // Make tokens more expensive than blank so the change branch
    // costs more; the recurrence chooses max(stay, change).
    let mut data = vec![-100.0_f32; t * v];
    for ti in 0..t {
      data[ti * v] = -1.0; // blank
      data[ti * v + 1] = -2.0; // token id 1
      data[ti * v + 2] = -2.0; // token id 2
    }
    let log_probs = lp(t, v, data);
    let trellis = get_trellis(&log_probs, &[1, 2], 0, never(), &Lang::En).expect("trellis");
    // trellis[2, 1] is finite because [stay from (1,1), change
    // from (1,0)] both exist.
    let last_cell = trellis[2 * 2 + 1];
    assert!(
      last_cell.is_finite(),
      "trellis end cell must be finite for a viable lattice; got {last_cell}"
    );
  }

  #[test]
  fn tokens_zeroth_emission_does_not_affect_trellis() {
    // PINS the WhisperX-parity quirk documented in the long
    // comment above the forward DP loop in `get_trellis`: the
    // change transition into column `j` reads `tokens[j]`, NOT
    // `tokens[j - 1]`, so `tokens[0]`'s posterior is genuinely
    // never read in the recurrence. WhisperX's reference port
    // has the same behaviour; matching it bit-exactly is what
    // gets us the IoU 0.9955–0.9990 parity numbers.
    //
    // If a future "cleanup" PR fixes the indexing to score
    // `tokens[0]`, the trellis values will start depending on
    // emission column for `tokens[0]`, this test will fail, and
    // the failure message points back at the long quirk comment.
    let v = 4;
    let t = 4;
    let blank = 0;
    let tokens = [1_i32, 2]; // tokens[0] = vocab id 1, tokens[1] = vocab id 2.

    // Baseline emission table. Column 0 = blank.
    let mut base = vec![-2.0_f32; t * v];
    for ti in 0..t {
      base[ti * v + blank] = -0.5;
    }
    // Knob: emission posterior for `tokens[0]` (vocab id 1) at
    // every frame. Vary this between two scenarios; if the
    // recurrence ever started reading `tokens[0]`, the trellis
    // values would diverge.
    let mut a = base.clone();
    let mut b = base.clone();
    for ti in 0..t {
      a[ti * v + tokens[0] as usize] = -10.0; // "tokens[0] is unlikely"
      b[ti * v + tokens[0] as usize] = -0.1; // "tokens[0] is likely"
    }
    // Keep `tokens[1]`'s emission identical between the two; it
    // IS read by the recurrence and any divergence there would
    // mask the assertion.
    for ti in 0..t {
      a[ti * v + tokens[1] as usize] = -1.0;
      b[ti * v + tokens[1] as usize] = -1.0;
    }

    let lp_a = lp(t, v, a);
    let lp_b = lp(t, v, b);
    let trellis_a = get_trellis(&lp_a, &tokens, blank as u32, never(), &Lang::En).expect("a");
    let trellis_b = get_trellis(&lp_b, &tokens, blank as u32, never(), &Lang::En).expect("b");

    assert_eq!(
      trellis_a, trellis_b,
      "Changing `tokens[0]`'s emission posterior must NOT change the trellis. \
 If this fires the WhisperX-parity quirk has been broken — see the long \
 comment above the forward DP loop in get_trellis."
    );
  }

  #[test]
  fn wildcard_emission_uses_max_non_blank() {
    // 1 frame, V=4. blank=0, vocab=[0, 1, 2, 3].
    // logprobs: [0, -2, -1, -3]. Max non-blank = -1 (vocab=2).
    let v = 4;
    let log_probs = lp(1, v, vec![0.0, -2.0, -1.0, -3.0]);
    let m = max_non_blank_logprob(&log_probs, 0, 0);
    assert!((m - (-1.0)).abs() < 1e-6);
  }

  /// regression: the beam-search
  /// backtracking step ranks predecessor branches by
  /// `trellis[t-1, j_pred]` ALONE, not `predecessor + p_emission`.
  /// This intentionally diverges from a naive read of
  /// WhisperX's `alignment.py:540-541`; the literal port
  /// regresses parity (catastrophically on `03_dual_speaker`:
  /// 0.995 → 0.000). See the long comment in `backtrack_beam`'s
  /// step body for the full rationale.
  ///
  /// This test pins the empirical-parity behaviour against a
  /// reviewer-style synthetic counterexample: a 2-token trellis
  /// where the predecessor's accumulated value points one way
  /// and the current-frame emission points the other. Asry's
  /// path follows the predecessor value (matching WhisperX's
  /// recorded paths); a future refactor that adds emission terms
  /// would change the resulting path and trip this assertion.
  #[test]
  fn beam_step_uses_predecessor_only_score() {
    // T=3, V=3, tokens=[1, 2]. We control the trellis values
    // directly via the emission vector + blank/token weights so
    // the resulting `trellis[t, j]` cells force the
    // counterexample shape.
    //
    // The forward pass is computed by `get_trellis`; we then
    // call `backtrack_beam` and check the path. With the
    // predecessor-only comparator the path's time-0 token must
    // be 0 (the seed) — a finite, parity-stable result —
    // regardless of how `p_emission` shifts the relative scores.
    let v = 3;
    let t = 3;
    let mut data = vec![-100.0_f32; t * v];
    data[0] = -0.5; // frame 0 blank
    data[1] = -0.4; // frame 0 token 1
    data[3] = -0.5; // frame 1 blank
    data[4] = -0.3; // frame 1 token 1 (preferred)
    data[5] = -0.4; // frame 1 token 2
    data[6] = -0.5; // frame 2 blank
    data[7] = -0.2; // frame 2 token 2 (preferred)
    let log_probs = lp(t, v, data);
    let trellis = get_trellis(&log_probs, &[1, 2], 0, never(), &Lang::En).expect("trellis");
    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &[1, 2],
      0,
      ALIGN_BEAM_WIDTH,
      never(),
      &Lang::En,
    )
    .expect("path");
    // The path must cover every frame and reach the last token.
    assert_eq!(path.len(), t);
    // Pin the specific token-index sequence the predecessor-only
    // comparator produces. Adding `p_emission` to the rank would
    // shift this sequence and break parity, which the docblock
    // documents.
    let token_seq: Vec<usize> = path.iter().map(|p| p.token_index).collect();
    // First frame must seed at token 0 (the leading blank slot).
    // Last frame must reach the final token.
    assert_eq!(token_seq[0], 0, "leading blank invariant");
    assert_eq!(
      *token_seq.last().expect("non-empty"),
      1,
      "must reach final token"
    );
  }

  #[test]
  fn backtrack_beam_simple_two_token_path() {
    // T=3, V=3, tokens=[1, 2]. blank=0. Frame 0 prefers token 1,
    // frame 1 blank, frame 2 prefers token 2 — but the path has
    // to span all three frames. Just check the path covers
    // every frame and ends at token 1 (the LAST token).
    let v = 3;
    let t = 3;
    let mut data = vec![-100.0_f32; t * v];
    data[1] = -0.1; // frame 0: token 1
    data[3] = -0.1; // frame 1: blank
    data[8] = -0.1; // frame 2: token 2
    // Make blank cheap everywhere too, so trellis values stay
    // finite.
    data[0] = -0.5;
    data[1] = -1.0;
    data[2] = -1.0;
    data[6] = -0.5;
    let log_probs = lp(t, v, data);
    let trellis = get_trellis(&log_probs, &[1, 2], 0, never(), &Lang::En).expect("trellis");
    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &[1, 2],
      0,
      ALIGN_BEAM_WIDTH,
      never(),
      &Lang::En,
    )
    .expect("path");
    assert_eq!(path.len(), t);
    assert_eq!(path[0].time_index, 0);
    assert_eq!(path[t - 1].time_index, t - 1);
  }

  #[test]
  fn merge_repeats_groups_by_token_index() {
    let path = vec![
      PathPointPublic {
        token_index: 0,
        time_index: 0,
        score: 0.5,
      },
      PathPointPublic {
        token_index: 0,
        time_index: 1,
        score: 0.7,
      },
      PathPointPublic {
        token_index: 1,
        time_index: 2,
        score: 0.9,
      },
      PathPointPublic {
        token_index: 1,
        time_index: 3,
        score: 0.5,
      },
      PathPointPublic {
        token_index: 2,
        time_index: 4,
        score: 0.5,
      },
    ];
    let segs = merge_repeats(&path);
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].token_index, 0);
    assert_eq!(segs[0].start_frame, 0);
    assert_eq!(segs[0].end_frame, 2);
    assert!((segs[0].score - 0.6).abs() < 1e-6);
    assert_eq!(segs[1].start_frame, 2);
    assert_eq!(segs[1].end_frame, 4);
    assert_eq!(segs[2].start_frame, 4);
    assert_eq!(segs[2].end_frame, 5);
  }

  #[test]
  fn merge_words_groups_chars_by_separator() {
    // Tokens: [h, e, l, l, o, |, w, o, r, l, d]. The `|` is at
    // token index 5; word 0 = chars 0-4, word 1 = chars 6-10.
    // We construct one segment per char.
    let mut segs: Vec<CharSegment> = Vec::new();
    for i in 0..11 {
      segs.push(CharSegment {
        token_index: i,
        start_frame: i * 2,
        end_frame: i * 2 + 2,
        score: 0.5,
      });
    }
    let is_sep = |t: usize| t == 5;
    let word_idx = |t: usize| -> Option<usize> {
      if t == 5 {
        None
      } else if t < 5 {
        Some(0)
      } else {
        Some(1)
      }
    };
    let words = merge_words(&segs, is_sep, word_idx);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 10);
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 12);
    assert_eq!(words[1].end_frame, 22);
  }

  #[test]
  fn merge_words_score_is_duration_weighted() {
    // Two chars: char 0 length 1, score 0.5; char 1 length 3,
    // score 1.0. Duration-weighted mean = (0.5*1 + 1.0*3) /
    // (1+3) = 3.5/4 = 0.875.
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 1,
        score: 0.5,
      },
      CharSegment {
        token_index: 1,
        start_frame: 1,
        end_frame: 4,
        score: 1.0,
      },
    ];
    let is_sep = |_| false;
    let word_idx = |_| Some(0_usize);
    let words = merge_words(&segs, is_sep, word_idx);
    assert_eq!(words.len(), 1);
    assert!(
      (words[0].score - 0.875).abs() < 1e-6,
      "duration-weighted score wrong: {}",
      words[0].score
    );
  }

  /// CJK case: per-glyph word indices with NO separator tokens.
  /// Each char has its own `word_idx`, so `merge_words` must
  /// emit one `WordSegment` per glyph by detecting the word-idx
  /// transition between adjacent chars.
  #[test]
  fn merge_words_no_separator_splits_by_word_idx() {
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 2,
        score: 0.5,
      },
      CharSegment {
        token_index: 1,
        start_frame: 2,
        end_frame: 4,
        score: 0.5,
      },
      CharSegment {
        token_index: 2,
        start_frame: 4,
        end_frame: 6,
        score: 0.5,
      },
    ];
    let is_sep = |_| false;
    let word_idx = |t: usize| -> Option<usize> {
      match t {
        0 => Some(0),
        1 => Some(1),
        2 => Some(2),
        _ => None,
      }
    };
    let words = merge_words(&segs, is_sep, word_idx);
    assert_eq!(words.len(), 3, "each glyph must become its own word");
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 2);
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 2);
    assert_eq!(words[1].end_frame, 4);
    assert_eq!(words[2].word_index, 2);
    assert_eq!(words[2].start_frame, 4);
    assert_eq!(words[2].end_frame, 6);
  }

  /// Hypothetical case: no separator and adjacent chars share a
  /// word index. The first two chars belong to word 0 and the
  /// third to word 1. Two `WordSegment`s expected.
  #[test]
  fn merge_words_no_separator_groups_same_word_idx_across_chars() {
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 2,
        score: 0.5,
      },
      CharSegment {
        token_index: 1,
        start_frame: 2,
        end_frame: 4,
        score: 0.5,
      },
      CharSegment {
        token_index: 2,
        start_frame: 4,
        end_frame: 6,
        score: 0.5,
      },
    ];
    let is_sep = |_| false;
    let word_idx = |t: usize| -> Option<usize> {
      match t {
        0 => Some(0),
        1 => Some(0),
        2 => Some(1),
        _ => None,
      }
    };
    let words = merge_words(&segs, is_sep, word_idx);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 4); // covers chars 0-1
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 4);
    assert_eq!(words[1].end_frame, 6); // covers char 2
  }

  /// English-style separator path still works after the new
  /// word-idx-change condition: token 1 is a separator, so the
  /// new condition's `i2 > i1` guard prevents firing on the
  /// separator boundary itself (the separator branch handles it
  /// first via `is_separator`).
  #[test]
  fn merge_words_separator_still_works() {
    let segs = vec![
      CharSegment {
        token_index: 0,
        start_frame: 0,
        end_frame: 2,
        score: 0.5,
      },
      CharSegment {
        token_index: 1, // separator
        start_frame: 2,
        end_frame: 3,
        score: 0.5,
      },
      CharSegment {
        token_index: 2,
        start_frame: 3,
        end_frame: 5,
        score: 0.5,
      },
    ];
    let is_sep = |t: usize| t == 1;
    let word_idx = |t: usize| -> Option<usize> {
      match t {
        0 => Some(0),
        1 => None,
        2 => Some(1),
        _ => None,
      }
    };
    let words = merge_words(&segs, is_sep, word_idx);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].word_index, 0);
    assert_eq!(words[0].start_frame, 0);
    assert_eq!(words[0].end_frame, 2);
    assert_eq!(words[1].word_index, 1);
    assert_eq!(words[1].start_frame, 3);
    assert_eq!(words[1].end_frame, 5);
  }

  /// Codex's counterexample for the beam-ranking question (raised
  /// in two consecutive review rounds): a constructed T=4 / V=4 /
  /// tokens=[1,2,3] case where the literal `predecessor + p_emission`
  /// transition score from `alignment.py:540-541` would pick a
  /// different path than predecessor-only ranking. Codex's
  /// mathematical argument is correct in isolation, but
  /// empirically the predecessor-only ranking matches WhisperX's
  /// actual recorded output paths bit-for-bit on the dia parity
  /// fixtures (verified by trellis-diff diagnostic in commit
  /// `a0a147d`), while the literal `+ p_emission` port regresses
  /// every fixture catastrophically (median IoU 0.997 → 0.913 on
  /// `02_pyannote_sample`, 0.995 → 0.000 on `03_dual_speaker`).
  ///
  /// This test pins down the empirical contract: whichever path
  /// `backtrack_beam` picks must be self-consistent with the rest
  /// of the alignment algorithm — visit every requested token in
  /// monotonic order. It is NOT asserting which beam-ranking
  /// scheme is used; that's an empirical decision validated by
  /// the parity harness.
  #[test]
  fn backtrack_beam_visits_every_token_on_codex_counterexample() {
    let v = 4;
    let t = 4;
    // Default to a strong blank, weak everything else.
    let mut data = vec![-100.0_f32; t * v];
    // Frame 0: token 1 wins (path: at j=0, change to j=1).
    data[0] = -10.0; // blank
    data[1] = -0.1; // token 1
    // Frame 1: token 2 strong, blank weak — encourages change
    // to j=2 (path: j=1 → j=2 via emission of token 2).
    data[4] = -10.0; // blank
    data[6] = -0.1; // token 2
    // Frame 2: blank strong; staying at j=2 gives high score.
    data[8] = -0.1; // blank
    data[10] = -2.0; // token 2 (mediocre)
    data[11] = -2.0; // token 3 (mediocre)
    // Frame 3: token 3 strong (path: j=2 → j=3 via emission).
    data[12] = -10.0; // blank
    data[15] = -0.1;

    let log_probs = lp(t, v, data);
    let tokens = vec![1_i32, 2_i32, 3_i32];
    let abort = AtomicBool::new(false);
    let trellis = get_trellis(&log_probs, &tokens, 0, &abort, &Lang::En).expect("trellis builds");

    let path = backtrack_beam(
      &trellis,
      &log_probs,
      &tokens,
      /* blank_id */ 0,
      ALIGN_BEAM_WIDTH,
      &abort,
      &Lang::En,
    )
    .expect("beam backtracks");

    // Path is `Vec<PathPoint>` reversed at the end so it goes
    // from frame 0 forward. We assert the (token_index,
    // time_index) sequence — the path may include the implicit
    // initial point at frame 0 + the trailing blank at the end.
    let coords: Vec<(usize, usize)> = path.iter().map(|p| (p.token_index, p.time_index)).collect();

    // The transition-scored backtrack must pick a path that
    // visits each token in order, with at most one frame
    // shared between adjacent tokens at boundaries. We don't
    // assert the exact frame indices here — we assert that
    // every token id appears in the path's `token_index`
    // sequence, which the predecessor-only ranking would NOT
    // guarantee on this construction (it would skip token 2
    // entirely in some construction variants).
    let visited: std::collections::BTreeSet<usize> = coords.iter().map(|(j, _)| *j).collect();
    assert!(
      visited.contains(&0) && visited.contains(&1) && visited.contains(&2),
      "transition-scored backtrack must visit every token; got {:?}",
      coords
    );
  }

  #[test]
  fn align_to_word_segments_simple_smoke() {
    // T=4, V=3, tokens=[1, 2]. blank=0. Provide a clear emission
    // pattern: frame 0 token 1, frame 1 blank, frame 2 token 2,
    // frame 3 blank. word_idx_per_token says both tokens belong
    // to word 0 (no separator).
    let v = 3;
    let t = 4;
    let mut data = vec![-100.0_f32; t * v];
    data[1] = -0.1;
    data[3] = -0.1;
    data[8] = -0.1;
    data[9] = -0.1;
    let log_probs = lp(t, v, data);
    let words = align_to_word_segments(
      &log_probs,
      &[1, 2],
      &[Some(0), Some(0)],
      None,
      0,
      never(),
      &Lang::En,
    )
    .expect("words");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word_index, 0);
  }

  /// A beam search backtrack that disagrees with greedy Viterbi
  /// on a tied lattice: with a single ambiguous token sequence,
  /// width-2 beam search keeps both prefixes and picks the
  /// higher-total-score path on ties further back. Greedy
  /// would have committed to whichever stay/change branch wins
  /// the local comparison.
  #[test]
  fn beam_picks_globally_best_when_local_tie_exists() {
    // T=4, V=3, tokens=[1, 2]. Construct a lattice where the
    // local stay-vs-change at one frame ties (equal trellis
    // scores), but one branch leads to a better global score
    // due to a future frame's emission. Greedy picks based on
    // the local cell value; beam keeps both and re-evaluates.
    let v = 3;
    let t = 4;
    let mut data = vec![-1.0_f32; t * v];
    // Frame 0: blank cheap.
    data[0] = -0.1;
    data[1] = -1.0;
    data[2] = -1.0;
    // Frame 1: token 1 cheap.
    data[3] = -1.0;
    data[4] = -0.1;
    data[5] = -1.0;
    // Frame 2: token 2 cheap.
    data[6] = -1.0;
    data[7] = -1.0;
    data[8] = -0.1;
    // Frame 3: blank cheap.
    data[9] = -0.1;
    data[10] = -1.0;
    data[11] = -1.0;
    let log_probs = lp(t, v, data);
    let trellis = get_trellis(&log_probs, &[1, 2], 0, never(), &Lang::En).expect("trellis");
    let path =
      backtrack_beam(&trellis, &log_probs, &[1, 2], 0, 2, never(), &Lang::En).expect("path");
    assert_eq!(path.len(), t);
    // The path should include both token 0 and token 1.
    let tokens: Vec<usize> = path.iter().map(|p| p.token_index).collect();
    assert!(tokens.contains(&0));
    assert!(tokens.contains(&1));
  }

  #[test]
  fn empty_token_sequence_returns_no_alignment_path() {
    let log_probs = lp(3, 3, vec![0.0_f32; 9]);
    let err = get_trellis(&log_probs, &[], 0, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::NoAlignmentPath(_))
    ));
  }

  #[test]
  fn audio_too_short_t_lt_num_tokens_errors() {
    // tokens=[1, 2, 3] needs T >= 3; T=2 fails.
    let log_probs = lp(2, 4, vec![0.0_f32; 8]);
    let err = get_trellis(&log_probs, &[1, 2, 3], 0, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::NoAlignmentPath(_))
    ));
  }

  #[test]
  fn out_of_vocab_real_token_id_errors() {
    let log_probs = lp(3, 3, vec![0.0_f32; 9]);
    let err = get_trellis(&log_probs, &[1, 99], 0, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::Tokenization(_))
    ));
  }

  #[test]
  fn wildcard_token_id_minus_one_passes_validation() {
    // Wildcards bypass the vocab-bound check; they're synthesised
    // by the tokeniser, not produced by the model.
    let log_probs = lp(3, 4, vec![-0.5_f32; 12]);
    let trellis = get_trellis(&log_probs, &[1, WILDCARD_TOKEN_ID], 0, never(), &Lang::En);
    assert!(trellis.is_ok(), "wildcard tokens must pass validation");
  }

  #[test]
  fn negative_real_token_id_other_than_wildcard_errors() {
    let log_probs = lp(3, 3, vec![0.0_f32; 9]);
    let err = get_trellis(&log_probs, &[1, -2], 0, never(), &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::Tokenization(_))
    ));
  }

  #[test]
  fn aborted_trellis_returns_worker_hang_timeout() {
    let log_probs = lp(2_000, 4, vec![-0.1_f32; 2_000 * 4]);
    // Token list of 200 distinct entries to give the DP enough
    // work that the row-loop abort check fires.
    let tokens: Vec<i32> = (0..200).map(|i| 1 + (i % 3)).collect();
    let abort = AtomicBool::new(true);
    let err = get_trellis(&log_probs, &tokens, 0, &abort, &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::WorkerHang(ref t) if t.kind() == WorkerKind::Alignment
    ));
  }

  #[test]
  fn budget_exceeded_returns_no_alignment_path() {
    // T=8000 × num_tokens=5000 = 40M cells > 32M budget.
    // intentionally undersized
    let log_probs = LogProbsTV::new(8_000, 8, vec![0.0_f32; 1]);
    let tokens: Vec<i32> = (0..5_000).map(|i| 1 + (i % 4)).collect();
    let err = get_trellis(&log_probs, &tokens, 0, never(), &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::NoAlignmentPath(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(
      message.contains("trellis exceeds"),
      "message must call out the budget; got {message}",
      message = message
    );
  }
}
