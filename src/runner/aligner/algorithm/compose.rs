//! Per-word state + surface-form recovery stages of the
//! alignment algorithm.

use alloc::{borrow::Cow, vec::Vec};
use core::{num::NonZeroU32, time::Duration};

use mediatime::{TimeRange, Timebase};
use smol_str::SmolStr;

use crate::{
  core::AlignmentResult,
  runner::aligner::algorithm::{encode::LogProbsTV, viterbi::ViterbiPath},
  time::SAMPLE_RATE_HZ,
  types::Word,
};

/// Per-word accumulator (M4 sparse vector).
#[derive(Clone, Copy)]
struct WordAccum {
  /// First *speech-supported* emission frame index. Range
  /// `[start_frame, end_frame)` therefore bookends speech
  /// support, not all token emissions.
  start_frame: u32,
  /// One past the last speech-supported emission frame.
  end_frame: u32,
  /// Log-probability sum over speech-supported emissions only.
  logprob_sum: f32,
  /// Count of speech-supported emissions. Score numerator's
  /// denominator (mean log-prob → exp() → score).
  speech_emissions: u32,
  /// Count of *all* token emissions for this word — silent and
  /// speech-supported. Denominator for the post-pass coverage
  /// ratio. Required to detect "fragmented speech support" (most
  /// of a word masked, one stray frame surviving) so we don't
  /// emit a high-confidence word over a bounding range that
  /// misrepresents the audio.
  total_emissions: u32,
}

/// Default minimum `speech_emissions / total_emissions` ratio
/// for [`Aligner::min_speech_coverage`](crate::Aligner::min_speech_coverage).
/// Half-coverage is the natural threshold — majority-speech
/// words stay; mostly-masked words drop.
pub const DEFAULT_MIN_SPEECH_COVERAGE: f32 = 0.5;

/// Default maximum contiguous silent run inside a word's
/// `[start_frame, end_frame)` span for
/// [`Aligner::max_intra_silent_run`](crate::Aligner::max_intra_silent_run).
/// 80 ms tolerates most unvoiced consonants (the closure of
/// `/t/`, `/k/`, `/p/` is typically 30–80 ms), glottal stops,
/// and VAD jitter (1–2 frames) while rejecting longer gaps
/// where a word's emissions straddle silence — usually a CTC
/// alignment artifact, not real speech.
///
/// At wav2vec2-base-960h's frame rate (`hop_samples=320` at
/// 16 kHz → 50 fps), this resolves to 4 frames. Models with a
/// different stride convert via the same `Duration` and
/// auto-correct.
pub const DEFAULT_MAX_INTRA_SILENT_RUN: Duration = Duration::from_millis(80);

/// Build a per-frame speech mask of length `n_frames`, marking
/// `true` exactly for frames whose audio sample range overlaps any
/// of the supplied chunk-local sub-segments. Used by
/// [`compose_words`] to drop CTC-forced word assignments that fall
/// entirely inside silence-masked audio.
///
/// Frame `t` represents samples `[t * hop_samples, (t + 1) *
/// hop_samples)` (an approximation of wav2vec2's effective stride);
/// any overlap with a sub-segment marks the frame as speech. The
/// silence_mask step has already zeroed those samples for non-speech,
/// so this mirrors the same boundary the audio carries.
///
/// `sub_segments` must be in chunk-local sample-index space — the
/// caller (alignment worker) wraps the segment range PTS in a
/// 1/16000 timebase so `start_pts` == `start_sample`.
pub(crate) fn build_speech_frames(
  n_frames: usize,
  hop_samples: u32,
  sub_segments: &[mediatime::TimeRange],
) -> alloc::vec::Vec<bool> {
  if hop_samples == 0 {
    return alloc::vec![false; n_frames];
  }
  let hop = hop_samples as i64;
  // A frame is marked "speech" only if at least half its
  // `hop_samples` are inside some VAD sub-segment. Pre-fix any
  // overlap, even 1 sample, promoted the whole frame — a tiny
  // VAD island inside an otherwise-silent frame let the
  // post-pass keep CTC-forced words whose ranges covered
  // mostly zero-masked audio. ≥50 % is the natural threshold
  // — frames whose majority of samples are silence don't
  // qualify; frames whose majority is speech do.
  let min_overlap_samples = hop / 2;
  let mut overlap_per_frame = alloc::vec![0_i64; n_frames];
  for seg in sub_segments {
    let seg_start = seg.start_pts().max(0);
    let seg_end = seg.end_pts().max(0);
    if seg_end <= seg_start {
      continue;
    }
    // Iterate every frame that touches the segment and
    // accumulate the per-frame overlap. Adjacent VAD segments
    // cumulatively contribute to the same frame, which matches
    // the spirit of the old "any overlap" rule for cases where
    // VAD splits a single voiced span across two segments.
    let frame_start = (seg_start / hop) as usize;
    let frame_end = ((seg_end + hop - 1) / hop) as usize;
    let upper = frame_end.min(n_frames);
    if frame_start >= upper {
      continue;
    }
    for f in frame_start..upper {
      let frame_lo = (f as i64) * hop;
      let frame_hi = frame_lo + hop;
      let overlap = seg_end.min(frame_hi) - seg_start.max(frame_lo);
      if overlap > 0 {
        overlap_per_frame[f] = overlap_per_frame[f].saturating_add(overlap);
      }
    }
  }
  overlap_per_frame
    .into_iter()
    .map(|o| o >= min_overlap_samples)
    .collect()
}

/// Walk the Viterbi path and accumulate per-word `(start_frame,
/// end_frame, logprob_sum, speech_emissions, total_emissions)`
/// into a `Vec<Option<...>>` indexed by normalised-word position.
///
/// Step 7:
/// - Skip frames whose state is a blank (`state % 2 == 0`).
/// - Skip frames whose mapped token's `word_idx_per_token == None`
///   (delimiters / `<unk>` / specials).
/// - **Skip frames over silence-masked audio** (`!speech_frames[t]`).
///   The CTC lattice is forced to visit every non-blank state to
///   reach the end, so words sitting entirely inside masked
///   silence would otherwise still get fabricated frame ranges
///   from whichever frames the path consumed them at. Filtering
///   to speech-supported frames here is what makes the silence
///   mask actually drop unsupported words from the output.
/// - For non-blank, mapped, speech-supported frames: open the
///   entry on first sight, extend `end_frame`, accumulate logprob.
///
/// Words that received no speech-supported emitting frames stay
/// `None`. They are dropped by `compose_words` (step 8/9), not
/// added to `Word`s.
fn accumulate_per_word(
  path: &ViterbiPath,
  log_probs: &LogProbsTV,
  word_idx_per_token: &[Option<usize>],
  n_words: usize,
  speech_frames: &[bool],
  min_speech_coverage: f32,
  max_silent_run_frames: usize,
) -> Vec<Option<WordAccum>> {
  let mut per_word: Vec<Option<WordAccum>> = alloc::vec![None; n_words];

  for (t_idx, &state) in path.state_per_frame.iter().enumerate() {
    if state % 2 == 0 {
      continue; // blank
    }
    let token_idx = state / 2;
    let Some(word_idx) = word_idx_per_token.get(token_idx).copied().flatten() else {
      continue; // delimiter / special; skip
    };
    let Some(slot) = per_word.get_mut(word_idx) else {
      // word_idx out of range — caller / tokeniser bug. Skip
      // rather than panic.
      continue;
    };

    let is_speech = speech_frames.get(t_idx).copied().unwrap_or(true);

    // Open the slot regardless of speech support so silent
    // emissions count toward `total_emissions` (the coverage
    // denominator). The bounding `[start_frame, end_frame)` is
    // still anchored to speech-supported frames — only those
    // contribute to `start_frame`/`end_frame`/`logprob_sum`/
    // `speech_emissions`.
    let entry = slot.get_or_insert_with(|| WordAccum {
      start_frame: 0,
      end_frame: 0,
      logprob_sum: 0.0,
      speech_emissions: 0,
      total_emissions: 0,
    });
    entry.total_emissions += 1;
    if is_speech {
      let token_id = path.tokens[token_idx];
      let lp = log_probs.at(t_idx, token_id as usize);
      if entry.speech_emissions == 0 {
        entry.start_frame = t_idx as u32;
      }
      entry.end_frame = (t_idx + 1) as u32;
      entry.logprob_sum += lp;
      entry.speech_emissions += 1;
    }
  }

  // Post-pass: drop fragmented and long-gap words.
  //
  // History:
  // - An earlier version dropped any word whose bounding span
  //   covered a silent frame. Too aggressive — a 1-frame VAD
  //   false-negative inside a real word killed the word.
  // - Removing the post-pass entirely was too lax — a word with
  //   most emissions masked and one surviving speech frame
  //   still emitted with a high-confidence score over a
  //   misleading bounding range.
  // - Current rule: drop on either signal — speech coverage
  //   below `min_speech_coverage` ("fragmented") *or* longest
  //   contiguous silent run inside the bounding span exceeds
  //   `max_silent_run_frames` ("long straddle"). Both
  //   thresholds are configurable on `Aligner` (defaults in
  //   `DEFAULT_MIN_SPEECH_COVERAGE` /
  //   `DEFAULT_MAX_INTRA_SILENT_RUN`); brief intra-word
  //   silences (unvoiced consonants, glottal stops, VAD jitter
  //   under the configured threshold) still keep the word.
  for slot in per_word.iter_mut() {
    let Some(accum) = slot else { continue };
    if accum.speech_emissions == 0 {
      *slot = None;
      continue;
    }
    let coverage = accum.speech_emissions as f32 / accum.total_emissions as f32;
    if coverage < min_speech_coverage {
      *slot = None;
      continue;
    }
    let start = accum.start_frame as usize;
    let end = accum.end_frame as usize;
    if let Some(slice) = speech_frames.get(start..end) {
      let mut max_run: usize = 0;
      let mut current: usize = 0;
      for &b in slice {
        if !b {
          current += 1;
          if current > max_run {
            max_run = current;
          }
        } else {
          current = 0;
        }
      }
      if max_run > max_silent_run_frames {
        *slot = None;
      }
    }
  }

  per_word
}

/// Compose the final `AlignmentResult` from per-word accumulators
/// and original-word surface forms.
///
/// `speech_frames` is a length-`T` vector marking which encoder
/// output frames overlap real speech (true) versus silence-masked
/// audio (false). Words whose entire CTC-assigned span sits in
/// silence drop from the output.
///
/// Step 8/9: for each `(i, slot)`:
/// - `Some` => build `Word { text: original_words[i].into(), range:
///   frames_to_output_range(start_frame, end_frame), score:
///   exp(logprob_sum / speech_emissions) }`.
/// - `None` => skip; the word had no speech-supported audio
///   (typically silence-masked or all-`<unk>`). It is *not* added
///   to `words`. The total chunk text on `Transcript.text` still
///   contains the word.
pub(crate) fn compose_words<F>(
  path: &ViterbiPath,
  log_probs: &LogProbsTV,
  word_idx_per_token: &[Option<usize>],
  original_words: &[Cow<'_, str>],
  speech_frames: &[bool],
  chunk_first_sample_in_stream: u64,
  hop_samples: u32,
  samples_to_output_range: F,
  min_speech_coverage: f32,
  max_intra_silent_run: Duration,
) -> AlignmentResult
where
  F: Fn(u64, u64) -> TimeRange,
{
  // Convert the wall-clock silent-run threshold into encoder
  // frames using the model's frame timebase (`hop_samples` per
  // 16 kHz analysis sample → seconds per frame). Done once per
  // alignment so `accumulate_per_word` can compare directly
  // against frame indices.
  let frame_tb = Timebase::new(hop_samples, NonZeroU32::new(SAMPLE_RATE_HZ).unwrap());
  let max_silent_run_frames = frame_tb.duration_to_pts(max_intra_silent_run) as usize;

  let n_words = original_words.len();
  let per_word = accumulate_per_word(
    path,
    log_probs,
    word_idx_per_token,
    n_words,
    speech_frames,
    min_speech_coverage,
    max_silent_run_frames,
  );

  let mut words: Vec<Word> = Vec::with_capacity(n_words);
  for (i, slot) in per_word.iter().enumerate() {
    let Some(accum) = slot else {
      continue;
    };
    let start_sample =
      chunk_first_sample_in_stream + (accum.start_frame as u64) * (hop_samples as u64);
    let end_sample = chunk_first_sample_in_stream + (accum.end_frame as u64) * (hop_samples as u64);
    let range = samples_to_output_range(start_sample, end_sample);

    let mean_lp = accum.logprob_sum / (accum.speech_emissions.max(1) as f32);
    let score = mean_lp.exp().clamp(0.0, 1.0);

    words.push(Word::new(SmolStr::new(&original_words[i]), range, score));
  }

  AlignmentResult::new(words)
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::num::NonZeroU32;
  use mediatime::Timebase;

  fn tb_ms() -> Timebase {
    Timebase::new(1, NonZeroU32::new(1000).unwrap())
  }

  fn lp_const(t: usize, v: usize, value: f32) -> LogProbsTV {
    LogProbsTV {
      t,
      v,
      data: alloc::vec![value; t * v],
    }
  }

  fn fake_samples_to_output_range(start: u64, end: u64) -> TimeRange {
    TimeRange::new(start as i64, end as i64, tb_ms())
  }

  #[test]
  fn missing_word_remains_none_and_drops_from_output() {
    // 2 words; only word 0 has emitting frames.
    let path = ViterbiPath {
      // states: [blank, y_0, blank, blank, blank, blank]
      state_per_frame: alloc::vec![0, 1, 2, 2, 2, 2],
      tokens: alloc::vec![10, 20], // token 0 = id 10 (word 0), token 1 = id 20 (word 1)
    };
    let log_probs = lp_const(6, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let words = result.words();
    assert_eq!(words.len(), 1, "silence-masked word must drop");
    assert_eq!(words[0].text(), "hello");
  }

  #[test]
  fn delimiter_token_is_skipped() {
    // 2 words separated by a delimiter token.
    // Tokens: [hello-token=10, delim=99, world-token=20]
    // word_idx_per_token: [Some(0), None, Some(1)]
    // n_states = 7: blank, 10, blank, 99, blank, 20, blank.
    let path = ViterbiPath {
      // visit each non-blank state once: states 1, 3, 5
      state_per_frame: alloc::vec![0, 1, 2, 3, 4, 5, 6],
      tokens: alloc::vec![10, 99, 20],
    };
    let log_probs = lp_const(7, 100, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), None, Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let words = result.words();
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text(), "hello");
    assert_eq!(words[1].text(), "world");
    // Delimiter at state 3 (token idx 1) carried no per-word
    // index; it was skipped, not added.
  }

  #[test]
  fn surface_form_preserved_not_normalized() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 1, 2],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, -0.5);
    let word_idx_per_token = alloc::vec![Some(0)];
    // Original surface form has casing + punctuation.
    let original = alloc::vec![Cow::Borrowed("Hello!")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words()[0].text(), "Hello!");
  }

  #[test]
  fn frame_to_output_range_uses_chunk_first_sample_offset() {
    // Confirm that chunk_first_sample_in_stream offsets the
    // output range. With chunk_first_sample = 8000 and
    // hop_samples = 320, frame 1 maps to sample 8320, frame 2
    // to sample 8640.
    let path = ViterbiPath {
      // states: [blank, y_0, y_0]; emit at frames 1, 2.
      state_per_frame: alloc::vec![0, 1, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, -0.5);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hi")];

    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      8_000,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let r = result.words()[0].range();
    // start_frame = 1 -> 8000 + 320 = 8320
    // end_frame = 3 -> 8000 + 960 = 8960
    assert_eq!(r.start_pts(), 8320);
    assert_eq!(r.end_pts(), 8960);
  }

  #[test]
  fn all_silence_frames_drop_every_word() {
    // The CTC lattice forces a successful path to visit every
    // non-blank state, so even a real Viterbi run would assign
    // every word to *some* frame. With every frame marked
    // non-speech, those force-emitted assignments must drop —
    // otherwise zero-masking silence would still produce
    // fabricated word timings.
    let path = ViterbiPath {
      // states: [blank, y_0, blank, y_1, blank]
      state_per_frame: alloc::vec![0, 1, 2, 3, 4],
      tokens: alloc::vec![10, 20],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];
    let speech_frames = alloc::vec![false; log_probs.t]; // all silence

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      result.words().is_empty(),
      "no words may emit when every frame is silence-masked; got {:?}",
      result.words()
    );
  }

  #[test]
  fn partial_silence_drops_only_the_silent_word() {
    // Word 0 is assigned frame 1 (speech), word 1 is assigned
    // frame 3 (silence). Only word 0 must emit.
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 1, 2, 3, 4],
      tokens: alloc::vec![10, 20],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];
    let speech_frames = alloc::vec![false, true, false, false, false];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    assert_eq!(result.words()[0].text(), "hello");
  }

  /// A word with speech-supported emissions on both sides of a
  /// brief silent gap must be kept. The accumulator's per-frame
  /// skip already anchors `start_frame`/`end_frame` to
  /// speech-supported frames; the emitted range bookends those
  /// frames even when a single silent frame falls inside.
  ///
  /// Path: word 0's token (state 1) emits at frames 0, 2, 4,
  /// with frame 2 masked silent (frames 1 and 3 are blank). An
  /// over-aggressive earlier rule dropped this word entirely;
  /// the current rule emits it with range [0, 5) — accumulator's
  /// first/last speech-supported frame indices.
  #[test]
  fn word_with_brief_silent_gap_is_kept_with_speech_supported_span() {
    // states: [y_0, blank, y_0, blank, y_0]
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 1, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hello")];
    // Speech at 0,1,3,4; silence at 2 (inside the word's span).
    let speech_frames = alloc::vec![true, true, false, true, true];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(
      result.words().len(),
      1,
      "speech-supported word with intra-silent gap must NOT drop; got {:?}",
      result.words()
    );
    let r = result.words()[0].range();
    // start_frame = 0, end_frame = 5 → [0*320, 5*320) = [0, 1600).
    assert_eq!(
      r.start_pts(),
      0,
      "range starts at first speech-supported frame"
    );
    assert_eq!(
      r.end_pts(),
      1_600,
      "range ends one past last speech-supported frame"
    );
  }

  /// A word whose token visits the path 5 times but only one
  /// frame is speech-supported (4 silent emissions skipped) must
  /// drop. Coverage = 1 / 5 = 0.20, below
  /// `MIN_SPEECH_COVERAGE = 0.5`. A previous implementation kept
  /// it — emitting a high-confidence word scored from the single
  /// surviving emission while ignoring the 4 masked-out token
  /// frames.
  #[test]
  fn fragmented_word_with_minority_speech_support_drops() {
    // 5 frames; word 0's token (state 1) emits at every frame.
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 1, 1, 1, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("missed")];
    // Only frame 0 is speech-supported; frames 1-4 are masked.
    let speech_frames = alloc::vec![true, false, false, false, false];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      result.words().is_empty(),
      "fragmented word (1/5 coverage) must drop; got {:?}",
      result.words()
    );
  }

  /// A word with speech emissions on both sides of a long
  /// silent gap must drop — the bounding
  /// `Word.range` would otherwise misrepresent where the word
  /// actually occurred. The 19-frame masked gap inside the
  /// span (~380 ms at 50 fps) blows past the default 80 ms
  /// (`DEFAULT_MAX_INTRA_SILENT_RUN`). Coverage alone passes
  /// (2 of 2 emissions are speech-supported), so the gap check
  /// is what catches this.
  #[test]
  fn word_spanning_long_silent_gap_drops() {
    // 21 frames; state 1 (token 0, word 0) emits at frame 0 and
    // frame 20 only; everything else is blank.
    let mut state_per_frame = alloc::vec![0_usize; 21];
    state_per_frame[0] = 1;
    state_per_frame[20] = 1;
    let path = ViterbiPath {
      state_per_frame,
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(21, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("split")];
    // Speech only at the two emission frames; everything else
    // is masked silence (a 19-frame masked run inside the span).
    let mut speech_frames = alloc::vec![false; 21];
    speech_frames[0] = true;
    speech_frames[20] = true;

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      result.words().is_empty(),
      "word with 19-frame intra-silent run must drop; got {:?}",
      result.words()
    );
  }

  /// Boundary check on the long-gap rule with the default
  /// 80 ms tolerance: a word with exactly 4 masked frames
  /// (4 × 20 ms = 80 ms at wav2vec2-base's 50 fps) inside its
  /// bounding span survives. The threshold is strict (`>`), so
  /// the run-at-threshold case is kept; one more frame would
  /// drop. Coverage 2/2 = 1.0 also passes.
  #[test]
  fn word_with_silent_run_at_threshold_is_kept() {
    // 6 frames; state 1 emits at frame 0 and frame 5, blank
    // elsewhere. The 4 silent middle frames sit at the
    // tolerance limit (80 ms / 20 ms per frame = 4 frames).
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 0, 0, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(6, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("ok")];
    let speech_frames = alloc::vec![true, false, false, false, false, true];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(
      result.words().len(),
      1,
      "4-frame intra-silent run is at the 80 ms threshold and must be kept; got {:?}",
      result.words()
    );
    let r = result.words()[0].range();
    assert_eq!(r.start_pts(), 0);
    assert_eq!(r.end_pts(), 6 * 320);
  }

  /// Configurable threshold: overriding `max_intra_silent_run`
  /// to 100 ms (5 frames at 50 fps) lets a 5-frame silent run
  /// through that the default 80 ms would drop. Verifies the
  /// new `Aligner::with_max_intra_silent_run` plumbing reaches
  /// the post-pass.
  #[test]
  fn longer_max_intra_silent_run_keeps_word_default_would_drop() {
    // 7 frames; speech at 0 and 6, masked silence between.
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 0, 0, 0, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(7, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("ok")];
    let speech_frames = alloc::vec![true, false, false, false, false, false, true];

    // With the default 80 ms (= 4 frames), 5-frame run drops.
    let default_result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      default_result.words().is_empty(),
      "default 80 ms threshold must drop a 5-frame (100 ms) silent run"
    );

    // Bumping the threshold to 100 ms lets the same word through.
    let permissive_result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      Duration::from_millis(100),
    );
    assert_eq!(
      permissive_result.words().len(),
      1,
      "100 ms threshold must keep a 5-frame silent run; got {:?}",
      permissive_result.words()
    );
  }

  /// Configurable coverage: bumping `min_speech_coverage` to
  /// 0.9 drops a word whose 4-of-5 emissions are speech-
  /// supported (coverage 0.8). The default 0.5 keeps it.
  #[test]
  fn stricter_min_speech_coverage_drops_word_default_would_keep() {
    // 5 frames; word 0's token (state 1) emits at every frame.
    // 4 of 5 are speech-supported.
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 1, 1, 1, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("ok")];
    // 4/5 speech-supported = coverage 0.8.
    let speech_frames = alloc::vec![true, true, false, true, true];

    // Default 0.5 keeps it.
    let default_result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(
      default_result.words().len(),
      1,
      "default 0.5 coverage must keep an 0.8-coverage word; got {:?}",
      default_result.words()
    );

    // Strict 0.9 drops it.
    let strict_result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      0.9,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      strict_result.words().is_empty(),
      "strict 0.9 coverage must drop an 0.8-coverage word; got {:?}",
      strict_result.words()
    );
  }

  /// A long word with brief intra-silence (1-frame VAD jitter)
  /// must be kept. Common in real audio when VAD misses a brief
  /// unvoiced consonant inside a word. Earlier implementations
  /// dropped this case — losing real speech.
  #[test]
  fn long_word_with_one_frame_silent_gap_is_kept() {
    // 11 frames; word 0's token emits at every odd frame
    // (0, 2, 4, 6, 8, 10). Silence at frame 6 only (one of the
    // emission frames mid-word).
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(11, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("alignment")];
    let speech_frames = alloc::vec![
      true, true, true, true, true, true, false, true, true, true, true
    ];

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(
      result.words().len(),
      1,
      "long word with brief intra-silent gap must NOT drop; got {:?}",
      result.words()
    );
    let w = &result.words()[0];
    assert_eq!(w.text(), "alignment");
    // start_frame = 0 (first speech-supported emission),
    // end_frame = 11 (one past frame 10's emission).
    assert_eq!(w.range().start_pts(), 0);
    assert_eq!(w.range().end_pts(), 11 * 320);
  }

  /// Companion: same path, no silence in the middle. Word
  /// emits normally with span [0, 5).
  #[test]
  fn word_with_only_blanks_in_span_emits_normally() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 0, 1, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(5, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hello")];
    let speech_frames = alloc::vec![true; 5]; // no silence

    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
  }

  #[test]
  fn build_speech_frames_marks_overlapping_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Sub-segment from sample 320 to 960 (frames 1..3 at
    // hop_samples = 320). Frames 0 and 3+ are silence.
    let segs = alloc::vec![TimeRange::new(320, 960, tb_16k)];
    let mask = build_speech_frames(/* n_frames: */ 5, /* hop_samples: */ 320, &segs);
    assert_eq!(mask, alloc::vec![false, true, true, false, false]);
  }

  #[test]
  fn build_speech_frames_handles_no_segments() {
    let mask = build_speech_frames(4, 320, &[]);
    assert_eq!(mask, alloc::vec![false; 4]);
  }

  /// A 1-sample VAD island inside an otherwise-silent frame
  /// must NOT promote the whole frame to speech. Pre-fix any
  /// overlap was sufficient, so a ≤ 1/320 sliver could let the
  /// silence-aware post-pass keep CTC-forced words whose
  /// ranges covered mostly zero-masked audio. Threshold:
  /// ≥ 50 % sample overlap (= 160 samples at hop=320).
  #[test]
  fn build_speech_frames_rejects_sub_threshold_vad_island() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // A 1-sample island at sample 10 inside frame 0.
    let segs = alloc::vec![TimeRange::new(10, 11, tb_16k)];
    let mask = build_speech_frames(5, 320, &segs);
    assert_eq!(mask, alloc::vec![false; 5]);
  }

  /// Boundary check: a VAD segment covering exactly half a
  /// frame (`hop_samples / 2 = 160` samples) is at the
  /// threshold and counts as speech (`>=`). One sample less
  /// drops the frame.
  #[test]
  fn build_speech_frames_threshold_is_inclusive() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Exactly 160 samples in frame 0.
    let segs_at = alloc::vec![TimeRange::new(0, 160, tb_16k)];
    assert_eq!(
      build_speech_frames(2, 320, &segs_at),
      alloc::vec![true, false],
      "exactly 50% overlap is at threshold and must count"
    );
    // 159 samples in frame 0 — just under threshold.
    let segs_under = alloc::vec![TimeRange::new(0, 159, tb_16k)];
    assert_eq!(
      build_speech_frames(2, 320, &segs_under),
      alloc::vec![false, false],
      "1 sample under threshold must drop the frame"
    );
  }

  /// Adjacent VAD segments contribute cumulatively to the same
  /// frame's overlap. Splits in the VAD output (e.g. a
  /// breath-between-words gap that VAD reports as two
  /// 80-sample segments) shouldn't lose the frame just because
  /// each segment alone is below the threshold.
  #[test]
  fn build_speech_frames_accumulates_overlap_across_adjacent_segments() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // Two 80-sample segments inside frame 0 — together = 160.
    let segs = alloc::vec![
      TimeRange::new(0, 80, tb_16k),
      TimeRange::new(160, 240, tb_16k),
    ];
    assert_eq!(
      build_speech_frames(2, 320, &segs),
      alloc::vec![true, false],
      "two sub-threshold segments inside one frame must accumulate to clear threshold"
    );
  }

  #[test]
  fn score_in_unit_interval() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 1, 2],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, 0.0); // logprob 0.0 => score = exp(0) = 1.0
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hi")];
    let speech_frames = alloc::vec![true; log_probs.t];
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let s = result.words()[0].score();
    assert!((0.0..=1.0).contains(&s));
  }
}
