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
  //
  // Use ceil-half so odd custom strides still need a strict
  // majority of samples (`hop=3` → 2 samples, not 1) and
  // clamp the threshold to ≥1 so `hop=1` doesn't trivialise
  // to "any-overlap-counts" — without the clamp, an empty
  // sub_segments list would pass the `>= 0` check for every
  // frame and mark the whole chunk as speech.
  let min_overlap_samples = ((hop + 1) / 2).max(1);
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
/// - Skip frames whose state is a blank (`state % 2 == 0`) for
///   the *emission* accumulators (logprob, counts), but use them
///   to extend the LEFT-adjacent token's word `end_frame` — see
///   below.
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
/// Blank-stay attribution (matches WhisperX's `merge_repeats` /
/// `merge_words` semantics):
///
/// In WhisperX, the CTC path's `token_index` indexes into the
/// cleaned character string `text_clean` (which includes `|`
/// word separators). When the path "stays" at index `j`, the
/// frame emits a blank but stays at character `j`. `merge_repeats`
/// groups those frames into a `Segment` for the j-th character;
/// `merge_words` then drops `|` segments and uses the last
/// non-separator character's segment end as the word's end. So
/// blank-stays attribute to **the character immediately to their
/// left** in the path's traversal order — never carrying through
/// a separator into the previous word.
///
/// Whispery's state machine encodes the same thing differently:
/// - State `2*i + 1` = emit token `i`.
/// - State `2*k`     = blank slot.
///   - `k = 0`: leading blank (before any token has emitted).
///   - `k ≥ 1`: blank slot between token `k-1` and token `k` —
///     i.e. WhisperX's stays at `token_index = k-1`.
///
/// So a blank frame at state `2*k` (`k ≥ 1`) attributes to
/// **token `k-1`**. If that token's `word_idx_per_token` is a
/// real word, the blank extends that word's `end_frame`. If
/// it's a delimiter / unmapped (e.g. WhisperX's `|`), the blank
/// attributes to the delimiter slot and DOES NOT extend any
/// word — exactly matching `merge_words`'s drop-`|` rule.
///
/// Blank-stay rules:
/// - Blank-stays *only* extend `end_frame`; they do **not**
///   contribute to `logprob_sum`, `speech_emissions`, or
///   `total_emissions`. Coverage and score remain anchored to
///   real emissions; otherwise the post-pass coverage semantics
///   would silently change.
/// - Blank-stays on non-speech frames are skipped; the speech-
///   frame mask still gates everything.
/// - Blank state `2*0` (`k = 0`, leading blank) belongs to no
///   token; skip.
/// - Blank state `2*k` (`k ≥ 1`) where token `k-1` maps to a
///   delimiter (`word_idx_per_token == None`) belongs to that
///   delimiter slot, not to any word; skip.
/// - Extension only happens if the corresponding word slot has
///   already been opened — i.e. the previous token emitted at
///   least once. Otherwise a blank in state `2*k` for which
///   token `k-1` has not yet emitted (e.g. fully masked) would
///   open and extend a phantom range. Aligns with WhisperX:
///   `merge_repeats` only includes frames where the path
///   actually visited that token_index.
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
    let is_speech = speech_frames.get(t_idx).copied().unwrap_or(true);
    let is_blank = state % 2 == 0;

    if is_blank {
      // Blank-stay attribution: state `2*k` (`k ≥ 1`) is the
      // blank slot AFTER token `k-1`; attribute to token `k-1`.
      // State `2*0` is the leading blank (before any token has
      // emitted) — belongs to no word.
      if !is_speech {
        continue;
      }
      let k = state / 2;
      if k == 0 {
        continue; // leading blank
      }
      let prev_token_idx = k - 1;
      let Some(word_idx) = word_idx_per_token
        .get(prev_token_idx)
        .copied()
        .flatten()
      else {
        // Previous token is a delimiter / unmapped; its
        // blank-stay slot belongs to that delimiter, not to
        // any word. Skip — same rule WhisperX's `merge_words`
        // applies to `|` segments.
        continue;
      };
      let Some(slot) = per_word.get_mut(word_idx) else {
        continue;
      };
      // Only extend if the word's slot is already open — i.e.
      // token `prev_token_idx` actually emitted. Otherwise a
      // word that was never visited (every emission masked or
      // skipped upstream) would get a phantom range from
      // adjacent blank-stays.
      if let Some(entry) = slot {
        entry.end_frame = (t_idx + 1) as u32;
      }
      continue;
    }

    // Non-blank frame.
    let token_idx = state / 2;
    let Some(word_idx) = word_idx_per_token.get(token_idx).copied().flatten() else {
      continue; // delimiter / special
    };
    let Some(slot) = per_word.get_mut(word_idx) else {
      // word_idx out of range — caller / tokeniser bug. Skip
      // rather than panic.
      continue;
    };

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
  // `n_samples` is the chunk's input audio length in 16 kHz
  // samples. Word ranges are clamped to
  // `[chunk_first_sample, chunk_first_sample + n_samples]` so
  // the stride validator's 2-frame overshoot tolerance can't
  // leak into emitted word timestamps. It also drives the
  // effective samples-per-frame ratio (`n_samples / (T-1)`)
  // that matches WhisperX's frame→time math; nominal
  // `hop_samples` alone introduced a ~40 ms drift over 30 s
  // because wav2vec2's CNN truncates one frame at the edge.
  // Tests should pass `log_probs.t * hop_samples` so the
  // effective ratio collapses back to ~`hop_samples`.
  n_samples: u64,
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

  // Clamp ceiling: word ranges must not extend past the
  // chunk's audio. The stride validator allows up to two
  // frames of overshoot for CNN edge effects, which would
  // otherwise leak into emitted word timestamps —
  // user-visible overlap with later audio. `saturating_add`
  // is a safety net for the `u64::MAX` test sentinel.
  let chunk_end_sample = chunk_first_sample_in_stream.saturating_add(n_samples);

  // Effective samples-per-frame from the actual encoder
  // output count, matching WhisperX's
  // `ratio = duration / (T - 1)` in `alignment.py`. Using
  // nominal `hop_samples` (320) introduced a ~40 ms drift over
  // a 30 s clip because wav2vec2's CNN truncates one frame at
  // the edge (n_samples=480 000 → T=1499 not 1500).
  let samples_per_frame = if log_probs.t >= 2 {
    (n_samples as f64) / ((log_probs.t - 1) as f64)
  } else {
    // Single-frame or empty chunk: effective ratio
    // undefined; fall back to nominal hop. Empty cases
    // already short-circuit upstream.
    hop_samples as f64
  };

  let mut words: Vec<Word> = Vec::with_capacity(n_words);
  for (i, slot) in per_word.iter().enumerate() {
    let Some(accum) = slot else {
      continue;
    };
    let start_sample = chunk_first_sample_in_stream
      + (accum.start_frame as f64 * samples_per_frame).round() as u64;
    let end_sample = chunk_first_sample_in_stream
      + (accum.end_frame as f64 * samples_per_frame).round() as u64;

    // If the word's first speech-supported frame is already
    // past the chunk's audio, drop the word — there's no
    // honest range to emit (the model placed every emission
    // in the overshoot region). This is rare but possible
    // when the encoder returns the maximum tolerated
    // overshoot.
    if start_sample >= chunk_end_sample {
      continue;
    }
    // Otherwise clamp the end so no Word range claims audio
    // past the chunk boundary.
    let clamped_end = end_sample.min(chunk_end_sample);
    let range = samples_to_output_range(start_sample, clamped_end);

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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words()[0].text(), "Hello!");
  }

  #[test]
  fn frame_to_output_range_uses_chunk_first_sample_offset() {
    // Confirm that chunk_first_sample_in_stream offsets the
    // output range. With chunk_first_sample = 8000,
    // hop_samples = 320, t = 3 and n_samples = 3*320 = 960,
    // the effective samples-per-frame is 960/(3-1) = 480.
    // Frame 1 maps to sample 8000 + 480 = 8480; frame 3 maps
    // to sample 8000 + 1440 = 9440 (clamped to 8000+960=8960).
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
      log_probs.t as u64 * 320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let r = result.words()[0].range();
    // start_frame = 1 -> 8000 + 480 = 8480
    // end_frame = 3 -> 8000 + 1440 = 9440, clamped to chunk_end = 8960
    assert_eq!(r.start_pts(), 8480);
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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

  /// Clamping word ranges to chunk bounds: when the encoder
  /// returns up to its 2-frame overshoot tolerance, raw
  /// `frame * hop` math would put the last word's end past
  /// the chunk's audio. The clamp at the compose boundary
  /// stops that leaking into emitted timestamps.
  #[test]
  fn word_ranges_are_clamped_to_chunk_bounds() {
    // 4 frames; word emits at every frame including the
    // overshoot frame 3.
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 1, 1, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(4, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("ok")];
    let speech_frames = alloc::vec![true; 4];

    // Chunk's actual audio length is 1000 samples (3 full
    // frames + 40 samples). Frame 3's [3*320, 4*320) =
    // [960, 1280) extends 280 samples past the chunk.
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      /* n_samples: */ 1000,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    let r = result.words()[0].range();
    // start_frame=0 → 0 (well within chunk).
    assert_eq!(r.start_pts(), 0);
    // end_frame=4 → raw 4*320=1280, clamped to chunk_end=1000.
    assert_eq!(
      r.end_pts(),
      1000,
      "end must clamp to chunk bound (1000), not raw 1280; got {}",
      r.end_pts()
    );
  }

  /// If the word's first speech-supported frame is past the
  /// chunk boundary entirely (every emission lands in the
  /// overshoot tail), the word drops — there's no honest
  /// range to report.
  #[test]
  fn word_entirely_in_overshoot_drops() {
    // 4 frames; word emits ONLY at frame 3 (the overshoot).
    let path = ViterbiPath {
      state_per_frame: alloc::vec![0, 0, 0, 1],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(4, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("late")];
    let speech_frames = alloc::vec![true; 4];

    // n_samples=900 — frames 0,1,2 fit ([0,960)≈960 samples)
    // and frame 3 starts at 960 which is past chunk_end=900.
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      /* n_samples: */ 900,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert!(
      result.words().is_empty(),
      "word emitting only in the overshoot tail must drop; got {:?}",
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
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
      log_probs.t as u64 * 320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
  }

  /// Trailing blank-stays at the *trailing-blank state* of a
  /// word's last token must extend the word's `end_frame` to
  /// match WhisperX's `merge_repeats` semantics. State `2*k`
  /// (`k ≥ 1`) is the blank slot between token `k-1` and token
  /// `k`; its frames attribute to token `k-1`.
  ///
  /// Path: state 1 (emit token 0) at frame 0; state 2 (blank
  /// slot AFTER token 0) at frames 1, 2 — all speech-supported.
  /// Pre-fix `end_frame` was 1 (one past the last non-blank
  /// emission); post-fix it is 3 (one past the last
  /// trailing-blank-stay attributable to token 0).
  ///
  /// `n_samples = 1500` keeps `samples_per_frame = 1500 / 2 =
  /// 750`. start_frame = 0 → 0; end_frame = 3 → 3 * 750 = 2250,
  /// clamped to chunk_end = 1500. Pre-fix end_pts would have
  /// read 750.
  ///
  /// Score and coverage are unaffected: the blank-stays do not
  /// contribute to `speech_emissions`, `total_emissions`, or
  /// `logprob_sum` — only to `end_frame`.
  #[test]
  fn word_end_extends_through_trailing_blank_stays_for_held_word() {
    let path = ViterbiPath {
      // Frame 0: emit token 0 (state 1). Frames 1, 2:
      // trailing-blank slot for token 0 (state 2 = 2*1 =
      // blank between token 0 and token 1).
      state_per_frame: alloc::vec![1, 2, 2],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hi")];
    let speech_frames = alloc::vec![true, true, true];

    let n_samples: u64 = 1_500;
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      n_samples,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    let r = result.words()[0].range();
    assert_eq!(r.start_pts(), 0);
    // end_frame = 3, raw end = 3 * 750 = 2250, clamped to 1500.
    assert_eq!(
      r.end_pts(),
      1_500,
      "trailing speech-supported blank-stays at state 2*(i+1) must extend token i's word; got {}",
      r.end_pts()
    );
  }

  /// Companion: trailing **non-speech** blank-stays do NOT
  /// extend the previous token's word `end_frame`. The
  /// speech-frame mask still gates attribution — extending into
  /// masked silence would smear the word's range over
  /// non-speech audio.
  ///
  /// Path: state 1 emit at frame 0, then state-2 blanks at
  /// frames 1, 2 with `speech_frames = [true, false, false]`.
  /// `end_frame` stays at 1 (one past the last non-blank
  /// speech-supported emission).
  #[test]
  fn trailing_non_speech_blank_stays_do_not_extend_word_end() {
    let path = ViterbiPath {
      state_per_frame: alloc::vec![1, 2, 2],
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(3, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("hi")];
    // Only frame 0 is speech-supported; trailing blanks are
    // masked silence.
    let speech_frames = alloc::vec![true, false, false];

    let n_samples: u64 = 1_500;
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      n_samples,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    let r = result.words()[0].range();
    assert_eq!(r.start_pts(), 0);
    // end_frame = 1, raw end = 1 * 750 = 750. If the speech-
    // mask gate were broken, this would read 2250 (clamped to
    // 1500) instead.
    assert_eq!(
      r.end_pts(),
      750,
      "non-speech blank-stays must NOT extend end_frame; got {}",
      r.end_pts()
    );
  }

  /// Blank slots between adjacent tokens attribute to the LEFT
  /// token (the token immediately before the blank slot in the
  /// CTC graph). Blanks at state `2*k` for `k ≥ 1` belong to
  /// token `k - 1`. Crucially, when token `k - 1` is a
  /// **delimiter** (`word_idx_per_token == None`), the blank
  /// belongs to the delimiter slot — not to any word — exactly
  /// matching WhisperX's `merge_words` rule that drops `|`
  /// separator segments.
  ///
  /// Path layout: tokens = [hello-token, delim, world-token]
  /// with `word_idx_per_token = [Some(0), None, Some(1)]`. The
  /// blank slots interleave: state 0 (leading), state 1
  /// (hello), state 2 (post-hello / pre-delim), state 3
  /// (delim), state 4 (post-delim / pre-world), state 5
  /// (world), state 6 (post-world).
  ///
  /// Frames: emit hello at 0; state-2 blank at 1; emit delim
  /// at 2; state-4 blank at 3; emit world at 4; state-6 blank
  /// at 5.
  ///
  /// Expected end_frames:
  /// - hello: 2 (state-2 blank attributes to token 0 = hello).
  /// - world: 6 (state-6 blank attributes to token 2 = world).
  ///
  /// State-4 blank attributes to token 1 (the delimiter); it
  /// must NOT extend hello's range past frame 2. This is the
  /// regression that bit the round-22 first attempt — naively
  /// "carry the previously-emitted real word" would wrongly
  /// extend hello through the inter-word blank, just as
  /// over-extending word ends through the post-`|` blank-stay
  /// region in the parity run.
  #[test]
  fn inter_word_blank_through_delimiter_does_not_extend_previous_word() {
    let path = ViterbiPath {
      // states: emit hello, blank2, emit delim, blank4, emit world, blank6
      state_per_frame: alloc::vec![1, 2, 3, 4, 5, 6],
      tokens: alloc::vec![10, 99, 20],
    };
    let log_probs = lp_const(6, 100, -1.0);
    let word_idx_per_token = alloc::vec![Some(0), None, Some(1)];
    let original = alloc::vec![Cow::Borrowed("hello"), Cow::Borrowed("world")];
    let speech_frames = alloc::vec![true; 6];

    // n_samples = 1500, samples_per_frame = 1500 / 5 = 300.
    let n_samples: u64 = 1_500;
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      n_samples,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 2);
    let w0 = result.words()[0].range();
    let w1 = result.words()[1].range();
    // hello: start_frame 0, end_frame 2 (extended through the
    // post-hello blank at state 2). Raw end = 2 * 300 = 600.
    assert_eq!(w0.start_pts(), 0);
    assert_eq!(
      w0.end_pts(),
      600,
      "hello must NOT extend through the inter-word delimiter blank; got {}",
      w0.end_pts()
    );
    // world: start_frame 4, end_frame 6 (extended through the
    // post-world blank at state 6). Raw end = 6 * 300 = 1800,
    // clamped to chunk_end = 1500.
    assert_eq!(w1.start_pts(), 4 * 300);
    assert_eq!(w1.end_pts(), 1_500);
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

  /// `hop_samples == 1` with no VAD segments must produce an
  /// all-false mask. Pre-fix the threshold floored to `0`
  /// (because `1 / 2 = 0`) and `0 >= 0` for every frame's zero
  /// overlap, so an empty `sub_segments` list marked every
  /// frame as speech — defeating the silence-mask drop and
  /// emitting forced-alignment timestamps over non-speech
  /// audio. Ceil-half + min-1 fixes both branches.
  #[test]
  fn build_speech_frames_hop_one_with_no_segments_is_all_silence() {
    let mask = build_speech_frames(8, 1, &[]);
    assert_eq!(
      mask,
      alloc::vec![false; 8],
      "hop=1, no VAD: every frame must be silence; got {:?}",
      mask
    );
  }

  /// Odd custom stride: with `hop=3` the threshold must be 2
  /// samples (ceil of half), not 1 (floor of half). A
  /// 1-sample overlap inside a 3-sample frame is below 50 %
  /// and must not count as speech.
  #[test]
  fn build_speech_frames_odd_hop_requires_strict_majority() {
    use core::num::NonZeroU32;
    use mediatime::{TimeRange, Timebase};

    let tb_16k = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    // 4 frames × hop=3 = 12 samples. VAD island [0, 1) is
    // 1 sample inside frame 0 — below the 2-sample threshold.
    let segs = alloc::vec![TimeRange::new(0, 1, tb_16k)];
    let mask = build_speech_frames(4, 3, &segs);
    assert_eq!(
      mask,
      alloc::vec![false; 4],
      "hop=3, 1-sample overlap is below ceil-half threshold; got {:?}",
      mask
    );

    // Boundary: 2 samples in frame 0 = exactly the threshold.
    let segs_at = alloc::vec![TimeRange::new(0, 2, tb_16k)];
    let mask_at = build_speech_frames(4, 3, &segs_at);
    assert_eq!(
      mask_at[0], true,
      "hop=3, 2-sample overlap (= ceil-half) is at threshold and must count"
    );
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
      log_probs.t as u64 * 320,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    let s = result.words()[0].score();
    assert!((0.0..=1.0).contains(&s));
  }

  /// Regression: emitted Word ranges must use the
  /// effective `n_samples / (T - 1)` samples-per-frame
  /// ratio (matching WhisperX's `alignment.py`), not the
  /// nominal `hop_samples`. With wav2vec2-base on 30 s of
  /// audio (480 000 samples → T = 1499 + 1 conceptual,
  /// here exercised as T = 1500 to confirm the formula),
  /// the per-frame difference is ~0.21 samples; over 1500
  /// frames this accumulates to the ~40 ms drift that
  /// previously misaligned every word vs WhisperX.
  #[test]
  fn compose_words_uses_effective_samples_per_frame_not_nominal_hop() {
    // 1500 frames; word 0's token (state 1) emits at frame
    // 100 only. All other frames are blank (state 0).
    let mut state_per_frame = alloc::vec![0_usize; 1500];
    state_per_frame[100] = 1;
    let path = ViterbiPath {
      state_per_frame,
      tokens: alloc::vec![10],
    };
    let log_probs = lp_const(1500, 30, -1.0);
    let word_idx_per_token = alloc::vec![Some(0)];
    let original = alloc::vec![Cow::Borrowed("ratio")];
    let speech_frames = alloc::vec![true; 1500];

    // n_samples = 480 000 → samples_per_frame =
    // 480 000 / (1500 - 1) = 480 000 / 1499 ≈ 320.2135.
    // Frame 100 maps to ≈ 32021 samples (NOT 32 000 as
    // nominal `100 * 320` would give).
    let result = compose_words(
      &path,
      &log_probs,
      &word_idx_per_token,
      &original,
      &speech_frames,
      0,
      320,
      480_000,
      fake_samples_to_output_range,
      DEFAULT_MIN_SPEECH_COVERAGE,
      DEFAULT_MAX_INTRA_SILENT_RUN,
    );
    assert_eq!(result.words().len(), 1);
    let r = result.words()[0].range();
    let start = r.start_pts();
    let expected = 32_021_i64; // round(100 * 480000/1499)
    let nominal = 32_000_i64; // what the buggy code returned
    assert!(
      (start - expected).abs() <= 1,
      "expected start within 1 sample of {expected} (effective ratio); \
       got {start}. Nominal `hop * frame` would have been {nominal}, \
       which is the regression this test guards against.",
    );
    assert_ne!(
      start, nominal,
      "compose must NOT use nominal hop_samples * frame; got {start} \
       (would mean we re-introduced the ~40 ms drift)",
    );
  }
}
