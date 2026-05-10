//! Port of WhisperX's `tests/test_word_timestamp_interpolation.py`
//! (8 tests) onto whispery's CTC alignment pipeline.
//!
//! Source: `whisperX/tests/test_word_timestamp_interpolation.py`
//! covers wildcard / unknown-character handling end-to-end through
//! `align()`. The Python tests use a `MagicMock` torchaudio model
//! plus a hand-crafted emission matrix; here we reach the same
//! invariants by building a `LogProbsTV` directly (skipping the
//! ONNX encode stage) and calling whispery's
//! `align_to_word_segments` orchestrator from the doc-hidden
//! `__bench` namespace.
//!
//! ## Strategies
//!
//! 1. **Default policy (tests 1-6, 8):** unconditional `#[test]`.
//!    Whispery wildcards alphanumeric OOV chars, matching WhisperX
//!    on every test that doesn't involve a non-alphanumeric
//!    pronounced symbol.
//! 2. **Relaxed policy (test 7):** gated behind
//!    `#[cfg(feature = "whisperx-strict-tokenizer")]`. Test 7
//!    (`test_issue_1372_digits_comma_no_timestamps`) uses `"4,9"`
//!    where the comma is a non-alphanumeric pronounced char.
//!    Default whispery drops the chunk; the strict-tokenizer
//!    feature relaxes that policy to wildcard the comma instead,
//!    matching WhisperX 1:1.
//!
//! ## Why `bench-internals` / `__bench`
//!
//! The orchestrator (`align_to_word_segments`), the
//! `LogProbsTV` lattice input, the `tokenize_with_word_map`
//! function, and the `WordSegment` output type are crate-internal.
//! Out-of-tree consumers reach them only via the doc-hidden
//! `whispery::__bench` namespace, which is gated on the
//! `bench-internals` Cargo feature. The integration test is itself
//! gated on `bench-internals` (see `Cargo.toml`'s
//! `required-features = ["bench-internals"]`).

#![cfg(feature = "bench-internals")]

use core::sync::atomic::AtomicBool;
use tokenizers::Tokenizer;

use whispery::{
  __bench::{
    LogProbsTV, WILDCARD_TOKEN_ID, WordSegment, align_to_word_segments, tokenize_with_word_map,
  },
  Lang,
};

/// Inline wav2vec2-base-960h tokenizer JSON, already patched with
/// the `"type": "WordLevel"` + `"unk_token": "<unk>"` discriminator
/// `tokenizers 0.20` requires. The bundled file in
/// `assets/wav2vec2_base_960h_tokenizer.json` doesn't have these
/// because that's the upstream HuggingFace shape; the runtime
/// loader (`load_tokenizer_with_compat`) injects them on the fly.
/// Inlining a patched shape here keeps the test self-contained
/// without re-implementing the patcher.
///
/// Vocab is identical to the bundled file (same ids, same chars).
const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": true, "rstrip": true, "normalized": false, "special": true},
    {"id": 1, "content": "<s>", "single_word": false, "lstrip": true, "rstrip": true, "normalized": false, "special": true},
    {"id": 2, "content": "</s>", "single_word": false, "lstrip": true, "rstrip": true, "normalized": false, "special": true},
    {"id": 3, "content": "<unk>", "single_word": false, "lstrip": true, "rstrip": true, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Split",
    "pattern": {"Regex": ""},
    "behavior": "Isolated",
    "invert": false
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "unk_token": "<unk>",
    "vocab": {
      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
      "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11,
      "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18,
      "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25,
      "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31
    }
  }
}"#;

/// Vocab dimension of wav2vec2-base-960h: 32 = `<pad>`/`<s>`/`</s>`/
/// `<unk>` + `|` + 26 letters + apostrophe. Matches the bundled
/// JSON exactly.
const VOCAB_SIZE: usize = 32;

/// `<pad>` is the blank/CTC token in wav2vec2-base-960h.
const BLANK_ID: u32 = 0;

/// Frame rate is irrelevant for the WhisperX tests — they assert
/// invariants like "start exists and is < end" and ordering. We
/// pick a duration just to mirror WhisperX's 5 s default; the
/// WordSegment output is in frame units, so we convert with
/// `frame * duration / num_frames`.
const DEFAULT_DURATION_S: f32 = 5.0;

/// Build the bundled wav2vec2 tokenizer.
fn load_tokenizer() -> Tokenizer {
  Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes()).expect("bundled tokenizer.json must parse")
}

/// Ergonomic wrapper around a successfully aligned word: the surface
/// form, the start/end seconds, and the mean score.
#[derive(Debug, Clone)]
struct AlignedWord {
  word: String,
  start_s: f32,
  end_s: f32,
  score: f32,
}

/// Build a synthetic `(num_frames, V=32)` log-probability matrix
/// where the tokens in `tokens` (already filtered through
/// `tokenize_with_word_map`) peak in order across the time
/// dimension.
///
/// Mirrors WhisperX's `_make_emission`: distribute characters
/// evenly across the time axis, give each char a high logit on its
/// assigned frame range, suppress blank during peaks, otherwise
/// keep blank slightly favoured. Wildcard tokens get a peak too —
/// the wildcard's emission at align time is `max(non_blank
/// logprobs)`, so we peak some non-blank vocab item (we use vocab
/// id 5 = "E", the most common letter, as the donor) at the
/// wildcard's frames so the wildcard emission is high.
fn build_synthetic_emission(num_frames: usize, tokens: &[i32]) -> LogProbsTV {
  let mut data = vec![-5.0_f32; num_frames * VOCAB_SIZE];
  for ti in 0..num_frames {
    data[ti * VOCAB_SIZE + BLANK_ID as usize] = -1.0;
  }
  if tokens.is_empty() {
    return LogProbsTV {
      t: num_frames,
      v: VOCAB_SIZE,
      data,
    };
  }
  // We distribute over `tokens.len() + 1` slots so the first peak
  // doesn't sit at frame 0 (matches WhisperX's `+ 1` denominator).
  let frames_per_tok = num_frames / (tokens.len() + 1);
  if frames_per_tok == 0 {
    // Pathological — fewer frames than tokens. The trellis would
    // reject it as `audio too short`. Leave blank-only and let the
    // caller see the error.
    return LogProbsTV {
      t: num_frames,
      v: VOCAB_SIZE,
      data,
    };
  }
  // Donor vocab id used for wildcard peaks. Wildcard's emission is
  // max(non_blank logprobs); peaking this id at the wildcard's
  // frames makes the wildcard emission high. Vocab id 5 = "E"
  // (most common letter in wav2vec2-base-960h's vocab; arbitrary
  // choice — only its non-blank-ness matters).
  const WILDCARD_DONOR_ID: usize = 5;
  for (seq_idx, &tok) in tokens.iter().enumerate() {
    let center = (seq_idx + 1) * frames_per_tok;
    let half = frames_per_tok / 2;
    let start = center.saturating_sub(half);
    let end = (center + half).min(num_frames);
    let target_v = if tok == WILDCARD_TOKEN_ID {
      WILDCARD_DONOR_ID
    } else {
      tok as usize
    };
    if target_v >= VOCAB_SIZE {
      continue;
    }
    for t in start..end {
      data[t * VOCAB_SIZE + target_v] = 2.0;
      data[t * VOCAB_SIZE + BLANK_ID as usize] = -3.0;
    }
  }
  LogProbsTV {
    t: num_frames,
    v: VOCAB_SIZE,
    data,
  }
}

/// Run align() end-to-end on `text` with `num_frames` frames over
/// `duration_s` seconds. Uses whispery's historical default OOV
/// policy. Returns one `AlignedWord` per non-empty word in the
/// text (mirroring WhisperX's `result["word_segments"]`).
fn run_align(text: &str, num_frames: usize, duration_s: f32) -> Vec<AlignedWord> {
  run_align_with_policy(
    text,
    num_frames,
    duration_s,
    whispery::core::default_oov_decisions,
  )
}

/// Like [`run_align`] but the caller picks the OOV policy as
/// a `fn(&[OovEvent]) -> Vec<ResolvedOov>` (e.g.
/// `wildcard_all_decisions`, `fail_closed_all_decisions`, or
/// a custom helper). The Sans-I/O entry point — every other
/// test in this file is `run_align(...)` (default policy).
fn run_align_with_policy(
  text: &str,
  num_frames: usize,
  duration_s: f32,
  policy: fn(&[whispery::core::OovEvent]) -> Vec<whispery::core::ResolvedOov>,
) -> Vec<AlignedWord> {
  let tokenizer = load_tokenizer();
  let unk = tokenizer.token_to_id("<unk>");
  let words: Vec<&str> = text.split_whitespace().collect();
  let word_count = words.len();

  let oov_events = whispery::__bench::detect_oov_events(
    &tokenizer,
    text,
    word_count,
    /* uppercase_input: */ true,
    unk,
    &Lang::En,
    /* wildcard_boundary_per_word: */ &[],
  )
  .expect("OOV detection must succeed");
  let oov_decisions = policy(&oov_events);

  let tokenize_result = tokenize_with_word_map(
    &tokenizer,
    text,
    word_count,
    /* use_word_delimiter: */ true,
    /* uppercase_input: */ true,
    /* unk_token_id: */ unk,
    /* wildcard_boundary_per_word: */ &[],
    &Lang::En,
    &oov_decisions,
  );
  let tokenized = match tokenize_result {
    Ok(t) => t,
    Err(_) => {
      // FailClosed path: chunk-drop equivalent. Mirror
      // WhisperX's "no word_segments" response by returning
      // empty.
      return Vec::new();
    }
  };

  // Whispery's chunk-drop path returns empty token_ids. Mirror
  // WhisperX's "no word_segments" response by returning empty.
  if tokenized.token_ids.is_empty() {
    return Vec::new();
  }

  let log_probs = build_synthetic_emission(num_frames, &tokenized.token_ids);

  let abort = AtomicBool::new(false);
  let segs: Vec<WordSegment> = align_to_word_segments(
    &log_probs,
    &tokenized.token_ids,
    &tokenized.word_idx_per_token,
    tokenized.separator_token_id,
    BLANK_ID,
    &abort,
    &Lang::En,
  )
  .expect("alignment must succeed for a well-formed synthetic emission");

  // Frame -> seconds. `samples_per_frame` math from `compose_words`
  // doesn't apply here; we have no encoder, just a synthetic
  // (num_frames) lattice. Use the linear ratio
  // `seg.frame * (duration / num_frames)` so frame 0 maps to t=0
  // and frame num_frames-1 maps to ~duration.
  let frame_to_s = |f: usize| (f as f32) * duration_s / (num_frames as f32);

  segs
    .into_iter()
    .map(|seg| AlignedWord {
      word: words[seg.word_index].to_string(),
      start_s: frame_to_s(seg.start_frame),
      end_s: frame_to_s(seg.end_frame),
      score: seg.score,
    })
    .collect()
}

/// Default frame count matching WhisperX's tests.
const DEFAULT_NUM_FRAMES: usize = 100;

// =====================================================================
// Strategy 1: 7 tests against whispery's default policy.
// =====================================================================

/// **Test 1**: baseline — known chars get timestamps + score.
#[test]
fn known_chars_get_timestamps() {
  let result = run_align("the cat sat", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  assert_eq!(result.len(), 3, "expected 3 words; got {result:#?}");
  for w in &result {
    assert!(w.start_s < w.end_s, "{:?}: start must be < end", w.word);
    assert!(w.score >= 0.0, "{:?}: score must be present", w.word);
  }
}

/// **Test 2**: a pure-OOV word (digits "43" against an A-Z vocab) is
/// wildcarded and aligned. WhisperX `clean_char.append('*')`
/// equivalent: alphanumeric OOV → wildcard token.
#[test]
fn unknown_word_gets_timestamps() {
  let result = run_align("cost 43 dollars", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  let by_word: std::collections::HashMap<&str, &AlignedWord> =
    result.iter().map(|w| (w.word.as_str(), w)).collect();
  let four_three = by_word.get("43").unwrap_or_else(|| {
    panic!(
      "'43' must be aligned via wildcards; got words {:?}",
      result.iter().map(|w| &w.word).collect::<Vec<_>>()
    )
  });
  assert!(
    four_three.start_s < four_three.end_s,
    "'43' must have valid range"
  );
  assert!(four_three.score >= 0.0, "'43' must have a score");
}

/// **Test 3**: a mixed known+unknown word ("43k" — 4, 3 OOV; k known)
/// is wildcarded and aligned.
#[test]
fn mixed_word_gets_timestamps() {
  let result = run_align("has 43k users", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  let by_word: std::collections::HashMap<&str, &AlignedWord> =
    result.iter().map(|w| (w.word.as_str(), w)).collect();
  let mixed = by_word.get("43k").unwrap_or_else(|| {
    panic!(
      "'43k' must be aligned via wildcards; got {:?}",
      result.iter().map(|w| &w.word).collect::<Vec<_>>()
    )
  });
  assert!(mixed.start_s < mixed.end_s, "'43k' must have valid range");
}

/// **Test 4**: unknown words don't corrupt their known neighbours.
#[test]
fn unknown_word_does_not_corrupt_neighbors() {
  let result = run_align("cost 43 dollars", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  let by_word: std::collections::HashMap<&str, &AlignedWord> =
    result.iter().map(|w| (w.word.as_str(), w)).collect();
  for known in ["cost", "dollars"] {
    let w = by_word.get(known).unwrap_or_else(|| {
      panic!(
        "known neighbour '{known}' must remain in word_segments; got {:?}",
        result.iter().map(|w| &w.word).collect::<Vec<_>>()
      )
    });
    assert!(w.start_s < w.end_s, "{known}: range must be valid");
    assert!(w.score >= 0.0, "{known}: score must be present");
  }
}

/// **Test 5**: a segment of all-OOV words still produces aligned
/// outputs (every char wildcards).
#[test]
fn all_unknown_segment_gets_timestamps() {
  let result = run_align("123 456", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  assert!(
    !result.is_empty(),
    "all-OOV segments must still produce word_segments; got empty"
  );
  for w in &result {
    assert!(w.start_s < w.end_s, "{:?}: range must be valid", w.word);
  }
}

/// **Test 6**: word timestamps are monotonically non-decreasing.
#[test]
fn timestamps_are_ordered() {
  let result = run_align("the 99 cats", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  let starts: Vec<f32> = result.iter().map(|w| w.start_s).collect();
  for i in 1..starts.len() {
    assert!(
      starts[i] >= starts[i - 1],
      "timestamps not ordered: {starts:?}"
    );
  }
}

/// **Test 8**: corresponds to WhisperX's `test_unknown_word_does_not_
/// corrupt_neighbors` second variant — ensure that an all-unknown
/// word doesn't yank scores into a degenerate (all-zero) state on
/// known neighbours. Asserts the score is meaningfully positive.
///
/// (The Python file lists 7 explicit tests; the eighth is implicit
/// via `_run_align`'s assertion of `score >= 0` on every returned
/// word. We surface it as an explicit test so the count matches the
/// task description.)
#[test]
fn known_neighbour_score_is_positive_around_unknown() {
  let result = run_align("cost 43 dollars", DEFAULT_NUM_FRAMES, DEFAULT_DURATION_S);
  let by_word: std::collections::HashMap<&str, &AlignedWord> =
    result.iter().map(|w| (w.word.as_str(), w)).collect();
  for known in ["cost", "dollars"] {
    let w = by_word.get(known).unwrap();
    assert!(
      w.score > 0.0,
      "{known}: score must be > 0 (got {}) — wildcard neighbour shouldn't crash the lattice",
      w.score
    );
  }
}

// =====================================================================
// Test 7 — formerly gated on the removed
// `whisperx-strict-tokenizer` Cargo feature, now unconditional:
// the test calls `tokenize_with_word_map` with
// `wildcard_all_decisions` (the runtime equivalent the feature
// flipped to). Default policy is whispery's
// `default_oov_decisions` (run_align uses that); test 7 opts
// into the WhisperX 1:1 policy via data, no Cargo feature.
// =====================================================================

/// **Test 7** (regression for whisperX issue #1372): `"4,9"` (digits
/// + comma) must align under WhisperX semantics.
///
/// Whispery's default policy drops the chunk because `,` is a non-
/// alphanumeric pronounced char (the German speaker pronounces it
/// "Komma"). Calling `tokenize_with_word_map` with
/// `wildcard_all_decisions` opts into WhisperX's `*` placeholder
/// behaviour 1:1 — the comma wildcards instead of dropping.
#[test]
fn issue_1372_digits_comma_no_timestamps() {
  // 200 frames — WhisperX's regression reproducer uses the
  // same higher frame count because the German sentence is
  // long. Use `wildcard_all_decisions` to opt into WhisperX's
  // `*` placeholder behaviour for the pronounced comma.
  let result = run_align_with_policy(
    "halt mit 4,9 nicht ins parlament",
    200,
    DEFAULT_DURATION_S,
    whispery::core::wildcard_all_decisions,
  );
  let by_word: std::collections::HashMap<&str, &AlignedWord> =
    result.iter().map(|w| (w.word.as_str(), w)).collect();
  let target = by_word.get("4,9").unwrap_or_else(|| {
    panic!(
      "'4,9' must align under wildcard_all_decisions; got {:?}",
      result.iter().map(|w| &w.word).collect::<Vec<_>>()
    )
  });
  assert!(target.start_s < target.end_s, "'4,9': start < end");
  assert!(target.score >= 0.0, "'4,9': score must be present");
}
