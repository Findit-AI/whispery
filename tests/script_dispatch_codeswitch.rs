//! Ja+Zh code-switch parity test for the script-dispatch pipeline.
//!
//! Runs whisper ASR + whispery alignment on a hand-picked
//! Japanese/Chinese mixed clip and asserts:
//!
//! 1. Per-word language assignment matches the hand-labelled
//!    ground truth.
//! 2. Per-word `t0_ms` / `t1_ms` are within ±50 ms on at least
//!    95 % of words.
//!
//! ## Fixture layout
//!
//! Files live at
//! `tests/parity_whisperx/fixtures/ja-zh-codeswitch/`:
//!
//! - `audio.wav` — 16 kHz mono PCM, ~10–30 s mixing Ja + Zh.
//! - `ground_truth.json` — hand-labelled per-word reference.
//!
//! See that directory's `README.md` for the full schema. When
//! either file is missing the test skips with a diagnostic
//! `[script_dispatch_codeswitch] fixture missing; skipping`
//! line — keeping the suite green for callers who haven't
//! materialised the fixture yet (CI / reviewers running this
//! branch before the audio + labels land).
//!
//! Required environment for the model paths:
//!
//! - `WHISPERY_WHISPER_MODEL` — `ggml-large-v3-turbo.bin` (or
//!   any multilingual whisper.cpp model).
//! - `WHISPERY_W2V_MODEL_JA` / `WHISPERY_W2V_TOKENIZER_JA` —
//!   Japanese wav2vec2 ONNX + tokenizer.
//! - `WHISPERY_W2V_MODEL_ZH` / `WHISPERY_W2V_TOKENIZER_ZH` —
//!   Chinese wav2vec2 ONNX + tokenizer.

#![cfg(feature = "alignment")]

use core::{num::NonZeroU32, time::Duration};
use std::path::{Path, PathBuf};

use mediatime::{Timebase, Timestamp};
use whispery::{
  Lang, LanguagePolicy, ManagedTranscriber, VadSegment, WhisperPoolOptions,
  runner::{Aligner, AlignerKey, AlignmentFallback, AlignmentSetBuilder, default_normalizer_for},
};

const WHISPER_MODEL: Option<&str> = option_env!("WHISPERY_WHISPER_MODEL");
const W2V_JA_MODEL: Option<&str> = option_env!("WHISPERY_W2V_MODEL_JA");
const W2V_JA_TOKENIZER: Option<&str> = option_env!("WHISPERY_W2V_TOKENIZER_JA");
const W2V_ZH_MODEL: Option<&str> = option_env!("WHISPERY_W2V_MODEL_ZH");
const W2V_ZH_TOKENIZER: Option<&str> = option_env!("WHISPERY_W2V_TOKENIZER_ZH");

/// Tolerance for the per-word boundary comparison.
const TIMING_TOLERANCE_MS: i64 = 50;
/// Minimum fraction of words that must satisfy the timing
/// tolerance. Per the spec ("±50 ms on 95 % of words").
const TIMING_HIT_RATE_FLOOR: f64 = 0.95;

/// One word in the hand-labelled reference. Constructed
/// manually from `serde_json::Value` so the test compiles with
/// the bare `alignment` feature (the crate's `serde` feature is
/// optional, but `serde_json` is always available as a
/// dev-dependency).
#[derive(Debug, Clone)]
struct GroundTruthWord {
  word: String,
  lang: String,
  t0_ms: i64,
  t1_ms: i64,
}

impl GroundTruthWord {
  /// Read one element of the ground-truth JSON array. Fails
  /// loud — a missing key or wrong type is an authoring bug in
  /// the fixture, not something to silently mask.
  fn from_json(v: &serde_json::Value) -> Self {
    Self {
      word: v
        .get("word")
        .and_then(serde_json::Value::as_str)
        .expect("ground_truth.json: word")
        .to_string(),
      lang: v
        .get("lang")
        .and_then(serde_json::Value::as_str)
        .expect("ground_truth.json: lang")
        .to_string(),
      t0_ms: v
        .get("t0_ms")
        .and_then(serde_json::Value::as_i64)
        .expect("ground_truth.json: t0_ms"),
      t1_ms: v
        .get("t1_ms")
        .and_then(serde_json::Value::as_i64)
        .expect("ground_truth.json: t1_ms"),
    }
  }
}

/// Resolve the fixture directory relative to this test file.
fn fixture_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("parity_whisperx")
    .join("fixtures")
    .join("ja-zh-codeswitch")
}

/// Load a 16 kHz mono PCM WAV into f32 samples. Mirrors
/// `tests/alignment_e2e.rs`'s loader so the two tests share
/// identical audio-decoding semantics.
fn read_wav_16k_mono_f32(path: &Path) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("open wav");
  let spec = reader.spec();
  assert_eq!(spec.sample_rate, 16_000, "fixture expected at 16 kHz");
  assert_eq!(spec.channels, 1, "fixture expected mono");
  match spec.sample_format {
    hound::SampleFormat::Int => reader
      .samples::<i16>()
      .map(|s| s.unwrap() as f32 / i16::MAX as f32)
      .collect(),
    hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
  }
}

/// Map a ground-truth lang code (`"ja"` / `"zh"`) onto the
/// matching [`Lang`].
fn parse_lang(s: &str) -> Option<Lang> {
  match s {
    "ja" => Some(Lang::Ja),
    "zh" => Some(Lang::Zh),
    _ => None,
  }
}

/// True when the test's required fixture files all exist on
/// disk. The harness short-circuits to a skip-with-diagnostic
/// when any are missing — `#[ignore]` would hide the test
/// completely; this approach makes the skip self-explaining.
fn fixture_complete(dir: &Path) -> bool {
  dir.join("audio.wav").is_file() && dir.join("ground_truth.json").is_file()
}

/// True when every required model env var is set. Mirrors the
/// other alignment integration tests' offline-mode guard.
fn models_complete() -> bool {
  WHISPER_MODEL.is_some()
    && W2V_JA_MODEL.is_some()
    && W2V_JA_TOKENIZER.is_some()
    && W2V_ZH_MODEL.is_some()
    && W2V_ZH_TOKENIZER.is_some()
}

/// End-to-end Ja+Zh code-switch parity. Marked `#[ignore]` so
/// it doesn't run on plain `cargo test`; opt in via `cargo test
/// -- --ignored` once the fixture and models are in place. The
/// inner skip-on-missing-fixture path keeps the test compiling
/// and the gate green even before the audio has been
/// materialised.
#[test]
#[ignore = "fixture pending: tests/parity_whisperx/fixtures/ja-zh-codeswitch (audio.wav + ground_truth.json) and Ja/Zh wav2vec2 models"]
fn ja_zh_codeswitch_per_word_parity() {
  let dir = fixture_dir();
  if !fixture_complete(&dir) {
    eprintln!(
      "[script_dispatch_codeswitch] fixture missing at {}; skipping",
      dir.display()
    );
    return;
  }
  if !models_complete() {
    eprintln!(
      "[script_dispatch_codeswitch] WHISPERY_WHISPER_MODEL / WHISPERY_W2V_MODEL_JA / \
       WHISPERY_W2V_TOKENIZER_JA / WHISPERY_W2V_MODEL_ZH / WHISPERY_W2V_TOKENIZER_ZH \
       not all set; skipping"
    );
    return;
  }

  // Unwraps OK after `models_complete()` and `fixture_complete()`
  // gates above.
  let whisper_model = WHISPER_MODEL.expect("WHISPERY_WHISPER_MODEL");
  let w2v_ja_model = W2V_JA_MODEL.expect("WHISPERY_W2V_MODEL_JA");
  let w2v_ja_tok = W2V_JA_TOKENIZER.expect("WHISPERY_W2V_TOKENIZER_JA");
  let w2v_zh_model = W2V_ZH_MODEL.expect("WHISPERY_W2V_MODEL_ZH");
  let w2v_zh_tok = W2V_ZH_TOKENIZER.expect("WHISPERY_W2V_TOKENIZER_ZH");

  let audio_path = dir.join("audio.wav");
  let truth_path = dir.join("ground_truth.json");

  let samples = read_wav_16k_mono_f32(&audio_path);
  let truth: Vec<GroundTruthWord> = {
    let bytes = std::fs::read(&truth_path).expect("read ground_truth.json");
    let raw: serde_json::Value =
      serde_json::from_slice(&bytes).expect("parse ground_truth.json");
    let arr = raw
      .as_array()
      .expect("ground_truth.json: top-level array");
    arr.iter().map(GroundTruthWord::from_json).collect()
  };
  assert!(
    !truth.is_empty(),
    "ground_truth.json must list at least one word"
  );

  // Build the per-language alignment registry. `default_normalizer_for`
  // returns the canonical Ja / Zh normaliser that matches the
  // wav2vec2 vocab the parity harness has been trained against.
  let ja_norm = default_normalizer_for(&Lang::Ja).expect("ja normalizer");
  let zh_norm = default_normalizer_for(&Lang::Zh).expect("zh normalizer");

  let aligner_ja = Aligner::from_paths(
    Lang::Ja,
    Path::new(w2v_ja_model),
    Path::new(w2v_ja_tok),
    ja_norm,
  )
  .expect("build Ja Aligner");
  let aligner_zh = Aligner::from_paths(
    Lang::Zh,
    Path::new(w2v_zh_model),
    Path::new(w2v_zh_tok),
    zh_norm,
  )
  .expect("build Zh Aligner");

  let alignment_set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::SkipChunk)
    .register(AlignerKey::Lang(Lang::Ja), aligner_ja)
    .register(AlignerKey::Lang(Lang::Zh), aligner_zh)
    .build();

  // Drive the runner. Same shape as the JFK alignment_e2e test:
  // single packet, single VAD segment covering the whole clip,
  // drain to completion.
  let pool_opts = WhisperPoolOptions::new(whisper_model);
  let mut transcriber = ManagedTranscriber::from_options(pool_opts)
    .expect("ManagedTranscriber::from_options")
    .chunk_size(Duration::from_secs(30))
    .language_policy(LanguagePolicy::Auto)
    .with_alignment(alignment_set)
    .build()
    .expect("ManagedTranscriber::build");

  let tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
  let starts_at = Timestamp::new(0, tb);
  let vad = VadSegment::new(0, samples.len() as u64);

  transcriber
    .process_packet(starts_at, &samples, &[vad], None)
    .expect("process_packet");
  transcriber.signal_eof().expect("signal_eof");
  transcriber.drain().expect("drain");

  // Collect every transcript the runner emitted.
  let mut got: Vec<(String, Lang, i64, i64)> = Vec::new();
  while let Some(transcript) = transcriber
    .poll_transcript()
    .expect("poll_transcript")
  {
    let chunk_lang = transcript.language().clone();
    for word in transcript.words() {
      let r = word.range();
      // Output timebase: the runner's `samples_to_output_range`
      // produces a `TimeRange` whose `Timebase` matches the
      // first push's timebase. We pushed `starts_at` at 1/16 000
      // (samples), so PTS values here are sample indices —
      // convert to milliseconds for the comparison.
      let t0_ms = (r.start_pts() * 1_000) / 16_000;
      let t1_ms = (r.end_pts() * 1_000) / 16_000;
      got.push((word.text().to_string(), chunk_lang.clone(), t0_ms, t1_ms));
    }
  }

  assert!(
    !got.is_empty(),
    "no aligned words came back from the runner"
  );

  // Assertion 1: same word count (or at least same total
  // characters). Whisper + dispatcher may differ on punctuation
  // / segmentation, so we accept either equality.
  let truth_chars: usize = truth.iter().map(|w| w.word.chars().count()).sum();
  let got_chars: usize = got.iter().map(|(t, _, _, _)| t.chars().count()).sum();
  assert!(
    got.len() == truth.len() || got_chars == truth_chars,
    "word-count / char-count mismatch: got {} words ({} chars), truth {} words ({} chars)",
    got.len(),
    got_chars,
    truth.len(),
    truth_chars,
  );

  // Assertion 2 + 3: walk the pairwise zip, asserting same
  // language and tracking how many words satisfy the timing
  // tolerance.
  let pair_len = got.len().min(truth.len());
  let mut hits = 0_usize;
  for ((got_text, got_lang, got_t0, got_t1), gt) in got.iter().zip(truth.iter()).take(pair_len) {
    let truth_lang = parse_lang(&gt.lang).unwrap_or_else(|| {
      panic!("unknown ground-truth lang code: {:?}", gt.lang);
    });
    assert_eq!(
      *got_lang, truth_lang,
      "language mismatch on word {got_text:?} (truth {:?})",
      gt.word
    );
    if (got_t0 - gt.t0_ms).abs() <= TIMING_TOLERANCE_MS
      && (got_t1 - gt.t1_ms).abs() <= TIMING_TOLERANCE_MS
    {
      hits += 1;
    }
  }
  let hit_rate = hits as f64 / pair_len as f64;
  assert!(
    hit_rate >= TIMING_HIT_RATE_FLOOR,
    "timing hit rate {:.3} < {:.3} (±{} ms tolerance)",
    hit_rate,
    TIMING_HIT_RATE_FLOOR,
    TIMING_TOLERANCE_MS,
  );
}

/// Compile-time check that the comparison harness builds even
/// when the fixture file is missing. This is the
/// `#[ignore]`-skip-path documented in the spec — keeping the
/// suite green on `cargo test` regardless of whether the audio
/// has been materialised yet.
#[test]
fn fixture_skip_path_compiles() {
  let dir = fixture_dir();
  let _ = fixture_complete(&dir);
  let _ = models_complete();
  // Smoke-check the lang-code parser doesn't panic on the two
  // codes the harness expects.
  assert_eq!(parse_lang("ja"), Some(Lang::Ja));
  assert_eq!(parse_lang("zh"), Some(Lang::Zh));
  assert_eq!(parse_lang("xx"), None);
}
