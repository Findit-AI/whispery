//! `whispery-parity-runner` — load a 16 kHz mono WAV, push it through
//! `ManagedTranscriber` with English-locked language + wav2vec2 forced
//! alignment, and dump word-level results to JSON. Pair with
//! `python/whisperx_runner.py` (same JSON schema, runner = "whisperx")
//! and `python/score.py` for IoU comparison.
//!
//! This binary is **NOT** part of `cargo test`. It's invoked from the
//! `run.sh` driver, which expects models at the locations populated by
//! whispery's `build.rs` when `WHISPERY_FETCH_MODEL=1` /
//! `WHISPERY_FETCH_W2V=1` are set (a one-time prep on a fresh machine).
//!
//! Models are found in this order:
//!   1. CLI flags (`--whisper-model`, `--w2v-model`, `--w2v-tokenizer`)
//!   2. Env vars (`WHISPER_MODEL_PATH`, `WAV2VEC2_ONNX_PATH`,
//!      `WAV2VEC2_TOKENIZER_PATH`)
//!   3. Auto-detected fixture dir (CARGO_TARGET_DIR or
//!      `$HOME/.cargo/target/whispery-test-fixtures/`)
//!
//! `ORT_DYLIB_PATH` is consumed by `ort` itself in `load-dynamic` mode
//! (whispery's pinned configuration); the runner doesn't touch it.

use std::{
  fs,
  io::{Read, Write},
  num::NonZeroU32,
  path::{Path, PathBuf},
  time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use hound::SampleFormat;
use serde_json::json;
use sha2::{Digest, Sha256};
// `mediatime` is reachable through whispery's re-exports, so we don't
// need to add it as a separate Cargo dependency. Goes through the
// crate-root re-export rather than `whispery::time` because the public
// API path is the SemVer-stable surface.
use whispery::{
  Lang, LanguagePolicy, ManagedTranscriber, TimeRange, Timebase, Timestamp, VadSegment,
  WhisperPoolOptions,
  runner::{Aligner, AlignerKey, AlignmentFallback, AlignmentSetBuilder, EnglishNormalizer},
};

#[derive(Parser, Debug)]
#[command(
  about = "Run whispery alignment on a 16 kHz mono WAV; emit JSON for side-by-side comparison with WhisperX."
)]
struct Args {
  /// Path to a 16 kHz mono WAV (s16le or f32le).
  wav_path: PathBuf,

  /// `ggml-tiny.en.bin` (or any English whisper.cpp model). Defaults
  /// to env `WHISPER_MODEL_PATH`, then to the build.rs fixture dir.
  #[arg(long)]
  whisper_model: Option<PathBuf>,

  /// `wav2vec2-base-960h.onnx`. Defaults to env `WAV2VEC2_ONNX_PATH`,
  /// then to the build.rs fixture dir.
  #[arg(long)]
  w2v_model: Option<PathBuf>,

  /// `wav2vec2-base-960h-tokenizer.json` (HuggingFace `tokenizer.json`
  /// format). Defaults to env `WAV2VEC2_TOKENIZER_PATH`, then to the
  /// build.rs fixture dir.
  #[arg(long)]
  w2v_tokenizer: Option<PathBuf>,

  /// If set, bypass whisper.cpp ASR entirely. Reads the WhisperX
  /// JSON output at this path, concatenates its `words[].text`
  /// into a single transcript, and feeds that string straight
  /// into [`Aligner::align_chunk`]. Used to exercise alignment
  /// parity in isolation while the upstream `whisper-rs`
  /// `failed to encode` bug (gating
  /// `tests/runner_e2e.rs` and `tests/alignment_e2e.rs`)
  /// blocks the full ASR-then-align pipeline.
  ///
  /// `--whisper-model` is ignored in this mode.
  #[arg(long)]
  inject_from: Option<PathBuf>,

  /// Output file (defaults to stdout).
  #[arg(long)]
  out: Option<PathBuf>,
}

fn fixture_dir() -> Option<PathBuf> {
  // Match the layout build.rs writes to: `<cargo_target_dir>/whispery-test-fixtures/`.
  // Cargo's default target dir on macOS / Linux is
  // `$CARGO_TARGET_DIR` -> `$CARGO_HOME/target/` -> `$HOME/.cargo/target/`.
  if let Ok(p) = std::env::var("CARGO_TARGET_DIR") {
    let candidate = PathBuf::from(p).join("whispery-test-fixtures");
    if candidate.is_dir() {
      return Some(candidate);
    }
  }
  if let Ok(p) = std::env::var("CARGO_HOME") {
    let candidate = PathBuf::from(p).join("target").join("whispery-test-fixtures");
    if candidate.is_dir() {
      return Some(candidate);
    }
  }
  if let Ok(home) = std::env::var("HOME") {
    let candidate = PathBuf::from(home)
      .join(".cargo")
      .join("target")
      .join("whispery-test-fixtures");
    if candidate.is_dir() {
      return Some(candidate);
    }
  }
  None
}

fn resolve_model(
  cli: Option<PathBuf>,
  env_var: &str,
  fixture_filename: &str,
) -> Result<PathBuf> {
  if let Some(p) = cli {
    return Ok(p);
  }
  if let Ok(p) = std::env::var(env_var) {
    return Ok(PathBuf::from(p));
  }
  if let Some(dir) = fixture_dir() {
    let candidate = dir.join(fixture_filename);
    if candidate.is_file() {
      return Ok(candidate);
    }
  }
  bail!(
    "couldn't find {fixture_filename}: pass --{} (or set ${env_var}, or run \
     `WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 cargo test --features alignment` once)",
    fixture_filename.replace('.', "-")
  );
}

fn read_wav_16k_mono_f32(path: &Path) -> Result<Vec<f32>> {
  let mut reader = hound::WavReader::open(path)
    .with_context(|| format!("open WAV at {}", path.display()))?;
  let spec = reader.spec();
  if spec.sample_rate != 16_000 {
    bail!(
      "{}: expected 16 kHz, got {} Hz",
      path.display(),
      spec.sample_rate
    );
  }
  if spec.channels != 1 {
    bail!(
      "{}: expected mono, got {} channels",
      path.display(),
      spec.channels
    );
  }
  Ok(match (spec.sample_format, spec.bits_per_sample) {
    (SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
      .collect::<Result<Vec<_>, _>>()?,
    (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    other => bail!(
      "{}: unsupported WAV sample format {:?} ({}-bit)",
      path.display(),
      other.0,
      other.1
    ),
  })
}

fn sha256_file(path: &Path) -> Result<String> {
  let mut f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
  let mut hasher = Sha256::new();
  let mut buf = [0u8; 64 * 1024];
  loop {
    let n = f.read(&mut buf)?;
    if n == 0 {
      break;
    }
    hasher.update(&buf[..n]);
  }
  Ok(format!("{:x}", hasher.finalize()))
}

fn main() -> Result<()> {
  let args = Args::parse();

  // `--inject-from` short-circuits the whisper.cpp dependency so we
  // can exercise alignment parity in isolation. The whisper.cpp model
  // isn't needed and we skip resolving it (the build.rs fixture may
  // legitimately not be populated when alignment is the only thing
  // being measured).
  if let Some(inject_path) = args.inject_from.clone() {
    return run_inject_mode(args, inject_path);
  }

  let whisper_model = resolve_model(
    args.whisper_model,
    "WHISPER_MODEL_PATH",
    "ggml-tiny.en.bin",
  )?;
  let w2v_model = resolve_model(
    args.w2v_model,
    "WAV2VEC2_ONNX_PATH",
    "wav2vec2-base-960h.onnx",
  )?;
  let w2v_tokenizer = resolve_model(
    args.w2v_tokenizer,
    "WAV2VEC2_TOKENIZER_PATH",
    "wav2vec2-base-960h-tokenizer.json",
  )?;

  eprintln!(
    "[whispery-parity] whisper={} w2v={} tok={}",
    whisper_model.display(),
    w2v_model.display(),
    w2v_tokenizer.display()
  );

  // Build the alignment registry first. `Aligner::from_paths` is
  // where all the ORT loading happens; surface its errors with full
  // context.
  let aligner = Aligner::from_paths(
    Lang::En,
    &w2v_model,
    &w2v_tokenizer,
    Box::new(EnglishNormalizer::new()),
  )
  .context("build wav2vec2 Aligner")?;

  let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::SkipChunk)
    .register(AlignerKey::Lang(Lang::En), aligner)
    .build();

  // Single-worker pool. The whole clip flows through one ASR worker
  // and one alignment worker; ordering and chunk identity are
  // therefore deterministic for a given input.
  let pool = WhisperPoolOptions::new(&whisper_model)
    .with_worker_count(1)
    .with_max_queued_chunks(8);

  let mut runner = ManagedTranscriber::from_options(pool)
    .context("build ManagedTranscriber from WhisperPoolOptions")?
    .chunk_size(Duration::from_secs(30))
    .language_policy(LanguagePolicy::Lock { hint: Lang::En })
    // Generous: the longest fixture is ~16 minutes. The clip is
    // chunked into 30 s pieces by the cut state machine, so each
    // worker call sees ≤30 s of audio.
    .worker_timeouts(Duration::from_secs(120), Duration::from_secs(120))
    // 10 s per chunk-of-audio + slack. Capped so a regression that
    // hangs surfaces cleanly rather than blocking the harness
    // forever.
    .drain_timeout(Duration::from_secs(60 * 30))
    .with_alignment(set)
    .build()
    .context("build runner")?;

  // Load + measure the audio. `clip_sha256` keys outputs to the
  // exact bytes scored, so a fixture change can't go undetected.
  let samples = read_wav_16k_mono_f32(&args.wav_path)?;
  let total_samples = samples.len() as u64;
  let duration_s = total_samples as f64 / 16_000.0;
  let clip_sha256 = sha256_file(&args.wav_path)?;

  // Caller's output timebase = mediatime's microsecond default
  // (1/48000 s tick). We use it consistently when emitting Word
  // ranges below; the conversion to seconds happens via
  // `Timestamp::seconds()` so any future timebase rescale stays
  // correct without hardcoding the denominator here.
  let tb = Timebase::new(1, NonZeroU32::new(48_000).unwrap());
  let starts_at = Timestamp::new(0, tb);

  // Single VAD segment covering the entire clip — we want the full
  // audio aligned, not a VAD-driven subset. Dia's parity does the
  // same with its `push_voice_range`.
  runner
    .process_packet(
      starts_at,
      &samples,
      &[VadSegment::new(0, total_samples)],
      None,
    )
    .context("process_packet")?;
  runner.signal_eof().context("signal_eof")?;
  runner.drain().context("drain")?;

  // Drain transcripts. Each carries words in time order; we
  // flatten across chunks because the Python side compares against
  // a single flat word list too.
  let mut all_words: Vec<serde_json::Value> = Vec::new();
  let mut transcript_count = 0usize;
  while let Some(t) = runner.poll_transcript() {
    transcript_count += 1;
    for w in t.words() {
      let r = w.range();
      // `start_pts() / end_pts()` are raw ticks. Reconstruct
      // `Timestamp`s in the same timebase, take `duration()` from
      // PTS zero, then read seconds via `Duration::as_secs_f64`.
      // Centralises tick→seconds conversion through mediatime so a
      // future timebase change here doesn't quietly desync the
      // emitted ranges.
      let start_s = Timestamp::new(r.start_pts(), tb)
        .duration()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
      let end_s = Timestamp::new(r.end_pts(), tb)
        .duration()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
      all_words.push(json!({
        "text": w.text(),
        "start_s": start_s,
        "end_s": end_s,
        "score": w.score(),
      }));
    }
  }

  // Drain stray errors so we know if any chunk silently produced
  // no words. They go to stderr so JSON on stdout stays clean.
  while let Some((chunk_id, failure)) = runner.poll_error() {
    eprintln!(
      "[whispery-parity] chunk {:?} failed: {failure:?}",
      chunk_id
    );
  }

  let payload = json!({
    "runner": "whispery",
    "clip_path": args.wav_path.display().to_string(),
    "clip_sha256": clip_sha256,
    "duration_s": duration_s,
    "transcript_count": transcript_count,
    "words": all_words,
  });

  let serialized = serde_json::to_string_pretty(&payload)?;
  match args.out {
    Some(path) => {
      let mut f = fs::File::create(&path)
        .with_context(|| format!("create output {}", path.display()))?;
      f.write_all(serialized.as_bytes())?;
      f.write_all(b"\n")?;
      eprintln!(
        "[whispery-parity] wrote {} words across {transcript_count} transcripts to {}",
        all_words_len(&payload),
        path.display()
      );
    }
    None => {
      println!("{serialized}");
    }
  }

  Ok(())
}

fn all_words_len(payload: &serde_json::Value) -> usize {
  payload
    .get("words")
    .and_then(|v| v.as_array())
    .map(|a| a.len())
    .unwrap_or(0)
}

/// Inject mode: skip whisper.cpp + ManagedTranscriber entirely. Read
/// the WhisperX JSON and mirror WhisperX's **per-segment** alignment
/// flow — for each `segments[]` entry, slice the audio to that
/// segment's `[start_s, end_s)` window and drive `Aligner::align_chunk`
/// on just that slice with just that segment's text.
///
/// This matches `alignment.py:237-289` (`f1 = int(t1 * SAMPLE_RATE);
/// f2 = int(t2 * SAMPLE_RATE); waveform_segment = audio[:, f1:f2]`).
/// The previous whole-clip approach gave CTC too many ambiguous paths
/// on long clips and the alignment drifted (median IoU 0.000 on
/// `03_dual_speaker` at 60 s).
///
/// Output schema is identical to the non-inject path so `score.py`
/// is mode-agnostic.
fn run_inject_mode(args: Args, inject_path: PathBuf) -> Result<()> {
  let w2v_model = resolve_model(
    args.w2v_model,
    "WAV2VEC2_ONNX_PATH",
    "wav2vec2-base-960h.onnx",
  )?;
  let w2v_tokenizer = resolve_model(
    args.w2v_tokenizer,
    "WAV2VEC2_TOKENIZER_PATH",
    "wav2vec2-base-960h-tokenizer.json",
  )?;

  eprintln!(
    "[whispery-parity:inject] inject_from={} w2v={} tok={}",
    inject_path.display(),
    w2v_model.display(),
    w2v_tokenizer.display()
  );

  // Load the WhisperX JSON. We need `segments[]` (with `start_s`,
  // `end_s`, `text`, and per-segment `words[]`) for the per-segment
  // alignment flow; if it's missing, fall back to a synthesised
  // single segment over the entire `words[]` for back-compat with
  // older WhisperX outputs.
  let injected: serde_json::Value = {
    let bytes = fs::read(&inject_path)
      .with_context(|| format!("read whisperX JSON {}", inject_path.display()))?;
    serde_json::from_slice(&bytes)
      .with_context(|| format!("parse whisperX JSON {}", inject_path.display()))?
  };
  let injected_words_total = injected
    .get("words")
    .and_then(|v| v.as_array())
    .map(|a| a.len())
    .unwrap_or(0);

  // Audio first — needed before we slice per-segment.
  let samples = read_wav_16k_mono_f32(&args.wav_path)?;
  let total_samples = samples.len();
  let duration_s = total_samples as f64 / 16_000.0;
  let clip_sha256 = sha256_file(&args.wav_path)?;

  // Build the segments list. Prefer `segments[]` (the per-segment
  // path WhisperX itself uses); fall back to one synthetic segment
  // over the full clip if absent.
  struct InjectedSegment {
    start_s: f64,
    end_s: f64,
    text: String,
  }

  // The parity runner's default mode is per-sentence
  // `segments[]` — that's what WhisperX itself ends up with
  // after `align()` plus its `PunktSentenceTokenizer` break-up.
  //
  // The `WHISPERY_PARITY_USE_RAW_SEGMENTS=1` env var switches
  // to `raw_asr_segments[]` (the un-broken ASR segments
  // WhisperX feeds to `align()`). Useful for diagnosing
  // hallucinated-repetition cases where the per-sentence
  // breakdown is a downstream derivation; on those clips the
  // raw mode produces a more apples-to-apples comparison
  // (both implementations align the same audio + text). On
  // clips without hallucination, per-sentence is closer to
  // what WhisperX's downstream consumers see.
  let use_raw_segments = std::env::var("WHISPERY_PARITY_USE_RAW_SEGMENTS")
    .map(|v| v != "0" && !v.is_empty())
    .unwrap_or(false);
  let segments: Vec<InjectedSegment> = if use_raw_segments
    && let Some(segs) = injected.get("raw_asr_segments").and_then(|v| v.as_array())
  {
    eprintln!(
      "[whispery-parity:inject] using raw_asr_segments ({} entries)",
      segs.len()
    );
    segs
      .iter()
      .filter_map(|s| {
        let start_s = s.get("start_s").and_then(|v| v.as_f64())?;
        let end_s = s.get("end_s").and_then(|v| v.as_f64())?;
        let text = s
          .get("text")
          .and_then(|v| v.as_str())
          .map(|s| s.trim().to_string())
          .unwrap_or_default();
        Some(InjectedSegment {
          start_s,
          end_s,
          text,
        })
      })
      .collect()
  } else if let Some(segs) = injected.get("segments").and_then(|v| v.as_array())
  {
    segs
      .iter()
      .filter_map(|s| {
        let start_s = s.get("start_s").and_then(|v| v.as_f64())?;
        let end_s = s.get("end_s").and_then(|v| v.as_f64())?;
        // Segment text: prefer the verbatim `text` field WhisperX
        // emits; if missing or empty, glue per-segment word texts.
        let text = s
          .get("text")
          .and_then(|v| v.as_str())
          .map(|s| s.trim().to_string())
          .filter(|t| !t.is_empty())
          .or_else(|| {
            s.get("words").and_then(|v| v.as_array()).map(|ws| {
              ws.iter()
                .filter_map(|w| w.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(" ")
            })
          })
          .unwrap_or_default();
        Some(InjectedSegment {
          start_s,
          end_s,
          text,
        })
      })
      .collect()
  } else {
    // Backwards-compat: no `segments[]`, fall back to a single
    // pseudo-segment containing all words over the whole clip.
    eprintln!(
      "[whispery-parity:inject] WARN: no `segments[]` in inject JSON; \
       falling back to single whole-clip segment (drift expected on >30s clips)"
    );
    let words = injected.get("words").and_then(|v| v.as_array());
    let text = words
      .map(|ws| {
        ws.iter()
          .filter_map(|w| w.get("text").and_then(|t| t.as_str()))
          .collect::<Vec<_>>()
          .join(" ")
      })
      .unwrap_or_default();
    vec![InjectedSegment {
      start_s: 0.0,
      end_s: duration_s,
      text,
    }]
  };

  eprintln!(
    "[whispery-parity:inject] {} segments, {} total injected words across {:.2}s",
    segments.len(),
    injected_words_total,
    duration_s
  );

  // Build the aligner directly — no Transcriber, no
  // ManagedTranscriber, no whisper.cpp.
  let mut aligner = Aligner::from_paths(
    Lang::En,
    &w2v_model,
    &w2v_tokenizer,
    Box::new(EnglishNormalizer::new()),
  )
  .context("build wav2vec2 Aligner")?;

  // VAD-style sub-segments are computed per-segment below (each
  // covers its own slice in chunk-local 16 kHz coordinates).
  let analysis_tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());

  // Caller's output timebase = 1/1000 (millisecond ticks). Chosen to
  // match WhisperX's seconds-as-floats with one decimal place's
  // worth of headroom; the JSON downconverts to seconds via
  // `tick / 1000.0` below. Picking ms (rather than the runner-mode
  // 1/48000) avoids tick-quantisation rounding when we display
  // boundaries to 3 decimal places.
  let ms_tb = Timebase::new(1, NonZeroU32::new(1_000).unwrap());

  let mut all_words: Vec<serde_json::Value> = Vec::new();
  let mut segments_aligned = 0usize;
  let mut segments_skipped_empty = 0usize;
  let mut segments_failed = 0usize;

  for (idx, seg) in segments.iter().enumerate() {
    if seg.text.trim().is_empty() {
      segments_skipped_empty += 1;
      continue;
    }

    // Mirror WhisperX: f1/f2 are 16 kHz sample indices over the
    // segment's `[t1, t2)` window. Clamp to the audio length
    // defensively (segment metadata can occasionally over-shoot the
    // clip end by a few samples on the very last segment).
    let f1 = (seg.start_s * 16_000.0).max(0.0) as usize;
    let f2_raw = (seg.end_s * 16_000.0).max(0.0) as usize;
    let f2 = f2_raw.min(total_samples);
    if f1 >= f2 {
      // Empty / pathological segment (start >= end after clamping,
      // or completely past the clip). Skip without erroring.
      segments_skipped_empty += 1;
      continue;
    }
    let segment_samples = &samples[f1..f2];

    // Single sub-segment covering the segment's full slice in
    // chunk-local 16 kHz coordinates. Same trick the previous
    // whole-clip path used; the aligner needs at least one VAD-style
    // sub-segment to drive its silence mask.
    let sub_segments = vec![TimeRange::new(
      0,
      segment_samples.len() as i64,
      analysis_tb,
    )];

    // `chunk_first_sample_in_stream = f1` so the
    // `samples_to_output_range` closure sees stream-coordinate
    // sample indices when the aligner converts wav2vec2 frame
    // indices back. This is exactly WhisperX's `t1` anchor:
    // `word.start_seconds = char_seg.start * (duration / (T-1)) + t1`.
    let sams_to_out = move |start: u64, end: u64| -> TimeRange {
      // 16 kHz samples → ms ticks: floor(sample * 1000 / 16000).
      TimeRange::new(
        (start as i64) * 1_000 / 16_000,
        (end as i64) * 1_000 / 16_000,
        ms_tb,
      )
    };

    let result = match aligner.align_chunk(
      segment_samples,
      &sub_segments,
      &seg.text,
      f1 as u64,
      sams_to_out,
    ) {
      Ok(r) => r,
      Err(e) => {
        // A segment whose alignment fails (e.g. all-OOV text →
        // empty AlignmentResult, or the silence-mask wipes
        // everything). Skip its words, keep going.
        eprintln!(
          "[whispery-parity:inject] segment {idx} ({:.3}-{:.3}s, \
           {} chars) failed: {e:?}; skipping",
          seg.start_s,
          seg.end_s,
          seg.text.len()
        );
        segments_failed += 1;
        continue;
      }
    };

    let mut seg_word_count = 0usize;
    for w in result.words() {
      let r = w.range();
      let start_s = r.start_pts() as f64 / 1_000.0;
      let end_s = r.end_pts() as f64 / 1_000.0;
      all_words.push(json!({
        "text": w.text(),
        "start_s": start_s,
        "end_s": end_s,
        "score": w.score(),
      }));
      seg_word_count += 1;
    }
    if seg_word_count == 0 {
      // Aligner returned successfully but produced zero words
      // (e.g. text was all-OOV or fully filtered by the silence
      // mask). Treat the same as a failure for diagnostic
      // bookkeeping; doesn't crash.
      segments_failed += 1;
    } else {
      segments_aligned += 1;
    }
  }

  eprintln!(
    "[whispery-parity:inject] aligned {} segments ({} skipped-empty, \
     {} failed) → {} output words",
    segments_aligned,
    segments_skipped_empty,
    segments_failed,
    all_words.len()
  );

  let payload = json!({
    "runner": "whispery",
    "mode": "inject",
    "clip_path": args.wav_path.display().to_string(),
    "clip_sha256": clip_sha256,
    "duration_s": duration_s,
    "transcript_count": segments_aligned,
    "injected_word_count": injected_words_total,
    "segments_total": segments.len(),
    "segments_aligned": segments_aligned,
    "segments_skipped_empty": segments_skipped_empty,
    "segments_failed": segments_failed,
    "words": all_words,
  });

  let serialized = serde_json::to_string_pretty(&payload)?;
  match args.out {
    Some(path) => {
      let mut f = fs::File::create(&path)
        .with_context(|| format!("create output {}", path.display()))?;
      f.write_all(serialized.as_bytes())?;
      f.write_all(b"\n")?;
      eprintln!(
        "[whispery-parity:inject] wrote {} aligned words ({} input) to {}",
        all_words.len(),
        injected_words_total,
        path.display()
      );
    }
    None => {
      println!("{serialized}");
    }
  }

  Ok(())
}
