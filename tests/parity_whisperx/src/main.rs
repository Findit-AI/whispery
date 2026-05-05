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
//!   3. Auto-detected `<whispery-crate>/models/` (canonical) or
//!      legacy `$CARGO_TARGET_DIR/whispery-test-fixtures/`
//!
//! `ORT_DYLIB_PATH` is consumed by `ort` itself in `load-dynamic` mode
//! (whispery's pinned configuration); the runner doesn't touch it.

use std::{
  fs,
  io::Write,
  num::NonZeroU32,
  path::{Path, PathBuf},
  sync::Once,
  time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use ffmpeg_next as ffmpeg;
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

/// Candidate directories the parity runner searches for ML model
/// files (whisper ggml, wav2vec2 ONNX, tokenizer.json). First-hit
/// wins. Order:
///
/// 1. `<whispery-crate>/models/` — the canonical in-tree location.
///    Resolved via `env!("CARGO_MANIFEST_DIR")` which expands at
///    compile time to the parity runner's own manifest dir
///    (`tests/parity_whisperx/`); two `..` segments up reach the
///    whispery crate root, then `models`.
/// 2. `$CARGO_TARGET_DIR/whispery-test-fixtures/` — legacy layout
///    used before the `models/` folder existed; kept for users
///    whose builds still cache there.
/// 3. `$CARGO_HOME/target/whispery-test-fixtures/` — same legacy
///    layout under the user's cargo home.
/// 4. `$HOME/.cargo/target/whispery-test-fixtures/` — final
///    legacy fallback for the macOS / Linux default.
fn models_dir() -> Option<PathBuf> {
  // 1. In-tree `<whispery-crate>/models/`.
  let in_tree = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("..")
    .join("..")
    .join("models");
  if in_tree.is_dir() {
    return Some(in_tree);
  }
  // 2-4. Legacy `target/whispery-test-fixtures/` locations.
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
  if let Some(dir) = models_dir() {
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

/// Idempotent guard for `ffmpeg::init()`. The runner has two `main`
/// entry-paths (regular + `--inject-from`) that both load audio; this
/// keeps each safe to call independently.
fn ffmpeg_init() -> Result<()> {
  static INIT: Once = Once::new();
  let mut init_err: Option<ffmpeg::Error> = None;
  INIT.call_once(|| {
    if let Err(e) = ffmpeg::init() {
      init_err = Some(e);
    }
  });
  if let Some(e) = init_err {
    Err(anyhow::anyhow!("ffmpeg::init failed: {e}"))
  } else {
    Ok(())
  }
}

/// Load an audio file as 16 kHz mono f32 via ffmpeg-next, mirroring
/// WhisperX's `load_audio` byte-for-byte: decode → resample to 16 kHz
/// mono `s16` (signed 16-bit, packed) → cast each sample to `f32` and
/// divide by exactly `32768.0`.
///
/// WhisperX's reference (`whisperx/audio.py:44-65`) shells out to:
///   `ffmpeg -nostdin -threads 0 -i FILE -f s16le -ac 1 -acodec pcm_s16le -ar 16000 -`
/// then runs `np.frombuffer(out, np.int16).astype(np.float32) / 32768.0`.
///
/// We use the `ffmpeg-next` bindings (libswresample under the hood)
/// rather than shelling out to keep the harness dependency-free at the
/// process level and to surface decoder/resampler errors as typed
/// errors instead of stderr text.
///
/// Returns `(samples, duration_s, sha256)` where `sha256` is computed
/// over the **little-endian f32 byte representation of the returned
/// samples**, i.e. the exact bytes whispery (and any production
/// caller) feeds into the model. Comparing this hash against the
/// WhisperX runner's own `clip_sha256` (computed by
/// `whisperx_runner.py` over the same float buffer) lets the harness
/// verify both runners saw byte-identical inputs.
fn read_audio_16k_mono_f32(path: &Path) -> Result<(Vec<f32>, f64, String)> {
  use ffmpeg::format::sample::{Sample, Type as SampleType};
  use ffmpeg::software::resampling::Context as Resampler;
  use ffmpeg::{ChannelLayout, codec::context::Context as CodecContext, frame, media};

  ffmpeg_init()?;

  let mut ictx = ffmpeg::format::input(path)
    .with_context(|| format!("open audio container at {}", path.display()))?;
  let stream = ictx
    .streams()
    .best(media::Type::Audio)
    .ok_or_else(|| anyhow::anyhow!("{}: no audio stream", path.display()))?;
  let stream_index = stream.index();

  // Build the decoder from the container's stream parameters.
  let codec_ctx = CodecContext::from_parameters(stream.parameters())
    .with_context(|| format!("decoder context for {}", path.display()))?;
  let mut decoder = codec_ctx
    .decoder()
    .audio()
    .with_context(|| format!("audio decoder for {}", path.display()))?;
  decoder
    .set_parameters(stream.parameters())
    .with_context(|| format!("decoder set_parameters for {}", path.display()))?;

  // WhisperX's exact ffmpeg invocation outputs `pcm_s16le` (signed
  // 16-bit little-endian, packed mono) at 16 kHz. ffmpeg-next's
  // `Sample::I16(Type::Packed)` is `AV_SAMPLE_FMT_S16` which is
  // little-endian on every platform we run on.
  const TARGET_RATE: u32 = 16_000;
  let target_format = Sample::I16(SampleType::Packed);
  let target_layout = ChannelLayout::MONO;

  // Resolve a non-empty source channel layout. PCM/WAV decoders
  // commonly emit frames with `ch_layout.order = UNSPEC` (only the
  // channel count is set); libswresample's `swr_alloc_set_opts2`
  // rejects that in FFuf 7+. Fall back to
  // `ChannelLayout::default(channels)` — exactly the recovery path
  // the indexer's `audio_service` runs when it hits
  // `Error::InputChanged`.
  let resolve_src_layout =
    |layout: ChannelLayout, channels: i32| -> ChannelLayout {
      if layout.is_empty() {
        ChannelLayout::default(channels)
      } else {
        layout
      }
    };

  let mut src_format = decoder.format();
  let mut src_rate = decoder.rate();
  let mut src_layout = resolve_src_layout(decoder.channel_layout(), decoder.channels() as i32);

  let build_resampler = |src_format,
                         src_layout,
                         src_rate|
   -> Result<Resampler> {
    Resampler::get(
      src_format,
      src_layout,
      src_rate,
      target_format,
      target_layout,
      TARGET_RATE,
    )
    .with_context(|| format!("init libswresample for {}", path.display()))
  };

  let mut resampler = build_resampler(src_format, src_layout, src_rate)?;

  let mut samples_f32: Vec<f32> = Vec::new();
  let mut decoded = frame::Audio::empty();

  // Push i16 samples from a packed-mono frame into `samples_f32`,
  // dividing by the literal `32768.0` exactly as WhisperX does.
  // `frame.plane::<i16>(0)` is only valid for packed-mono i16
  // (which is what we requested above), and exposes the first
  // `frame.samples()` interleaved samples.
  let push_resampled = |frame: &frame::Audio, dst: &mut Vec<f32>| {
    let n = frame.samples();
    if n == 0 {
      return;
    }
    let plane: &[i16] = frame.plane::<i16>(0);
    debug_assert!(plane.len() >= n);
    dst.reserve(n);
    for &s in &plane[..n] {
      dst.push(s as f32 / 32768.0_f32);
    }
  };

  // Run a decoded frame through the resampler. Handles
  // `InputChanged` / `OutputChanged` by rebuilding the resampler
  // against the new source params (mirrors `audio_service`'s
  // recovery path) — this is what makes the loader robust to
  // PCM/WAV decoders that set `ch_layout` only after the first
  // frame, instead of in `set_parameters`.
  //
  // We don't `flush()` between frames: that's reserved for
  // EOF, and the rate ratios libswresample picks for 16 kHz →
  // 16 kHz mono → mono are 1:1, so there's nothing buffered
  // between calls anyway. (Calling `flush` mid-stream returns
  // `Error::OutputChanged` because libswresample treats a
  // post-flush `run` as a stream restart.)
  let run_resample = |decoded: &frame::Audio,
                      resampler: &mut Resampler,
                      samples_f32: &mut Vec<f32>,
                      src_format: &mut Sample,
                      src_layout: &mut ChannelLayout,
                      src_rate: &mut u32|
   -> Result<()> {
    let mut resampled = frame::Audio::empty();
    match resampler.run(decoded, &mut resampled) {
      Ok(_) => {
        push_resampled(&resampled, samples_f32);
      }
      Err(ffmpeg::Error::InputChanged | ffmpeg::Error::OutputChanged) => {
        // Decoder gave us a frame whose params don't match what
        // the resampler was opened with. Re-derive params from
        // the actual frame, rebuild the resampler, retry once.
        *src_format = decoded.format();
        *src_layout = resolve_src_layout(
          decoded.channel_layout(),
          decoded.channels() as i32,
        );
        *src_rate = decoded.rate();
        *resampler = build_resampler(*src_format, *src_layout, *src_rate)?;
        let mut retried = frame::Audio::empty();
        resampler
          .run(decoded, &mut retried)
          .context("libswresample::run after rebuild")?;
        push_resampled(&retried, samples_f32);
      }
      Err(e) => return Err(anyhow::anyhow!("libswresample::run: {e}")),
    }
    Ok(())
  };

  // Helper: PCM/WAV decoders frequently emit frames with
  // `ch_layout.order = UNSPEC` even when the codec parameters (and
  // therefore the resampler we just opened) carry a concrete
  // layout. libswresample then trips `Error::InputChanged` on the
  // first frame because it sees a layout mismatch. Patch the
  // frame's layout to match what the resampler expects (the
  // resolved `src_layout` we computed above) before each `run`
  // call. This is an idempotent no-op for decoders that DO set
  // the layout explicitly.
  let fixup_frame_layout = |frame: &mut frame::Audio, src_layout: ChannelLayout| {
    if frame.channel_layout().is_empty() {
      frame.set_channel_layout(src_layout);
    }
  };

  // Pull packets from the input, send them to the decoder, then
  // drain any newly-available frames. Mirrors the standard
  // `transcode-audio` example pattern in `ffmpeg-next/examples/`.
  for (s, packet) in ictx.packets() {
    if s.index() != stream_index {
      continue;
    }
    decoder.send_packet(&packet).context("decoder.send_packet")?;
    while decoder.receive_frame(&mut decoded).is_ok() {
      fixup_frame_layout(&mut decoded, src_layout);
      run_resample(
        &decoded,
        &mut resampler,
        &mut samples_f32,
        &mut src_format,
        &mut src_layout,
        &mut src_rate,
      )?;
    }
  }
  decoder.send_eof().context("decoder.send_eof")?;
  while decoder.receive_frame(&mut decoded).is_ok() {
    fixup_frame_layout(&mut decoded, src_layout);
    run_resample(
      &decoded,
      &mut resampler,
      &mut samples_f32,
      &mut src_format,
      &mut src_layout,
      &mut src_rate,
    )?;
  }

  // Final flush of libswresample after the decoder is fully
  // drained. At 16 kHz → 16 kHz there's no resampling buffer to
  // empty, but for fixtures whose source rate ≠ 16 kHz (which we
  // resample down to 16 kHz) libswresample may have queued tail
  // samples that haven't been emitted yet.
  //
  // `Error::OutputChanged` here means "no buffered samples; the
  // implicit flush already happened", which is the common case
  // for the rate-1:1 PCM fixtures the harness uses today. We
  // treat it as a successful no-op rather than a hard error so
  // those fixtures load cleanly.
  loop {
    let mut tail = frame::Audio::empty();
    match resampler.flush(&mut tail) {
      Ok(_) => {
        if tail.samples() == 0 {
          break;
        }
        push_resampled(&tail, &mut samples_f32);
      }
      Err(ffmpeg::Error::OutputChanged) => break,
      Err(e) => {
        return Err(anyhow::anyhow!("libswresample::flush at EOF: {e}"));
      }
    }
  }

  if samples_f32.is_empty() {
    bail!(
      "{}: ffmpeg-next decoded zero samples; corrupt or empty audio?",
      path.display()
    );
  }

  let duration_s = samples_f32.len() as f64 / TARGET_RATE as f64;

  // Hash the f32 bytes (LE) the model will see. This is what makes
  // the parity check meaningful: WhisperX's runner emits the same
  // hash if and only if both pipelines decoded to byte-identical
  // float buffers. Comparing file-byte hashes (the previous
  // behaviour) couldn't catch loader-quantization divergences like
  // the `hound` f32-direct vs ffmpeg s16-quantized one this commit
  // closes.
  let mut hasher = Sha256::new();
  // Safety: `f32` is `Copy + 'static`, layout is well-defined as
  // 4 little-endian bytes per sample on every target this harness
  // ships to (macOS / Linux x86_64+aarch64). `bytemuck` would be
  // tidier but pulling in a dep for one cast isn't worth it here.
  let bytes = unsafe {
    std::slice::from_raw_parts(
      samples_f32.as_ptr() as *const u8,
      samples_f32.len() * std::mem::size_of::<f32>(),
    )
  };
  hasher.update(bytes);
  let sha = format!("{:x}", hasher.finalize());

  Ok((samples_f32, duration_s, sha))
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

  // Load + measure the audio. `clip_sha256` is computed over the
  // f32 bytes whispery hands to the model — comparing it to
  // `whisperx_runner.py`'s own clip_sha256 (computed over the same
  // float buffer) verifies both runners decoded the audio
  // byte-identically, which is what closes the audio-loader
  // divergence the README documented at parity-runner v1.
  let (samples, duration_s, clip_sha256) = read_audio_16k_mono_f32(&args.wav_path)?;
  let total_samples = samples.len() as u64;

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
  while let Some(t) = runner.poll_transcript().expect("poll_transcript") {
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
  while let Some((chunk_id, failure)) = runner.poll_error().expect("poll_error") {
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
  // Load the WhisperX JSON FIRST so we can read the top-level
  // `language` field and pick the matching wav2vec2 ONNX. Falling
  // back to "en" preserves the legacy behaviour (whisperx_runner.py
  // outputs without a `language` field, or with `language: en`).
  let injected: serde_json::Value = {
    let bytes = fs::read(&inject_path)
      .with_context(|| format!("read whisperX JSON {}", inject_path.display()))?;
    serde_json::from_slice(&bytes)
      .with_context(|| format!("parse whisperX JSON {}", inject_path.display()))?
  };
  let language_code = injected
    .get("language")
    .and_then(|v| v.as_str())
    .unwrap_or("en")
    .to_lowercase();
  let lang = whispery::Lang::from_iso639_1(&language_code);

  // Pick the aligner fixture for the detected language. The
  // English path stays where it always was; Ja/Zh use the
  // FinDIT-Studio-mirrored fixtures (build.rs writes
  // WHISPERY_W2V_{JA,ZH}_{MODEL,TOKENIZER} when
  // `WHISPERY_FETCH_W2V=1`). Future languages just add a match
  // arm + the corresponding option_env! pair.
  let (w2v_model, w2v_tokenizer) = match lang {
    whispery::Lang::Ja => {
      let model = option_env!("WHISPERY_W2V_JA_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
          models_dir()
            .map(|d| d.join("jonatasgrosman--wav2vec2-large-xlsr-53-japanese.onnx"))
        })
        .ok_or_else(|| anyhow::anyhow!(
          "no Japanese wav2vec2 fixture; run `WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 \
           cargo build --features alignment` to fetch from FinDIT-Studio"
        ))?;
      let tok = option_env!("WHISPERY_W2V_JA_TOKENIZER")
        .map(PathBuf::from)
        .or_else(|| {
          models_dir().map(|d| {
            d.join("jonatasgrosman--wav2vec2-large-xlsr-53-japanese-tokenizer.json")
          })
        })
        .ok_or_else(|| anyhow::anyhow!("no Japanese tokenizer fixture"))?;
      (model, tok)
    }
    whispery::Lang::Zh | whispery::Lang::Yue => {
      let model = option_env!("WHISPERY_W2V_ZH_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
          models_dir().map(|d| {
            d.join("jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn.onnx")
          })
        })
        .ok_or_else(|| anyhow::anyhow!(
          "no Chinese wav2vec2 fixture; run `WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 \
           cargo build --features alignment` to fetch from FinDIT-Studio"
        ))?;
      let tok = option_env!("WHISPERY_W2V_ZH_TOKENIZER")
        .map(PathBuf::from)
        .or_else(|| {
          models_dir().map(|d| {
            d.join(
              "jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn-tokenizer.json",
            )
          })
        })
        .ok_or_else(|| anyhow::anyhow!("no Chinese tokenizer fixture"))?;
      (model, tok)
    }
    _ => {
      // Fall through to English / CLI-supplied paths for any
      // language we don't have a multi-lang fixture for. Behaves
      // exactly like pre-multilingual parity-runner: the user can
      // still pass --w2v-model + --w2v-tokenizer for an arbitrary
      // language fixture.
      let model = resolve_model(
        args.w2v_model,
        "WAV2VEC2_ONNX_PATH",
        "wav2vec2-base-960h.onnx",
      )?;
      let tok = resolve_model(
        args.w2v_tokenizer,
        "WAV2VEC2_TOKENIZER_PATH",
        "wav2vec2-base-960h-tokenizer.json",
      )?;
      (model, tok)
    }
  };

  eprintln!(
    "[whispery-parity:inject] inject_from={} language={} w2v={} tok={}",
    inject_path.display(),
    lang.as_str(),
    w2v_model.display(),
    w2v_tokenizer.display()
  );
  let injected_words_total = injected
    .get("words")
    .and_then(|v| v.as_array())
    .map(|a| a.len())
    .unwrap_or(0);

  // Audio first — needed before we slice per-segment. SHA-256 is
  // over the f32 byte buffer (see `read_audio_16k_mono_f32` doc).
  let (samples, duration_s, clip_sha256) = read_audio_16k_mono_f32(&args.wav_path)?;
  let total_samples = samples.len();

  // Build the segments list. Prefer `segments[]` (the per-segment
  // path WhisperX itself uses); fall back to one synthetic segment
  // over the full clip if absent.
  struct InjectedSegment {
    start_s: f64,
    end_s: f64,
    text: String,
  }

  // The parity runner's default mode is `raw_asr_segments[]` —
  // the un-broken ASR segments WhisperX itself feeds to its own
  // `align()`. Using these gives apples-to-apples parity: both
  // implementations align the SAME audio + text + segment
  // anchors, so any disagreement is the algorithm itself.
  //
  // Earlier the default was per-sentence `segments[]` (the
  // POST-alignment WhisperX output, run through its
  // `PunktSentenceTokenizer` break-up). That mode hands whispery
  // dramatically smaller chunks than WhisperX itself aligned
  // against — by the time the segments[] view is built,
  // WhisperX has already done its own forced alignment, then
  // SPLIT into 10–25× more segments. Asking whispery to
  // re-align each tiny sub-chunk independently exercises the
  // wav2vec2 encoder on inputs WhisperX never inferenced on,
  // and amplifies ORT-vs-PyTorch numerical drift into per-word
  // 60–250 ms shifts. Median IoU dropped to 0.196 on
  // 03_dual_speaker.
  //
  // To opt back into per-sentence mode (e.g., to compare what
  // a downstream WhisperX consumer actually sees, or to debug
  // hallucinated-repetition splitting), set
  // `WHISPERY_PARITY_USE_PER_SENTENCE_SEGMENTS=1`. This is the
  // legacy `WHISPERY_PARITY_USE_RAW_SEGMENTS=0` behaviour
  // (which is also still honoured for backwards compat).
  let use_per_sentence = std::env::var("WHISPERY_PARITY_USE_PER_SENTENCE_SEGMENTS")
    .map(|v| v != "0" && !v.is_empty())
    .unwrap_or(false)
    || matches!(
      std::env::var("WHISPERY_PARITY_USE_RAW_SEGMENTS").as_deref(),
      Ok("0") | Ok("")
    );
  let use_raw_segments = !use_per_sentence;
  if use_raw_segments {
    // Fail loud if the inject JSON doesn't carry `raw_asr_segments[]`.
    // Falling back silently to per-sentence `segments[]` is the
    // bug that put 03_dual_speaker at median 0.196 / 70 below-0.5
    // (whispery re-aligning WhisperX's POST-alignment sub-chunks
    // independently amplifies the inherent ORT-vs-PyTorch
    // numerical drift). Caller must either fix the inject JSON
    // or set `WHISPERY_PARITY_USE_PER_SENTENCE_SEGMENTS=1` to
    // explicitly opt into per-sentence mode (with the
    // understanding that parity numbers degrade).
    if injected.get("raw_asr_segments").is_none() {
      bail!(
        "inject JSON has no `raw_asr_segments[]`. Default parity mode aligns \
         against the un-broken ASR segments WhisperX itself feeds to its own \
         `align()`; per-sentence mode (the legacy `segments[]` view) is a \
         downstream derivation that diverges from WhisperX's own forced \
         alignment by 60–250 ms per word. Either re-run \
         `whisperx_runner.py` so the JSON includes `raw_asr_segments[]`, or \
         set `WHISPERY_PARITY_USE_PER_SENTENCE_SEGMENTS=1` and accept the \
         degraded parity bar."
      );
    }
  }
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
  // ManagedTranscriber, no whisper.cpp. The normalizer is picked
  // by language via `default_normalizer_for(lang)`; for languages
  // whispery doesn't yet ship a built-in normalizer for (Arabic,
  // Korean, etc.) we fall back to EnglishNormalizer with a
  // diagnostic log.
  let normalizer = whispery::default_normalizer_for(&lang).unwrap_or_else(|| {
    eprintln!(
      "[whispery-parity:inject] no built-in normalizer for {:?}; \
       falling back to EnglishNormalizer (parity numbers will be \
       meaningless on non-Latin scripts)",
      lang
    );
    Box::new(EnglishNormalizer::new())
  });
  let mut aligner = Aligner::from_paths(lang.clone(), &w2v_model, &w2v_tokenizer, normalizer)
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
