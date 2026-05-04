//! `Aligner` — per-language wav2vec2 forced-alignment engine.

use alloc::string::String;
use core::time::Duration;
use std::path::Path;

use mediatime::TimeRange;
use ort::session::{RunOptions, Session};
use tokenizers::Tokenizer;

use crate::{
  core::AlignmentResult,
  runner::{RunnerError, aligner::normalizer::DynTextNormalizer},
  time::SAMPLE_RATE_HZ,
  types::{Lang, WorkFailure},
};

/// Per-language forced-alignment engine. Loads a wav2vec2 ONNX
/// model, its HuggingFace tokenizer, and the language's text
/// normaliser. Each instance is heavyweight (ONNX session +
/// tokenizer state); the [`crate::AlignmentSet`] registry keeps one
/// per registered language, gated behind `Mutex<Aligner>` so the
/// single alignment worker can drive any language without copying.
///
/// Fields are private; access is via getters per the findit-studio
/// convention.
///
/// **Concurrency.** `Aligner` is `Send` (every field is `Send`) but
/// not `Sync` (`ort::Session::run` requires `&mut self`). The
/// registry stores `Mutex<Aligner>` which collapses to a no-op lock
/// in the v1 single-worker case.
pub struct Aligner {
  session: Session,
  tokenizer: Tokenizer,
  language: Lang,
  normalizer: DynTextNormalizer,
  sample_rate: u32,
  hop_samples: u32,
  blank_token_id: u32,
  /// `<unk>` / `[UNK]` token id, when the tokenizer exposes one.
  /// `tokenize_with_word_map` uses this to reject out-of-vocab
  /// word tokens up-front rather than feeding `<unk>` ids into the
  /// CTC graph and silently producing garbage alignments.
  unk_token_id: Option<u32>,
  /// Whether the tokenizer's vocab covers ASCII uppercase but not
  /// lowercase (e.g., `wav2vec2-base-960h`). When true,
  /// tokenisation uppercases ASCII before encoding so the
  /// (lowercase-emitting) [`crate::EnglishNormalizer`] doesn't
  /// produce a stream of `<unk>`s on every English word.
  vocab_uppercase_only: bool,
  /// Tokenizer vocab size (including added tokens) captured at
  /// construction time. The model's ORT output `V` dimension
  /// must match this exactly — otherwise Viterbi reads
  /// posteriors from columns that don't correspond to the
  /// tokenizer's tokens, emitting believable but corrupt
  /// timings. Validated per-call in [`Self::align`] via
  /// [`validate_vocab_dim`].
  tokenizer_vocab_size: usize,
  /// Minimum `speech_emissions / total_emissions` ratio required
  /// for a word to survive the alignment composer's post-pass.
  /// Words whose CTC visit has too many masked frames drop —
  /// they would otherwise emit a high-confidence range over
  /// mostly-silent audio. Defaults to
  /// [`compose::DEFAULT_MIN_SPEECH_COVERAGE`](crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE).
  /// Override via [`Self::with_min_speech_coverage`] when
  /// stricter or more permissive thresholds are needed (e.g.,
  /// tighter for closed-caption use, looser for noisy audio).
  min_speech_coverage: f32,
  /// Maximum allowed contiguous silent run inside a word's
  /// `[start_frame, end_frame)` bounding span. Defaults to
  /// [`compose::DEFAULT_MAX_INTRA_SILENT_RUN`](crate::runner::aligner::algorithm::compose::DEFAULT_MAX_INTRA_SILENT_RUN).
  /// Stored as a `Duration` so the threshold stays meaningful
  /// across alignment models with different frame strides;
  /// converted to encoder frames at align time using
  /// `hop_samples` and the 16 kHz analysis timebase.
  max_intra_silent_run: Duration,
}

impl Aligner {
  /// Construct from on-disk paths.
  ///
  /// `model_path` points to a wav2vec2 ONNX export with input
  /// shape `(1, T)` (raw f32 samples) and output shape `(1, T',
  /// V)` (logits). `tokenizer_path` points to the matching
  /// HuggingFace `tokenizer.json`.
  ///
  /// The blank-token id is read from the tokenizer's `<pad>` /
  /// `[PAD]` entry (the standard wav2vec2 convention). If the
  /// model uses a non-standard blank token, override via a
  /// future `with_blank_token_id` method (not in v1 scope).
  ///
  /// `sample_rate` defaults to 16 000 (wav2vec2's universal
  /// pre-processing target). `hop_samples` defaults to 320 (=
  /// 20 ms @ 16 kHz, the wav2vec2-base/large convention).
  /// Custom-strided models may pass overrides via a future
  /// builder.
  ///
  /// Returns [`RunnerError::AlignerLoad`] on any I/O or parse
  /// failure.
  pub fn from_paths(
    language: Lang,
    model_path: &Path,
    tokenizer_path: &Path,
    normalizer: DynTextNormalizer,
  ) -> Result<Self, RunnerError> {
    let session = Session::builder()
      .map_err(|e| RunnerError::AlignerLoad {
        message: alloc::format!("Session::builder failed: {e:?}"),
      })?
      .commit_from_file(model_path)
      .map_err(|e| RunnerError::AlignerLoad {
        message: alloc::format!("commit_from_file({}) failed: {e:?}", model_path.display()),
      })?;
    let tokenizer = load_tokenizer_with_compat(tokenizer_path)?;

    let blank_token_id =
      detect_blank_token_id(&tokenizer).ok_or_else(|| RunnerError::AlignerLoad {
        message: String::from(
          "tokenizer has no <pad> / [PAD] entry; cannot determine CTC blank token",
        ),
      })?;
    let unk_token_id = tokenizer
      .token_to_id("<unk>")
      .or_else(|| tokenizer.token_to_id("[UNK]"));
    // wav2vec2-base-960h's vocab is uppercase-only; en/de/fr CTC
    // checkpoints typically follow the same convention. Detect by
    // probing a single ASCII letter pair — sufficient because the
    // vocab either has both cases (mixed-case alphabet) or one
    // (case-folded alphabet).
    let vocab_uppercase_only =
      tokenizer.token_to_id("A").is_some() && tokenizer.token_to_id("a").is_none();

    // When the normaliser declares `use_word_delimiter == true`
    // (the English-shape default), the tokenizer MUST expose a
    // `|` token. See [`validate_word_delimiter_present`] for the
    // rationale.
    validate_word_delimiter_present(&tokenizer, normalizer.use_word_delimiter())?;

    // Snapshot the tokenizer's vocab size (including added
    // tokens) so per-align validation can reject ORT outputs
    // whose `V` dim doesn't match — otherwise the per-token id
    // checks in `ctc_viterbi` would pass whenever the chunk's
    // tokens happen to fit, then read posteriors from
    // mis-aligned columns.
    let tokenizer_vocab_size = tokenizer.get_vocab_size(true);

    Ok(Self {
      session,
      tokenizer,
      language,
      normalizer,
      sample_rate: 16_000,
      hop_samples: 320,
      blank_token_id,
      unk_token_id,
      vocab_uppercase_only,
      tokenizer_vocab_size,
      min_speech_coverage: crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE,
      max_intra_silent_run:
        crate::runner::aligner::algorithm::compose::DEFAULT_MAX_INTRA_SILENT_RUN,
    })
  }

  /// Detected language for this aligner.
  pub const fn language(&self) -> &Lang {
    &self.language
  }

  /// Audio sample rate the model expects (16 kHz for wav2vec2).
  pub const fn sample_rate(&self) -> u32 {
    self.sample_rate
  }

  /// Frame stride in 16 kHz samples (320 = 20 ms by default).
  pub const fn hop_samples(&self) -> u32 {
    self.hop_samples
  }

  /// CTC blank-token id detected at construction time.
  pub const fn blank_token_id(&self) -> u32 {
    self.blank_token_id
  }

  /// Set [`Self::sample_rate`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. A zero sample rate would make the
  /// frame-timebase math collapse and downstream PTS conversions
  /// produce nonsense ranges; failing fast at the construction
  /// boundary is correct.
  pub const fn set_sample_rate(&mut self, value: u32) {
    assert!(value > 0, "sample_rate must be > 0");
    self.sample_rate = value;
  }

  /// Builder-style override for [`Self::sample_rate`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See [`Self::set_sample_rate`].
  pub const fn with_sample_rate(mut self, value: u32) -> Self {
    assert!(value > 0, "sample_rate must be > 0");
    self.sample_rate = value;
    self
  }

  /// Set [`Self::hop_samples`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. A zero hop would collapse the
  /// frame→sample conversion in `compose_words` (every word
  /// would land at the chunk's first sample), corrupting word
  /// timings silently. Fail fast.
  pub const fn set_hop_samples(&mut self, value: u32) {
    assert!(value > 0, "hop_samples must be > 0");
    self.hop_samples = value;
  }

  /// Builder-style override for [`Self::hop_samples`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See [`Self::set_hop_samples`].
  pub const fn with_hop_samples(mut self, value: u32) -> Self {
    assert!(value > 0, "hop_samples must be > 0");
    self.hop_samples = value;
    self
  }

  /// Minimum `speech_emissions / total_emissions` ratio
  /// required for a word to survive the alignment composer's
  /// post-pass. See the field doc on [`Self`] for the
  /// motivation. Default: `0.5`
  /// (`DEFAULT_MIN_SPEECH_COVERAGE`).
  pub const fn min_speech_coverage(&self) -> f32 {
    self.min_speech_coverage
  }

  /// Set [`Self::min_speech_coverage`]. Values outside `[0.0,
  /// 1.0]` are stored verbatim — out-of-range values
  /// effectively disable the coverage check (`< 0.0` always
  /// passes; `> 1.0` always fails).
  pub const fn set_min_speech_coverage(&mut self, value: f32) {
    self.min_speech_coverage = value;
  }

  /// Builder-style override for [`Self::min_speech_coverage`].
  pub const fn with_min_speech_coverage(mut self, value: f32) -> Self {
    self.min_speech_coverage = value;
    self
  }

  /// Maximum allowed contiguous silent run inside a word's
  /// bounding span. Default: 80 ms
  /// (`DEFAULT_MAX_INTRA_SILENT_RUN`). See the field doc on
  /// [`Self`] for the rationale.
  pub const fn max_intra_silent_run(&self) -> Duration {
    self.max_intra_silent_run
  }

  /// Set [`Self::max_intra_silent_run`].
  pub const fn set_max_intra_silent_run(&mut self, value: Duration) {
    self.max_intra_silent_run = value;
  }

  /// Builder-style override for [`Self::max_intra_silent_run`].
  pub const fn with_max_intra_silent_run(mut self, value: Duration) -> Self {
    self.max_intra_silent_run = value;
    self
  }

  /// Public alignment entrypoint for parity / benchmarking
  /// tooling that drives the pipeline without
  /// [`crate::ManagedTranscriber`]. Constructs default
  /// [`RunOptions`] + abort flag internally and delegates to the
  /// crate-private [`Self::align`].
  ///
  /// Inputs match [`Self::align`] minus the
  /// `abort_flag` / `run_options` infrastructure. See that
  /// method's doc-comment for argument semantics.
  ///
  /// Returns [`WorkFailure::AlignmentFailed`] with kind
  /// [`crate::types::AlignmentFailureKind::ModelInferenceFailed`]
  /// if [`RunOptions::new`] fails (rare; ORT initialisation
  /// hiccup) — same shape as the worker's internal mapping so
  /// downstream error handling stays uniform across entrypoints.
  pub fn align_chunk<F>(
    &mut self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    chunk_first_sample_in_stream: u64,
    samples_to_output_range: F,
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    use crate::types::AlignmentFailureKind;

    let abort_flag = core::sync::atomic::AtomicBool::new(false);
    let run_options = RunOptions::new().map_err(|e| WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("RunOptions::new failed: {e:?}"),
      language: self.language.clone(),
    })?;
    self.align(
      samples,
      sub_segments,
      text,
      chunk_first_sample_in_stream,
      samples_to_output_range,
      &abort_flag,
      &run_options,
    )
  }

  /// Crate-private alignment entrypoint.
  ///
  /// Inputs:
  /// - `samples`: the chunk's 16 kHz f32 mono audio.
  /// - `sub_segments`: VAD sub-segments **in chunk-local 1/16000
  ///   timebase**, so `start_pts()` / `end_pts()` are chunk-local
  ///   16 kHz sample indices in `[0, samples.len()]`. The silence
  ///   mask in step 0 (and `build_speech_frames` in step 7) reads
  ///   the PTS values directly as sample offsets — they are NOT
  ///   in any output / wall-clock timebase. Out-of-range PTS get
  ///   clamped to `[0, samples.len()]`. The internal worker dispatch
  ///   path (`managed_transcriber.rs`) converts output-timebase
  ///   sub-segments to this form before calling. External callers
  ///   that drive `align_chunk` (parity / benchmarking tooling)
  ///   must respect the same contract; a non-1/16000 timebase
  ///   trips a `debug_assert`.
  /// - `text`: Whisper's transcribed text.
  /// - `chunk_first_sample_in_stream`: the chunk's first 16 kHz
  ///   sample index in stream coordinates (used to convert
  ///   wav2vec2 frame indices back to stream sample indices).
  /// - `samples_to_output_range`: callback bridging stream sample
  ///   indices to output-timebase `TimeRange`s. The core's
  ///   `SampleBuffer::samples_to_output_range` is `pub(crate)`;
  ///   the worker constructs a closure over it.
  pub(crate) fn align<F>(
    &mut self,
    samples: &[f32],
    sub_segments: &[TimeRange],
    text: &str,
    chunk_first_sample_in_stream: u64,
    samples_to_output_range: F,
    abort_flag: &core::sync::atomic::AtomicBool,
    run_options: &RunOptions,
  ) -> Result<AlignmentResult, WorkFailure>
  where
    F: Fn(u64, u64) -> TimeRange,
  {
    use core::sync::atomic::Ordering;

    use crate::{
      runner::aligner::algorithm::{
        compose::{build_speech_frames, compose_words},
        encode::encode_log_softmax,
        tokenize::tokenize_with_word_map,
        trellis_beam::align_to_word_segments,
      },
      types::AlignmentFailureKind,
    };

    // Helper: produce a WorkerHangTimeout when the watchdog has
    // already flipped abort_flag. `elapsed` is left as ZERO here;
    // `run_one_alignment` (the worker) holds the canonical
    // Instant::now() reference and overwrites unconditionally
    // when abort_flag is set, so the value here is purely
    // diagnostic. We keep the in-`align` checks anyway so a long
    // encode (1+ seconds for 30 s of audio) bails out at the
    // next stage boundary instead of compounding the hang by
    // running CTC + Viterbi + compose on probably-bogus data.
    let timed_out = || -> WorkFailure {
      WorkFailure::WorkerHangTimeout {
        kind: crate::types::WorkerKind::Alignment,
        elapsed: core::time::Duration::ZERO,
      }
    };

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Sub-400-sample early-out. wav2vec2's first conv layer has
    // a 400-sample receptive field, so for shorter inputs the
    // encode path pads to 400 below before invoking ORT. The
    // encoder then emits `T` frames whose stride is governed by
    // the PADDED length, not the original `samples.len()`.
    // Downstream `samples_per_frame = samples.len() / (T-1)`
    // (the WhisperX effective ratio) uses the original length,
    // so padded-tensor frames get projected onto the original
    // audio at the wrong stride — `compose_words` emits word
    // ranges shrunk by `samples.len() / padded.len()` and
    // `build_speech_frames` classifies frames against intervals
    // that don't match the encoder's own view. Codex round-25
    // flagged this; the simplest safe response is to refuse
    // alignment for chunks that can't satisfy the receptive
    // field directly. At 25 ms or less, a chunk cannot realistically
    // contain a meaningful CTC path through any non-trivial
    // transcript anyway. Match the empty-text / empty-tokens
    // short-circuits below by returning an empty
    // `AlignmentResult` rather than `Err` — alignment becoming
    // a no-op on degenerate input, not a data-loss path.
    const WAV2VEC2_MIN_SAMPLES: usize = 400;
    if samples.len() < WAV2VEC2_MIN_SAMPLES {
      return Ok(AlignmentResult::new(alloc::vec::Vec::new()));
    }

    // Step 0: silence-aware preprocessing.
    //
    // `sub_segments` are in chunk-local 1/16000 timebase per the
    // method-level contract — `start_pts()` / `end_pts()` are
    // chunk-local sample indices, NOT output-timebase ticks.
    // Build a per-sample boolean speech mask for the silence-aware
    // normaliser; once that returns the buffer it's already been
    // (1) normalised over speech samples only and (2) zeroed at
    // non-speech positions, so the silence-mask invariant survives
    // all the way to ORT. A previous two-step approach
    // (`build_masked_samples` then a non-mask-aware normalise
    // inside `encode_log_softmax`) had the intermediate zeros
    // mean-shifted by the normaliser, so masked regions became
    // `(0 - mean) / std` ≠ 0 by the time they reached the model.
    let speech_mask = build_speech_mask(samples.len(), sub_segments, &self.language)?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 1: normalise.
    //
    // `NormalizationError::EmptyText` (punctuation-only or
    // whitespace-only ASR output) is *not* an error here — it
    // mirrors the empty-tokens short-circuit below. Returning
    // `Ok(empty AlignmentResult)` lets the cached ASR
    // transcript surface as `Transcript { text, words: [] }`
    // instead of `Event::Error`. Otherwise this would be a
    // data-loss path that contradicts the `AlignmentResult`
    // contract.
    let normalized = match self.normalizer.normalize(text) {
      Ok(nt) => nt,
      Err(crate::runner::aligner::normalizer::NormalizationError::EmptyText) => {
        return Ok(AlignmentResult::new(alloc::vec::Vec::new()));
      }
      Err(crate::runner::aligner::normalizer::NormalizationError::RuleFailed { detail }) => {
        return Err(WorkFailure::AlignmentFailed {
          kind: AlignmentFailureKind::NormalizationFailed,
          message: detail,
          language: self.language.clone(),
        });
      }
    };

    let n_words = normalized.original_words().len();

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 2: tokenise with word index map. The normaliser's
    // `use_word_delimiter` policy gates inter-word `|` insertion
    // (true for word-segmented English; false for char-segmented
    // Chinese/Japanese where whitespace is an indexing artefact).
    // `vocab_uppercase_only` triggers ASCII case projection so a
    // lowercase normaliser doesn't feed <unk>s into a vocab like
    // wav2vec2-base-960h's. `unk_token_id` is the per-character
    // skip target.
    let tokenized = tokenize_with_word_map(
      &self.tokenizer,
      normalized.normalized(),
      n_words,
      self.normalizer.use_word_delimiter(),
      self.vocab_uppercase_only,
      self.unk_token_id,
      normalized.wildcard_chars_per_word(),
      &self.language,
    )?;

    // No-alignable-tokens short-circuit: a chunk like `"1000"`
    // against the uppercase-only English vocab legitimately
    // produces zero in-vocab tokens (every digit is <unk>).
    // Returning an empty `AlignmentResult` makes the dispatch
    // emit the cached ASR transcript with `words: []` instead
    // of converting it into `Event::Error` — alignment becoming
    // optional, not a data-loss path.
    if tokenized.token_ids.is_empty() {
      return Ok(AlignmentResult::new(alloc::vec::Vec::new()));
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 3-4: encode + log-softmax. The alignment worker's
    // watchdog calls `RunOptions::terminate()` on
    // [`crate::runner::alignment_pool::AlignWorkItem::align_timeout`];
    // ORT then returns from `Session::run_with_options` with an
    // error rather than blocking the worker indefinitely. The
    // post-encode `abort_flag` check below catches the watchdog's
    // race-window cases (terminate fired, run already returning
    // success).
    //
    // **WhisperX parity:** WhisperX's `alignment.py` feeds the
    // **raw** waveform to `Wav2Vec2ForCTC.forward` (line 255 — the
    // HF processor's mean/var normalisation step is skipped). The
    // wav2vec2-base architecture has GroupNorm on the first conv
    // layer so it tolerates unnormalised audio in `[-1, 1]`, but
    // the resulting emissions differ materially from the
    // processor-normalised path: per-frame argmax disagrees on
    // ~14 % of frames over a 24 s segment, and individual blank
    // log-probabilities differ by up to 5+ nats. To match the
    // de facto reference's frame-level timing decisions we drop
    // the pre-encode mean/var normalisation and feed the silence-
    // masked but otherwise raw audio buffer to ORT. The model's
    // GroupNorm absorbs the global scale; the silence-mask
    // contract — `false` positions → exactly `0.0_f32` going
    // into the encoder — is preserved by zeroing non-speech
    // samples before handoff.
    let normalized_samples: alloc::vec::Vec<f32> = samples
      .iter()
      .zip(speech_mask.iter())
      .map(|(&s, &is_speech)| if is_speech { s } else { 0.0_f32 })
      .collect();

    // wav2vec2's CNN front-end has a minimum input length (the
    // receptive field of the first stride-conv) of 400 samples
    // at 16 kHz. WhisperX's `align()` pads with zeros to 400 if
    // the slice is shorter (`alignment.py:243-247`). Without
    // this padding, the model's first conv produces a degenerate
    // output for very short segments — typical for a 1-2 word
    // segment after Whisper splits on a brief utterance — and
    // the encoder either errors out or emits T=0 frames. We
    // append zeros to the silence-mask-normalised buffer; the
    // padded samples are zero (silent) by construction, so the
    // existing speech-mask doesn't need updating to track them.
    let padded_samples: alloc::borrow::Cow<'_, [f32]> = if normalized_samples.len() < 400 {
      let mut buf = alloc::vec::Vec::with_capacity(400);
      buf.extend_from_slice(&normalized_samples);
      buf.resize(400, 0.0_f32);
      alloc::borrow::Cow::Owned(buf)
    } else {
      alloc::borrow::Cow::Borrowed(&normalized_samples[..])
    };
    let log_probs = encode_log_softmax(
      &mut self.session,
      &padded_samples,
      run_options,
      &self.language,
    )?;

    // Diagnostic: when the parity harness sets
    // `WHISPERY_PARITY_DUMP_TRELLIS` to a directory, write a
    // per-segment `wy_seg<N>.emission.bin` and (after the trellis
    // step below) `wy_seg<N>.trellis.bin` plus a
    // `wy_seg<N>.tokens.json` companion. The `<N>` counter is a
    // monotonic integer drawn from a process-global atomic so each
    // alignment call against the harness gets a unique slot. The
    // hot path bails out cheaply when the env var isn't set; this
    // adds a single env lookup per `align_chunk` and is gated on
    // an opt-in env var, so production runs are unaffected.
    //
    // Lives behind the `parity-dump-emission` feature so the env
    // hook + `serde_json` formatter don't compile into the prod
    // `Aligner` for downstream consumers.
    #[cfg(feature = "parity-dump-emission")]
    {
      use core::sync::atomic::AtomicUsize;
      static SEG_COUNTER: AtomicUsize = AtomicUsize::new(0);
      if let Ok(dir) = std::env::var("WHISPERY_PARITY_DUMP_TRELLIS") {
        let n = SEG_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir_path);
        let em_path = dir_path.join(alloc::format!("wy_seg{n}.emission.bin"));
        if let Ok(mut f) = std::fs::File::create(&em_path) {
          use std::io::Write;
          let _ = f.write_all(&(log_probs.t as u32).to_le_bytes());
          let _ = f.write_all(&(log_probs.v as u32).to_le_bytes());
          // Write as f32 LE one cell at a time. The dump path is
          // diagnostic-only; the per-cell `to_le_bytes` is acceptable
          // overhead for the few-K-cells * once-per-segment frequency.
          let mut buf: alloc::vec::Vec<u8> =
            alloc::vec::Vec::with_capacity(log_probs.data.len() * 4);
          for v in &log_probs.data {
            buf.extend_from_slice(&v.to_le_bytes());
          }
          let _ = f.write_all(&buf);
        }
        let tok_path = dir_path.join(alloc::format!("wy_seg{n}.tokens.json"));
        if let Ok(mut f) = std::fs::File::create(&tok_path) {
          use std::io::Write;
          // Hand-format JSON to avoid the serde_json prod dep.
          let mut payload = alloc::format!("{{\"blank_id\":{},\"tokens\":[", self.blank_token_id);
          for (i, t) in tokenized.token_ids.iter().enumerate() {
            if i > 0 {
              payload.push(',');
            }
            payload.push_str(&alloc::format!("{}", t));
          }
          payload.push_str(&alloc::format!(
            "],\"n_samples\":{},\"T\":{},\"V\":{}}}",
            padded_samples.len(),
            log_probs.t,
            log_probs.v
          ));
          let _ = f.write_all(payload.as_bytes());
        }
      }
    }

    // Two-sided stride check: the encoded time `T * hop_samples`
    // must lie within `samples.len() ± 2*hop_samples`. Catches
    // both stride-too-small (T*hop overshoots — `compose_words`
    // would emit ranges past the chunk's audio) and
    // stride-too-large (T*hop undershoots — `compose_words`
    // would compress every word into the first portion of the
    // chunk). Fatal: the only recovery is fixing the model /
    // `hop_samples` config, not retrying.
    crate::runner::aligner::algorithm::encode::validate_stride_extent(
      log_probs.t,
      self.hop_samples,
      samples.len(),
      &self.language,
    )?;

    // Vocab-axis check: model output `V` must equal the
    // tokenizer's vocab size. A mismatch (e.g. wrong CTC head
    // wired into the export, or a hidden-states tensor leaked
    // out as the logits output) would otherwise let the
    // per-token id check inside `ctc_viterbi` pass whenever
    // the chunk's token ids happened to fit, then read
    // posteriors from columns that don't correspond to the
    // tokenizer's tokens — emitting plausible but corrupt
    // timings.
    crate::runner::aligner::algorithm::encode::validate_vocab_dim(
      log_probs.v,
      self.tokenizer_vocab_size,
      &self.language,
    )?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 5-6: WhisperX-bit-exact trellis + beam-search
    // backtrack + char→word grouping. Returns
    // `Vec<WordSegment>` directly; the legacy
    // `state_per_frame` lattice encoding is gone. Same
    // cooperative-cancellation contract as before — the DP
    // checks `abort_flag` periodically so a hallucinated long
    // token sequence can't run past `align_timeout` and starve
    // every chunk queued behind it on the single worker.
    let word_segments = align_to_word_segments(
      &log_probs,
      &tokenized.token_ids,
      &tokenized.word_idx_per_token,
      tokenized.separator_token_id,
      self.blank_token_id,
      abort_flag,
      &self.language,
    )?;

    // Companion to the emission dump above: rebuild the trellis
    // diagnostically and dump it. We don't capture it from
    // `align_to_word_segments` to avoid leaking the trellis allocation
    // into a prod-facing return type. Recomputation is O(T*N) and
    // only fires when the env var is set on a parity harness run.
    #[cfg(feature = "parity-dump-emission")]
    {
      use core::sync::atomic::AtomicUsize;
      static TRELLIS_COUNTER: AtomicUsize = AtomicUsize::new(0);
      if let Ok(dir) = std::env::var("WHISPERY_PARITY_DUMP_TRELLIS") {
        let n = TRELLIS_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir_path = std::path::PathBuf::from(dir);
        let trellis = crate::runner::aligner::algorithm::trellis_beam::get_trellis(
          &log_probs,
          &tokenized.token_ids,
          self.blank_token_id,
          abort_flag,
          &self.language,
        );
        if let Ok(trellis) = trellis {
          let path = dir_path.join(alloc::format!("wy_seg{n}.trellis.bin"));
          if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            let _ = f.write_all(&(log_probs.t as u32).to_le_bytes());
            let _ = f.write_all(&(tokenized.token_ids.len() as u32).to_le_bytes());
            let mut buf: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(trellis.len() * 4);
            for v in &trellis {
              buf.extend_from_slice(&v.to_le_bytes());
            }
            let _ = f.write_all(&buf);
          }
        }
      }
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 7-9: per-word state + surface-form recovery. The
    // speech-frame mask comes from the same `sub_segments` the
    // silence-mask step zeroed, so words whose CTC-forced
    // assignment lands entirely inside masked silence drop from
    // the result rather than emit fabricated timings.
    //
    // `samples_per_frame` is computed once and passed to both
    // `build_speech_frames` (which uses it to map encoder frames
    // back to sample ranges for VAD overlap classification) and
    // `compose_words` (which uses the same mapping to emit word
    // timestamps). Codex round-21 flagged the previous mismatch
    // — `build_speech_frames` used nominal `hop_samples` while
    // `compose_words` used the WhisperX effective ratio
    // `n_samples / (T - 1)`. On a 30 s chunk where wav2vec2
    // truncates one frame (T=1499 vs nominal 1500), the drift
    // hits ~40 ms by the chunk end, enough to misclassify
    // boundary words.
    let samples_per_frame = crate::runner::aligner::algorithm::compose::effective_samples_per_frame(
      samples.len() as u64,
      log_probs.t,
      self.hop_samples,
    );
    let speech_frames = build_speech_frames(
      log_probs.t,
      samples_per_frame,
      samples.len() as u64,
      sub_segments,
    );
    Ok(compose_words(
      &word_segments,
      normalized.original_words(),
      &speech_frames,
      chunk_first_sample_in_stream,
      self.hop_samples,
      // Pass the chunk's input audio length so word ranges
      // get clamped to the chunk boundary (the stride
      // validator's 2-frame overshoot tolerance can't leak
      // into emitted word timestamps).
      samples.len() as u64,
      log_probs.t,
      samples_to_output_range,
      self.min_speech_coverage,
      self.max_intra_silent_run,
    ))
  }
}

/// Build a per-sample boolean speech mask for `Aligner::align`'s
/// step 0. `sub_segments` are in chunk-local 1/16000 timebase per
/// the `align` contract; `start_pts` / `end_pts` are sample indices
/// that get clamped to `[0, n_samples]` via i64 saturation.
///
/// Two contract details worth highlighting:
///
/// 1. A non-1/16000 timebase fails the chunk in BOTH debug and
///    release with a `WorkFailure::AlignmentFailed`. Previously
///    the check was a `debug_assert!` only, so release builds
///    silently misinterpreted (e.g.) a millisecond-timebase PTS
///    as a sample index, masking the wrong samples and producing
///    plausible-but-wrong word alignments. Internal callers
///    always wrap in 1/16000 (`managed_transcriber.rs`); external
///    callers of `align_chunk` are documented to do the same and
///    now hit a clear runtime error if they don't.
/// 2. `i64 → usize` is via `.clamp(0, n_samples_i64) as usize`, NOT
///    `as u64 as usize`. The old cast wrapped negative `start_pts`
///    to a huge u64, which then got clamped to `n_samples` and the
///    `if end > start` guard dropped the sub-segment entirely.
///    Negative-overlap ranges (sub-segment whose head extends past
///    the chunk start) now get their head trimmed and their tail
///    masked, matching `compose::build_speech_frames`'s `.max(0)`.
fn build_speech_mask(
  n_samples: usize,
  sub_segments: &[TimeRange],
  language: &Lang,
) -> Result<alloc::vec::Vec<bool>, WorkFailure> {
  use crate::types::AlignmentFailureKind;
  let mut mask = alloc::vec![false; n_samples];
  let n_samples_i64 = n_samples as i64;
  for &seg in sub_segments {
    if seg.timebase().num() != 1 || seg.timebase().den().get() != SAMPLE_RATE_HZ {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "Aligner::align expects sub_segments in chunk-local 1/{} timebase, \
           got {}/{}; caller passed sub_segments in the wrong timebase \
           (samples will not match audio if we proceed).",
          SAMPLE_RATE_HZ,
          seg.timebase().num(),
          seg.timebase().den().get(),
        ),
        language: language.clone(),
      });
    }
    let start = seg.start_pts().clamp(0, n_samples_i64) as usize;
    let end = seg.end_pts().clamp(0, n_samples_i64) as usize;
    if end > start {
      for slot in &mut mask[start..end] {
        *slot = true;
      }
    }
  }
  Ok(mask)
}

/// Read the CTC blank-token id from a HuggingFace tokenizer.
fn detect_blank_token_id(tok: &Tokenizer) -> Option<u32> {
  // Standard wav2vec2 convention: pad token == CTC blank.
  if let Some(id) = tok.token_to_id("<pad>") {
    return Some(id);
  }
  if let Some(id) = tok.token_to_id("[PAD]") {
    return Some(id);
  }
  if let Some(id) = tok.token_to_id("<blank>") {
    return Some(id);
  }
  None
}

/// Default per-job timeout for one chunk's alignment. Surfaced
/// via the `worker_timeouts(_, align)` builder hook.
pub(crate) const DEFAULT_ALIGN_TIMEOUT: Duration = Duration::from_secs(30);

/// Validate that the tokenizer exposes the wav2vec2 `|`
/// word-delimiter token whenever the normaliser declared
/// `use_word_delimiter == true`.
///
/// Without this check, a missing `|` token slips through silently
/// — `tokenize_with_word_map` would simply emit no inter-word
/// delimiter, glueing adjacent words together in the CTC graph.
/// Word timings would then be plausible but wrong with no
/// configuration error visible to the caller.
///
/// Char-segmented normalisers (`use_word_delimiter == false`)
/// don't need the delimiter and pass through.
///
/// Pulled out as a free function so unit tests can exercise it
/// against an in-memory tokenizer without spinning up ORT.
fn validate_word_delimiter_present(
  tokenizer: &Tokenizer,
  use_word_delimiter: bool,
) -> Result<(), RunnerError> {
  if !use_word_delimiter {
    return Ok(());
  }
  if tokenizer.token_to_id("|").is_some() {
    return Ok(());
  }
  Err(RunnerError::AlignerLoad {
    message: String::from(
      "tokenizer is missing the `|` word-delimiter token, but the language's normaliser \
       declared `use_word_delimiter = true`. wav2vec2 word-segmented vocabularies require \
       a `|` token between spoken words. Either swap to a tokenizer that exposes `|`, or \
       supply a normaliser whose `use_word_delimiter` returns false (char-level segmentation).",
    ),
  })
}

/// Load a HuggingFace tokenizer.json with `tokenizers 0.20`
/// compatibility shimming.
///
/// The canonical wav2vec2 tokenizer.json (e.g.,
/// `facebook/wav2vec2-base-960h`, `onnx-community/wav2vec2-base-960h-ONNX`)
/// ships in an older HF format whose `model` object carries
/// only `vocab` — no `type` discriminator. `tokenizers 0.20`'s
/// `ModelUntagged` deserialiser rejects that with `data did not
/// match any variant of untagged enum ModelUntagged`. The repo's
/// `build.rs` patches the build-time fixture, but a downstream
/// consumer following the public `Aligner::from_paths` API with
/// their own tokenizer file would have hit the same load
/// failure.
///
/// We try the raw file first so already-compliant tokenizer
/// JSONs (BPE / Unigram models, or modern WordLevel exports
/// with `type`) take the fast path. On failure, we attempt one
/// patch — inject `"type": "WordLevel"` and `"unk_token":
/// "<unk>"` immediately inside the `"model": {` block — and
/// retry. If the retry still fails we surface the *original*
/// error, not the patched-version error, since the patch is
/// only meaningful for the wav2vec2 shape.
fn load_tokenizer_with_compat(path: &Path) -> Result<Tokenizer, RunnerError> {
  let bytes = std::fs::read(path).map_err(|e| RunnerError::AlignerLoad {
    message: alloc::format!("read tokenizer {}: {e}", path.display()),
  })?;

  let original_err = match Tokenizer::from_bytes(&bytes) {
    Ok(tok) => return Ok(tok),
    Err(e) => alloc::format!("{e:?}"),
  };

  if let Some(patched) = inject_wordlevel_model_type(&bytes)
    && let Ok(tok) = Tokenizer::from_bytes(&patched)
  {
    return Ok(tok);
  }

  Err(RunnerError::AlignerLoad {
    message: alloc::format!(
      "Tokenizer::from_file({}) failed: {original_err}",
      path.display()
    ),
  })
}

/// Inject `"type": "WordLevel"` and `"unk_token": "<unk>"` into
/// the `model` object of an HF tokenizer.json. Returns `None` if
/// the file already has a `type:` (no patch needed) or if we
/// can't find the `"model": {` boundary (different schema —
/// don't guess).
fn inject_wordlevel_model_type(bytes: &[u8]) -> Option<alloc::vec::Vec<u8>> {
  let s = core::str::from_utf8(bytes).ok()?;
  // Find `"model"`'s opening brace. Robust to whitespace.
  let model_idx = s.find("\"model\"")?;
  let after_model = &s[model_idx..];
  let brace_offset = after_model.find('{')?;
  let brace_pos = model_idx + brace_offset;

  // Already patched / already typed: leave it alone.
  let model_body = &s[brace_pos..];
  // Find the matching closing brace, conservatively by depth.
  let mut depth = 0_i32;
  let mut close_pos = None;
  for (i, c) in model_body.char_indices() {
    match c {
      '{' => depth += 1,
      '}' => {
        depth -= 1;
        if depth == 0 {
          close_pos = Some(brace_pos + i);
          break;
        }
      }
      _ => {}
    }
  }
  let close_pos = close_pos?;
  if s[brace_pos..close_pos].contains("\"type\"") {
    return None; // already discriminated
  }

  // Inject the discriminator fields right after `{`.
  let injection = "\n        \"type\": \"WordLevel\",\n        \"unk_token\": \"<unk>\",";
  let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(bytes.len() + injection.len());
  out.extend_from_slice(s[..=brace_pos].as_bytes());
  out.extend_from_slice(injection.as_bytes());
  out.extend_from_slice(s[brace_pos + 1..].as_bytes());
  Some(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  // Unit tests for `from_paths` are tricky: they require real
  // wav2vec2 ONNX + tokenizer.json files. The end-to-end test
  // exercises the actual loader against the build.rs-fetched
  // fixture. Here we lock in the type-level invariants and the
  // blank-token-id detection helper.

  /// Regression: the upstream wav2vec2 tokenizer.json (HF format,
  /// no `model.type` discriminator) loaded directly via
  /// `Aligner::from_paths` used to fail with `tokenizers 0.20`'s
  /// ModelUntagged deserialiser. The build.rs fixture got
  /// patched, but a downstream consumer loading their own copy
  /// from HuggingFace would have hit a load-time error.
  ///
  /// Fix: `load_tokenizer_with_compat` patches in-memory and
  /// retries. This test exercises that path with the canonical
  /// minimal upstream shape — exactly what Hugging Face serves
  /// for `facebook/wav2vec2-base-960h`'s `tokenizer.json`.
  #[test]
  fn load_tokenizer_with_compat_handles_unpatched_hf_format() {
    // Minimal upstream HF tokenizer.json shape — `model` has
    // only `vocab`, no `type` discriminator. `tokenizers 0.20`
    // rejects this raw; the compat shim must inject the
    // missing fields and retry.
    let raw = br#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
      "post_processor": null,
      "decoder": null,
      "model": {
        "vocab": {
          "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
          "A": 5, "B": 6, "C": 7
        }
      }
    }"#;
    // Confirm the raw form really does fail (otherwise the
    // shim is exercising nothing). If `tokenizers` upstream
    // ever relaxes its parser, this assert catches it.
    assert!(
      Tokenizer::from_bytes(raw).is_err(),
      "tokenizers 0.20 unexpectedly accepted raw upstream HF format; \
       the compat shim is no longer necessary"
    );

    // Shim must accept and patch.
    let patched =
      inject_wordlevel_model_type(raw).expect("inject_wordlevel_model_type must succeed");
    let tok = Tokenizer::from_bytes(&patched).expect("patched JSON must parse");
    assert_eq!(tok.token_to_id("A"), Some(5));
    assert_eq!(tok.token_to_id("<unk>"), Some(3));
  }

  /// The shim must NOT mangle a tokenizer that already carries
  /// a `type` discriminator (modern HF format, BPE / Unigram
  /// models). It returns `None` and leaves the file untouched.
  #[test]
  fn load_tokenizer_with_compat_skips_already_patched_input() {
    let already_typed = br#"{
      "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "A": 1},
        "unk_token": "<unk>"
      }
    }"#;
    assert!(inject_wordlevel_model_type(already_typed).is_none());
  }

  // --- Word-delimiter validation ---

  /// In-memory tokenizer with a `|` token. Use for "valid"
  /// cases where the delimiter check should pass.
  fn tokenizer_with_pipe_delimiter() -> Tokenizer {
    let json = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "<pad>": 1, "|": 2, "A": 3, "B": 4},
        "unk_token": "<unk>"
      }
    }"#;
    Tokenizer::from_bytes(json.as_bytes()).expect("parse")
  }

  /// Same shape WITHOUT the `|` token. Reproduces the
  /// configuration mistake the delimiter check catches.
  fn tokenizer_without_pipe_delimiter() -> Tokenizer {
    let json = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"<unk>": 0, "<pad>": 1, "A": 2, "B": 3},
        "unk_token": "<unk>"
      }
    }"#;
    Tokenizer::from_bytes(json.as_bytes()).expect("parse")
  }

  #[test]
  fn delimiter_check_passes_when_token_present_and_required() {
    let tok = tokenizer_with_pipe_delimiter();
    assert!(validate_word_delimiter_present(&tok, true).is_ok());
  }

  #[test]
  fn delimiter_check_fails_when_required_but_missing() {
    let tok = tokenizer_without_pipe_delimiter();
    let err = validate_word_delimiter_present(&tok, true).unwrap_err();
    let RunnerError::AlignerLoad { message } = err else {
      panic!("expected AlignerLoad");
    };
    assert!(
      message.contains("`|` word-delimiter"),
      "must call out the missing delimiter; got {message:?}"
    );
  }

  #[test]
  fn delimiter_check_passes_for_char_segmented_normalizers() {
    // CJK-shape normaliser: `use_word_delimiter == false`.
    // Missing `|` is fine — char-segmented inputs don't use
    // inter-word delimiters in the CTC graph.
    let tok = tokenizer_without_pipe_delimiter();
    assert!(validate_word_delimiter_present(&tok, false).is_ok());
  }

  // --- build_speech_mask: silence-mask coordinate contract ---

  fn analysis_tb() -> mediatime::Timebase {
    mediatime::Timebase::new(1, core::num::NonZeroU32::new(SAMPLE_RATE_HZ).unwrap())
  }

  #[test]
  fn build_speech_mask_marks_inrange_segments() {
    // Plain in-range segment: bits set exactly inside [start, end).
    let segs = [TimeRange::new(2, 5, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, false, true, true, true, false, false, false]
    );
  }

  #[test]
  fn build_speech_mask_clamps_negative_overlap_to_zero() {
    // Regression: pre-fix, `as u64 as usize` wrapped negative
    // start_pts to a huge value, then `.min(samples.len())`
    // clamped to len, and `if end > start` dropped the segment
    // entirely. Now the head trims to 0 and the tail (within
    // the chunk) gets masked.
    let segs = [TimeRange::new(-3, 4, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![true, true, true, true, false, false, false, false]
    );
  }

  #[test]
  fn build_speech_mask_clamps_overshoot_to_buffer_end() {
    // end_pts past `n_samples` clamps to len; start in range.
    let segs = [TimeRange::new(5, 100, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, false, false, false, false, true, true, true]
    );
  }

  #[test]
  fn build_speech_mask_drops_fully_negative_range() {
    // Both bounds negative: clamps to [0, 0), no bits set.
    let segs = [TimeRange::new(-10, -3, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_drops_fully_overshoot_range() {
    // Both bounds past len: clamps to [len, len), no bits set.
    let segs = [TimeRange::new(20, 30, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_zero_width_range_is_dropped() {
    // start == end: `if end > start` skips, no bits set.
    // (`TimeRange::new` panics on `end < start`, so a literal
    // inverted-range case can't be constructed via the public
    // API and isn't reachable through the silence-mask path.)
    let segs = [TimeRange::new(5, 5, analysis_tb())];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(mask, vec![false; 8]);
  }

  #[test]
  fn build_speech_mask_unions_overlapping_segments() {
    // Mask is a per-sample OR of all segments; overlap is fine.
    let segs = [
      TimeRange::new(1, 4, analysis_tb()),
      TimeRange::new(3, 6, analysis_tb()),
    ];
    let mask = build_speech_mask(8, &segs, &Lang::En).expect("ok");
    assert_eq!(
      mask,
      vec![false, true, true, true, true, true, false, false]
    );
  }

  #[test]
  fn build_speech_mask_empty_buffer_returns_empty_mask() {
    let segs = [TimeRange::new(0, 0, analysis_tb())];
    let mask = build_speech_mask(0, &segs, &Lang::En).expect("ok");
    assert!(mask.is_empty());
  }

  #[test]
  fn build_speech_mask_errors_on_non_analysis_timebase() {
    // Promoted from the previous `debug_assert!`-only check: a
    // non-1/16000 timebase now fails the chunk in BOTH debug and
    // release. Codex round-20 round-tripped this as a
    // medium-severity finding because release builds silently
    // misinterpreted (e.g.) a millisecond-timebase PTS as a
    // 16 kHz sample index, masking the wrong samples and
    // producing plausible-but-wrong word alignments.
    let ms_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(1000).unwrap());
    let segs = [TimeRange::new(0, 100, ms_tb)];
    let err = build_speech_mask(16_000, &segs, &Lang::En).expect_err("must error");
    match err {
      WorkFailure::AlignmentFailed { kind, message, .. } => {
        assert_eq!(
          kind,
          crate::types::AlignmentFailureKind::ModelInferenceFailed
        );
        assert!(
          message.contains("chunk-local 1/16000 timebase"),
          "error message must cite the contract; got: {message}"
        );
        assert!(
          message.contains("1/1000"),
          "error message must cite the offending timebase; got: {message}"
        );
      }
      other => panic!("expected AlignmentFailed, got {other:?}"),
    }
  }

  #[test]
  fn build_speech_mask_errors_on_output_timebase() {
    // Codex's example was milliseconds (1/1000); a 1/48000
    // (output-rate) PTS is the more realistic foot-gun: a
    // production caller passing the output-timebase ranges they
    // were going to emit, instead of converting back to
    // chunk-local 1/16000. Same fail-loud behaviour required.
    let out_tb = mediatime::Timebase::new(1, core::num::NonZeroU32::new(48_000).unwrap());
    let segs = [TimeRange::new(0, 1000, out_tb)];
    let err = build_speech_mask(16_000, &segs, &Lang::En).expect_err("must error");
    assert!(matches!(err, WorkFailure::AlignmentFailed { .. }));
  }

  #[test]
  fn aligner_is_send_not_sync() {
    // Aligner is Send (each field — Session, Tokenizer, Lang,
    // DynTextNormalizer, primitives — is Send). It must not
    // be Sync because Session::run requires &mut self.
    fn assert_send<T: Send>() {}
    // We can't easily assert !Sync at the type level without
    // negative trait bounds; the Mutex<Aligner> in
    // AlignmentSet is the runtime check.
    assert_send::<Aligner>();
  }

  /// Regression: punctuation-only ASR text normalises to empty,
  /// but alignment must NOT turn the successful ASR transcript
  /// into `Event::Error`. The fix short-circuits `EmptyText` to
  /// `Ok(empty AlignmentResult)` inside `Aligner::align`; this
  /// test exercises that path without ONNX inference (the
  /// short-circuit returns before `encode_log_softmax` runs).
  ///
  /// Skips when the build.rs fixture isn't present (offline /
  /// `WHISPERY_OFFLINE=1`); aligner_load already verifies the
  /// fixture loads, so we know `Aligner::from_paths` succeeds
  /// when the env vars are set.
  #[test]
  fn empty_normalised_text_returns_empty_alignment_result() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::normalizers::EnglishNormalizer;

    let model_path = match option_env!("WHISPERY_W2V_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("WHISPERY_W2V_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let mut aligner = Aligner::from_paths(
      Lang::En,
      Path::new(model_path),
      Path::new(tokenizer_path),
      alloc::boxed::Box::new(EnglishNormalizer::new()),
    )
    .expect("Aligner::from_paths");

    // 16 kHz silence buffer — never read because `EmptyText`
    // short-circuits before encode runs.
    let samples = alloc::vec![0.0_f32; 16_000];
    let sub_segments: alloc::vec::Vec<TimeRange> = alloc::vec::Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");

    // Punctuation-only input → EnglishNormalizer returns
    // `EmptyText`; align must surface as Ok(empty), not Err.
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        /* text: */ "!!!...",
        /* chunk_first_sample_in_stream: */ 0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
      )
      .expect("EmptyText must short-circuit to Ok, not propagate as AlignmentFailed");
    assert!(
      result.words().is_empty(),
      "empty normalisation must yield zero words; got {:?}",
      result.words()
    );
  }

  /// Codex round-25 regression: a chunk shorter than wav2vec2's
  /// 400-sample receptive field must NOT enter the encode path,
  /// because the encoder would pad to 400 and emit `T` frames
  /// whose stride is governed by the padded length, while
  /// `samples_per_frame` downstream would use the original
  /// `samples.len()`. The two views disagree by exactly the
  /// padding ratio. Skipping these chunks (returning empty
  /// `AlignmentResult`) is the simplest safe response — at
  /// 25 ms or less, the chunk cannot contain a meaningful
  /// CTC path through any non-trivial transcript.
  #[test]
  fn sub_400_sample_chunk_short_circuits_to_empty_result() {
    use core::sync::atomic::AtomicBool;

    use mediatime::{TimeRange, Timebase};

    use crate::runner::aligner::normalizers::EnglishNormalizer;

    let model_path = match option_env!("WHISPERY_W2V_MODEL") {
      Some(p) => p,
      None => return,
    };
    let tokenizer_path = match option_env!("WHISPERY_W2V_TOKENIZER") {
      Some(p) => p,
      None => return,
    };

    let mut aligner = Aligner::from_paths(
      Lang::En,
      Path::new(model_path),
      Path::new(tokenizer_path),
      alloc::boxed::Box::new(EnglishNormalizer::new()),
    )
    .expect("Aligner::from_paths");

    // 200 samples at 16 kHz = 12.5 ms. wav2vec2 needs ≥400.
    let samples = alloc::vec![0.0_f32; 200];
    let sub_segments: alloc::vec::Vec<TimeRange> = alloc::vec::Vec::new();
    let abort = AtomicBool::new(false);
    let run_options = ort::session::RunOptions::new().expect("RunOptions::new");

    // Realistic transcript text — would normalise + tokenise
    // fine, but the sub-400-sample guard fires before encode.
    let result = aligner
      .align(
        &samples,
        &sub_segments,
        /* text: */ "hello world",
        /* chunk_first_sample_in_stream: */ 0,
        |start, end| {
          TimeRange::new(
            start as i64,
            end as i64,
            Timebase::new(1, core::num::NonZeroU32::new(16_000).unwrap()),
          )
        },
        &abort,
        &run_options,
      )
      .expect("sub-400-sample chunks must Ok(empty), not propagate as AlignmentFailed");
    assert!(
      result.words().is_empty(),
      "sub-400-sample chunk must yield zero words; got {:?}",
      result.words()
    );
  }
}
