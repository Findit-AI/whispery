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
  types::{Lang, WorkFailure},
};

/// Per-language forced-alignment engine. Loads a wav2vec2 ONNX
/// model, its HuggingFace tokenizer, and the language's text
/// normaliser. Each instance is heavyweight (ONNX session +
/// tokenizer state); the [`crate::AlignmentSet`] registry keeps one
/// per registered language, gated behind `Mutex<Aligner>` (spec
/// §6.3.3) so the single alignment worker can drive any language
/// without copying.
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

    // Codex round-18 [medium]: when the normaliser declares
    // `use_word_delimiter == true` (the English-shape default),
    // the tokenizer MUST expose a `|` token. See
    // [`validate_word_delimiter_present`] for the rationale.
    validate_word_delimiter_present(&tokenizer, normalizer.use_word_delimiter())?;

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
      min_speech_coverage: crate::runner::aligner::algorithm::compose::DEFAULT_MIN_SPEECH_COVERAGE,
      max_intra_silent_run: crate::runner::aligner::algorithm::compose::DEFAULT_MAX_INTRA_SILENT_RUN,
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
  /// boundary is correct. Codex round-23.
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
  /// timings silently. Fail fast. Codex round-23.
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

  // The crate-private `align` method is implemented across Tasks
  // 10-14. The signature is fixed here so other modules can
  // declare it as a dependency.

  /// Crate-private alignment entrypoint. Implemented incrementally
  /// in Tasks 10-14.
  ///
  /// Inputs:
  /// - `samples`: the chunk's 16 kHz f32 mono audio.
  /// - `sub_segments`: VAD sub-segments inside the chunk, in the
  ///   caller's output timebase. Used by the silence mask in step 0.
  /// - `text`: Whisper's transcribed text.
  /// - `chunk_first_sample_in_stream`: the chunk's first 16 kHz
  ///   sample index in stream coordinates (used to convert
  ///   wav2vec2 frame indices back to stream sample indices).
  /// - `samples_to_output_range`: callback bridging stream sample
  ///   indices to output-timebase `TimeRange`s. Plan A's
  ///   `SampleBuffer::samples_to_output_range` is `pub(crate)`;
  ///   the worker constructs a closure over it (see Task 21).
  ///
  /// Implemented in Tasks 10-14; this stub is the API contract.
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
        viterbi::ctc_viterbi,
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

    // Step 0: silence-aware preprocessing.
    //
    // The chunk-local sub_segments come in as `TimeRange`s in a
    // 1/16 kHz timebase, so `start_pts()` / `end_pts()` are
    // chunk-local sample indices. Build a per-sample boolean
    // speech mask for the silence-aware normaliser; once that
    // returns the buffer it's already been (1) normalised over
    // speech samples only and (2) zeroed at non-speech
    // positions, so the silence-mask invariant survives all the
    // way to ORT. Codex round-14 [high]: pre-fix this was two
    // steps — `build_masked_samples` then a non-mask-aware
    // normalise inside `encode_log_softmax`. The intermediate
    // zeros got mean-shifted by the normaliser, so masked
    // regions became `(0 - mean) / std` ≠ 0 by the time they
    // reached the model.
    let mut speech_mask = alloc::vec![false; samples.len()];
    for &seg in sub_segments {
      let start = (seg.start_pts() as u64 as usize).min(samples.len());
      let end = (seg.end_pts() as u64 as usize).min(samples.len());
      if end > start {
        for slot in &mut speech_mask[start..end] {
          *slot = true;
        }
      }
    }

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
    // instead of `Event::Error`. Codex round-12 [medium] flagged
    // this as a data-loss path that contradicts the
    // `AlignmentResult` contract.
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
    // The samples we hand to ORT have already gone through the
    // silence-aware normaliser above — `encode_log_softmax`
    // expects pre-normalised input. (Codex round-14 [high]: the
    // pre-fix internal-normalise inside `encode_log_softmax`
    // broke the silence mask.)
    let normalized_samples =
      crate::runner::aligner::algorithm::normalize::normalize_with_silence_mask(
        samples,
        &speech_mask,
      );
    let log_probs = encode_log_softmax(
      &mut self.session,
      &normalized_samples,
      run_options,
      &self.language,
    )?;

    // Codex round-23 [high]: validate the encoder's stride
    // against the input audio length. wav2vec2's CNN
    // downsamples by `hop_samples` (320 for *-base / *-large),
    // so the encoded "time" `T * hop_samples` should be at
    // most `samples.len()` plus a couple of frames of
    // receptive-field slack. A grossly larger T means the
    // model export uses a different stride or `hop_samples`
    // is misconfigured — `compose_words`'s `frame * hop`
    // mapping would otherwise emit plausible-looking word
    // ranges past the chunk's audio boundary with no error
    // surfaced. Fatal because the only recovery is fixing
    // the model/config, not retrying.
    let frame_extent = (log_probs.t as u64).saturating_mul(self.hop_samples as u64);
    let chunk_extent = samples.len() as u64;
    let max_allowed = chunk_extent.saturating_add(2 * self.hop_samples as u64);
    if frame_extent > max_allowed {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "ORT output stride mismatch: T={} × hop={} = {} sample-equivalents \
           exceeds chunk ({} samples) + 2-frame slack ({}); model export \
           uses a different stride or `hop_samples` is misconfigured",
          log_probs.t,
          self.hop_samples,
          frame_extent,
          chunk_extent,
          max_allowed,
        ),
        language: self.language.clone(),
      });
    }

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 5-6: CTC lattice + Viterbi. The DP checks
    // `abort_flag` once per frame row so a pathological (long
    // chunk × long token sequence) lattice can't run past the
    // alignment worker's `align_timeout` and starve every
    // chunk queued behind it on the single worker.
    let path = ctc_viterbi(
      &log_probs,
      &tokenized.token_ids,
      self.blank_token_id,
      abort_flag,
      &self.language,
    )?;

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Steps 7-9: per-word state + surface-form recovery. The
    // speech-frame mask comes from the same `sub_segments` the
    // silence-mask step zeroed, so words whose CTC-forced
    // assignment lands entirely inside masked silence drop from
    // the result rather than emit fabricated timings.
    let speech_frames = build_speech_frames(log_probs.t, self.hop_samples, sub_segments);
    Ok(compose_words(
      &path,
      &log_probs,
      &tokenized.word_idx_per_token,
      normalized.original_words(),
      &speech_frames,
      chunk_first_sample_in_stream,
      self.hop_samples,
      samples_to_output_range,
      self.min_speech_coverage,
      self.max_intra_silent_run,
    ))
  }
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
/// via the `worker_timeouts(_, align)` builder hook in Plan B.
pub(crate) const DEFAULT_ALIGN_TIMEOUT: Duration = Duration::from_secs(30);

/// Validate that the tokenizer exposes the wav2vec2 `|`
/// word-delimiter token whenever the normaliser declared
/// `use_word_delimiter == true`.
///
/// Codex round-18 [medium]: pre-fix a missing `|` token slipped
/// through silently — `tokenize_with_word_map` would simply emit
/// no inter-word delimiter, glueing adjacent words together in
/// the CTC graph. Word timings would then be plausible but
/// wrong with no configuration error visible to the caller.
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
/// failure. Codex round-13 [high].
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
  // wav2vec2 ONNX + tokenizer.json files. Task 25's end-to-end
  // test exercises the actual loader against the build.rs-fetched
  // fixture. Here we lock in the type-level invariants and the
  // blank-token-id detection helper.

  /// Codex round-13 [high] regression: the upstream wav2vec2
  /// tokenizer.json (HF format, no `model.type` discriminator)
  /// loaded directly via `Aligner::from_paths` used to fail
  /// with `tokenizers 0.20`'s ModelUntagged deserialiser. The
  /// build.rs fixture got patched, but a downstream consumer
  /// loading their own copy from HuggingFace would have hit a
  /// load-time error.
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

  // --- Codex round-18 [medium]: word-delimiter validation ---

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
  /// configuration mistake the round-18 check catches.
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

  /// Codex round-12 [medium] regression: punctuation-only ASR
  /// text normalises to empty, but alignment must NOT turn the
  /// successful ASR transcript into `Event::Error`. The fix
  /// short-circuits `EmptyText` to `Ok(empty AlignmentResult)`
  /// inside `Aligner::align`; this test exercises that path
  /// without ONNX inference (the short-circuit returns before
  /// `encode_log_softmax` runs).
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
}
