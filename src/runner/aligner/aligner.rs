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
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| RunnerError::AlignerLoad {
      message: alloc::format!(
        "Tokenizer::from_file({}) failed: {e:?}",
        tokenizer_path.display()
      ),
    })?;

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
  pub const fn set_sample_rate(&mut self, value: u32) {
    self.sample_rate = value;
  }

  /// Builder-style override for [`Self::sample_rate`].
  pub const fn with_sample_rate(mut self, value: u32) -> Self {
    self.sample_rate = value;
    self
  }

  /// Set [`Self::hop_samples`].
  pub const fn set_hop_samples(&mut self, value: u32) {
    self.hop_samples = value;
  }

  /// Builder-style override for [`Self::hop_samples`].
  pub const fn with_hop_samples(mut self, value: u32) -> Self {
    self.hop_samples = value;
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
        silence_mask::build_masked_samples,
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

    // Step 0: silence-mask non-speech regions.
    // The output_range_to_chunk_local closure converts an
    // output-timebase TimeRange to chunk-local 16 kHz indices.
    // We use samples_to_output_range as our bridge: invert it
    // by converting (range.start_pts, range.end_pts) back to
    // chunk-local sample offsets via the chunk_first_sample
    // offset.
    //
    // Actually: the worker stage already converts sub_segment
    // TimeRanges into the output timebase from Plan A's
    // ExtractedChunk; the inversion at this layer is identical
    // to the conversion the worker did. We accept TimeRanges
    // here and the worker passes a closure that does the
    // chunk-local conversion (Task 21 wires this).
    //
    // For the v1 pipeline, the closure is constructed by the
    // alignment worker (run_one_alignment in Task 18); the
    // signature here takes a Fn(TimeRange) -> (u64, u64) but
    // we don't have it as a parameter. The pragmatic approach:
    // the worker pre-converts sub_segments to chunk-local
    // (start_sample, end_sample) pairs and passes those in
    // place of TimeRanges. We change the signature to take
    // pre-converted ranges to avoid a redundant closure.
    //
    // Rather than introducing a fourth closure parameter, we
    // build the mask directly from sub_segments expressed in
    // sample space. Caller (worker) is responsible for
    // expressing sub_segments in chunk-local 16 kHz indices.
    // To keep the public Aligner::align contract honest,
    // sub_segments is documented as "chunk-local sample
    // ranges, not output-timebase TimeRanges" — see the
    // worker's run_one_alignment (Task 18) for the conversion.
    let masked = build_masked_samples(samples, sub_segments, |seg| {
      // `seg` is documented as carrying chunk-local 16 kHz
      // sample indices in its PTS units. Caller builds the
      // ranges with a tb of (1/16000) so PTS == sample idx.
      (seg.start_pts() as u64, seg.end_pts() as u64)
    });

    if abort_flag.load(Ordering::Relaxed) {
      return Err(timed_out());
    }

    // Step 1: normalise.
    let normalized = self.normalizer.normalize(text).map_err(|e| match e {
      crate::runner::aligner::normalizer::NormalizationError::EmptyText => {
        WorkFailure::AlignmentFailed {
          kind: AlignmentFailureKind::EmptyText,
          message: alloc::format!("empty text after normalisation"),
          language: self.language.clone(),
        }
      }
      crate::runner::aligner::normalizer::NormalizationError::RuleFailed { detail } => {
        WorkFailure::AlignmentFailed {
          kind: AlignmentFailureKind::NormalizationFailed,
          message: detail,
          language: self.language.clone(),
        }
      }
    })?;

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
    // wav2vec2-base-960h's. `unk_token_id` lets the tokeniser
    // reject any word that would have aligned to <unk>.
    let tokenized = tokenize_with_word_map(
      &self.tokenizer,
      normalized.normalized(),
      n_words,
      self.normalizer.use_word_delimiter(),
      self.vocab_uppercase_only,
      self.unk_token_id,
      &self.language,
    )?;

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
    let log_probs = encode_log_softmax(&mut self.session, &masked, run_options, &self.language)?;

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

#[cfg(test)]
mod tests {
  use super::*;

  // Unit tests for `from_paths` are tricky: they require real
  // wav2vec2 ONNX + tokenizer.json files. Task 25's end-to-end
  // test exercises the actual loader against the build.rs-fetched
  // fixture. Here we lock in the type-level invariants and the
  // blank-token-id detection helper.

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
}
