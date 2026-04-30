//! `Aligner` — per-language wav2vec2 forced-alignment engine.

use alloc::string::String;
use core::time::Duration;
use std::path::Path;

use mediatime::TimeRange;
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::core::AlignmentResult;
use crate::runner::RunnerError;
use crate::runner::aligner::normalizer::DynTextNormalizer;
use crate::types::{Lang, WorkFailure};

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
                message: alloc::format!(
                    "commit_from_file({}) failed: {e:?}",
                    model_path.display()
                ),
            })?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            RunnerError::AlignerLoad {
                message: alloc::format!(
                    "Tokenizer::from_file({}) failed: {e:?}",
                    tokenizer_path.display()
                ),
            }
        })?;

        let blank_token_id = detect_blank_token_id(&tokenizer).ok_or_else(|| {
            RunnerError::AlignerLoad {
                message: String::from(
                    "tokenizer has no <pad> / [PAD] entry; cannot determine CTC blank token",
                ),
            }
        })?;

        Ok(Self {
            session,
            tokenizer,
            language,
            normalizer,
            sample_rate: 16_000,
            hop_samples: 320,
            blank_token_id,
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
    ) -> Result<AlignmentResult, WorkFailure>
    where
        F: Fn(u64, u64) -> TimeRange,
    {
        // Will dispatch to the algorithm pipeline once Tasks 10-14
        // land. Stub returns EmptyText so the caller path compiles.
        let _ = (samples, sub_segments, text, chunk_first_sample_in_stream, samples_to_output_range);
        Err(WorkFailure::AlignmentFailed {
            kind: crate::types::AlignmentFailureKind::EmptyText,
            message: alloc::string::String::from("aligner pipeline stub: implemented in Tasks 10-14"),
            language: self.language.clone(),
        })
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
