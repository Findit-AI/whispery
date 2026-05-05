//! Runner — wires the Sans-I/O core to whisper-rs (and, with
//! `feature = "alignment"`, to wav2vec2 forced alignment).

mod errors;
mod managed_transcriber;
mod whisper_pool;

// `pub(crate)` (rather than `mod`) so the crate-root
// `#[cfg(feature = "bench-internals")] pub mod __bench` can
// re-export this module's `pub(crate)` SIMD/scalar kernels.
// Outside the bench gate the module's items are only visible
// through the curated `pub use` re-exports below.
#[cfg(feature = "alignment")]
pub(crate) mod aligner;
#[cfg(feature = "alignment")]
mod alignment_pool;

pub use errors::RunnerError;
pub use managed_transcriber::{ManagedTranscriber, ManagedTranscriberBuilder};
pub use whisper_pool::WhisperPoolOptions;

#[cfg(feature = "alignment")]
pub use aligner::{
  Aligner, AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet, AlignmentSetBuilder,
  ChineseNormalizer, DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE, DynTextNormalizer,
  EnglishNormalizer, JapaneseNormalizer, NormalizationError, NormalizedText, TextNormalizer,
  bundled, default_normalizer_for,
};
