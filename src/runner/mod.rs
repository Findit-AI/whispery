//! Runner — wires the Sans-I/O core to whisper-rs (and, with
//! `feature = "alignment"`, to wav2vec2 forced alignment).

mod errors;
mod managed_transcriber;
mod whisper_pool;

#[cfg(feature = "alignment")]
mod aligner;
#[cfg(feature = "alignment")]
mod alignment_pool;

pub use errors::RunnerError;
pub use managed_transcriber::{ManagedTranscriber, ManagedTranscriberBuilder};
pub use whisper_pool::WhisperPoolConfig;

#[cfg(feature = "alignment")]
pub use aligner::{
    Aligner, AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet, AlignmentSetBuilder,
    ChineseNormalizer, DynTextNormalizer, EnglishNormalizer, JapaneseNormalizer,
    NormalizationError, NormalizedText, TextNormalizer,
};
