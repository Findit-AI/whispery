//! Aligner subsystem — wav2vec2 forced alignment via ort.

mod key;
mod normalizer;
mod normalizers;

pub use key::{AlignerKey, AlignmentFallback};
pub use normalizer::{DynTextNormalizer, NormalizationError, NormalizedText, TextNormalizer};
pub use normalizers::{ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer};
