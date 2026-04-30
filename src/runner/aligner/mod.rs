//! Aligner subsystem — wav2vec2 forced alignment via ort.

mod aligner;
mod algorithm;
mod key;
mod normalizer;
mod normalizers;

pub use aligner::Aligner;
pub use key::{AlignerKey, AlignmentFallback};
pub use normalizer::{DynTextNormalizer, NormalizationError, NormalizedText, TextNormalizer};
pub use normalizers::{ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer};
