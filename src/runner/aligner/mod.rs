//! Aligner subsystem — wav2vec2 forced alignment via ort.

mod aligner;
mod algorithm;
mod builder;
mod key;
mod normalizer;
mod normalizers;
mod set;

pub use aligner::Aligner;
pub use builder::AlignmentSetBuilder;
pub use key::{AlignerKey, AlignmentFallback};
pub use normalizer::{DynTextNormalizer, NormalizationError, NormalizedText, TextNormalizer};
pub use normalizers::{ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer};
pub use set::{AlignmentLookup, AlignmentSet};
