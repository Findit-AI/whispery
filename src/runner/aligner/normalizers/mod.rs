//! Concrete `TextNormalizer` implementations.
//!
//! Spec §6.3 names English / Chinese / Japanese as the v1
//! supported set. Future versions add more languages by adding
//! files here and re-exporting from `runner::aligner`.

mod chinese;
mod english;
mod japanese;
#[cfg(test)]
mod tests;

pub use chinese::ChineseNormalizer;
pub use english::EnglishNormalizer;
pub use japanese::JapaneseNormalizer;
