//! English text normaliser — back-compat thin wrapper.
//!
//! As of step 7 of the script-dispatch design, the English-only
//! normaliser is a special case of the general
//! [`crate::runner::aligner::normalizers::LatinNormalizer`]
//! parameterised by `Lang::En`. This file keeps the
//! [`EnglishNormalizer`] symbol stable so existing call sites
//! (`asry::EnglishNormalizer`, `runner::EnglishNormalizer`,
//! tests, examples, parity binaries) keep compiling unchanged.
//!
//! Behaviour is identical to the legacy `EnglishNormalizer`:
//! lowercase + boundary-punct strip, apostrophes survive inside
//! contractions, hyphen / em-dash / slash splits emit per-piece
//! surface spans, empty input returns `EmptyText`.

use crate::runner::aligner::{
  normalizer::{NormalizationError, NormalizedText, TextNormalizer},
  normalizers::latin::LatinNormalizer,
};

/// English wav2vec2-base-960h-shaped normaliser.
///
/// Lowercase Latin + ASCII boundary punctuation strip +
/// per-piece surface spans on hyphen / em-dash / slash splits.
/// Apostrophes survive inside contractions (`don't` stays one
/// word).
///
/// Behaviour is delegated to
/// [`LatinNormalizer::english`][LatinNormalizer::english]. The
/// distinct type is kept so existing
/// `Box::new(EnglishNormalizer::new())` call sites compile
/// without change.
#[derive(Default, Clone, Copy, Debug)]
pub struct EnglishNormalizer;

impl EnglishNormalizer {
  /// Construct an English normaliser. `const fn` for use in
  /// static lookup tables.
  pub const fn new() -> Self {
    Self
  }
}

impl TextNormalizer for EnglishNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    LatinNormalizer::english().normalize(text)
  }

  fn use_word_delimiter(&self) -> bool {
    LatinNormalizer::english().use_word_delimiter()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // Back-compat smoke tests: a few representative cases from the
  // original `english.rs` test surface. The full English contract
  // is exercised by `latin::tests::en_*` — these stay here so
  // consumers grep-ing for "EnglishNormalizer" still see an
  // obvious test entry point under that name.

  #[test]
  fn lowercase_and_strip_punct() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello, World!").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
  }

  #[test]
  fn contraction_stays_one_word() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Don't go.").unwrap();
    assert_eq!(nt.normalized(), "don't go");
    assert_eq!(nt.original_words()[0], "Don't");
  }

  #[test]
  fn empty_input_errors() {
    let n = EnglishNormalizer::new();
    let err = n.normalize("   .,!?  ").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn delegates_use_word_delimiter() {
    let n = EnglishNormalizer::new();
    assert!(n.use_word_delimiter());
  }

  /// Surface-form invariant must hold for the simplest case.
  /// Original words must reflect the source text verbatim.
  #[test]
  fn original_words_preserve_source_casing() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello World").unwrap();
    assert_eq!(nt.original_words()[0], "Hello");
    assert_eq!(nt.original_words()[1], "World");
  }
}
