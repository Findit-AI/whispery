//! Surface-form preservation + special-token skipping regressions.

#![cfg(feature = "alignment")]

use asry::{EnglishNormalizer, TextNormalizer};

#[test]
fn english_preserves_casing_in_original_words() {
  let n = EnglishNormalizer::new();
  let nt = n.normalize("The QUICK brown Fox").unwrap();
  assert_eq!(nt.normalized(), "the quick brown fox");
  let originals: Vec<&str> = nt.original_words().iter().map(|c| c.as_ref()).collect();
  assert_eq!(originals, vec!["The", "QUICK", "brown", "Fox"]);
}

#[test]
fn english_preserves_punctuation_in_original_words() {
  let n = EnglishNormalizer::new();
  let nt = n.normalize("Hello, world!").unwrap();
  assert_eq!(nt.normalized(), "hello world");
  let originals: Vec<&str> = nt.original_words().iter().map(|c| c.as_ref()).collect();
  assert_eq!(originals, vec!["Hello,", "world!"]);
}

/// Contractions are preserved as a single normalised word with
/// the apostrophe character intact. wav2vec2-base-960h's vocab
/// has the `'` glyph (id 27) and was trained on LibriSpeech
/// transcripts that write `DON'T` directly; expanding to
/// `do not` would force the CTC graph to insert a `|` word
/// delimiter the speaker never pronounces, shifting timings
/// and emitting duplicate `Word.text()` entries that
/// downstream consumers can't safely dedupe.
#[test]
fn contraction_preserved_as_single_word() {
  let n = EnglishNormalizer::new();
  let nt = n.normalize("Don't go.").unwrap();
  assert_eq!(nt.normalized(), "don't go");
  let originals: Vec<&str> = nt.original_words().iter().map(|c| c.as_ref()).collect();
  assert_eq!(originals, vec!["Don't", "go."]);
}

// Special-token skipping is enforced by:
//   src/runner/aligner/algorithm/compose.rs::tests::delimiter_token_is_skipped
// We re-document the contract here.

#[test]
fn delimiter_token_skipping_documented() {
  // Token-level delimiters (the `|` token in wav2vec2 vocabs)
  // must have word_idx_per_token=None and must be skipped by
  // step 7's per-word accumulator. Verified directly by the
  // compose.rs unit test; this integration shell documents the
  // contract.
}
