//! Cross-cutting normaliser tests.

#![cfg(test)]

use crate::runner::aligner::normalizer::{DynTextNormalizer, TextNormalizer};
use crate::runner::aligner::normalizers::{ChineseNormalizer, EnglishNormalizer, JapaneseNormalizer};

fn assert_send<T: Send>() {}

#[test]
fn all_normalizers_are_send() {
    assert_send::<EnglishNormalizer>();
    assert_send::<ChineseNormalizer>();
    assert_send::<JapaneseNormalizer>();
    // The boxed dyn must also be Send (the alignment worker
    // requires it for crossing thread boundaries).
    assert_send::<DynTextNormalizer>();
}

#[test]
fn english_word_count_matches_original_words() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello, World! Don't go.").unwrap();
    assert_eq!(
        nt.original_words().len(),
        nt.normalized().split_whitespace().count(),
        "original_words.len() must equal whitespace-token count of normalised text"
    );
}

#[test]
fn chinese_word_count_matches_original_words() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("你好世界 Hello").unwrap();
    assert_eq!(
        nt.original_words().len(),
        nt.normalized().split_whitespace().count(),
    );
}

#[test]
fn japanese_word_count_matches_original_words() {
    let n = JapaneseNormalizer::new();
    let nt = n.normalize("日本語 USA 勉強").unwrap();
    assert_eq!(
        nt.original_words().len(),
        nt.normalized().split_whitespace().count(),
    );
}

#[test]
fn boxed_dyn_normalizer_dispatches() {
    let n: DynTextNormalizer = Box::new(EnglishNormalizer::new());
    let nt = n.normalize("Hi.").unwrap();
    assert_eq!(nt.normalized(), "hi");
}
