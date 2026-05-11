//! Korean text normaliser (character-level).

use std::borrow::Cow;

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// Korean normaliser: per-character segmentation across the
/// Hangul Unicode blocks, plus CJK + ASCII punctuation strip.
///
/// **Hangul blocks treated as segmenting characters.** A single
/// character is one alignment unit when it falls in:
///
/// - Hangul Jamo (`U+1100`–`U+11FF`): the conjoining-form
///   leading / vowel / trailing consonants used by IME stacks
///   before composition into syllable blocks.
/// - Hangul Compatibility Jamo (`U+3130`–`U+318F`): the
///   non-conjoining display forms used in dictionaries and as
///   bullet points.
/// - Hangul Syllables (`U+AC00`–`U+D7AF`): the precomposed LV /
///   LVT syllables that make up the bulk of modern Korean text.
///   Each syllable (e.g. `안`) is one alignment unit; we do NOT
///   decompose into jamo.
///
/// **v1 scope.** No morphological / eojeol-level segmentation,
/// no NFD jamo decomposition. The wav2vec2 KO models commonly
/// used in v1 are character-level CTC over precomposed syllables,
/// so per-syllable segmentation is the correct granularity for
/// this stage. Future v2 can plug in a jamo-decomposed normaliser
/// by adding a new variant; the trait is already general enough.
///
/// **Half-width vs. full-width Latin:** kept as per-character
/// tokens (matches whisperX's `LANGUAGES_WITHOUT_SPACES` contract
/// for `ko`). Latin chars get lowercased before emit so they hit
/// the wav2vec2 KO vocab the same way whisperX's `char_.lower()`
/// lookup does.
///
/// **Punctuation:** drops the same CJK + ASCII punctuation set
/// the JA / ZH normalisers strip, plus the Korean middle-dot
/// `·` (`U+00B7`) which appears in proper-noun separators.
#[derive(Default, Clone, Copy, Debug)]
pub struct KoreanNormalizer;

impl KoreanNormalizer {
  /// Construct a Korean normaliser.
  pub const fn new() -> Self {
    Self
  }
}

fn is_ko_punct(c: char) -> bool {
  matches!(
    c,
    '\u{3002}' // 。
            | '\u{3001}' // 、
            | '\u{FF0C}' // ，
            | '\u{FF01}' // ！
            | '\u{FF1F}' // ？
            | '\u{FF1B}' // ；
            | '\u{FF1A}' // ：
            | '\u{2026}' // …
            | '\u{300C}' // 「
            | '\u{300D}' // 」
            | '\u{300E}' // 『
            | '\u{300F}' // 』
            | '\u{FF08}' // (
            | '\u{FF09}' // )
            | '\u{30FB}' // ・
            | '\u{00B7}' // · Korean middle-dot
            | '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')'
            | '[' | ']' | '{' | '}' | '-'
  )
}

impl TextNormalizer for KoreanNormalizer {
  /// Korean is character-segmented across Hangul syllables /
  /// jamo: the whitespace this normaliser emits between every
  /// glyph is purely an indexing device. Returning `false` here
  /// keeps the tokeniser from forcing `|` between every glyph in
  /// the CTC alignment graph.
  fn use_word_delimiter(&self) -> bool {
    false
  }

  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();

    // WhisperX-matching per-character iteration. Every non-skipped
    // char (Hangul syllable, jamo, Latin, digit, ...) becomes its
    // own word; whitespace and KO/CJK/ASCII punctuation are
    // dropped. Latin chars get lowercased before emit so they
    // hit the wav2vec2 KO vocab the same way whisperX's
    // `char_.lower()` lookup does.
    //
    // Hangul syllables (U+AC00..U+D7AF) are emitted whole — we
    // do NOT decompose to L+V+T jamo. The KO wav2vec2 vocab the
    // FinDIT-Studio ONNX ships is `kresnik/wav2vec2-large-xlsr-
    // korean`'s precomposed-syllable-level vocab (1205 entries:
    // 1202 syllable blocks plus `|`, `[UNK]`, `[PAD]`). The
    // upstream `jonatasgrosman/wav2vec2-large-xlsr-53-korean`
    // referenced in earlier comments was removed from HF; kresnik
    // is the de-facto replacement (604k+ downloads).
    for c in text.chars() {
      if c.is_whitespace() || is_ko_punct(c) {
        continue;
      }
      let lowered: String = c.to_lowercase().collect();
      if !normalized.is_empty() {
        normalized.push(' ');
      }
      normalized.push_str(&lowered);
      if c.is_ascii_alphabetic() {
        original_words.push(Cow::Owned(lowered));
      } else {
        let mut buf = [0u8; 4];
        let s: &str = c.encode_utf8(&mut buf);
        original_words.push(Cow::Owned(String::from(s)));
      }
    }

    if original_words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    Ok(NormalizedText::new(normalized, original_words))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn hangul_syllables_per_char() {
    // 안녕하세요 — five precomposed Hangul syllables in U+AC00..U+D7AF.
    let n = KoreanNormalizer::new();
    let nt = n.normalize("안녕하세요").unwrap();
    assert_eq!(nt.normalized(), "안 녕 하 세 요");
    assert_eq!(nt.original_words().len(), 5);
    assert_eq!(nt.original_words()[0], "안");
    assert_eq!(nt.original_words()[4], "요");
  }

  #[test]
  fn hangul_jamo_per_char() {
    // Conjoining jamo block (U+1100..U+11FF). Each jamo becomes
    // its own word (we don't compose them).
    let n = KoreanNormalizer::new();
    let nt = n.normalize("\u{1100}\u{1161}\u{11A8}").unwrap();
    assert_eq!(nt.original_words().len(), 3);
  }

  #[test]
  fn hangul_compatibility_jamo_per_char() {
    // Compatibility jamo (U+3130..U+318F) — the dictionary /
    // bullet-point form. ㄱ ㄴ ㄷ.
    let n = KoreanNormalizer::new();
    let nt = n.normalize("ㄱㄴㄷ").unwrap();
    assert_eq!(nt.normalized(), "ㄱ ㄴ ㄷ");
    assert_eq!(nt.original_words().len(), 3);
  }

  #[test]
  fn mixed_hangul_and_latin() {
    let n = KoreanNormalizer::new();
    // "USA에서" — three Latin letters, then two Hangul syllables.
    let nt = n.normalize("USA에서").unwrap();
    // Per-character (matches whisperX's LANGUAGES_WITHOUT_SPACES
    // for `ko`): each Latin letter is its own word, lowercased.
    // 에 서 segment as before.
    assert_eq!(nt.normalized(), "u s a 에 서");
    assert_eq!(nt.original_words().len(), 5);
    assert_eq!(nt.original_words()[0], "u");
    assert_eq!(nt.original_words()[1], "s");
    assert_eq!(nt.original_words()[2], "a");
    assert_eq!(nt.original_words()[3], "에");
  }

  #[test]
  fn mixed_hangul_and_digits() {
    let n = KoreanNormalizer::new();
    let nt = n.normalize("3개").unwrap();
    // Digit `3` is its own word; Hangul syllable `개` is its own.
    assert_eq!(nt.normalized(), "3 개");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "3");
    assert_eq!(nt.original_words()[1], "개");
  }

  #[test]
  fn korean_punctuation_stripped() {
    let n = KoreanNormalizer::new();
    let nt = n.normalize("안녕, 세계!").unwrap();
    // Comma, space, exclamation all dropped; surviving:
    // 안 녕 세 계
    assert_eq!(nt.normalized(), "안 녕 세 계");
    assert_eq!(nt.original_words().len(), 4);
  }

  #[test]
  fn cjk_punctuation_stripped() {
    let n = KoreanNormalizer::new();
    let nt = n.normalize("안녕。세계、").unwrap();
    assert_eq!(nt.normalized(), "안 녕 세 계");
    assert_eq!(nt.original_words().len(), 4);
  }

  #[test]
  fn empty_after_punct_only_errors() {
    let n = KoreanNormalizer::new();
    let err = n.normalize("。、！？").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn does_not_use_word_delimiter() {
    // Char-segmented across Hangul: whitespace between glyphs is
    // an indexing artefact and must NOT trigger `|` insertion in
    // CTC tokenisation.
    let n = KoreanNormalizer::new();
    assert!(!n.use_word_delimiter());
  }
}
