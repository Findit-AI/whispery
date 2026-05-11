//! Japanese text normaliser (character-level).

use std::borrow::Cow;

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// Japanese normaliser: per-character segmentation across kanji /
/// hiragana / katakana, plus CJK + ASCII punctuation strip.
///
/// **v1 scope.** No morphological analysis (MeCab/fugashi) — that
/// requires a runtime dictionary and a non-trivial native
/// dependency. The wav2vec2 JA models commonly used in v1 are
/// character-level CTC, so per-character segmentation is the
/// correct granularity for this stage. Future v2 can plug in a
/// MeCab-backed normaliser by adding a new `MeCabJapaneseNormalizer`
/// variant; the trait is already general enough.
///
/// **Half-width vs. full-width Latin:** kept as whitespace tokens
/// (no half/full-width folding) so loanwords like "コーヒー"
/// segment per-katakana but "USA" stays as one token.
///
/// **Voice marks:** `゛` and `゜` (combining sound marks) are
/// preserved on the previous character because Unicode normalises
/// them as part of the same grapheme; the simple `chars()` walk
/// emits them as separate "words" only if they appear *standalone*,
/// which is rare in clean Whisper output.
#[derive(Default, Clone, Copy, Debug)]
pub struct JapaneseNormalizer;

impl JapaneseNormalizer {
  /// Construct a Japanese normaliser.
  pub const fn new() -> Self {
    Self
  }
}

fn is_jp_punct(c: char) -> bool {
  matches!(
    c,
    '\u{3002}' // 。
            | '\u{3001}' // 、
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
            | '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')'
            | '[' | ']' | '{' | '}' | '-'
  )
}

impl TextNormalizer for JapaneseNormalizer {
  /// Japanese is character-segmented across kanji / hiragana /
  /// katakana: the whitespace this normaliser emits between every
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
    // char (kanji, hiragana, katakana, Latin, digit, ...) becomes
    // its own word; whitespace and JP/ASCII punctuation are
    // dropped. Latin chars get lowercased before emit so they
    // hit the wav2vec2-large-xlsr-53-japanese vocab the same way
    // whisperX's `char_.lower()` lookup does.
    //
    // The `is_japanese_segmenting_char` helper is no longer
    // used inside the loop — kept for the docstring's intent
    // (kanji/hiragana/katakana are the *expected* chars from
    // whisper-ASR transcripts of Japanese audio) but we don't
    // gate on it: any non-skipped char becomes a word, matching
    // whisperX exactly.
    for c in text.chars() {
      if c.is_whitespace() || is_jp_punct(c) {
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
  fn hiragana_per_char() {
    let n = JapaneseNormalizer::new();
    let nt = n.normalize("ありがとう").unwrap();
    assert_eq!(nt.normalized(), "あ り が と う");
    assert_eq!(nt.original_words().len(), 5);
  }

  #[test]
  fn katakana_per_char() {
    let n = JapaneseNormalizer::new();
    let nt = n.normalize("コーヒー").unwrap();
    assert_eq!(nt.normalized(), "コ ー ヒ ー");
  }

  #[test]
  fn kanji_per_char() {
    let n = JapaneseNormalizer::new();
    let nt = n.normalize("日本語").unwrap();
    assert_eq!(nt.normalized(), "日 本 語");
  }

  #[test]
  fn mixed_kanji_kana_strip_punct() {
    let n = JapaneseNormalizer::new();
    let nt = n.normalize("私は日本語を話します。").unwrap();
    // 私 は 日 本 語 を 話 し ま す
    assert_eq!(nt.original_words().len(), 10);
  }

  #[test]
  fn latin_chars_segment_per_character() {
    let n = JapaneseNormalizer::new();
    let nt = n.normalize("USA で勉強").unwrap();
    // Per-character (matches whisperX's LANGUAGES_WITHOUT_SPACES
    // for `ja`): each Latin letter is its own word, lowercased.
    // で 勉 強 segment as before.
    assert_eq!(nt.normalized(), "u s a で 勉 強");
    assert_eq!(nt.original_words().len(), 6);
    assert_eq!(nt.original_words()[0], "u");
    assert_eq!(nt.original_words()[1], "s");
    assert_eq!(nt.original_words()[2], "a");
    assert_eq!(nt.original_words()[3], "で");
  }

  #[test]
  fn empty_after_punct_only_errors() {
    let n = JapaneseNormalizer::new();
    let err = n.normalize("。、！？").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn does_not_use_word_delimiter() {
    // Char-segmented across kanji/hiragana/katakana: whitespace
    // between glyphs is an indexing artefact and must NOT trigger
    // `|` insertion in CTC tokenisation.
    let n = JapaneseNormalizer::new();
    assert!(!n.use_word_delimiter());
  }
}
