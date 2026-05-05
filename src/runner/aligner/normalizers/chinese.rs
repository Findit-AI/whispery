//! Chinese text normaliser (character-level, WhisperX-compatible).

use alloc::{borrow::Cow, string::String, vec::Vec};

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// Chinese normaliser: every character becomes its own word.
///
/// **Per-character segmentation, including Latin.** Mirrors
/// WhisperX's `LANGUAGES_WITHOUT_SPACES` contract for `zh`:
/// `whisperx/alignment.py` iterates the source text character
/// by character and treats EACH char as its own alignment
/// unit (`word_idx += 1` after every char), regardless of
/// script. So a Latin loanword `"Python"` in Chinese text
/// emits 6 separate alignment entries (`p`, `y`, `t`, `h`, `o`,
/// `n`) — same as whisperX. This guarantees per-word IoU
/// parity in the testaudioset suite when both runners share
/// the same wav2vec2 ZH ONNX.
///
/// **Lowercase ASCII letters** before emitting, because the
/// jonatasgrosman ZH vocab is uppercase-only (whisperX does
/// `char_.lower()` then dictionary lookup; the lowercase form
/// is what ends up in tokens).
///
/// **Skip:** whitespace, ASCII punctuation, and CJK full-width
/// punctuation (`。 ， ！ ？ …`). Han glyphs themselves are
/// never stripped.
///
/// **Surface preservation:** `original_words` carries each
/// emitted character as-is (Han kept verbatim, Latin
/// lowercased) so step 9 of the alignment algorithm emits the
/// expected glyph. Pipelines that need traditional-vs-simplified
/// fidelity for Han chars get it; Latin mid-Chinese loses
/// case info (matches whisperX).
#[derive(Default, Clone, Copy, Debug)]
pub struct ChineseNormalizer;

impl ChineseNormalizer {
  /// Construct a Chinese normaliser.
  pub const fn new() -> Self {
    Self
  }
}

fn is_cjk_punct(c: char) -> bool {
  matches!(
    c,
    '\u{3002}' // 。
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
            | '\u{3001}' // 、
            | '\u{30FB}' // ・
  )
}

fn is_ascii_punct(c: char) -> bool {
  matches!(
    c,
    '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '-'
  )
}

fn is_punct_either(c: char) -> bool {
  is_cjk_punct(c) || is_ascii_punct(c)
}

impl TextNormalizer for ChineseNormalizer {
  /// Chinese is character-segmented: the whitespace this normaliser
  /// emits between every glyph is purely an indexing device,
  /// not a real word boundary. Returning `false` here keeps the
  /// tokeniser from forcing `|` between every glyph in the CTC
  /// alignment graph.
  fn use_word_delimiter(&self) -> bool {
    false
  }

  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();

    // WhisperX-matching per-character iteration. Every non-skipped
    // char becomes its own word; Latin letters get lowercased on
    // both the surface (`original_words`) and the normalised
    // string before emit. See type-level doc.
    for c in text.chars() {
      if c.is_whitespace() || is_punct_either(c) {
        continue;
      }
      // Lowercase Latin (matches whisperX's `char.lower()`).
      // Han glyphs and other non-Latin chars lowercase to
      // themselves under Unicode `to_lowercase`, so this is a
      // no-op for them and we don't special-case the script.
      // The `to_lowercase()` iterator can yield multiple
      // codepoints for a single input char (e.g. ß → ss); for
      // the wav2vec2 ZH vocab those expansions don't occur in
      // practice (Han + ASCII letters + digits), so we collect
      // them all into one alignment word — preserving the
      // 1-input-char ↔ 1-output-word contract.
      let lowered: String = c.to_lowercase().collect();
      if !normalized.is_empty() {
        normalized.push(' ');
      }
      normalized.push_str(&lowered);
      // For the surface form: ASCII letters use the lowered
      // string (matches whisperX, vocab is upper-only and we
      // lowered for the lookup); other chars use the raw glyph
      // so traditional-vs-simplified Han stays intact.
      if c.is_ascii_alphabetic() {
        original_words.push(Cow::Owned(lowered));
      } else {
        let mut buf = [0u8; 4];
        let s: &str = c.encode_utf8(&mut buf);
        let owned = String::from(s);
        original_words.push(Cow::Owned(owned));
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
  fn pure_chinese_segments_per_glyph() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("你好世界").unwrap();
    assert_eq!(nt.normalized(), "你 好 世 界");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "你");
    assert_eq!(nt.original_words()[3], "界");
  }

  #[test]
  fn cjk_punctuation_stripped() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("你好，世界。").unwrap();
    assert_eq!(nt.normalized(), "你 好 世 界");
    assert_eq!(nt.original_words().len(), 4);
  }

  #[test]
  fn mixed_chinese_and_latin() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("我用 Python 写代码").unwrap();
    // Per-character: each Han glyph AND each Latin letter is
    // its own word. "Python" → 6 separate words p y t h o n.
    // Whitespace is dropped. Matches whisperX's
    // LANGUAGES_WITHOUT_SPACES contract for `zh`.
    assert_eq!(nt.normalized(), "我 用 p y t h o n 写 代 码");
    assert_eq!(nt.original_words().len(), 11);
    assert_eq!(nt.original_words()[2], "p");
    assert_eq!(nt.original_words()[7], "n");
    assert_eq!(nt.original_words()[10], "码");
  }

  #[test]
  fn empty_after_punct_only_errors() {
    let n = ChineseNormalizer::new();
    let err = n.normalize("。，！？").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn surface_glyph_preserved_in_original_words() {
    let n = ChineseNormalizer::new();
    let nt = n.normalize("龜").unwrap(); // Traditional turtle
    assert_eq!(nt.original_words()[0], "龜");
  }

  #[test]
  fn does_not_use_word_delimiter() {
    // Char-segmented: whitespace between glyphs is an indexing
    // device, not a real word boundary. Tokenisation must NOT
    // insert `|` between every Han glyph.
    let n = ChineseNormalizer::new();
    assert!(!n.use_word_delimiter());
  }
}
