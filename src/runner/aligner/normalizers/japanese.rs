//! Japanese text normaliser (character-level). See spec §6.3.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

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

fn is_japanese_segmenting_char(c: char) -> bool {
    let code = c as u32;
    matches!(
        code,
        // Kanji
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0xF900..=0xFAFF
            // Hiragana
            | 0x3040..=0x309F
            // Katakana
            | 0x30A0..=0x30FF
            // Half-width katakana
            | 0xFF66..=0xFF9D
    )
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
    fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
        let mut normalized = String::with_capacity(text.len());
        let mut original_words: Vec<Cow<'a, str>> = Vec::new();

        let mut latin_run_start: Option<usize> = None;

        let flush_latin_run =
            |start: usize, end: usize, normalized: &mut String, words: &mut Vec<Cow<'a, str>>| {
                let raw = &text[start..end];
                let stripped = raw
                    .trim_start_matches(is_jp_punct)
                    .trim_end_matches(is_jp_punct);
                if stripped.is_empty() {
                    return;
                }
                let lower = stripped.to_lowercase();
                if !normalized.is_empty() {
                    normalized.push(' ');
                }
                normalized.push_str(&lower);
                let original = &text[start..end];
                words.push(Cow::Borrowed(original));
            };

        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let c = match text[i..].chars().next() {
                Some(c) => c,
                None => break,
            };
            let len = c.len_utf8();

            if c.is_whitespace() {
                if let Some(start) = latin_run_start.take() {
                    flush_latin_run(start, i, &mut normalized, &mut original_words);
                }
            } else if is_japanese_segmenting_char(c) {
                if let Some(start) = latin_run_start.take() {
                    flush_latin_run(start, i, &mut normalized, &mut original_words);
                }
                if !normalized.is_empty() {
                    normalized.push(' ');
                }
                let glyph = &text[i..i + len];
                normalized.push_str(glyph);
                original_words.push(Cow::Borrowed(glyph));
            } else if is_jp_punct(c) {
                if let Some(start) = latin_run_start.take() {
                    flush_latin_run(start, i, &mut normalized, &mut original_words);
                }
            } else {
                if latin_run_start.is_none() {
                    latin_run_start = Some(i);
                }
            }
            i += len;
        }
        if let Some(start) = latin_run_start.take() {
            flush_latin_run(start, bytes.len(), &mut normalized, &mut original_words);
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
    fn latin_run_stays_as_token() {
        let n = JapaneseNormalizer::new();
        let nt = n.normalize("USA で勉強").unwrap();
        // USA -> "usa" (lowercased latin run); で 勉 強 segment
        assert_eq!(nt.normalized(), "usa で 勉 強");
    }

    #[test]
    fn empty_after_punct_only_errors() {
        let n = JapaneseNormalizer::new();
        let err = n.normalize("。、！？").unwrap_err();
        assert!(matches!(err, NormalizationError::EmptyText));
    }
}
