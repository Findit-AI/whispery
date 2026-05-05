//! Concrete `TextNormalizer` implementations.
//!
//! v1 ships English / Chinese / Japanese. Future versions add
//! more languages by adding files here and re-exporting from
//! `runner::aligner`.

mod chinese;
mod english;
mod japanese;
#[cfg(test)]
mod tests;

pub use chinese::ChineseNormalizer;
pub use english::EnglishNormalizer;
pub use japanese::JapaneseNormalizer;

use crate::runner::aligner::normalizer::TextNormalizer;
use crate::types::Lang;

/// Pick a built-in normalizer best suited to the given language,
/// or `None` if whispery doesn't yet ship one whose vocabulary
/// rules match the language's typical wav2vec2 tokenizer.
///
/// Mirrors WhisperX's `DEFAULT_ALIGN_MODELS_*` coverage: every
/// language whisperX has a default model for can plug a matching
/// normalizer through this function — except those whose script
/// needs custom rules whispery hasn't shipped yet (Arabic,
/// Cyrillic, Korean, Devanagari, Hebrew, Telugu, Malayalam,
/// Georgian, Greek, Thai). For those, `None` is returned and the
/// caller must `Box::new` a custom `TextNormalizer` impl.
///
/// Three buckets:
///
/// 1. **English-shape** — Latin script, lowercase a-z + apostrophe,
///    `|` word delimiter on the wav2vec2 vocab. Covers en, fr, de,
///    es, it, nl, pt, ca, da, sv, id, eu, gl, hr, sl, sk, hu, fi,
///    no, nn, ro, tl, vi, tr, cs, pl, lv. The
///    `EnglishNormalizer` strips ASCII punctuation and casefolds
///    Latin script — that's correct for any of those (and good
///    enough for Latin-extended diacritics, which the
///    wav2vec2-large-xlsr-53-* tokenizers commonly accept inside
///    their vocab).
///
/// 2. **Chinese-character** (`zh`): per-char segmentation,
///    no `|` word delimiter, `ChineseNormalizer`.
///
/// 3. **Japanese kana/kanji** (`ja`): per-char segmentation
///    over kanji + hiragana + katakana, no word delimiter,
///    `JapaneseNormalizer`.
///
/// Returning `None` is whispery saying "we don't have a
/// pre-built normalizer for this language; you must supply one
/// yourself". This is intentional — silently picking the wrong
/// normalizer (e.g., feeding Arabic text through the
/// `EnglishNormalizer`) would bake in bugs that only surface
/// during alignment as nonsensical IoU.
pub fn default_normalizer_for(
  lang: &Lang,
) -> Option<alloc::boxed::Box<dyn TextNormalizer>> {
  use alloc::boxed::Box;
  match lang {
    // Chinese: char-level segmentation, no word delimiter.
    Lang::Zh | Lang::Yue => Some(Box::new(ChineseNormalizer::new())),

    // Japanese: kanji/hiragana/katakana char-level, no delimiter.
    Lang::Ja => Some(Box::new(JapaneseNormalizer::new())),

    // English-shape Latin script: ASCII lowercase + boundary
    // punctuation strip + `|` word delimiter. Works for any
    // language whose wav2vec2 tokenizer is character-or-grapheme
    // level over Latin (with or without diacritics).
    Lang::En
    | Lang::Fr
    | Lang::De
    | Lang::Es
    | Lang::It
    | Lang::Nl
    | Lang::Pt
    | Lang::Ca
    | Lang::Da
    | Lang::Sv
    | Lang::Id
    | Lang::Eu
    | Lang::Gl
    | Lang::Hr
    | Lang::Sl
    | Lang::Sk
    | Lang::Hu
    | Lang::Fi
    | Lang::No
    | Lang::Nn
    | Lang::Ro
    | Lang::Tl
    | Lang::Vi
    | Lang::Tr
    | Lang::Cs
    | Lang::Pl
    | Lang::Lv => Some(Box::new(EnglishNormalizer::new())),

    // Languages WhisperX supports but whispery has no normalizer
    // for yet (different scripts that need custom punctuation /
    // casing / RTL rules). Caller must supply a custom
    // TextNormalizer.
    //
    // - Arabic (ar): RTL, diacritic-strip, custom punctuation
    // - Hebrew (he): RTL, niqqud handling
    // - Russian (ru), Ukrainian (uk): Cyrillic case-folding
    // - Korean (ko): Hangul jamo decomposition
    // - Hindi (hi): Devanagari + virama / nukta
    // - Telugu (te), Malayalam (ml): Brahmi-derived scripts
    // - Greek (el): polytonic→monotonic
    // - Persian (fa): RTL with Arabic-derived script + Persian
    //   digits
    // - Urdu (ur): RTL, Persian-derived
    // - Georgian (ka): Mkhedruli / Asomtavruli
    _ => None,
  }
}

#[cfg(test)]
mod default_normalizer_tests {
  use super::*;

  #[test]
  fn english_languages_get_english_normalizer() {
    for lang in [Lang::En, Lang::Fr, Lang::De, Lang::Es, Lang::Pt, Lang::Sv] {
      let n = default_normalizer_for(&lang);
      assert!(n.is_some(), "{lang:?} must resolve to EnglishNormalizer");
      // Sanity: this normalizer asserts use_word_delimiter (the
      // English-shape contract).
      assert!(
        n.unwrap().use_word_delimiter(),
        "Latin-script normalizer must enable | word delimiter for {lang:?}",
      );
    }
  }

  #[test]
  fn cjk_languages_get_per_character_normalizers() {
    for lang in [Lang::Zh, Lang::Yue, Lang::Ja] {
      let n = default_normalizer_for(&lang);
      assert!(n.is_some(), "{lang:?} must resolve to a normalizer");
      assert!(
        !n.unwrap().use_word_delimiter(),
        "CJK normalizer must NOT enable | word delimiter for {lang:?}",
      );
    }
  }

  #[test]
  fn unsupported_languages_return_none() {
    // Arabic, Korean, Hindi, Russian — distinct scripts requiring
    // custom normalizers whispery hasn't shipped.
    for lang in [Lang::Ar, Lang::Ko, Lang::Hi, Lang::Ru, Lang::He, Lang::El] {
      assert!(
        default_normalizer_for(&lang).is_none(),
        "{lang:?} must return None — no built-in normalizer yet",
      );
    }
  }
}
