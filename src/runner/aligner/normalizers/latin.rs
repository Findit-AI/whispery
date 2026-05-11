//! Latin-script text normaliser, parameterised by [`Lang`].
//!
//! Generalises the original `english.rs` normaliser into a
//! single per-language-quirks-driven implementation. The
//! word-segmentation core is unchanged; per-language data
//! (extra letters, opening-punctuation glyphs, apostrophe
//! semantics) flips a small number of switches.
//!
//! # Backward compatibility
//!
//! [`crate::EnglishNormalizer`] is now a thin wrapper that
//! constructs a `LatinNormalizer::new(Lang::En)` — every existing
//! consumer keeps compiling. The old `english.rs` test contract
//! (lowercase + boundary-punct strip + apostrophes-survive-inside,
//! per-piece surface spans on hyphen / em-dash / slash splits)
//! is preserved verbatim by the `Lang::En` configuration.
//!
//! # Per-language quirks
//!
//! - Spanish (`Es`): opens with `¿` / `¡` (must strip at the
//!   word boundary just like ASCII `?` / `!`); accepts `ñ` / `Ñ`
//!   inline.
//! - German (`De`): umlauts `ä` / `ö` / `ü`, sharp s `ß`, plus
//!   capitalised forms — all ordinary letters, no extra rules
//!   beyond letting them survive lowercasing.
//! - French (`Fr`): apostrophes glued to the *preceding* word
//!   (`l'eau` → `l' eau`) so the wav2vec2-large-xlsr-53-french
//!   tokeniser sees the apostrophe as a clitic boundary. This
//!   matches WhisperX's behaviour for `fr` (see WhisperX
//!   `align.py::PUNKT_ABBREVIATIONS` + the `WhisperX`
//!   `LANGUAGES_WITHOUT_SPACES` list, which excludes `fr`).
//! - Italian (`It`): apostrophes glued to the preceding word
//!   (`dell'arte` → `dell' arte`), same rationale as French —
//!   the jonatasgrosman wav2vec2-large-xlsr-53-italian tokeniser
//!   was trained on transcripts where clitic apostrophes attach
//!   to the article, not the noun.
//! - Portuguese (`Pt`): cedilla `ç`, tilde-vowels `ã` / `õ`,
//!   acute / grave accents — all surface as ordinary letters
//!   under the wav2vec2-large-xlsr-53-portuguese vocab; no
//!   special boundary rules beyond letting them through
//!   `is_word_punct`'s strip phase.

use std::borrow::Cow;

use crate::{
  runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer},
  types::Lang,
};

/// Per-language behaviour switches consumed by the shared
/// segmentation logic. Constructed by [`LatinNormalizer::new`]
/// based on the input [`Lang`].
#[derive(Clone, Copy, Debug)]
struct LatinRules {
  /// If `true`, an apostrophe (`'`, `\u{2019}`) at the END of a
  /// whitespace-bounded token is treated as a word boundary that
  /// SPLITS the token: `l'eau` → two pieces (`l'`, `eau`). The
  /// apostrophe stays attached to the LEFT piece, matching the
  /// way romance-language wav2vec2 tokenisers were trained
  /// (clitic article + noun).
  ///
  /// English (`Lang::En`) keeps `false` — `don't` stays one word
  /// because the wav2vec2-base-960h vocab encodes `'` inline.
  splits_clitic_apostrophe: bool,
}

impl LatinRules {
  fn for_lang(lang: &Lang) -> Self {
    match lang {
      Lang::Fr | Lang::It => Self {
        splits_clitic_apostrophe: true,
      },
      _ => Self {
        splits_clitic_apostrophe: false,
      },
    }
  }
}

/// Latin-script normaliser parameterised by [`Lang`]. See module
/// docs for the per-language quirks each variant enables.
///
/// **Surface-form invariant.** The normaliser does not expand
/// contractions or fold diacritics — `"don't"` stays one word
/// (En) and `"café"` stays `"café"` (any Latin lang). The
/// `original_words` map points each normalised-word index back
/// to the source-text substring exactly as Whisper produced it.
///
/// **Punctuation handling.** ASCII `[ . , ! ? ; : " ' ( ) [ ] { } - — – ]`
/// plus Spanish opening punctuation `¿` / `¡` are stripped from
/// word boundaries. Internal apostrophes inside English
/// contractions (e.g., `'` in `don't`) survive into the
/// normalised form so the wav2vec2 tokenizer aligns the
/// apostrophe character directly. For French / Italian, clitic
/// apostrophes split: `l'eau` → `l'` + `eau`.
///
/// **Empty result.** If normalisation produces zero words,
/// `normalize` returns [`NormalizationError::EmptyText`].
/// `Aligner::align` short-circuits this to
/// `Ok(AlignmentResult::new(Vec::new()))` so a punctuation-only
/// transcript surfaces as `Transcript { text, words: [] }`
/// rather than `Event::Error`.
#[derive(Clone, Copy, Debug)]
pub struct LatinNormalizer {
  rules: LatinRules,
}

impl LatinNormalizer {
  /// Construct a Latin normaliser for `lang`. Not `const fn`
  /// because [`Lang`] is not `Drop`-free in const context (the
  /// `Other(SmolStr)` variant carries a heap-or-inline pointer).
  pub fn new(lang: Lang) -> Self {
    Self {
      rules: LatinRules::for_lang(&lang),
    }
  }

  /// Construct a Latin normaliser using `Lang::En` rules. Kept as
  /// a `const fn` shorthand for the common case + back-compat
  /// with consumers that don't carry a `Lang` value (e.g.
  /// [`crate::EnglishNormalizer`]).
  pub const fn english() -> Self {
    Self {
      rules: LatinRules {
        splits_clitic_apostrophe: false,
      },
    }
  }
}

fn is_word_punct(c: char) -> bool {
  matches!(
    c,
    '.' | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | '"'
            | '\'' // ASCII apostrophe — see `strip_word_punct` note below.
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '-'
            | '\u{2014}' // em dash
            | '\u{2013}' // en dash
            | '\u{201C}' // left double quote
            | '\u{201D}' // right double quote
            | '\u{2018}' // left single quote
            | '\u{2019}' // right single quote (Unicode apostrophe)
            // Spanish opening punctuation. `¿` / `¡` open
            // questions and exclamations and must strip the same
            // way `?` / `!` do at the trailing edge — wav2vec2's
            // vocab doesn't carry these glyphs, so leaving them
            // inline forces an `<unk>` rejection on the whole
            // chunk.
            | '\u{00BF}' // ¿
            | '\u{00A1}' // ¡
  )
}

// Note on ASCII `'` boundary handling. `is_word_punct` includes
// `'`, but `strip_word_punct` only trims leading/trailing matches.
// Internal apostrophes inside English contractions like `don't`
// survive the trim — wav2vec2-base-960h aligns them as a single
// word with the `'` glyph emitted inline.
//
// For Romance languages (French, Italian) where the apostrophe
// is a CLITIC boundary, the trailing-apostrophe-as-word-boundary
// rule fires before the punctuation trim, splitting `l'eau` into
// `l'` + `eau`. The trim then strips the dangling `'` from the
// right edge of `l'` only if no other letters precede it.

fn strip_word_punct(s: &str) -> &str {
  let trimmed_left = s.trim_start_matches(is_word_punct);
  trimmed_left.trim_end_matches(is_word_punct)
}

/// True for characters that join two real words inside a single
/// whitespace-bounded token but are themselves never spoken,
/// e.g. `hello-world`, `two—three`, `and/or`. The wav2vec2 vocab
/// doesn't cover these glyphs, so the normaliser treats them as
/// word boundaries: each side becomes its own normalised word,
/// both pointing back to their own surface slices.
///
/// We do NOT split on apostrophes here — apostrophe handling is
/// a per-language rule (English keeps them inline, French /
/// Italian split as a clitic boundary) and runs in a separate
/// pass before this generic separator split.
fn is_internal_separator(c: char) -> bool {
  matches!(
    c,
    '-' | '/' | '\u{2010}' // hyphen
                | '\u{2013}' // en-dash
                | '\u{2014}' // em-dash
                | '\u{2015}' // horizontal bar
  )
}

fn is_clitic_apostrophe(c: char) -> bool {
  matches!(c, '\'' | '\u{2019}')
}

fn lowercase_for_match(s: &str) -> String {
  s.to_lowercase()
}

/// Split a whitespace-bounded token at clitic apostrophes that
/// glue an article / preposition to the following word
/// (`l'eau` → `["l'", "eau"]`). The apostrophe stays attached
/// to the LEFT piece because the wav2vec2-large-xlsr-53-{fr,it}
/// tokenisers were trained on transcripts where the clitic
/// surface form keeps its apostrophe.
///
/// Returns owned `(piece_str, byte_offset_within_token)` pairs:
/// the offsets let the caller reconstruct borrowed slices into
/// the original text. We split greedily on every clitic
/// apostrophe inside the token; consecutive apostrophes are
/// rare in real text but handled by yielding empty pieces that
/// the caller filters out.
fn split_at_clitic_apostrophes(token: &str) -> Vec<(String, usize)> {
  let mut pieces: Vec<(String, usize)> = Vec::new();
  let mut current_start: usize = 0;
  let chars: Vec<(usize, char)> = token.char_indices().collect();
  let mut i = 0usize;
  while i < chars.len() {
    let (byte_idx, ch) = chars[i];
    if is_clitic_apostrophe(ch) {
      // Take chars [current_start ..= byte_idx] (apostrophe
      // included on the left piece). The next piece starts at
      // the next char's byte offset (or end-of-token).
      let after_apos = byte_idx + ch.len_utf8();
      let left = &token[current_start..after_apos];
      pieces.push((String::from(left), current_start));
      current_start = after_apos;
    }
    i += 1;
  }
  if current_start < token.len() {
    pieces.push((String::from(&token[current_start..]), current_start));
  } else if pieces.is_empty() {
    // All-empty token — push the original empty so callers see
    // a single zero-length entry rather than nothing.
    pieces.push((String::new(), 0));
  }
  pieces
}

impl TextNormalizer for LatinNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();
    let mut wildcards_per_word: Vec<crate::runner::aligner::normalizer::WildcardBoundary> =
      Vec::new();

    for (token_start, raw_token) in token_spans(text) {
      // Per-language clitic-apostrophe split runs FIRST so the
      // resulting sub-tokens go through the same boundary-punct
      // strip + separator-split pipeline as ordinary tokens.
      // English (rules.splits_clitic_apostrophe = false) skips
      // this and feeds the whole raw_token through unchanged,
      // matching the legacy `EnglishNormalizer` contract.
      //
      // The `is_clitic_left` flag marks pieces whose RIGHT edge
      // ends in the clitic apostrophe by design (e.g., `l'` from
      // `l'eau`). For those pieces we skip the trailing
      // apostrophe strip — the apostrophe IS the surface form.
      let token_pieces: Vec<(String, usize, bool)> =
        if self.rules.splits_clitic_apostrophe && raw_token.chars().any(is_clitic_apostrophe) {
          let split = split_at_clitic_apostrophes(raw_token);
          let last_idx = split.len().saturating_sub(1);
          split
            .into_iter()
            .enumerate()
            .filter(|(_, (p, _))| !p.is_empty())
            .map(|(i, (p, off))| {
              // Clitic-left pieces are every piece EXCEPT the last
              // — the splitter always puts the apostrophe at the
              // end of the piece it's emitted with, so any non-
              // final piece ends in `'`.
              let is_clitic_left = i != last_idx;
              (p, off, is_clitic_left)
            })
            .collect()
        } else {
          vec![(String::from(raw_token), 0usize, false)]
        };

      for (sub_token, sub_offset, is_clitic_left) in &token_pieces {
        let sub_start = token_start + sub_offset;
        let sub_len = sub_token.len();
        // Clitic-left pieces (`l'`, `dell'`) intentionally end
        // in `'`. Strip leading punctuation as usual but keep
        // the trailing apostrophe — it's the surface form the
        // wav2vec2-large-xlsr-53-{fr,it} tokenizer expects.
        let stripped = if *is_clitic_left {
          sub_token.trim_start_matches(is_word_punct)
        } else {
          strip_word_punct(sub_token)
        };
        if stripped.is_empty() {
          continue;
        }

        // Reconstruct the borrowed slice for the original sub-
        // token (without punctuation strip, so Whisper's
        // surface form is preserved verbatim — punctuation
        // included).
        let original_slice: &'a str = &text[sub_start..sub_start + sub_len];

        // Wildcards: count of source chars stripped from each
        // boundary (prefix vs suffix). Source-order placement
        // matters — WhisperX inserts a `*` placeholder per
        // source char at its actual position, so leading
        // punctuation like `"hello` keeps its `*` BEFORE the
        // letters and trailing punctuation like `hello"` keeps
        // its `*` AFTER.
        //
        // Clitic-left pieces have `suffix_stripped = 0` because
        // we explicitly do NOT strip the trailing apostrophe.
        let trimmed_left = sub_token.trim_start_matches(is_word_punct);
        let prefix_stripped: u32 =
          (sub_token.chars().count() - trimmed_left.chars().count()) as u32;
        let suffix_stripped: u32 = if *is_clitic_left {
          0
        } else {
          (trimmed_left.chars().count() - stripped.chars().count()) as u32
        };

        // Split on internal separators (`Hello-World` →
        // `["Hello", "World"]`). Each piece is a real word the
        // wav2vec2 vocab can encode.
        if stripped.contains(is_internal_separator) {
          let pieces: Vec<&str> = stripped
            .split(is_internal_separator)
            .filter(|p| !p.is_empty())
            .collect();
          let last_idx = pieces.len().saturating_sub(1);
          for (pi, piece_orig) in pieces.iter().enumerate() {
            let piece_lower = lowercase_for_match(piece_orig);
            if !normalized.is_empty() {
              normalized.push(' ');
            }
            normalized.push_str(&piece_lower);
            // `piece_orig` is a slice into `stripped`, which is
            // itself a slice into `sub_token` (an owned
            // `String`). Use Cow::Owned so the lifetime is
            // self-contained — the borrow checker can't trace
            // the slice back to `text`.
            original_words.push(Cow::Owned(String::from(*piece_orig)));
            let prefix = if pi == 0 { prefix_stripped } else { 0 };
            let suffix = if pi == last_idx { suffix_stripped } else { 0 };
            wildcards_per_word.push(crate::runner::aligner::normalizer::WildcardBoundary::new(
              prefix, suffix,
            ));
          }
        } else {
          // No internal separator — the sub-token is one piece.
          let lower = lowercase_for_match(stripped);
          if !normalized.is_empty() {
            normalized.push(' ');
          }
          normalized.push_str(&lower);
          original_words.push(Cow::Borrowed(original_slice));
          wildcards_per_word.push(crate::runner::aligner::normalizer::WildcardBoundary::new(
            prefix_stripped,
            suffix_stripped,
          ));
        }
      }
    }

    if original_words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    Ok(NormalizedText::with_wildcards(
      normalized,
      original_words,
      wildcards_per_word,
    ))
  }
}

/// Iterate `(byte_offset, slice)` for whitespace-separated
/// tokens. Equivalent to `text.split_whitespace()` but yields
/// starting byte offsets so callers can reconstruct borrowed
/// slices into the input.
fn token_spans(text: &str) -> impl Iterator<Item = (usize, &str)> + '_ {
  let mut iter = text.split_whitespace();
  core::iter::from_fn(move || {
    let token = iter.next()?;
    let token_start = (token.as_ptr() as usize).saturating_sub(text.as_ptr() as usize);
    Some((token_start, token))
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  // --- English (Lang::En) parity tests --------------------------
  // These mirror the original `english.rs` test surface verbatim.
  // The `LatinNormalizer::new(Lang::En)` configuration must match
  // the legacy `EnglishNormalizer` exactly.

  fn en() -> LatinNormalizer {
    LatinNormalizer::new(Lang::En)
  }

  #[test]
  fn en_lowercase_and_strip_punct() {
    let nt = en().normalize("Hello, World!").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello,");
    assert_eq!(nt.original_words()[1], "World!");
  }

  #[test]
  fn en_contraction_stays_one_word_with_apostrophe_inline() {
    let nt = en().normalize("Don't go.").unwrap();
    assert_eq!(nt.normalized(), "don't go");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Don't");
    assert_eq!(nt.original_words()[1], "go.");
  }

  #[test]
  fn en_em_dash_splits_into_per_piece_surface_spans() {
    let nt = en().normalize("hello\u{2014}world").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "hello");
    assert_eq!(nt.original_words()[1], "world");
  }

  #[test]
  fn en_hyphen_compound_splits_into_per_piece_surface_spans() {
    let nt = en().normalize("Hello-World").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello");
    assert_eq!(nt.original_words()[1], "World");
  }

  #[test]
  fn en_slash_alternation_splits_into_per_piece_surface_spans() {
    let nt = en().normalize("and/or").unwrap();
    assert_eq!(nt.normalized(), "and or");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "and");
    assert_eq!(nt.original_words()[1], "or");
  }

  #[test]
  fn en_empty_input_errors() {
    let err = en().normalize("   .,!?  ").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn en_casing_preserved_in_original_words() {
    let nt = en().normalize("The Quick BROWN Fox.").unwrap();
    assert_eq!(nt.normalized(), "the quick brown fox");
    assert_eq!(nt.original_words()[1], "Quick");
    assert_eq!(nt.original_words()[2], "BROWN");
    assert_eq!(nt.original_words()[3], "Fox.");
  }

  #[test]
  fn en_contraction_inside_sentence_stays_intact() {
    let nt = en().normalize("I won't be late.").unwrap();
    assert_eq!(nt.normalized(), "i won't be late");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "I");
    assert_eq!(nt.original_words()[1], "won't");
    assert_eq!(nt.original_words()[2], "be");
    assert_eq!(nt.original_words()[3], "late.");
  }

  #[test]
  fn en_apostrophe_word_passes_through_lowercased() {
    let nt = en().normalize("O'Brien rocks.").unwrap();
    assert_eq!(nt.normalized(), "o'brien rocks");
  }

  #[test]
  fn en_boundary_ascii_apostrophes_are_stripped() {
    let nt = en().normalize("'hello'").unwrap();
    assert_eq!(nt.normalized(), "hello");
    assert_eq!(nt.original_words()[0], "'hello'");
  }

  #[test]
  fn en_boundary_apostrophe_around_contraction_keeps_internal() {
    let nt = en().normalize("'don't'").unwrap();
    assert_eq!(nt.normalized(), "don't");
    assert_eq!(nt.original_words().len(), 1);
    assert_eq!(nt.original_words()[0], "'don't'");
  }

  #[test]
  fn en_trailing_possessive_apostrophe_strips() {
    let nt = en().normalize("the dogs'.").unwrap();
    assert_eq!(nt.normalized(), "the dogs");
  }

  #[test]
  fn en_uses_word_delimiter() {
    assert!(en().use_word_delimiter());
  }

  #[test]
  fn en_em_dash_and_slash_split() {
    let nt = en()
      .normalize("two\u{2014}three and/or four\u{2013}five")
      .unwrap();
    assert_eq!(nt.normalized(), "two three and or four five");
    assert_eq!(nt.original_words().len(), 6);
  }

  #[test]
  fn en_pure_separator_token_is_dropped() {
    let nt = en().normalize("hello --- world").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
  }

  #[test]
  fn en_collapses_consecutive_internal_separators() {
    let nt = en().normalize("foo\u{2014}/-bar").unwrap();
    assert_eq!(nt.normalized(), "foo bar");
    assert_eq!(nt.original_words().len(), 2);
  }

  // --- Spanish (Lang::Es) ---------------------------------------

  /// Spanish opens questions / exclamations with `¿` / `¡`. Both
  /// must strip at the boundary just like ASCII `?` / `!` —
  /// wav2vec2-large-xlsr-53-spanish's vocab doesn't carry these
  /// glyphs, so leaving them inline forces an `<unk>` rejection.
  #[test]
  fn es_strips_inverted_question_and_exclamation() {
    let n = LatinNormalizer::new(Lang::Es);
    let nt = n.normalize("¿Cómo estás? ¡Hola!").unwrap();
    assert_eq!(nt.normalized(), "cómo estás hola");
    assert_eq!(nt.original_words().len(), 3);
    assert_eq!(nt.original_words()[0], "¿Cómo");
    assert_eq!(nt.original_words()[1], "estás?");
    assert_eq!(nt.original_words()[2], "¡Hola!");
  }

  /// `ñ` is an ordinary letter — must survive the lowercase pass
  /// inline (no diacritic folding).
  #[test]
  fn es_preserves_enye() {
    let n = LatinNormalizer::new(Lang::Es);
    let nt = n.normalize("España niño").unwrap();
    assert_eq!(nt.normalized(), "españa niño");
  }

  // --- German (Lang::De) ----------------------------------------

  /// Umlauts (`ä`, `ö`, `ü`) and sharp s (`ß`) are ordinary
  /// letters. The wav2vec2-large-xlsr-53-german vocab carries
  /// them as inline tokens — folding to `ae` / `oe` / `ue` /
  /// `ss` would mismatch the tokenizer.
  #[test]
  fn de_preserves_umlauts_and_sharp_s() {
    let n = LatinNormalizer::new(Lang::De);
    let nt = n.normalize("Mädchen Größe heißt Tür").unwrap();
    assert_eq!(nt.normalized(), "mädchen größe heißt tür");
  }

  /// Capital umlauts lowercase correctly via the standard
  /// Unicode pass.
  #[test]
  fn de_lowercases_capital_umlauts() {
    let n = LatinNormalizer::new(Lang::De);
    let nt = n.normalize("ÄRGER ÖFTER ÜBER").unwrap();
    assert_eq!(nt.normalized(), "ärger öfter über");
  }

  // --- French (Lang::Fr) ----------------------------------------

  /// French clitic articles glue to the following word with an
  /// apostrophe (`l'eau`). The wav2vec2-large-xlsr-53-french
  /// tokenizer was trained on transcripts where this is a TWO-
  /// word boundary (article + noun), so the normaliser splits at
  /// the apostrophe and keeps the apostrophe attached to the
  /// LEFT (clitic) piece.
  ///
  /// Reference: WhisperX `align.py` excludes `fr` from
  /// `LANGUAGES_WITHOUT_SPACES`, so each whitespace-or-clitic
  /// boundary becomes its own word in the CTC graph.
  #[test]
  fn fr_splits_clitic_apostrophe() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("l'eau d'argent").unwrap();
    assert_eq!(nt.normalized(), "l' eau d' argent");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "l'");
    assert_eq!(nt.original_words()[1], "eau");
    assert_eq!(nt.original_words()[2], "d'");
    assert_eq!(nt.original_words()[3], "argent");
  }

  /// Accented vowels (`é`, `è`, `ê`, `à`, `ç`, etc.) are ordinary
  /// letters — the wav2vec2-large-xlsr-53-french vocab carries
  /// them inline.
  #[test]
  fn fr_preserves_accented_vowels_and_cedilla() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("Café à côté ça va.").unwrap();
    assert_eq!(nt.normalized(), "café à côté ça va");
  }

  /// Capitalisation is preserved in `original_words` and folded
  /// to lowercase in `normalized`. Same contract as English.
  #[test]
  fn fr_preserves_casing_in_original_words() {
    let n = LatinNormalizer::new(Lang::Fr);
    let nt = n.normalize("L'Hôtel est ouvert.").unwrap();
    assert_eq!(nt.normalized(), "l' hôtel est ouvert");
    assert_eq!(nt.original_words()[0], "L'");
    assert_eq!(nt.original_words()[1], "Hôtel");
  }

  // --- Italian (Lang::It) ---------------------------------------

  /// Italian has the same clitic-apostrophe behaviour as French
  /// (`dell'arte`, `un'altra`). The wav2vec2-large-xlsr-53-italian
  /// tokeniser expects the apostrophe attached to the article.
  #[test]
  fn it_splits_clitic_apostrophe() {
    let n = LatinNormalizer::new(Lang::It);
    let nt = n.normalize("dell'arte un'altra").unwrap();
    assert_eq!(nt.normalized(), "dell' arte un' altra");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "dell'");
    assert_eq!(nt.original_words()[1], "arte");
  }

  /// Accented vowels (`à`, `è`, `é`, `ì`, `ò`, `ù`) survive as
  /// ordinary letters.
  #[test]
  fn it_preserves_accented_vowels() {
    let n = LatinNormalizer::new(Lang::It);
    let nt = n.normalize("Città però così già più").unwrap();
    assert_eq!(nt.normalized(), "città però così già più");
  }

  // --- Portuguese (Lang::Pt) ------------------------------------

  /// Portuguese has cedilla `ç` and tilde-vowels `ã` / `õ` plus
  /// acute / grave accents. All ordinary letters under the
  /// wav2vec2-large-xlsr-53-portuguese vocab.
  #[test]
  fn pt_preserves_cedilla_and_tilde_vowels() {
    let n = LatinNormalizer::new(Lang::Pt);
    let nt = n.normalize("Coração não são informação").unwrap();
    assert_eq!(nt.normalized(), "coração não são informação");
  }

  /// Acute / grave accent forms lowercase correctly.
  #[test]
  fn pt_lowercases_accented_vowels() {
    let n = LatinNormalizer::new(Lang::Pt);
    let nt = n.normalize("Á É Í Ó Ú À").unwrap();
    assert_eq!(nt.normalized(), "á é í ó ú à");
  }

  /// Portuguese uses regular ASCII apostrophes only as a literary
  /// elision marker and is NOT in the clitic-split set — `d'água`
  /// stays one word like English `don't`.
  #[test]
  fn pt_apostrophe_does_not_split() {
    let n = LatinNormalizer::new(Lang::Pt);
    let nt = n.normalize("d'água").unwrap();
    assert_eq!(nt.normalized(), "d'água");
    assert_eq!(nt.original_words().len(), 1);
  }

  // --- Cross-cutting --------------------------------------------

  #[test]
  fn all_latin_use_word_delimiter() {
    for lang in [Lang::En, Lang::Es, Lang::Fr, Lang::De, Lang::It, Lang::Pt] {
      let n = LatinNormalizer::new(lang.clone());
      assert!(
        n.use_word_delimiter(),
        "Latin-script normaliser must enable | word delimiter for {lang:?}"
      );
    }
  }
}
