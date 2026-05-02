//! English text normaliser. See spec §6.3.

use alloc::{borrow::Cow, string::String, vec::Vec};

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// English normaliser: lowercase + strip ASCII punct, surface-
/// form preserved.
///
/// Surface-form invariant (spec §6.3.2 step 9): the `original_words`
/// map points each normalised-word index back to the original
/// substring of the input text. The normaliser does **not** expand
/// contractions — `"don't"` stays one word with normalised form
/// `"don't"` (lowercased) and a single `original_words` entry
/// pointing back to `"Don't"`.
///
/// **Why not expand contractions?** wav2vec2-base-960h's vocab has
/// the apostrophe glyph (`'` at id 27) and was trained on
/// LibriSpeech transcripts that write `DON'T` directly. Expanding
/// to `"do not"` forces the CTC graph to insert a `|` word
/// delimiter between `do` and `not`, which the speaker never
/// pronounces — the model then assigns word boundaries inside a
/// single phonetic blob, shifting timings and producing duplicate
/// `Word.text()` values that consumers can't safely dedupe (real
/// repeated words exist too). Codex round-22.
///
/// **Punctuation handling:** ASCII punctuation `[ . , ! ? ; : " ' (
/// ) [ ] { } - — – ]` is stripped from word boundaries (leading and
/// trailing). Internal apostrophes inside contractions (e.g., the
/// `'` in `"don't"`) survive into the normalised form so the
/// wav2vec2 tokenizer aligns the apostrophe character directly.
///
/// **Empty result:** if normalisation produces zero words (input
/// was all whitespace/punctuation), `normalize` returns
/// [`NormalizationError::EmptyText`]. `Aligner::align` treats
/// this as a non-fatal short-circuit and returns
/// `Ok(AlignmentResult::new(Vec::new()))`, so the cached ASR
/// transcript surfaces as `Transcript { text, words: [] }`
/// rather than `Event::Error` — alignment is optional, not a
/// data-loss path on punctuation-only ASR output.
#[derive(Default, Clone, Copy, Debug)]
pub struct EnglishNormalizer;

impl EnglishNormalizer {
  /// Construct an English normaliser. `const fn` for use in
  /// static lookup tables.
  pub const fn new() -> Self {
    Self
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
            | '\'' // ASCII apostrophe — see comment below
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
            | '\u{2019}' // right single quote
  )
}

// Note on ASCII `'` boundary handling: `is_word_punct` includes
// `'`, but `strip_word_punct` only trims leading/trailing matches
// (`trim_start_matches` / `trim_end_matches`). Internal apostrophes
// inside contractions like `don't` survive the trim — wav2vec2
// aligns them as a single word with the `'` character emitted
// inline. Codex round-16 [medium] flagged that the previous list
// omitted ASCII `'` entirely, so quoted text like `'hello'`
// kept the surrounding apostrophes in the normalised string;
// wav2vec2's tokeniser then forced unspoken apostrophe states
// into the CTC graph, stealing frames from real-word states.

fn strip_word_punct(s: &str) -> &str {
  let trimmed_left = s.trim_start_matches(is_word_punct);
  trimmed_left.trim_end_matches(is_word_punct)
}

/// True for characters that join two real words inside a single
/// whitespace-bounded token but are themselves never spoken, e.g.
/// `hello-world`, `two—three`, `and/or`. The wav2vec2 vocab
/// doesn't cover these glyphs, and CJK-style "preserve the
/// punctuation" makes no sense here, so the normaliser treats
/// them as word boundaries: each side becomes its own normalised
/// word, both pointing back to the same original surface slice.
///
/// We deliberately do *not* split on apostrophes (they can be
/// part of real surface forms like contractions) or on periods
/// (`U.S.A.` would explode into noise) — those need targeted
/// handling rather than a generic split rule.
fn is_internal_separator(c: char) -> bool {
  matches!(
    c,
    '-' | '/' | '\u{2010}' // hyphen
                | '\u{2013}' // en-dash
                | '\u{2014}' // em-dash
                | '\u{2015}' // horizontal bar
  )
}

fn lowercase_for_match(s: &str) -> String {
  s.to_lowercase()
}

impl TextNormalizer for EnglishNormalizer {
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError> {
    let mut normalized = String::with_capacity(text.len());
    let mut original_words: Vec<Cow<'a, str>> = Vec::new();

    for (word_start, word) in token_spans(text) {
      let stripped = strip_word_punct(word);
      if stripped.is_empty() {
        continue;
      }

      // Reconstruct the borrowed slice for the original word
      // (without punctuation strip, so Whisper's surface form
      // is preserved verbatim — punctuation included). Used
      // for the no-separator path; the with-separator path
      // emits per-piece surface slices so each emitted Word
      // carries its own substring.
      let original_slice: &'a str = &text[word_start..word_start + word.len()];

      // Split on internal separators (`Hello-World` →
      // `["Hello", "World"]`). Each piece is a real word the
      // wav2vec2 vocab can encode; without this the literal
      // `-` would survive into the normalised text and the
      // tokeniser's `<unk>` rejection would fail the whole
      // chunk on a single hyphen. The apostrophe is *not* an
      // internal separator, so `don't` stays one piece and
      // aligns as a single word with the `'` character emitted
      // inline by the wav2vec2 tokenizer (Codex round-22).
      //
      // Codex round-23 [medium]: each piece carries its own
      // surface span, not the full hyphenated word. Pre-fix
      // every split piece pointed back to the same
      // `original_slice`, so `Hello-World` emitted two
      // `Word.text() == "Hello-World"` entries that downstream
      // consumers couldn't dedupe (real repeated words also
      // exist). The split runs over `stripped` (no boundary
      // punct) so each `piece_orig` is a borrow into the input
      // text — surface form is preserved per piece.
      if stripped.contains(is_internal_separator) {
        for piece_orig in stripped.split(is_internal_separator).filter(|p| !p.is_empty()) {
          let piece_lower = lowercase_for_match(piece_orig);
          if !normalized.is_empty() {
            normalized.push(' ');
          }
          normalized.push_str(&piece_lower);
          original_words.push(Cow::Borrowed(piece_orig));
        }
      } else {
        // No internal separator — the word is one piece.
        // Preserve the full original_slice (with any
        // boundary punctuation) for surface-form display.
        let lower = lowercase_for_match(stripped);
        if !normalized.is_empty() {
          normalized.push(' ');
        }
        normalized.push_str(&lower);
        original_words.push(Cow::Borrowed(original_slice));
      }
    }

    if original_words.is_empty() {
      return Err(NormalizationError::EmptyText);
    }
    Ok(NormalizedText::new(normalized, original_words))
  }
}

/// Iterate `(byte_offset, slice)` for whitespace-separated tokens.
/// Equivalent to `text.split_whitespace()` but yields starting
/// byte offsets so callers can reconstruct borrowed slices.
fn token_spans(text: &str) -> impl Iterator<Item = (usize, &str)> + '_ {
  let mut idx = 0;
  let mut iter = text.split_whitespace();
  core::iter::from_fn(move || {
    let token = iter.next()?;
    // text.split_whitespace() returns slices that point into
    // `text`; recover the offset by subtracting base ptrs.
    let token_start = (token.as_ptr() as usize).saturating_sub(text.as_ptr() as usize);
    idx = token_start + token.len();
    Some((token_start, token))
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lowercase_and_strip_punct() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello, World!").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello,");
    assert_eq!(nt.original_words()[1], "World!");
  }

  /// Codex round-22 [high]: contractions stay one normalised
  /// word with the apostrophe character preserved inline. The
  /// previous round expanded `"Don't" → "do not"` and emitted
  /// duplicate `Word.text()` entries spanning a forced `|`
  /// delimiter that the speaker never pronounced — shifting
  /// timings and breaking dedupe semantics for downstream
  /// consumers. wav2vec2-base-960h's vocab has the apostrophe
  /// glyph; aligning `don't` directly is the supported path.
  #[test]
  fn contraction_stays_one_word_with_apostrophe_inline() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Don't go.").unwrap();
    // "Don't" stays one word; trailing period strips off "go.".
    assert_eq!(nt.normalized(), "don't go");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Don't");
    assert_eq!(nt.original_words()[1], "go.");
  }

  /// Codex round-23 [medium]: em-dash-glued tokens split into
  /// per-piece surface spans. Pre-fix every piece pointed back
  /// to the full `hello—world` slice, so the emitted alignment
  /// would carry two `Word.text() == "hello—world"` entries —
  /// indistinguishable from a real repetition.
  #[test]
  fn em_dash_splits_into_per_piece_surface_spans() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("hello\u{2014}world").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "hello");
    assert_eq!(nt.original_words()[1], "world");
  }

  /// Codex round-23 [medium]: hyphen-glued compound. Each piece
  /// must carry its own surface span — the full `Hello-World`
  /// would corrupt downstream word indexes / highlights and
  /// break dedupe semantics.
  #[test]
  fn hyphen_compound_splits_into_per_piece_surface_spans() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello-World").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello");
    assert_eq!(nt.original_words()[1], "World");
  }

  /// Slash-separated alternation (`and/or`). Same per-piece
  /// surface contract.
  #[test]
  fn slash_alternation_splits_into_per_piece_surface_spans() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("and/or").unwrap();
    assert_eq!(nt.normalized(), "and or");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "and");
    assert_eq!(nt.original_words()[1], "or");
  }

  #[test]
  fn empty_input_errors() {
    let n = EnglishNormalizer::new();
    let err = n.normalize("   .,!?  ").unwrap_err();
    assert!(matches!(err, NormalizationError::EmptyText));
  }

  #[test]
  fn casing_preserved_in_original_words() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("The Quick BROWN Fox.").unwrap();
    assert_eq!(nt.normalized(), "the quick brown fox");
    assert_eq!(nt.original_words()[1], "Quick");
    assert_eq!(nt.original_words()[2], "BROWN");
    assert_eq!(nt.original_words()[3], "Fox.");
  }

  #[test]
  fn contraction_inside_sentence_stays_intact() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("I won't be late.").unwrap();
    assert_eq!(nt.normalized(), "i won't be late");
    assert_eq!(nt.original_words().len(), 4);
    assert_eq!(nt.original_words()[0], "I");
    assert_eq!(nt.original_words()[1], "won't");
    assert_eq!(nt.original_words()[2], "be");
    assert_eq!(nt.original_words()[3], "late.");
  }

  #[test]
  fn apostrophe_word_passes_through_lowercased() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("O'Brien rocks.").unwrap();
    // Apostrophe survives lowercasing and is *not* an internal
    // separator; the word stays whole.
    assert_eq!(nt.normalized(), "o'brien rocks");
  }

  /// Codex round-16 [medium]: quoted text like `'hello'` must
  /// have its surrounding apostrophes stripped at boundaries so
  /// the wav2vec2 tokeniser doesn't inject unspoken apostrophe
  /// states into the CTC graph. Internal apostrophes inside
  /// contractions / proper nouns must still survive.
  #[test]
  fn boundary_ascii_apostrophes_are_stripped() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("'hello'").unwrap();
    assert_eq!(nt.normalized(), "hello");
    // Surface form preservation is unchanged: original_words[0]
    // still carries the input verbatim.
    assert_eq!(nt.original_words()[0], "'hello'");
  }

  /// Quoted contraction: leading/trailing `'` strip but the
  /// internal one survives, leaving the contraction intact as a
  /// single normalised word that wav2vec2 aligns directly.
  #[test]
  fn boundary_apostrophe_around_contraction_keeps_internal() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("'don't'").unwrap();
    assert_eq!(nt.normalized(), "don't");
    assert_eq!(nt.original_words().len(), 1);
    assert_eq!(nt.original_words()[0], "'don't'");
  }

  /// Mixed: trailing apostrophe + word + sentence punctuation.
  /// `dogs'.` (possessive plural) is a realistic case.
  #[test]
  fn trailing_possessive_apostrophe_strips() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("the dogs'.").unwrap();
    // "the" and "dogs" — the trailing `'` and `.` both strip.
    assert_eq!(nt.normalized(), "the dogs");
  }

  #[test]
  fn uses_word_delimiter() {
    // Word-segmented English: whitespace separates real spoken
    // words. Tokenisation must insert `|` between them so the
    // CTC graph aligns the same way the model was trained.
    let n = EnglishNormalizer::new();
    assert!(n.use_word_delimiter());
  }

  // Codex round-23 [medium]: the old
  // `hyphenated_word_splits_into_pieces` test pinned the
  // pre-fix behaviour where every split piece pointed back to
  // the full `Hello-World` surface — see
  // `hyphen_compound_splits_into_per_piece_surface_spans`
  // earlier in this module for the corrected contract.

  #[test]
  fn em_dash_and_slash_split() {
    let n = EnglishNormalizer::new();
    // em-dash, en-dash, slash all behave the same.
    let nt = n
      .normalize("two\u{2014}three and/or four\u{2013}five")
      .unwrap();
    assert_eq!(nt.normalized(), "two three and or four five");
    assert_eq!(nt.original_words().len(), 6);
  }

  #[test]
  fn pure_separator_token_is_dropped() {
    // A whitespace-bounded token that's only separators
    // produces no words (matches the pre-existing
    // strip-only-punct behaviour).
    let n = EnglishNormalizer::new();
    let nt = n.normalize("hello --- world").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
  }

  #[test]
  fn collapses_consecutive_internal_separators() {
    // Em-dash + slash + hyphen back-to-back inside a single
    // token still splits into the two real words.
    let n = EnglishNormalizer::new();
    let nt = n.normalize("foo\u{2014}/-bar").unwrap();
    assert_eq!(nt.normalized(), "foo bar");
    assert_eq!(nt.original_words().len(), 2);
  }
}
