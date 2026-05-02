//! English text normaliser. See spec §6.3.

use alloc::{borrow::Cow, string::String, vec::Vec};

use crate::runner::aligner::normalizer::{NormalizationError, NormalizedText, TextNormalizer};

/// English normaliser: lowercase + strip ASCII punct + expand a
/// canonical contraction table.
///
/// Surface-form invariant (spec §6.3.2 step 9): the `original_words`
/// map points each normalised-word index back to the original
/// substring of the input text. When a contraction expands to
/// multiple normalised words (e.g., `"don't"` → `"do not"`), every
/// expanded word maps back to the same source slice — so the
/// emitted [`crate::Word`] entries carry the original `"don't"`
/// twice (once for the time range covering `"do"`, once for the
/// range covering `"not"`). Downstream consumers can dedupe by
/// `Word.text == prior.text` if needed.
///
/// **Punctuation handling:** ASCII punctuation `[ . , ! ? ; : " ' (
/// ) [ ] { } - — – ]` is stripped from word boundaries (leading and
/// trailing). Internal apostrophes inside contractions (e.g., the
/// `'` in `"don't"`) are *not* stripped — they trigger expansion
/// instead.
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

/// Canonical contractions table. Order matters only when prefixes
/// collide (we apply the longest-match rule); the table is small
/// enough that linear scan is fine.
const CONTRACTIONS: &[(&str, &str)] = &[
  ("won't", "will not"),
  ("can't", "can not"),
  ("shan't", "shall not"),
  ("ain't", "is not"),
  ("don't", "do not"),
  ("doesn't", "does not"),
  ("didn't", "did not"),
  ("isn't", "is not"),
  ("aren't", "are not"),
  ("wasn't", "was not"),
  ("weren't", "were not"),
  ("hasn't", "has not"),
  ("haven't", "have not"),
  ("hadn't", "had not"),
  ("wouldn't", "would not"),
  ("couldn't", "could not"),
  ("shouldn't", "should not"),
  ("mustn't", "must not"),
  ("needn't", "need not"),
  ("mightn't", "might not"),
  ("oughtn't", "ought not"),
  ("i'm", "i am"),
  ("i've", "i have"),
  ("i'll", "i will"),
  ("i'd", "i would"),
  ("you're", "you are"),
  ("you've", "you have"),
  ("you'll", "you will"),
  ("you'd", "you would"),
  ("he's", "he is"),
  ("she's", "she is"),
  ("it's", "it is"),
  ("we're", "we are"),
  ("we've", "we have"),
  ("we'll", "we will"),
  ("we'd", "we would"),
  ("they're", "they are"),
  ("they've", "they have"),
  ("they'll", "they will"),
  ("they'd", "they would"),
  ("there's", "there is"),
  ("that's", "that is"),
  ("what's", "what is"),
  ("who's", "who is"),
  ("let's", "let us"),
  ("here's", "here is"),
  ("how's", "how is"),
  ("where's", "where is"),
];

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
// inside contractions like `don't` survive the trim and trigger
// the contraction-expansion table downstream — that's the desired
// behaviour. Codex round-16 [medium] flagged that the previous
// list omitted ASCII `'` entirely, so quoted text like `'hello'`
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

fn expand_contraction(lower: &str) -> Option<&'static str> {
  CONTRACTIONS
    .iter()
    .find(|(k, _)| *k == lower)
    .map(|(_, v)| *v)
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
      let lower = lowercase_for_match(stripped);

      // Reconstruct the borrowed slice for the original word
      // (without punctuation strip, so Whisper's surface form
      // is preserved verbatim — punctuation included).
      let original_slice: &'a str = &text[word_start..word_start + word.len()];

      if let Some(expansion) = expand_contraction(&lower) {
        // The contraction expands to N normalised words, each
        // pointing back to the same original slice (so callers
        // see the apostrophe-preserved `"don't"` for every
        // expanded position).
        let expanded_words: Vec<&str> = expansion.split_whitespace().collect();
        for expanded in expanded_words {
          if !normalized.is_empty() {
            normalized.push(' ');
          }
          normalized.push_str(expanded);
          original_words.push(Cow::Borrowed(original_slice));
        }
      } else {
        // Split on internal separators (`hello-world` →
        // `["hello", "world"]`). Each piece is a real word the
        // wav2vec2 vocab can encode; without this the literal
        // `-` would survive into the normalised text and the
        // tokeniser's `<unk>` rejection would fail the whole
        // chunk on a single hyphen. Each piece points back to
        // the same original surface slice — same pattern as
        // contraction expansion above.
        for piece in lower.split(is_internal_separator).filter(|p| !p.is_empty()) {
          if !normalized.is_empty() {
            normalized.push(' ');
          }
          normalized.push_str(piece);
          original_words.push(Cow::Borrowed(original_slice));
        }
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

  #[test]
  fn expands_contraction_and_duplicates_surface() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Don't go.").unwrap();
    // "Don't" → "do not"; "go" stripped of trailing period.
    assert_eq!(nt.normalized(), "do not go");
    assert_eq!(nt.original_words().len(), 3);
    assert_eq!(nt.original_words()[0], "Don't"); // do
    assert_eq!(nt.original_words()[1], "Don't"); // not
    assert_eq!(nt.original_words()[2], "go.");
  }

  #[test]
  fn em_dash_strips_at_word_boundary() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("hello\u{2014}world").unwrap();
    // The em dash is in the middle, so split_whitespace doesn't
    // split it; only edge punctuation strips. The whole token
    // becomes "hello—world" → "hello—world" lowercased (dash is
    // not stripped from the middle).
    // For v1 we accept that internal punctuation is preserved.
    // Whisper rarely emits em-dash-glued words.
    assert_eq!(nt.original_words()[0], "hello\u{2014}world");
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
  fn contraction_inside_sentence() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("I won't be late.").unwrap();
    assert_eq!(nt.normalized(), "i will not be late");
    assert_eq!(nt.original_words()[1], "won't");
    assert_eq!(nt.original_words()[2], "won't");
  }

  #[test]
  fn unknown_apostrophe_token_passes_through_lowercased() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("O'Brien rocks.").unwrap();
    // "O'Brien" is not in CONTRACTIONS; lowercased pass-through
    // preserves the apostrophe in the normalised form.
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

  /// Quoted contraction: leading/trailing `'` strips, but the
  /// internal one survives long enough to trigger the
  /// contraction-expansion table.
  #[test]
  fn boundary_apostrophe_around_contraction_keeps_internal() {
    let n = EnglishNormalizer::new();
    let nt = n.normalize("'don't'").unwrap();
    assert_eq!(nt.normalized(), "do not");
    // Both expanded words point back to the same surface slice
    // (which still has the surrounding quotes — that's what the
    // user originally typed).
    assert_eq!(nt.original_words()[0], "'don't'");
    assert_eq!(nt.original_words()[1], "'don't'");
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

  #[test]
  fn hyphenated_word_splits_into_pieces() {
    // Adversarial regression: before this fix, `hello-world` kept
    // the literal `-` in the normalised text, the tokeniser
    // produced `<unk>` for the hyphen, and the new <unk>
    // rejection failed the whole chunk on a single internal
    // separator.
    let n = EnglishNormalizer::new();
    let nt = n.normalize("Hello-World").unwrap();
    assert_eq!(nt.normalized(), "hello world");
    // Both halves preserve the original surface form.
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello-World");
    assert_eq!(nt.original_words()[1], "Hello-World");
  }

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
