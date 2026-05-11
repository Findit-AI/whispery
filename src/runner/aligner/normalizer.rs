//! Text-normaliser trait + canonical error type.

use std::{borrow::Cow, string::String, vec::Vec};

use smol_str::SmolStr;

/// Why text normalisation failed. Wrapped into
/// [`WorkFailure::Alignment(AlignmentError::Normalization(_))`](crate::types::AlignmentError::Normalization)
/// at the aligner boundary; the diagnostic text on the resulting
/// [`AlignmentFailure`](crate::types::AlignmentFailure) is the
/// `Display` rendering of this enum.
#[derive(Clone, Debug, thiserror::Error)]
pub enum NormalizationError {
  /// Input was empty after stripping whitespace and punctuation;
  /// alignment has nothing to do.
  #[error("normalised text is empty")]
  EmptyText,
  /// Language-specific rule failed (e.g., a contraction-expansion
  /// table could not parse a token). `detail` carries the
  /// implementation's verbatim message.
  #[error("normaliser rule failed: {detail}")]
  RuleFailed {
    /// Verbatim error from the language-specific normaliser.
    detail: SmolStr,
  },
}

/// Normalised text + back-pointer to original surface forms.
#[derive(Clone, Debug)]
pub struct NormalizedText<'a> {
  /// Normalised text the aligner tokenises against the wav2vec2
  /// vocab. Lowercased, punctuation-stripped, contractions
  /// expanded per the language's rules. Whitespace separates
  /// normalised words.
  normalized: String,
  /// Surface forms in normalised-word-index order. The i-th
  /// entry is the original-text slice (with casing and
  /// punctuation as Whisper produced them) that the i-th
  /// normalised word corresponds to. When normalisation expands
  /// a contraction (e.g., `"don't"` → `"do not"`), both
  /// expanded normalised words point back to the same source
  /// slice. Step 9 of the alignment algorithm uses this map to
  /// recover `Word.text`.
  original_words: Vec<Cow<'a, str>>,
  /// Per-word `(prefix, suffix)` count of "wildcard chars" —
  /// surface-form chars that are NOT pronounced (boundary
  /// punctuation the normaliser stripped) but still occupy
  /// frames in the audio. WhisperX includes these as wildcard
  /// tokens (`*` placeholder + token id `-1`) IN SOURCE ORDER,
  /// so leading punctuation like `"hello` keeps its `*` BEFORE
  /// the encoded chars while trailing punctuation like `hello"`
  /// keeps its `*` AFTER. Flagged that an earlier
  /// design carrying only a TOTAL count caused
  /// `tokenize_with_word_map` to push every wildcard at the end
  /// of the word's encoded chars, making leading and trailing
  /// punctuation indistinguishable in the CTC graph.
  ///
  /// Empty (zero-length) means "no wildcard padding tracked";
  /// every word interpreted as `WildcardBoundary { prefix: 0,
  /// suffix: 0 }`.
  wildcard_boundary_per_word: Vec<WildcardBoundary>,
}

/// Per-word boundary wildcard counts produced by a
/// [`TextNormalizer`] when it strips leading / trailing
/// punctuation. The downstream tokeniser
/// (`tokenize_with_word_map`) consults this per word to decide
/// how many `WILDCARD_TOKEN_ID` tokens to emit on each side of
/// the encoded letter chars; CTC alignment treats those
/// wildcards as "match any non-blank vocab item" so the
/// alignment doesn't lose frames over the stripped char.
///
/// Replaces the historical `(u32, u32)` tuple shape. Carrying a
/// named struct gives the boundary count a stable type identity
/// (the prefix vs suffix distinction can't accidentally swap on
/// destructuring, and downstream code reads as
/// `boundary.prefix` instead of `boundary.0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct WildcardBoundary {
  /// Number of leading-punctuation chars the normaliser
  /// stripped from this word's source form. The tokeniser
  /// emits this many `WILDCARD_TOKEN_ID` tokens BEFORE the
  /// word's encoded letter chars so leading punctuation like
  /// `"hello` aligns its `*` placeholders ahead of `h, e, l,
  /// l, o`.
  prefix: u32,
  /// Number of trailing-punctuation chars the normaliser
  /// stripped from this word's source form. Tokens emitted
  /// AFTER the encoded chars; mirrors the prefix story for
  /// trailing punct like `hello"`.
  suffix: u32,
}

impl WildcardBoundary {
  /// Convenience: zero on both sides — i.e. "no boundary
  /// wildcards for this word".
  pub const NONE: Self = Self {
    prefix: 0,
    suffix: 0,
  };

  /// Construct from leading + trailing wildcard counts.
  #[must_use]
  pub const fn new(prefix: u32, suffix: u32) -> Self {
    Self { prefix, suffix }
  }

  /// Number of leading-punctuation wildcards.
  #[must_use]
  pub const fn prefix(&self) -> u32 {
    self.prefix
  }

  /// Number of trailing-punctuation wildcards.
  #[must_use]
  pub const fn suffix(&self) -> u32 {
    self.suffix
  }
}

impl<'a> NormalizedText<'a> {
  /// Construct from a normalised text + original-word slices.
  /// `wildcard_boundary_per_word` defaults to empty (no wildcard
  /// padding) when the normaliser doesn't track it.
  pub const fn new(normalized: String, original_words: Vec<Cow<'a, str>>) -> Self {
    Self {
      normalized,
      original_words,
      wildcard_boundary_per_word: Vec::new(),
    }
  }

  /// Construct with explicit per-word [`WildcardBoundary`]
  /// counts. Length must match `original_words`. Panics on
  /// length mismatch because the count is structurally tied
  /// to word indexing.
  pub fn with_wildcards(
    normalized: String,
    original_words: Vec<Cow<'a, str>>,
    wildcard_boundary_per_word: Vec<WildcardBoundary>,
  ) -> Self {
    assert_eq!(
      original_words.len(),
      wildcard_boundary_per_word.len(),
      "wildcard_boundary_per_word must align 1:1 with original_words"
    );
    Self {
      normalized,
      original_words,
      wildcard_boundary_per_word,
    }
  }

  /// Normalised text the aligner feeds the tokeniser.
  pub fn normalized(&self) -> &str {
    &self.normalized
  }

  /// Surface forms in normalised-word-index order.
  pub fn original_words(&self) -> &[Cow<'a, str>] {
    &self.original_words
  }

  /// Per-word [`WildcardBoundary`] counts, or empty if the
  /// normaliser didn't track them. See [`Self::with_wildcards`]
  /// / [`Self::new`] for how this is populated.
  pub fn wildcard_boundary_per_word(&self) -> &[WildcardBoundary] {
    &self.wildcard_boundary_per_word
  }
}

/// Language-specific text normaliser.
///
/// Implementations must be `Send` because each `Aligner` lives
/// inside a `Mutex<Aligner>` that crosses thread boundaries to the
/// alignment worker.
pub trait TextNormalizer: Send {
  /// Returns `(normalised_text, original_words)`. The map's i-th
  /// entry gives the original surface form for the i-th word in
  /// the normalised text.
  fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError>;

  /// Whether whitespace in the normaliser's output represents real
  /// word boundaries that the wav2vec2 CTC graph should align
  /// against (via the tokeniser's `|` word-delimiter token).
  ///
  /// Returns `true` for word-segmented languages (English): the
  /// tokeniser inserts `|` between every pair of normalised words,
  /// matching how the model was trained.
  ///
  /// Returns `false` for character-segmented normalisers (Chinese,
  /// Japanese) that emit whitespace between every character as an
  /// indexing device only — those characters were never separated
  /// by a delimiter in speech, so forcing the CTC graph to align
  /// `|` between every Han/kana glyph would systematically corrupt
  /// the alignment. The character-level model is expected to align
  /// directly across glyphs without an inter-glyph delimiter.
  ///
  /// Default: `true`.
  fn use_word_delimiter(&self) -> bool {
    true
  }
}

/// Boxed `dyn TextNormalizer` for the [`crate::Aligner`]'s
/// per-language normaliser slot.
pub type DynTextNormalizer = std::boxed::Box<dyn TextNormalizer>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn normalized_text_round_trip() {
    let nt = NormalizedText::new(
      String::from("hello world"),
      vec![Cow::Borrowed("Hello"), Cow::Borrowed("world.")],
    );
    assert_eq!(nt.normalized(), "hello world");
    assert_eq!(nt.original_words().len(), 2);
    assert_eq!(nt.original_words()[0], "Hello");
  }

  #[test]
  fn normalization_error_displays_kinds() {
    assert!(NormalizationError::EmptyText.to_string().contains("empty"));
    assert!(
      NormalizationError::RuleFailed {
        detail: "bad contraction".into()
      }
      .to_string()
      .contains("bad contraction")
    );
  }
}
