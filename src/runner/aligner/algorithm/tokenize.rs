//! Step 1-2 of the alignment algorithm: tokenisation + per-token
//! word-index map.

use alloc::{string::String, vec::Vec};

use tokenizers::Tokenizer;

use crate::{
  runner::aligner::algorithm::trellis_beam::WILDCARD_TOKEN_ID,
  types::{AlignmentFailureKind, Lang, WorkFailure},
};

/// Result of tokenising the normalised text.
///
/// `pub` for the `feature = "bench-internals"` re-export at the
/// crate root — out-of-tree code only sees this type through the
/// doc-hidden `whispery::__bench` namespace.
#[derive(Debug)]
pub struct TokenizedText {
  /// Vocab indices in tokenisation order (Y in spec terms),
  /// stored as `i32` because the wildcard sentinel
  /// `WILDCARD_TOKEN_ID = -1` is allowed: an alphanumeric char
  /// the model dictionary doesn't know becomes a wildcard, and
  /// the trellis emits `max(non_blank_logprobs)` for that frame.
  /// All non-wildcard ids are non-negative and fit in `u32`.
  pub token_ids: Vec<i32>,
  /// Per-token mapping back to the normalised-word index. `None`
  /// for tokens that have no natural word index (word-delimiter
  /// `|`, special tokens like `<s>`, `<pad>`, `<unk>`).
  pub word_idx_per_token: Vec<Option<usize>>,
  /// The wav2vec2 word-delimiter `|` token id, when the
  /// tokenizer exposes one and the normaliser opted in. The
  /// trellis-beam orchestrator uses this to recognise
  /// separators in `merge_words`.
  pub separator_token_id: Option<u32>,
}

/// Tokenise `normalized` against the wav2vec2 tokeniser, building a
/// per-token word-index map.
///
/// The wav2vec2 vocab uses a single character per token (one of:
/// letter, digit, apostrophe, the word-delimiter `|`, or a special
/// like `<s>`, `<pad>`, `<unk>`, `</s>`). For word-segmented
/// languages the model is trained to align `|` between every pair
/// of spoken words.
///
/// We tokenise word-by-word (not the whole sentence at once) to
/// trivially get the word index — each word's encoded tokens map
/// to the word's index, and the inter-word `|` is appended with
/// `None` between words **only when the source normaliser declared
/// that whitespace represents real word boundaries**.
///
/// `use_word_delimiter` is the [`crate::TextNormalizer::use_word_delimiter`]
/// signal: `true` for English (whitespace = real word break, insert
/// `|`); `false` for character-segmented languages (Chinese,
/// Japanese) where whitespace is an indexing artefact only and must
/// not introduce CTC-graph delimiters that were never spoken.
///
/// `uppercase_input` projects ASCII to uppercase before encoding;
/// set when the vocab covers `A`-`Z` only (the case for
/// `wav2vec2-base-960h`). Without this projection a lowercase
/// normaliser would feed every English letter through `<unk>`,
/// producing a CTC graph that cannot meaningfully align word
/// boundaries — the bug that motivated this parameter.
///
/// `unk_token_id`, when supplied, is used to detect out-of-vocab
/// characters per char. The handling is **WhisperX-style**:
/// alphanumeric chars not in the vocab become wildcard tokens
/// (`WILDCARD_TOKEN_ID = -1`) and the trellis aligns them to
/// whichever non-blank vocab item carries the highest
/// log-probability at each frame. This matches WhisperX's
/// `clean_char.append('*')` placeholder + `tokens =
/// [model_dictionary.get(c, -1) for c in text_clean]` flow.
///
/// **Whispery-specific guard (default policy):** non-alphanumeric
/// chars (e.g. `&` in `AT&T`) that aren't on the whitelist of
/// skippable internal punctuation still drop the whole chunk's
/// alignment. WhisperX would silently align them as wildcards too,
/// but those chars represent semantic content the model can't
/// reasonably align (the `&` in `AT&T` is "and"); silently aligning
/// to whichever vocab item happens to win the frame would produce
/// honest-looking but wrong word ranges. Whispery fails closed at
/// chunk granularity instead.
///
/// **`whisperx-strict-tokenizer` feature (relaxed policy):** when
/// the `whisperx-strict-tokenizer` Cargo feature is enabled, this
/// guard is removed and non-alphanumeric pronounced OOV chars
/// become wildcards too — matching WhisperX's
/// `clean_char.append('*')` behaviour 1:1. Opt in only if downstream
/// consumers expect WhisperX-bit-equivalent output and accept the
/// silent-misalignment risk on pronounced symbols.
///
/// Internal punctuation that's never pronounced as a separate
/// sound and is safe to strip before encoding so the
/// surrounding letters still align. Currently just the period:
/// `U.S.A.`, `D.C.`, `etc.` are pronounced as their letters.
fn is_skippable_internal_punct(c: char) -> bool {
  c == '.'
}

/// Whether an OOV char should escalate to a chunk-level drop or
/// just become a wildcard token. Mirrors the policy in the
/// [`tokenize_with_word_map`] doc-comment.
///
/// **Default policy (no `whisperx-strict-tokenizer`):**
/// Wildcard targets:
/// - Alphanumeric chars (any letter or digit) the model
///   dictionary doesn't have.
/// - The apostrophe (already in most vocabs; if it's missing
///   from a custom one, we'd rather wildcard than drop).
///
/// Chunk-drop targets:
/// - Pronounced symbols (`&`, `@`, `%`, `,`, ...) — the audio
///   contains a real word the wav2vec2 model could in theory
///   align, but we have no honest way to know which vocab item
///   it's pronounced as.
///
/// **Relaxed policy (`whisperx-strict-tokenizer` enabled):**
/// EVERY OOV char becomes a wildcard. This matches WhisperX's
/// `clean_char.append('*')` 1:1 — same per-frame ambiguity but
/// preserves alignment continuity at the cost of honesty about
/// pronounced symbols.
fn allow_wildcard(c: char) -> bool {
  if cfg!(feature = "whisperx-strict-tokenizer") {
    // Relaxed policy: everything wildcards, mirroring WhisperX's
    // `clean_char.append('*')` for any char not in the model
    // dictionary.
    true
  } else {
    c.is_alphanumeric() || c == '\'' || c == '\u{2019}'
  }
}

/// `pub` for the `feature = "bench-internals"` re-export at the
/// crate root. Out-of-tree code only reaches this through
/// `whispery::__bench`, which is doc-hidden and gated on the
/// `bench-internals` feature.
pub fn tokenize_with_word_map(
  tokenizer: &Tokenizer,
  normalized: &str,
  word_count: usize,
  use_word_delimiter: bool,
  uppercase_input: bool,
  unk_token_id: Option<u32>,
  // Per-word `(prefix, suffix)` count of wildcard tokens to
  // inject around the word's encoded chars. Prefix wildcards
  // are pushed BEFORE the encoded chars; suffix wildcards
  // (plus any internal-skipped chars discovered during this
  // pass) are pushed AFTER. Mirrors WhisperX's `*` placeholder
  // for unpronounced boundary punctuation IN SOURCE ORDER, so
  // leading punctuation like `"hello` keeps its `*` before the
  // letters and trailing `hello"` keeps it after — Codex
  // round-28 flagged that an earlier total-count design pushed
  // every wildcard at the end of the word, making leading vs
  // trailing punctuation indistinguishable in the CTC graph.
  //
  // Empty slice means "zero wildcards for every word" (legacy
  // / non-English-normaliser path); whispery's
  // [`crate::EnglishNormalizer`] populates it from the
  // boundary-punctuation strip count.
  wildcard_boundary_per_word: &[(u32, u32)],
  language: &Lang,
) -> Result<TokenizedText, WorkFailure> {
  let mut token_ids: Vec<i32> = Vec::with_capacity(normalized.len() + word_count * 2);
  let mut word_idx_per_token: Vec<Option<usize>> = Vec::with_capacity(token_ids.capacity());

  let words: Vec<&str> = normalized.split_whitespace().collect();
  if words.len() != word_count {
    // Sanity: caller's claimed word_count must match the
    // normalised text. Off-by-one here would mis-index Word
    // emission in step 9.
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::TokenizationFailed,
      message: alloc::format!(
        "word_count mismatch: caller={}, normalized has {}",
        word_count,
        words.len()
      ),
      language: language.clone(),
    });
  }
  if !wildcard_boundary_per_word.is_empty() && wildcard_boundary_per_word.len() != word_count {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::TokenizationFailed,
      message: alloc::format!(
        "wildcard_boundary_per_word.len() = {} != word_count = {}",
        wildcard_boundary_per_word.len(),
        word_count
      ),
      language: language.clone(),
    });
  }

  // Per-char tokenisation. We can't encode the whole word at
  // once and inspect after-the-fact: we need to know which
  // *char* produced an `<unk>` so we can decide between
  // wildcard-and-keep vs drop-the-chunk.
  let mut per_word_tokens: Vec<Vec<i32>> = Vec::with_capacity(words.len());
  let mut tmp_buf = String::with_capacity(8);
  for (wi, word) in words.iter().enumerate() {
    let (prefix_wildcards, suffix_wildcards) = wildcard_boundary_per_word
      .get(wi)
      .copied()
      .unwrap_or((0, 0));
    let mut word_tokens: Vec<i32> = Vec::with_capacity(word.len());
    // Push prefix wildcards BEFORE the encoded chars so leading
    // punctuation like `"hello` aligns its `*` placeholders
    // ahead of `h, e, l, l, o`. The CTC graph then matches the
    // source order, mirroring WhisperX's approach.
    for _ in 0..prefix_wildcards {
      word_tokens.push(WILDCARD_TOKEN_ID);
    }
    let mut internal_skipped = 0_u32;
    for ch in word.chars() {
      if is_skippable_internal_punct(ch) {
        // Track each skipped internal punctuation char so we
        // can append a wildcard token at the end of the word
        // (matching WhisperX's `*` placeholder for chars not
        // in the model dictionary). The wildcard's frames
        // come AFTER the word's letter chars; not perfectly
        // interspersed like WhisperX (which keeps the order),
        // but the word's TOTAL frame range is comparable.
        internal_skipped += 1;
        continue;
      }
      let projected = if uppercase_input {
        ch.to_ascii_uppercase()
      } else {
        ch
      };
      tmp_buf.clear();
      tmp_buf.push(projected);
      let encoding = tokenizer
        .encode(tmp_buf.as_str(), /* add_special_tokens = */ false)
        .map_err(|e| WorkFailure::AlignmentFailed {
          kind: AlignmentFailureKind::TokenizationFailed,
          message: alloc::format!("encode({:?}) failed: {e:?}", projected),
          language: language.clone(),
        })?;
      let ids = encoding.get_ids();
      // Single-char encode usually yields exactly one token.
      // If for some reason it doesn't (a tokenizer with
      // multi-char merges, or a model that decomposes
      // characters), treat the whole encoded sequence as a
      // unit — but that's unusual for wav2vec2 vocabs. An
      // empty result means the tokenizer dropped the char
      // entirely; we treat that as if it produced an `<unk>`.
      let is_unk_or_empty = ids.is_empty()
        || match unk_token_id {
          Some(unk) => ids.iter().any(|&id| id == unk),
          None => false,
        };
      if is_unk_or_empty {
        if allow_wildcard(ch) {
          word_tokens.push(WILDCARD_TOKEN_ID);
        } else {
          // Non-alphanumeric semantic OOV. Drop the chunk.
          // See module-level comment for the rationale; we
          // can't honestly align an unspoken-as-itself
          // symbol to whichever vocab item wins the frame.
          return Ok(TokenizedText {
            token_ids: Vec::new(),
            word_idx_per_token: Vec::new(),
            separator_token_id: None,
          });
        }
      } else {
        for &id in ids {
          word_tokens.push(id as i32);
        }
      }
    }
    // Append SUFFIX wildcards: trailing-punct strips from the
    // normaliser plus any internal-skipped chars discovered in
    // this pass (currently just `.` for abbreviations). The
    // suffix bucket sits AFTER the encoded chars so trailing
    // punctuation like `hello"` keeps its `*` after the
    // letters. Internal-skipped chars also land in the suffix
    // bucket — placing them exactly in source order would
    // require interleaving wildcards mid-word during
    // encoding, which the current per-char tokeniser flow
    // doesn't support. The suffix-leaning approximation is
    // strictly better than the previous all-at-end design for
    // the common leading-vs-trailing case Codex round-28
    // flagged; internal punct (rare) takes the same hit it
    // already had.
    let total_suffix = (suffix_wildcards + internal_skipped) as usize;
    for _ in 0..total_suffix {
      word_tokens.push(WILDCARD_TOKEN_ID);
    }
    per_word_tokens.push(word_tokens);
  }

  // Pass 2: flatten into the final token stream, inserting the
  // `|` delimiter only between adjacent NON-EMPTY groups when
  // the normaliser opted in. The orphan-delimiter rule still
  // applies — empty groups (every char unprintable / dropped to
  // skippable internal punct) leave no stray `|` for the trellis
  // to attribute frames to.
  let delim_id = if use_word_delimiter {
    tokenizer.token_to_id("|")
  } else {
    None
  };
  let mut last_emitted_word: Option<usize> = None;
  for (word_idx, group) in per_word_tokens.iter().enumerate() {
    if group.is_empty() {
      continue;
    }
    if last_emitted_word.is_some()
      && let Some(d) = delim_id
    {
      token_ids.push(d as i32);
      word_idx_per_token.push(None);
    }
    for &id in group {
      token_ids.push(id);
      word_idx_per_token.push(Some(word_idx));
    }
    last_emitted_word = Some(word_idx);
  }

  // An empty token list is *not* an error. A chunk like `"...."`
  // (skippable punctuation only) legitimately produces zero
  // tokens. Returning `TokenizationFailed` here would convert
  // the successful ASR `Transcript` into an `Event::Error` at the
  // dispatch layer.
  Ok(TokenizedText {
    token_ids,
    word_idx_per_token,
    separator_token_id: delim_id,
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Lang;

  /// Inline WordLevel tokenizer matching the wav2vec2-base-960h
  /// shape (uppercase-only ASCII alphabet plus `<unk>`, `<pad>`,
  /// `|`).
  const UPPERCASE_TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": {
      "type": "Split",
      "pattern": {"Regex": ""},
      "behavior": "Isolated",
      "invert": false
    },
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "WordLevel",
      "vocab": {
        "<unk>": 0,
        "<pad>": 1,
        "|": 2,
        "A": 3, "B": 4, "C": 5, "D": 6, "E": 7, "F": 8, "G": 9,
        "H": 10, "I": 11, "J": 12, "K": 13, "L": 14, "M": 15,
        "N": 16, "O": 17, "P": 18, "Q": 19, "R": 20, "S": 21,
        "T": 22, "U": 23, "V": 24, "W": 25, "X": 26, "Y": 27, "Z": 28
      },
      "unk_token": "<unk>"
    }
  }"#;

  fn uppercase_tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(UPPERCASE_TOKENIZER_JSON.as_bytes())
      .expect("inline WordLevel tokenizer must parse")
  }

  #[test]
  fn english_lowercase_word_uppercases_for_uppercase_only_vocab() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
    )
    .expect("tokenisation must succeed with uppercase projection");

    assert_eq!(result.token_ids.len(), 5);
    let unk_i32 = unk.unwrap() as i32;
    assert!(
      result.token_ids.iter().all(|&id| id != unk_i32),
      "no <unk> ids; got {:?}",
      result.token_ids
    );
    let expected = ['H', 'E', 'L', 'L', 'O'].map(|c| {
      tok
        .token_to_id(&c.to_string())
        .expect("uppercase letter in vocab") as i32
    });
    assert_eq!(result.token_ids, expected.to_vec());
  }

  /// Punctuation-only input now maps to a single trailing
  /// wildcard token (matching WhisperX's `*` placeholder for
  /// chars not in the model dictionary). Pre-port this returned
  /// zero tokens because the legacy tokeniser stripped the `.`
  /// before encoding; the new policy retains stripped chars as
  /// wildcards so the word's audio frames still factor into the
  /// alignment.
  ///
  /// Real all-punctuation chunks short-circuit upstream — the
  /// English normaliser returns `EmptyText` because every word
  /// strips down to empty. So this test pins the
  /// `tokenize_with_word_map` boundary behaviour, not what the
  /// runner sees end-to-end.
  #[test]
  fn skippable_punctuation_only_yields_one_wildcard_token() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(&tok, ".", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids, alloc::vec![WILDCARD_TOKEN_ID]);
    assert_eq!(result.word_idx_per_token, alloc::vec![Some(0)]);
  }

  /// Internal periods strip cleanly.
  #[test]
  fn internal_periods_in_abbreviation_strip_to_letters() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "U.S.A",
      /* word_count: */ 1,
      /* use_word_delimiter: */ true,
      /* uppercase_input: */ true,
      /* unk_token_id: */ unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
    )
    .expect("U.S.A. must tokenise via per-char strip");

    // 3 letter ids + 2 internal-period wildcards = 5 tokens.
    // (The internal periods, which the tokeniser skips on the
    // whitelist, become trailing wildcards on the same word
    // so the word's frame range still covers the period
    // frames — matching WhisperX's `*` placeholder behaviour
    // for chars not in the model dictionary.)
    assert_eq!(result.token_ids.len(), 5);
    let unk_i32 = unk.unwrap() as i32;
    assert!(
      result.token_ids.iter().all(|&id| id != unk_i32),
      "no <unk> ids must reach the lattice; got {:?}",
      result.token_ids
    );
    // First three tokens are the letter ids; last two are
    // wildcard sentinels for the skipped internal periods.
    let expected_letters: [i32; 3] =
      ['U', 'S', 'A'].map(|c| tok.token_to_id(&c.to_string()).unwrap() as i32);
    assert_eq!(&result.token_ids[..3], &expected_letters[..]);
    assert_eq!(result.token_ids[3], WILDCARD_TOKEN_ID);
    assert_eq!(result.token_ids[4], WILDCARD_TOKEN_ID);
    // All 5 tokens belong to word 0.
    assert_eq!(result.word_idx_per_token, alloc::vec![Some(0); 5]);
  }

  /// **NEW behaviour**: alphanumeric OOV chars become wildcards
  /// (matching WhisperX's `*` placeholder + `-1` token id),
  /// instead of dropping the whole chunk.
  ///
  /// Pre-port: `B2B` against the A-Z vocab dropped the whole
  /// chunk's alignment because the digit `2` was an `<unk>`.
  /// Now: `B`, wildcard, `B` — 3 tokens, all attributed to word 0.
  #[test]
  fn partial_oov_alphanumeric_word_uses_wildcard() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "B2B", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 3);
    let b_id = tok.token_to_id("B").unwrap() as i32;
    assert_eq!(result.token_ids[0], b_id);
    assert_eq!(result.token_ids[1], WILDCARD_TOKEN_ID);
    assert_eq!(result.token_ids[2], b_id);
    assert_eq!(result.word_idx_per_token, alloc::vec![Some(0); 3]);
  }

  /// Same: a fully-alphanumeric all-OOV word (digits) maps to
  /// all-wildcards; the chunk does NOT drop. WhisperX-style
  /// permissive alignment.
  #[test]
  fn all_digit_word_against_uppercase_vocab_uses_wildcards() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "1000", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 4);
    assert!(
      result.token_ids.iter().all(|&id| id == WILDCARD_TOKEN_ID),
      "every digit must become a wildcard; got {:?}",
      result.token_ids
    );
  }

  /// Whispery-specific guard preserved: non-alphanumeric
  /// pronounced char (`&` in `AT&T`) still drops the chunk.
  /// WhisperX would silently align it; whispery fails closed
  /// because the `&` is pronounced as "and" and aligning to
  /// whichever vocab item wins the frame produces a wrong range.
  ///
  /// Skipped under `whisperx-strict-tokenizer` because that
  /// feature deliberately relaxes the chunk-drop policy.
  #[cfg(not(feature = "whisperx-strict-tokenizer"))]
  #[test]
  fn ampersand_oov_drops_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "AT&T", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert!(
      result.token_ids.is_empty(),
      "AT&T's `&` is non-alphanumeric semantic OOV; whole chunk drops"
    );
  }

  /// Accented letter is alphanumeric (per `char::is_alphanumeric`)
  /// and is an OOV against the A-Z-only vocab. Per the new
  /// policy this becomes a wildcard, NOT a chunk drop. The audio
  /// for `é` aligns to whichever vocab item the encoder thinks
  /// is most likely at that frame — typically `E`.
  #[test]
  fn accented_letter_uses_wildcard() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_word_map(&tok, "café", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 4);
    let expected_letters = ['C', 'A', 'F'];
    for (i, c) in expected_letters.iter().enumerate() {
      assert_eq!(
        result.token_ids[i],
        tok.token_to_id(&c.to_string()).unwrap() as i32
      );
    }
    assert_eq!(result.token_ids[3], WILDCARD_TOKEN_ID);
  }

  /// Sanity: digits in a chunk-middle word still survive (no
  /// chunk drop). Pre-port this dropped the whole chunk because
  /// any partial-OOV word triggered the closed-fail policy.
  #[test]
  fn middle_digit_word_no_longer_drops_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(&tok, "hi 1000 world", 3, true, true, unk, &[], &Lang::En)
      .expect("ok");
    // Words: hi (2), |, wildcards (4), |, world (5). 2 + 1 + 4 + 1 + 5 = 13.
    assert_eq!(result.token_ids.len(), 13);
    // Three distinct word indices represented (0, 1, 2).
    let word_indices: alloc::collections::BTreeSet<usize> = result
      .word_idx_per_token
      .iter()
      .filter_map(|w| *w)
      .collect();
    assert_eq!(word_indices.len(), 3);
  }

  #[test]
  fn separator_token_id_is_returned() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let pipe = tok.token_to_id("|").expect("|");

    let result = tokenize_with_word_map(
      &tok,
      "hello world",
      2,
      /* use_word_delimiter: */ true,
      true,
      unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.separator_token_id, Some(pipe));
  }

  #[test]
  fn separator_token_id_none_when_normaliser_opts_out() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello world",
      2,
      /* use_word_delimiter: */ false,
      true,
      unk,
      /* wildcard_boundary_per_word: */ &[],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.separator_token_id, None);
  }

  /// **Wildcards-per-word integration**: when the normaliser
  /// reports e.g. 1 stripped boundary char (a comma, period,
  /// etc.) for a word, `tokenize_with_word_map` appends one
  /// wildcard token to that word's group. The wildcard sits
  /// AFTER the word's letter chars and shares the same word
  /// index, so `merge_words` in the trellis layer extends the
  /// word's frame range through the wildcard's frames.
  #[test]
  fn trailing_wildcards_land_after_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    // "hello" with 1 SUFFIX wildcard reported → 5 letters + 1 wildcard.
    // Trailing punctuation case: `hello"` → letters then wildcard.
    let result = tokenize_with_word_map(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: */ &[(0, 1)],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.token_ids.len(), 6);
    assert_eq!(
      result.token_ids[5], WILDCARD_TOKEN_ID,
      "suffix wildcard must land at the END"
    );
    assert!(
      result.token_ids[..5]
        .iter()
        .all(|&id| id != WILDCARD_TOKEN_ID),
      "no leading wildcards expected when prefix=0; got tokens {:?}",
      result.token_ids
    );
    assert_eq!(result.word_idx_per_token, alloc::vec![Some(0); 6]);
  }

  /// Codex round-28 regression: leading punctuation like `"hello`
  /// must place its wildcard BEFORE the encoded letters, not
  /// after them. Without this distinction, `"hello` (prefix=1)
  /// and `hello"` (suffix=1) would produce identical token
  /// sequences `[h,e,l,l,o,*]` — making the CTC graph push the
  /// `*` into the trailing-frames zone for both cases and
  /// biasing word-end timing.
  #[test]
  fn leading_wildcards_land_before_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: prefix=1, suffix=0: */ &[(1, 0)],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.token_ids.len(), 6);
    assert_eq!(
      result.token_ids[0], WILDCARD_TOKEN_ID,
      "prefix wildcard must land at the START; got tokens {:?}",
      result.token_ids
    );
    assert!(
      result.token_ids[1..]
        .iter()
        .all(|&id| id != WILDCARD_TOKEN_ID),
      "no trailing wildcards expected when suffix=0; got tokens {:?}",
      result.token_ids
    );
    assert_eq!(result.word_idx_per_token, alloc::vec![Some(0); 6]);
  }

  /// Paired punctuation: `(hello)` → prefix=1, suffix=1 → both
  /// ends carry exactly one wildcard, matching source order.
  #[test]
  fn paired_wildcards_bracket_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_word_map(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: */ &[(1, 1)],
      &Lang::En,
    )
    .expect("ok");
    assert_eq!(result.token_ids.len(), 7);
    assert_eq!(result.token_ids[0], WILDCARD_TOKEN_ID, "prefix at start");
    assert_eq!(result.token_ids[6], WILDCARD_TOKEN_ID, "suffix at end");
    assert!(
      result.token_ids[1..6]
        .iter()
        .all(|&id| id != WILDCARD_TOKEN_ID),
      "interior must be encoded chars only; got {:?}",
      result.token_ids
    );
  }

  /// Wildcards-per-word length must match the word_count or
  /// the function surfaces TokenizationFailed (configuration
  /// bug — caller wired the normaliser's output incorrectly).
  #[test]
  fn wildcard_boundary_per_word_length_mismatch_errors() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let err = tokenize_with_word_map(
      &tok,
      "hello world",
      2,
      true,
      true,
      unk,
      &[(1, 0), (2, 1), (3, 0)], // length 3 but word_count = 2
      &Lang::En,
    )
    .expect_err("length mismatch must surface TokenizationFailed");
    assert!(matches!(
      err,
      WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::TokenizationFailed,
        ..
      }
    ));
  }
}
