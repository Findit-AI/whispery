//! Step 1-2 of the alignment algorithm: tokenisation + per-token
//! word-index map.

use smol_str::format_smolstr;
use tokenizers::Tokenizer;

use crate::{
  runner::aligner::algorithm::trellis_beam::WILDCARD_TOKEN_ID,
  types::{AlignmentError, AlignmentFailure, Lang, WorkFailure},
};

/// Result of tokenising the normalised text.
///
/// `pub` for the `feature = "bench-internals"` re-export at the
/// crate root — out-of-tree code only sees this type through the
/// doc-hidden `asry::__bench` namespace.
#[derive(Debug)]
pub struct TokenizedText {
  /// Vocab indices in tokenisation order (Y in spec terms),
  /// stored as `i32` because the wildcard sentinel
  /// `WILDCARD_TOKEN_ID = -1` is allowed: an alphanumeric char
  /// the model dictionary doesn't know becomes a wildcard, and
  /// the trellis emits `max(non_blank_logprobs)` for that frame.
  /// All non-wildcard ids are non-negative and fit in `u32`.
  token_ids: Vec<i32>,
  /// Per-token mapping back to the normalised-word index. `None`
  /// for tokens that have no natural word index (word-delimiter
  /// `|`, special tokens like `<s>`, `<pad>`, `<unk>`).
  word_idx_per_token: Vec<Option<usize>>,
  /// The wav2vec2 word-delimiter `|` token id, when the
  /// tokenizer exposes one and the normaliser opted in. The
  /// trellis-beam orchestrator uses this to recognise
  /// separators in `merge_words`.
  separator_token_id: Option<u32>,
}

impl TokenizedText {
  /// Construct from the three component vectors.
  #[must_use]
  pub const fn new(
    token_ids: Vec<i32>,
    word_idx_per_token: Vec<Option<usize>>,
    separator_token_id: Option<u32>,
  ) -> Self {
    Self {
      token_ids,
      word_idx_per_token,
      separator_token_id,
    }
  }

  /// Vocab indices in tokenisation order.
  #[must_use]
  pub fn token_ids(&self) -> &[i32] {
    &self.token_ids
  }

  /// Per-token mapping back to the normalised-word index.
  #[must_use]
  pub fn word_idx_per_token(&self) -> &[Option<usize>] {
    &self.word_idx_per_token
  }

  /// The wav2vec2 word-delimiter `|` token id, when present.
  #[must_use]
  pub const fn separator_token_id(&self) -> Option<u32> {
    self.separator_token_id
  }
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
/// **Asry-specific guard (default policy):** non-alphanumeric
/// chars (e.g. `&` in `AT&T`) that aren't on the whitelist of
/// skippable internal punctuation still drop the whole chunk's
/// alignment. WhisperX would silently align them as wildcards too,
/// but those chars represent semantic content the model can't
/// reasonably align (the `&` in `AT&T` is "and"); silently aligning
/// to whichever vocab item happens to win the frame would produce
/// honest-looking but wrong word ranges. Asry fails closed at
/// chunk granularity instead.
///
/// **WhisperX-equivalent runtime policy:** callers that want
/// `clean_char.append('*')` 1:1 (every OOV → wildcard) supply
/// `crate::core::wildcard_all_decisions(&events)` instead of
/// `default_oov_decisions(&events)`. The Cargo feature that
/// used to flip this globally (`whisperx-strict-tokenizer`)
/// was removed in the Sans-I/O OOV refactor — per-deployment
/// policy is now data, not a compile-time flag.
///
/// Internal punctuation that's never pronounced as a separate
/// sound and is safe to strip before encoding so the
/// surrounding letters still align. Currently just the period:
/// `U.S.A.`, `D.C.`, `etc.` are pronounced as their letters.
fn is_skippable_internal_punct(c: char) -> bool {
  c == '.'
}

/// Consume the next caller decision for a wildcard-generating
/// position; surface a typed `TokenizationFailed` if the
/// caller pre-sized too small. (parity
/// loop) [high]: shared by the boundary-prefix /
/// internal-punct / symbol-OOV / boundary-suffix sites so
/// they all consult the same indexed slice.
fn consume_oov_decision(
  oov_decisions: &[crate::core::ResolvedOov],
  oov_consumed: &mut usize,
  language: &Lang,
  site_label: &str,
) -> Result<crate::core::OovDecision, WorkFailure> {
  let decision = oov_decisions
    .get(*oov_consumed)
    .map(|r| r.decision())
    .ok_or_else(|| {
      WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
        format_smolstr!(
          "oov_decisions ran out at index {} ({site_label}); call detect_oov_events \
 first to size the decisions vec correctly",
          *oov_consumed,
        ),
        language.clone(),
      )))
    })?;
  *oov_consumed += 1;
  Ok(decision)
}

/// Build a `SemanticOutOfVocab` failure for a boundary-punct
/// wildcard the caller refused via `OovDecision::FailClosed`.
fn boundary_fail_closed(language: &Lang, position: &str) -> WorkFailure {
  WorkFailure::Alignment(AlignmentError::SemanticOutOfVocab(AlignmentFailure::new(
    format_smolstr!(
      "BoundaryPunct ({position}) resolved as FailClosed by caller policy; \
 chunk word alignment dropped (ASR text preserved)."
    ),
    language.clone(),
  )))
}

// `allow_wildcard` was deleted in the Sans-I/O OOV refactor
// (the `whisperx-strict-tokenizer` Cargo feature went with it).
// Policy is now caller-supplied as data — see
// `crate::core::oov`:
// * `default_oov_decisions` — historical default
// (alphanumeric/apostrophe → wildcard, pronounced → fail-closed).
// * `wildcard_all_decisions` — replaces the removed
// `whisperx-strict-tokenizer` feature (WhisperX 1:1).
// * `fail_closed_all_decisions` — strictest.
//
// `tokenize_with_word_map` consumes the resulting
// `&[ResolvedOov]` per OOV position in `detect_oov_events`
// order, validating each `ResolvedOov.event` against the
// freshly-detected event at the same position.

/// `pub` for the `feature = "bench-internals"` re-export at the
/// crate root. Out-of-tree code only reaches this through
/// `asry::__bench`, which is doc-hidden and gated on the
/// `bench-internals` feature.
#[allow(
  clippy::too_many_arguments,
  reason = "8 args mirror the wav2vec2 tokenisation contract \
 (tokenizer, text, word_count, delimiter flag, casing \
 flag, unk id, wildcard map, output buffer); each is a \
 distinct semantic input from a different upstream pass"
)]
/// Sans-I/O OOV detection — runs the same per-character
/// iteration as [`tokenize_with_word_map`] but emits an
/// [`OovEvent`] for each char that would otherwise hit the
/// "is unknown or empty" branch, instead of making a policy
/// decision.
///
/// **Order invariant.** Events are emitted in the order
/// `tokenize_with_word_map` encounters them. Callers that
/// later supply `&[ResolvedOov]` to the apply-decisions form
/// must produce decisions in the same order, with each
/// `ResolvedOov.event` matching the event at its position.
///
/// `is_skippable_internal_punct` chars (currently `.`) are
/// NOT surfaced — they're stripped silently and replaced with
/// a wildcard token at the same position; that's a tokenizer-
/// internal detail, not a policy decision.
///
/// pinned the contract that pronounced
/// non-alphanumeric OOV chars (`&`, `@`, `%`, `,`) are
/// observable rather than silently dropped — this function
/// makes them observable to the caller as data.
///
/// Returns an error only on tokenizer-engine failures
/// (`encode(..)` itself errored). OOV detection itself never
/// fails — every detected char becomes an event.
pub fn detect_oov_events(
  tokenizer: &Tokenizer,
  normalized: &str,
  word_count: usize,
  uppercase_input: bool,
  unk_token_id: Option<u32>,
  language: &Lang,
  // Per-word boundary-punct counts as supplied to
  // `tokenize_with_word_map`. Must be either empty (=
  // "no boundary wildcards") or `word_count`-long. Codex
  // boundary-punct
  // wildcards are surfaced as `OovKind::BoundaryPunct`
  // events so strict callers
  // (`fail_closed_all_decisions`) can refuse them.
  wildcard_boundary_per_word: &[crate::runner::aligner::normalizer::WildcardBoundary],
) -> Result<Vec<crate::core::OovEvent>, WorkFailure> {
  use crate::core::OovEvent;

  let mut events: Vec<OovEvent> = Vec::new();
  let words: Vec<&str> = normalized.split_whitespace().collect();
  if words.len() != word_count {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "word_count mismatch: caller={}, normalized has {}",
          word_count,
          words.len(),
        ),
        language.clone(),
      ),
    )));
  }
  // Boundary-per-word slice MUST be either empty or
  // word_count-long; surface a hard error otherwise rather
  // than risk indexing-skew between detect + tokenize.
  if !wildcard_boundary_per_word.is_empty() && wildcard_boundary_per_word.len() != word_count {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "wildcard_boundary_per_word.len() = {} != word_count = {}",
          wildcard_boundary_per_word.len(),
          word_count,
        ),
        language.clone(),
      ),
    )));
  }
  let mut tmp_buf = String::with_capacity(8);
  let mut char_index: usize = 0;
  for (word_index, word) in words.iter().enumerate() {
    let boundary = wildcard_boundary_per_word
      .get(word_index)
      .copied()
      .unwrap_or(crate::runner::aligner::normalizer::WildcardBoundary::NONE);
    let prefix_wildcards = boundary.prefix();
    let suffix_wildcards = boundary.suffix();
    // Boundary-prefix wildcards: surface BEFORE the word's
    // chars so the event order matches `tokenize_with_word_map`'s
    // emit order exactly.
    for _ in 0..prefix_wildcards {
      events.push(OovEvent::new(
        crate::core::OovKind::BoundaryPunct,
        char_index,
        word_index,
        language.clone(),
      ));
    }
    for ch in word.chars() {
      if is_skippable_internal_punct(ch) {
        events.push(OovEvent::new(
          crate::core::OovKind::InternalPunct(ch),
          char_index,
          word_index,
          language.clone(),
        ));
        char_index += 1;
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
        .map_err(|e| {
          WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
            format_smolstr!("encode({:?}) failed: {e:?}", projected),
            language.clone(),
          )))
        })?;
      let ids = encoding.get_ids();
      let is_unk_or_empty = ids.is_empty()
        || match unk_token_id {
          Some(unk) => ids.contains(&unk),
          None => false,
        };
      if is_unk_or_empty {
        events.push(OovEvent::new(
          crate::core::OovKind::Symbol(ch),
          char_index,
          word_index,
          language.clone(),
        ));
      }
      char_index += 1;
    }
    // Boundary-suffix wildcards: surface AFTER the word's
    // chars (mirrors tokenize order).
    for _ in 0..suffix_wildcards {
      events.push(OovEvent::new(
        crate::core::OovKind::BoundaryPunct,
        char_index,
        word_index,
        language.clone(),
      ));
    }
    // Word separator counted as one char in the index space
    // so neighbouring words' char_index values stay
    // intuitive for callers. Last word doesn't add a trailing
    // separator.
    if word_index + 1 < words.len() {
      char_index += 1;
    }
  }
  Ok(events)
}

/// Tokenise `normalized` against the wav2vec2 tokeniser into
/// `TokenizedText` (token-id stream + per-token word index +
/// separator id).
///
/// This is the second half of the Sans-I/O OOV resolution
/// flow: callers run [`detect_oov_events`] first to get a
/// `Vec<OovEvent>`, decide on each via a policy helper from
/// [`crate::core::oov`] (or a custom closure), and supply the
/// resulting `&[ResolvedOov]` here. The function recomputes
/// events for the chunk's text and refuses to apply a payload
/// whose embedded `event` does not match the freshly-detected
/// event at the same position.
///
/// Validation surfaces three flavours of `TokenizationFailed`:
/// * length mismatch — caller pre-sized too few or too many
/// decisions for this text;
/// * per-position identity mismatch — caller's payload was
/// produced for different text (different char_index,
/// word_index, kind, or language) and would silently
/// misalign if applied;
/// * mid-loop too-short consumption — defense-in-depth if the
/// preflight is somehow bypassed.
///
/// `OovDecision::Wildcard` pushes `WILDCARD_TOKEN_ID = -1`;
/// `OovDecision::FailClosed` returns `SemanticOutOfVocab`.
///
/// `pub` for the `feature = "bench-internals"` re-export at
/// the crate root. Out-of-tree code only reaches this through
/// `asry::__bench`, which is doc-hidden and gated on
/// `bench-internals`.
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
  // / non-English-normaliser path); asry's
  // [`crate::EnglishNormalizer`] populates it from the
  // boundary-punctuation strip count.
  wildcard_boundary_per_word: &[crate::runner::aligner::normalizer::WildcardBoundary],
  language: &Lang,
  // Caller's per-OOV-event resolved decisions, indexed by the
  // order [`detect_oov_events`] would have produced them.
  // Required: produce via `detect_oov_events` + a policy
  // helper from `crate::core::oov` (e.g.
  // `default_oov_decisions`, `wildcard_all_decisions`). An
  // empty slice means "no OOV expected"; encountering one
  // anyway raises `TokenizationFailed`. Each
  // `ResolvedOov.event` must match the freshly-detected event
  // at the same position — a stale-but-same-length payload
  // from a different chunk fails the per-position identity
  // check rather than silently misaligning.
  oov_decisions: &[crate::core::ResolvedOov],
) -> Result<TokenizedText, WorkFailure> {
  // + round-9
  // [high]: pre-validate length AND per-position event
  // identity BEFORE applying any decisions.
  //
  // Round 7 caught the FailClosed-early-return-skips-length-
  // check bug. Round 9 noted that a same-length-but-stale
  // payload (e.g. `[Wildcard]` produced for a digit OOV
  // applied to `&` OOV) would still pass a count-only check
  // and silently misalign. Binding decisions to events via
  // `ResolvedOov` makes that detectable: we recompute the
  // events for THIS chunk's text and require each supplied
  // `ResolvedOov.event` to match the recomputed event by
  // identity (kind, char_index, word_index, language).
  //
  // Cost: one duplicate per-char tokenizer.encode pass.
  // Tokenize is microsecond-scale, correctness trumps perf.
  let pre_events = detect_oov_events(
    tokenizer,
    normalized,
    word_count,
    uppercase_input,
    unk_token_id,
    language,
    wildcard_boundary_per_word,
  )?;
  if pre_events.len() != oov_decisions.len() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "oov_decisions length {} does not match the {} OOV events detected for this \
 text; this typically means the caller passed decisions from a different \
 chunk's text. Re-run `detect_oov_events` for the chunk's normalised text \
 and re-decide before calling `tokenize_with_word_map`.",
          oov_decisions.len(),
          pre_events.len(),
        ),
        language.clone(),
      ),
    )));
  }
  // identity
  // equality must compare POSITIONAL fields only (kind,
  // char_index, word_index) — not `language`. Under
  // `AlignerKey::Any` fallback, `AlignmentSet::detect_oov`
  // stamps events with the caller-REQUESTED language so
  // caller policy can switch on it (e.g. wildcard-en /
  // fail-closed-ko), but the inner Aligner re-detects events
  // here with its own CONSTRUCTION language. Comparing the
  // language field would reject every Any-fallback chunk
  // containing an OOV. Positional fields uniquely identify
  // the position in the chunk; language is policy metadata,
  // not positional identity.
  for (i, (pre, resolved)) in pre_events.iter().zip(oov_decisions.iter()).enumerate() {
    if !resolved.event().matches_position(pre) {
      return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
        AlignmentFailure::new(
          format_smolstr!(
            "oov_decisions[{i}] was produced for a different OOV event than the one \
 this chunk's text actually has at position {i}: supplied={:?} but \
 detected={:?}. This typically means the caller reused decisions from a \
 previous chunk whose OOV count happened to match. Re-run \
 `detect_oov_events` for THIS chunk's normalised text and re-decide.",
            resolved.event(),
            pre,
          ),
          language.clone(),
        ),
      )));
    }
  }

  let mut oov_consumed: usize = 0;
  let mut token_ids: Vec<i32> = Vec::with_capacity(normalized.len() + word_count * 2);
  let mut word_idx_per_token: Vec<Option<usize>> = Vec::with_capacity(token_ids.capacity());

  let words: Vec<&str> = normalized.split_whitespace().collect();
  if words.len() != word_count {
    // Sanity: caller's claimed word_count must match the
    // normalised text. Off-by-one here would mis-index Word
    // emission in step 9.
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "word_count mismatch: caller={}, normalized has {}",
          word_count,
          words.len()
        ),
        language.clone(),
      ),
    )));
  }
  if !wildcard_boundary_per_word.is_empty() && wildcard_boundary_per_word.len() != word_count {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "wildcard_boundary_per_word.len() = {} != word_count = {}",
          wildcard_boundary_per_word.len(),
          word_count
        ),
        language.clone(),
      ),
    )));
  }

  // Per-char tokenisation. We can't encode the whole word at
  // once and inspect after-the-fact: we need to know which
  // *char* produced an `<unk>` so we can decide between
  // wildcard-and-keep vs drop-the-chunk.
  let mut per_word_tokens: Vec<Vec<i32>> = Vec::with_capacity(words.len());
  let mut tmp_buf = String::with_capacity(8);
  for (wi, word) in words.iter().enumerate() {
    let boundary = wildcard_boundary_per_word
      .get(wi)
      .copied()
      .unwrap_or(crate::runner::aligner::normalizer::WildcardBoundary::NONE);
    let prefix_wildcards = boundary.prefix();
    let suffix_wildcards = boundary.suffix();
    let mut word_tokens: Vec<i32> = Vec::with_capacity(word.len());
    // Push prefix wildcards BEFORE the encoded chars so leading
    // punctuation like `"hello` aligns its `*` placeholders
    // ahead of `h, e, l, l, o`. The CTC graph then matches the
    // source order, mirroring WhisperX's approach. Codex
    // each wildcard
    // consults `oov_decisions` so strict callers
    // (`fail_closed_all_decisions`) genuinely fail-close on
    // structural wildcards too.
    for _ in 0..prefix_wildcards {
      let decision = consume_oov_decision(
        oov_decisions,
        &mut oov_consumed,
        language,
        "BoundaryPunct (prefix)",
      )?;
      match decision {
        crate::core::OovDecision::Wildcard => word_tokens.push(WILDCARD_TOKEN_ID),
        crate::core::OovDecision::FailClosed => {
          return Err(boundary_fail_closed(language, "prefix"));
        }
      }
    }
    for ch in word.chars() {
      if is_skippable_internal_punct(ch) {
        // emit the wildcard
        // token immediately at the source position of the
        // skipped internal punctuation. this counted
        // skipped chars and appended all wildcards at the end
        // of the word, breaking WhisperX token-order parity for
        // dotted acronyms like "U.S.A": WhisperX interleaves
        // [U, *, S, *, A, *] (one `*` per `.`), the  // emitted [U, S, A, *, *, *]. The total token count
        // matched but the per-position CTC frame attribution
        // shifted, moving boundaries on the following word.
        // Now we keep the source order — wildcards land at the
        // exact byte position of the punct char. The OOV
        // decision (parity-loop [high])
        // governs whether the wildcard is actually emitted or
        // surfaces as `SemanticOutOfVocab`.
        let decision =
          consume_oov_decision(oov_decisions, &mut oov_consumed, language, "InternalPunct")?;
        match decision {
          crate::core::OovDecision::Wildcard => word_tokens.push(WILDCARD_TOKEN_ID),
          crate::core::OovDecision::FailClosed => {
            return Err(WorkFailure::Alignment(AlignmentError::SemanticOutOfVocab(
              AlignmentFailure::new(
                format_smolstr!(
                  "InternalPunct {ch:?} resolved as FailClosed by caller policy; \
 chunk word alignment dropped (ASR text preserved)."
                ),
                language.clone(),
              ),
            )));
          }
        }
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
        .map_err(|e| {
          WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
            format_smolstr!("encode({:?}) failed: {e:?}", projected),
            language.clone(),
          )))
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
          Some(unk) => ids.contains(&unk),
          None => false,
        };
      if is_unk_or_empty {
        let decision = consume_oov_decision(oov_decisions, &mut oov_consumed, language, "Symbol")?;
        match decision {
          crate::core::OovDecision::Wildcard => {
            word_tokens.push(WILDCARD_TOKEN_ID);
          }
          crate::core::OovDecision::FailClosed => {
            // Non-alphanumeric semantic OOV the caller policy
            // told us to drop. We surface a typed failure so
            // the dispatch recovery preserves the ASR
            // transcript while making the drop observable.
            // introduced this
            // kind; the Sans-I/O OOV refactor preserves it.
            return Err(WorkFailure::Alignment(AlignmentError::SemanticOutOfVocab(
              AlignmentFailure::new(
                format_smolstr!(
                  "OOV {ch:?} resolved as FailClosed by caller policy; \
 chunk word alignment dropped (ASR text preserved)."
                ),
                language.clone(),
              ),
            )));
          }
        }
      } else {
        // validate every model
        // id fits an `i32` AND is non-negative before storing
        // alongside the `WILDCARD_TOKEN_ID = -1` sentinel.
        // `id as i32` aliased `u32::MAX` to `-1`, which
        // the trellis would then treat as a wildcard instead of
        // a real model token — silent misalignment for sparse
        // / malformed tokenizers. `i32::try_from` returns the
        // out-of-range case as a `TokenizationFailed` so the
        // caller learns about the tokenizer/model mismatch.
        for &id in ids {
          let signed_id = i32::try_from(id).map_err(|_| {
            WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
              format_smolstr!(
                "tokenizer returned id {} which exceeds i32::MAX or aliases the wildcard \
 sentinel; tokenizer / model mismatch?",
                id
              ),
              language.clone(),
            )))
          })?;
          if signed_id < 0 {
            return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
              AlignmentFailure::new(
                format_smolstr!(
                  "tokenizer returned negative-after-cast id {} (raw {}); refusing to alias \
 wildcard sentinel",
                  signed_id,
                  id
                ),
                language.clone(),
              ),
            )));
          }
          word_tokens.push(signed_id);
        }
      }
    }
    // Append SUFFIX wildcards from the normaliser's trailing-
    // punct strip count. Internal-punct wildcards are emitted
    // in source order inside the loop above ( // round-8 fix), so this branch only handles boundary
    // wildcards now.
    // [high]: each suffix wildcard consults `oov_decisions`
    // — see the prefix loop above for the rationale.
    for _ in 0..suffix_wildcards {
      let decision = consume_oov_decision(
        oov_decisions,
        &mut oov_consumed,
        language,
        "BoundaryPunct (suffix)",
      )?;
      match decision {
        crate::core::OovDecision::Wildcard => word_tokens.push(WILDCARD_TOKEN_ID),
        crate::core::OovDecision::FailClosed => {
          return Err(boundary_fail_closed(language, "suffix"));
        }
      }
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
      // Same overflow / sentinel-alias guard as the per-char
      // path above ([high]).
      let signed_d = i32::try_from(d).map_err(|_| {
        WorkFailure::Alignment(AlignmentError::Tokenization(AlignmentFailure::new(
          format_smolstr!(
            "tokenizer returned `|` delimiter id {} which exceeds i32::MAX",
            d
          ),
          language.clone(),
        )))
      })?;
      if signed_d < 0 {
        return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
          AlignmentFailure::new(
            format_smolstr!(
              "tokenizer returned negative-after-cast `|` delimiter id {} (raw {})",
              signed_d,
              d
            ),
            language.clone(),
          ),
        )));
      }
      token_ids.push(signed_d);
      word_idx_per_token.push(None);
    }
    for &id in group {
      token_ids.push(id);
      word_idx_per_token.push(Some(word_idx));
    }
    last_emitted_word = Some(word_idx);
  }

  // every
  // supplied decision must correspond to an OOV the
  // tokenizer encountered. The loop only checked
  // `oov_decisions.get(oov_consumed)` for the too-short
  // case; a stale / superset decision vec from a previous
  // chunk could leak in and silently apply the wrong prefix
  // policy (e.g. decisions for `"4&"` = [Wildcard,
  // FailClosed] applied to current text `"&"` would
  // wildcard the `&` instead of fail-closing). Reject the
  // mismatch loudly.
  if oov_consumed != oov_decisions.len() {
    return Err(WorkFailure::Alignment(AlignmentError::Tokenization(
      AlignmentFailure::new(
        format_smolstr!(
          "oov_decisions length {} does not match the {} OOV chars actually \
 encountered; this typically means the caller passed decisions \
 from a different chunk's text. Re-run `detect_oov_events` for \
 the chunk's normalised text and re-decide.",
          oov_decisions.len(),
          oov_consumed,
        ),
        language.clone(),
      ),
    )));
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

  /// Convenience wrapper that mirrors the historical default
  /// policy (alphanumeric / apostrophe → wildcard, pronounced
  /// → fail-closed) for tests written against
  /// `tokenize_with_word_map` before slice 4 made decisions
  /// caller-supplied. Calls `detect_oov_events` + the
  /// `default_oov_decisions` helper.
  fn tokenize_with_default_oov(
    tokenizer: &Tokenizer,
    normalized: &str,
    word_count: usize,
    use_word_delimiter: bool,
    uppercase_input: bool,
    unk_token_id: Option<u32>,
    wildcard_boundary_per_word: &[crate::runner::aligner::normalizer::WildcardBoundary],
    language: &Lang,
  ) -> Result<TokenizedText, WorkFailure> {
    let events = detect_oov_events(
      tokenizer,
      normalized,
      word_count,
      uppercase_input,
      unk_token_id,
      language,
      wildcard_boundary_per_word,
    )?;
    let decisions = crate::core::default_oov_decisions(&events);
    tokenize_with_word_map(
      tokenizer,
      normalized,
      word_count,
      use_word_delimiter,
      uppercase_input,
      unk_token_id,
      wildcard_boundary_per_word,
      language,
      &decisions,
    )
  }

  // -- detect_oov_events tests --------------------------------

  /// In-vocab text produces no events (the tokenizer encodes
  /// every char without hitting `<unk>`).
  #[test]
  fn detect_oov_events_empty_for_in_vocab_text() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(
      &tok,
      "hello",
      1,
      /* uppercase_input: */ true,
      unk,
      &Lang::En,
      &[],
    )
    .expect("ok");
    assert!(
      events.is_empty(),
      "in-vocab text should produce 0 events; got {events:?}"
    );
  }

  /// Pronounced symbols (`&`, `,`, `@`) and digits show up as
  /// events in the order they appear in the text.
  #[test]
  fn detect_oov_events_collects_in_source_order() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[]).expect("ok");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind(), &crate::core::OovKind::Symbol('&'));
    assert_eq!(events[0].word_index(), 0);
    assert_eq!(events[0].language(), &Lang::En);
  }

  /// a
  /// decisions slice longer than the actual OOV count must
  /// reject loudly. The loop only checked the
  /// too-short case via `oov_decisions.get(oov_consumed)`;
  /// extras at the tail were silently ignored. The risk:
  /// stale decisions from a previous chunk leak in (e.g.
  /// `[Wildcard, FailClosed]` from `"4&"` applied to current
  /// `"&"` would wildcard the `&` instead of fail-closing).
  #[test]
  fn tokenize_with_word_map_rejects_too_long_oov_decisions() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    // "AT&T" has exactly one OOV (`&`); supply two ResolvedOov.
    // The first matches the real event so length mismatch (not
    // identity mismatch) is the failing predicate; the second
    // is a synthetic event that will never be reached.
    let real_event = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[])
      .expect("ok")
      .pop()
      .expect("AT&T has one OOV");
    let extra_event =
      crate::core::OovEvent::new(crate::core::OovKind::Symbol('?'), 99, 99, Lang::En);
    let too_long = vec![
      crate::core::ResolvedOov::new(real_event, crate::core::OovDecision::Wildcard),
      crate::core::ResolvedOov::new(extra_event, crate::core::OovDecision::Wildcard),
    ];
    let result =
      tokenize_with_word_map(&tok, "AT&T", 1, true, true, unk, &[], &Lang::En, &too_long);
    match result {
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(payload))) => {
        assert!(
          payload.message().contains("oov_decisions length 2")
            && payload.message().contains("1 OOV events detected"),
          "diagnostic should cite the length mismatch; got {message}",
          message = payload.message(),
        );
      }
      other => panic!("expected TokenizationFailed mismatch; got {other:?}"),
    }
  }

  /// when a
  /// stale too-long decisions vec STARTS with `FailClosed`,
  /// the pre-flight length check must reject it as
  /// `TokenizationFailed` BEFORE the loop's `FailClosed`
  /// early-return surfaces as `SemanticOutOfVocab`.  /// the early-return skipped the post-loop length check
  /// entirely, so a one-OOV chunk with a 2-decision payload
  /// got `SemanticOutOfVocab` (recoverable, drops words
  /// silently) instead of `TokenizationFailed` (the actual
  /// stale-payload diagnostic).
  #[test]
  fn tokenize_with_word_map_rejects_too_long_decisions_even_when_first_is_fail_closed() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let real_event = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[])
      .expect("ok")
      .pop()
      .expect("AT&T has one OOV");
    let extra_event =
      crate::core::OovEvent::new(crate::core::OovKind::Symbol('?'), 99, 99, Lang::En);
    let too_long = vec![
      crate::core::ResolvedOov::new(real_event, crate::core::OovDecision::FailClosed),
      crate::core::ResolvedOov::new(extra_event, crate::core::OovDecision::Wildcard),
    ];
    let result =
      tokenize_with_word_map(&tok, "AT&T", 1, true, true, unk, &[], &Lang::En, &too_long);
    match result {
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(_))) => {
        // Correct: stale-payload mismatch surfaces as
        // TokenizationFailed (loud), not SemanticOutOfVocab
        // (silent recoverable empty-words drop).
      }
      Err(WorkFailure::Alignment(AlignmentError::SemanticOutOfVocab(_))) => panic!(
        "stale too-long decisions starting with FailClosed must surface as \
 TokenizationFailed (the loud diagnostic); SemanticOutOfVocab is the \
 silent recoverable path that masks the bug"
      ),
      other => panic!("expected TokenizationFailed mismatch; got {other:?}"),
    }
  }

  /// a stale
  /// decisions vec whose length matches the chunk's OOV count
  /// but whose embedded events were produced for DIFFERENT
  /// text must be rejected as `TokenizationFailed`.
  ///
  /// The preflight only checked length; a `[Wildcard]`
  /// decision originally produced for the digit OOV in `"4"`
  /// could be reused against `&` in `"AT&T"` and the dispatcher
  /// would happily wildcard the `&` even though the default
  /// policy would fail-closed on it. Binding decisions to
  /// events via `ResolvedOov` makes the per-position identity
  /// check catch this.
  #[test]
  fn tokenize_with_word_map_rejects_stale_same_length_decisions() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    // Decisions produced for `"4"` (digit OOV at char_index=0).
    let stale_for_digit = detect_oov_events(&tok, "4", 1, true, unk, &Lang::En, &[]).expect("ok");
    assert_eq!(stale_for_digit.len(), 1);
    let stale_resolved = vec![crate::core::ResolvedOov::new(
      stale_for_digit[0].clone(),
      crate::core::OovDecision::Wildcard,
    )];
    // Apply against `"AT&T"` (one OOV but it's `&` at
    // char_index=2, word_index=0) — same length, different
    // event.
    let result = tokenize_with_word_map(
      &tok,
      "AT&T",
      1,
      true,
      true,
      unk,
      &[],
      &Lang::En,
      &stale_resolved,
    );
    match result {
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(payload))) => {
        assert!(
          payload.message().contains("different OOV event"),
          "diagnostic should cite the per-position identity mismatch; got {message}",
          message = payload.message(),
        );
      }
      other => panic!("expected TokenizationFailed identity mismatch; got {other:?}"),
    }
  }

  /// under
  /// `AlignerKey::Any` fallback, `AlignmentSet::detect_oov`
  /// stamps events with the CALLER-requested language so
  /// caller policy can switch on it; the inner Aligner
  /// re-detects events with its own CONSTRUCTION language.
  /// The identity check must therefore compare positional
  /// fields only — including `language` would reject every
  /// Any-fallback chunk that contains an OOV.
  ///
  /// This test simulates the Any-fallback shape: tokenizer
  /// language is `En` (the fallback aligner) but the
  /// supplied `ResolvedOov.event.language` is `Ko` (the
  /// caller's requested language). Same positional fields,
  /// different language. Must succeed (apply the wildcard),
  /// not surface as `TokenizationFailed`.
  #[test]
  fn tokenize_with_word_map_accepts_mismatched_language_under_any_fallback() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    // The fallback aligner detects against its own language
    // (En), producing `Symbol('&')` at char_index=2, word_idx=0.
    let pre = detect_oov_events(&tok, "AT&T", 1, true, unk, &Lang::En, &[])
      .expect("ok")
      .pop()
      .expect("AT&T has one OOV");
    assert_eq!(pre.language(), &Lang::En);
    // Caller's payload was produced via `AlignmentSet::detect_oov`
    // for a Korean-tagged run; the event language stamp is `Ko`
    // but the positional fields match the En detection above.
    let resolved = vec![crate::core::ResolvedOov::new(
      crate::core::OovEvent::new(
        pre.kind().clone(),
        pre.char_index(),
        pre.word_index(),
        Lang::Ko,
      ),
      crate::core::OovDecision::Wildcard,
    )];
    let result =
      tokenize_with_word_map(&tok, "AT&T", 1, true, true, unk, &[], &Lang::En, &resolved);
    assert!(
      result.is_ok(),
      "Any-fallback identity check must compare positional fields \
 only (kind/char_index/word_index), not language. Got: {result:?}",
    );
  }

  /// Multi-word text with mixed OOV: `4` (digit, alphanumeric)
  /// + `,` (pronounced symbol) yields two events with the
  /// expected `word_index` values.
  #[test]
  fn detect_oov_events_tracks_word_index() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(&tok, "AT&T cost 43", 3, true, unk, &Lang::En, &[]).expect("ok");
    let chars: Vec<Option<char>> = events.iter().map(|e| e.char()).collect();
    let words: Vec<usize> = events.iter().map(|e| e.word_index()).collect();
    assert_eq!(chars, vec![Some('&'), Some('4'), Some('3')]);
    assert_eq!(words, vec![0, 2, 2]);
  }

  /// Internal-skippable punct (`.`) is NOT surfaced — it's a
  /// tokenizer-internal detail (always wildcarded at the
  /// position).
  #[test]
  fn detect_oov_events_surfaces_internal_punct() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let events = detect_oov_events(&tok, "U.S.A", 1, true, unk, &Lang::En, &[]).expect("ok");
    // internal-
    // punct wildcards are now surfaced as `OovKind::InternalPunct`
    // events so strict policies (`fail_closed_all_decisions`)
    // can refuse them. `U.S.A` has two `.` chars.
    let kinds: Vec<crate::core::OovKind> = events.iter().map(|e| e.kind().clone()).collect();
    assert_eq!(
      kinds,
      vec![
        crate::core::OovKind::InternalPunct('.'),
        crate::core::OovKind::InternalPunct('.'),
      ],
      "U.S.A should surface 2 InternalPunct events; got {events:?}",
    );
  }

  /// Word-count mismatch surfaces as `TokenizationFailed`,
  /// matching `tokenize_with_word_map`'s contract.
  #[test]
  fn detect_oov_events_word_count_mismatch_errors() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");
    let result = detect_oov_events(&tok, "hello world", 1, true, unk, &Lang::En, &[]);
    assert!(matches!(
      result,
      Err(WorkFailure::Alignment(AlignmentError::Tokenization(_)))
    ));
  }

  // -- tokenize_with_word_map tests ---------------------------

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
      &[],
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

    let result =
      tokenize_with_default_oov(&tok, ".", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids, vec![WILDCARD_TOKEN_ID]);
    assert_eq!(result.word_idx_per_token, vec![Some(0)]);
  }

  /// Internal periods strip to wildcard tokens in **source
  /// order** (the
  /// implementation appended internal-punct wildcards at the
  /// end of the word, breaking WhisperX token-order parity for
  /// dotted acronyms; the post-fix layout is [U, *, S, *, A] —
  /// matching WhisperX's per-position `*` placeholder for
  /// chars not in the model dictionary).
  #[test]
  fn internal_periods_in_abbreviation_strip_to_letters() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_default_oov(
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

    // 3 letter ids + 2 internal-period wildcards = 5 tokens,
    // interleaved in source order: [U, *, S, *, A].
    assert_eq!(result.token_ids.len(), 5);
    let unk_i32 = unk.unwrap() as i32;
    assert!(
      result.token_ids.iter().all(|&id| id != unk_i32),
      "no <unk> ids must reach the lattice; got {:?}",
      result.token_ids
    );
    let id_of = |c: char| tok.token_to_id(&c.to_string()).unwrap() as i32;
    assert_eq!(
      result.token_ids,
      vec![
        id_of('U'),
        WILDCARD_TOKEN_ID,
        id_of('S'),
        WILDCARD_TOKEN_ID,
        id_of('A'),
      ],
      "internal-punct wildcards must land in source order, not appended at end"
    );
    // All 5 tokens belong to word 0.
    assert_eq!(result.word_idx_per_token, vec![Some(0); 5]);
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
      tokenize_with_default_oov(&tok, "B2B", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 3);
    let b_id = tok.token_to_id("B").unwrap() as i32;
    assert_eq!(result.token_ids[0], b_id);
    assert_eq!(result.token_ids[1], WILDCARD_TOKEN_ID);
    assert_eq!(result.token_ids[2], b_id);
    assert_eq!(result.word_idx_per_token, vec![Some(0); 3]);
  }

  /// Same: a fully-alphanumeric all-OOV word (digits) maps to
  /// all-wildcards; the chunk does NOT drop. WhisperX-style
  /// permissive alignment.
  #[test]
  fn all_digit_word_against_uppercase_vocab_uses_wildcards() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result =
      tokenize_with_default_oov(&tok, "1000", 1, true, true, unk, &[], &Lang::En).expect("ok");
    assert_eq!(result.token_ids.len(), 4);
    assert!(
      result.token_ids.iter().all(|&id| id == WILDCARD_TOKEN_ID),
      "every digit must become a wildcard; got {:?}",
      result.token_ids
    );
  }

  /// Asry-specific guard preserved: non-alphanumeric
  /// pronounced char (`&` in `AT&T`) still drops the chunk's
  /// alignment. WhisperX would silently align it; asry
  /// fails closed because the `&` is pronounced as "and" and
  /// aligning to whichever vocab item wins the frame produces
  /// a wrong range.
  ///
  /// this returned
  /// `Ok(empty TokenizedText)`, which `Aligner::align` treated
  /// as a successful empty alignment — silent loss with no
  /// observable failure. Post-fix the chunk-drop is surfaced
  /// as `::SemanticOutOfVocab`, classified
  /// as recoverable so the dispatch still preserves the ASR
  /// transcript but the failure is observable in telemetry.
  ///
  /// This test runs under the historical default policy
  /// (`tokenize_with_default_oov` — alphanumeric → wildcard,
  /// pronounced → fail-closed). Callers who want the
  /// WhisperX wildcard-everything behaviour now opt in at
  /// runtime via `wildcard_all_decisions` (see
  /// `whisperx_unit_parity::issue_1372_digits_comma_no_timestamps`)
  /// instead of via a Cargo feature; the cfg gate this test
  /// previously carried is gone with the removed
  /// `whisperx-strict-tokenizer` feature.
  #[test]
  fn ampersand_oov_drops_chunk() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let outcome = tokenize_with_default_oov(&tok, "AT&T", 1, true, true, unk, &[], &Lang::En);
    match outcome {
      Err(crate::types::WorkFailure::Alignment(AlignmentError::SemanticOutOfVocab(payload))) => {
        assert_eq!(payload.language(), &Lang::En);
        let message = payload.message();
        assert!(
          message.contains("'&'") || message.contains("\"&\""),
          "diagnostic should cite the offending char; got {message}",
        );
      }
      other => panic!("expected SemanticOutOfVocab AlignmentFailed; got {other:?}"),
    }
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
      tokenize_with_default_oov(&tok, "café", 1, true, true, unk, &[], &Lang::En).expect("ok");
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

    let result =
      tokenize_with_default_oov(&tok, "hi 1000 world", 3, true, true, unk, &[], &Lang::En)
        .expect("ok");
    // Words: hi (2), |, wildcards (4), |, world (5). 2 + 1 + 4 + 1 + 5 = 13.
    assert_eq!(result.token_ids.len(), 13);
    // Three distinct word indices represented (0, 1, 2).
    let word_indices: std::collections::BTreeSet<usize> = result
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
      &[],
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
      &[],
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
    let result = tokenize_with_default_oov(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: */
      &[crate::runner::aligner::normalizer::WildcardBoundary::new(
        0, 1,
      )],
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
    assert_eq!(result.word_idx_per_token, vec![Some(0); 6]);
  }

  /// regression: leading punctuation like `"hello`
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

    let result = tokenize_with_default_oov(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: prefix=1, suffix=0: */
      &[crate::runner::aligner::normalizer::WildcardBoundary::new(
        1, 0,
      )],
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
    assert_eq!(result.word_idx_per_token, vec![Some(0); 6]);
  }

  /// Paired punctuation: `(hello)` → prefix=1, suffix=1 → both
  /// ends carry exactly one wildcard, matching source order.
  #[test]
  fn paired_wildcards_bracket_encoded_chars() {
    let tok = uppercase_tokenizer();
    let unk = tok.token_to_id("<unk>");

    let result = tokenize_with_default_oov(
      &tok,
      "hello",
      1,
      true,
      true,
      unk,
      /* wildcard_boundary_per_word: */
      &[crate::runner::aligner::normalizer::WildcardBoundary::new(
        1, 1,
      )],
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
      &[
        crate::runner::aligner::normalizer::WildcardBoundary::new(1, 0),
        crate::runner::aligner::normalizer::WildcardBoundary::new(2, 1),
        crate::runner::aligner::normalizer::WildcardBoundary::new(3, 0),
      ], // length 3 but word_count = 2
      &Lang::En,
      &[],
    )
    .expect_err("length mismatch must surface TokenizationFailed");
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::Tokenization(_))
    ));
  }
}
