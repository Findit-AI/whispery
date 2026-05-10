//! Sans-I/O types for resolving out-of-vocab characters
//! during wav2vec2 alignment tokenization.
//!
//! ## Where these flow
//!
//! Whispery's alignment dispatcher is Sans-I/O: the library
//! never owns ASR / alignment workers, never calls back into
//! caller code, and never blocks on user policy. Per-chunk
//! OOV policy is supplied alongside the alignment work item
//! itself, not via a separate command/response round-trip:
//!
//! ```text
//! AlignmentSet::detect_oov(text, lang)             -> Vec<OovEvent>
//!  └─ caller pairs each event with a decision via a helper
//!     from this module (or its own loop) producing a per-
//!     chunk Vec<ResolvedOov>
//!
//! AlignWorkItem { runs, oov_decisions: Vec<Vec<ResolvedOov>>, .. }
//!  └─ alignment pool reads `oov_decisions[run_idx]` for each
//!     run and threads it into `tokenize_with_word_map`; the
//!     dispatcher recomputes events for the chunk's text and
//!     refuses to apply a payload whose events do not match
//!     by identity (kind / char_index / word_index / language).
//!     Length, outer-shape, OR per-position identity mismatch
//!     fails loudly as
//!     `AlignmentFailureKind::TokenizationFailed` instead of
//!     silently mis-aligning a stale-but-same-length payload.
//! ```
//!
//! For whole-chunk alignment use [`AlignmentSet::detect_oov`]
//! and supply a single inner `Vec<ResolvedOov>`. For per-run
//! alignment use [`AlignmentSet::detect_oov_per_run`] and
//! supply one inner vec per run.
//!
//! [`AlignmentSet::detect_oov`]: crate::runner::aligner::AlignmentSet::detect_oov
//! [`AlignmentSet::detect_oov_per_run`]: crate::runner::aligner::AlignmentSet::detect_oov_per_run
//!
//! ## Why the caller decides
//!
//! WhisperX wildcards every OOV char (matches `*` placeholder
//! 1:1). That produces continuous alignment but plausible-but-
//! wrong word ranges on pronounced symbols (`&` in `AT&T` is
//! pronounced as the word "and"; aligning it to whichever vocab
//! item wins the frame yields confidently-wrong timing).
//! Whispery's earlier defaults baked the policy into the
//! tokenizer (`whisperx-strict-tokenizer` Cargo feature
//! flipped between fail-closed and wildcard-all) — that
//! denied the caller per-language / per-deployment / per-call
//! choice.
//!
//! Surfacing OOV as data passes the policy decision back to the
//! caller: the library detects OOV chars and returns them as
//! events; the caller applies whatever policy fits their
//! workflow (fail-closed, wildcard-all, fail on `&` but
//! wildcard digits, consult an ops dashboard, etc.). The
//! default policy lives in caller-side helper functions in
//! this module, not inside the library's hot path.

use crate::types::Lang;

/// What kind of wildcard-generating position this event
/// describes. Lets caller policy treat structural wildcards
/// (tokenizer-mechanical positions where a glyph was stripped
/// during normalisation) differently from semantic OOV
/// (chars the model dictionary doesn't have).
///
/// Codex round-37 round-3 (parity loop) [high]: introduced so
/// `fail_closed_all_decisions` truly fails on every wildcard
/// (pre-fix, boundary + internal-punct wildcards bypassed the
/// OOV policy entirely — strict callers got wildcard tokens
/// without their consent).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OovKind {
  /// Semantic OOV: the tokenizer encountered a char (digit,
  /// letter, pronounced symbol) that the wav2vec2 vocab
  /// doesn't have. Carries the offending char for per-class
  /// policy (e.g. wildcard alphanumeric, fail-closed `&`).
  Symbol(char),
  /// Boundary-punctuation wildcard: the per-language
  /// normaliser stripped a leading or trailing punct char
  /// during normalisation; the tokenizer mechanically pads
  /// the word with a wildcard at the same position to
  /// preserve CTC alignment count. The original char is
  /// already gone by the time this event is emitted.
  BoundaryPunct,
  /// Internal-punctuation wildcard: a `.` (or other
  /// `is_skippable_internal_punct` char) appears inside a
  /// word; whispery emits a wildcard at the source position
  /// so dotted acronyms like `U.S.A` align as `U * S * A`.
  /// Carries the offending char for callers that want to
  /// distinguish (e.g. allow `.` but fail other internal
  /// punct).
  InternalPunct(char),
}

/// One wildcard-generating position detected during
/// tokenization.
///
/// Returned by `AlignmentSet::detect_oov[_per_run]`. The
/// caller produces a matching [`OovDecision`] for each event
/// (in the same order) and threads it into the alignment work
/// item via `AlignWorkItem.oov_decisions`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OovEvent {
  /// What kind of wildcard-generating position this is —
  /// semantic OOV vs. structural (boundary / internal-punct).
  pub kind: OovKind,
  /// Zero-based char index in the chunk's normalised text.
  /// Boundary-punct events index the word's leading/trailing
  /// position (post-normalisation); internal-punct events
  /// index the source position of the punct char.
  pub char_index: usize,
  /// Zero-based word index (separator-counted) the position
  /// belongs to. Useful for callers that want per-word
  /// policy.
  pub word_index: usize,
  /// The language being aligned. Caller policy can switch on
  /// this (e.g. wildcard-all under `Lang::En` but fail-closed
  /// under `Lang::Ko`).
  pub language: Lang,
}

impl OovEvent {
  /// Convenience accessor: the offending char when the kind
  /// is `Symbol` or `InternalPunct`. Returns `None` for
  /// `BoundaryPunct` (the original char was stripped during
  /// normalisation and is no longer recoverable).
  #[must_use]
  pub fn char(&self) -> Option<char> {
    match self.kind {
      OovKind::Symbol(c) | OovKind::InternalPunct(c) => Some(c),
      OovKind::BoundaryPunct => None,
    }
  }

  /// Per-position identity check used by `tokenize_with_word_map`
  /// to validate a `ResolvedOov` payload against the chunk's
  /// freshly-detected events.
  ///
  /// Compares the three positional fields (`kind`,
  /// `char_index`, `word_index`) but **not** `language`.
  /// Codex round-37 round-10 (parity loop) [high]: the
  /// `language` field is a caller-policy stamp that
  /// `AlignmentSet::detect_oov` overrides under
  /// `AlignerKey::Any` fallback to the caller-requested
  /// language; the inner `Aligner` always re-detects with its
  /// own construction language. Including `language` in
  /// identity equality made every Any-fallback chunk with an
  /// OOV fail `TokenizationFailed`, even though the events
  /// describe the same text position. Positional fields are
  /// the actual identity; language is metadata for caller
  /// policy.
  #[must_use]
  pub fn matches_position(&self, other: &OovEvent) -> bool {
    self.kind == other.kind
      && self.char_index == other.char_index
      && self.word_index == other.word_index
  }
}

/// Caller's decision for one [`OovEvent`].
///
/// The caller produces one decision per event in the same
/// order. Length / shape mismatches against the chunk's
/// detected events surface as
/// [`AlignmentFailureKind::TokenizationFailed`](crate::types::AlignmentFailureKind::TokenizationFailed)
/// — the alignment dispatcher refuses to apply stale or
/// out-of-shape decisions silently.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OovDecision {
  /// Match WhisperX's `clean_char.append('*')`: emit
  /// `WILDCARD_TOKEN_ID = -1`. The CTC trellis aligns this
  /// position to whichever non-blank vocab item carries the
  /// highest log-probability at each frame. Continuous
  /// alignment at the cost of plausible-but-wrong timing on
  /// pronounced symbols.
  Wildcard,
  /// Drop the chunk's word alignment entirely; the ASR
  /// transcript still ships in the resulting
  /// [`Transcript`](crate::types::Transcript) but
  /// `Transcript::words()` is empty for this chunk. Surfaces
  /// as [`AlignmentFailureKind::SemanticOutOfVocab`](crate::types::AlignmentFailureKind::SemanticOutOfVocab)
  /// in the chunk's failure record. Honest at the cost of
  /// dropped timing.
  FailClosed,
}

/// One resolved OOV: the original event paired with the
/// caller's decision.
///
/// The dispatcher refuses to apply a `ResolvedOov` payload
/// whose embedded `event` does not match the freshly-detected
/// event at the same position via [`OovEvent::matches_position`]
/// (compares `kind`, `char_index`, `word_index` — but NOT
/// `language`, which is caller-policy metadata, not
/// positional identity). This binds the decision to the text
/// it was made for: a stale `[Wildcard]` decision produced
/// for digit OOV `[(Symbol('4'), …)]` cannot be applied to
/// `&` OOV `[(Symbol('&'), …)]` in a different chunk, even
/// if the lengths happen to match — the kind mismatch fails
/// the per-position identity check.
///
/// Codex round-37 round-9 (parity loop) [high]: prior shape
/// passed bare `Vec<OovDecision>` which carried no event
/// identity, so a stale same-length decisions vec would
/// silently bypass policy.
///
/// Codex round-37 round-10 (parity loop) [high]: identity
/// initially included `language`, which broke
/// `AlignerKey::Any` fallback (`AlignmentSet::detect_oov`
/// patches event language to the caller's requested lang,
/// but the fallback `Aligner` re-detects with its own
/// construction lang).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOov {
  /// The OOV event the decision was made for. Must match the
  /// freshly-detected event at the same position or the
  /// dispatcher rejects the payload.
  pub event: OovEvent,
  /// What to do with this position.
  pub decision: OovDecision,
}

/// Default Sans-I/O policy:
/// * Semantic OOV: alphanumeric / apostrophe → wildcard;
///   pronounced symbol → fail-closed.
/// * Boundary-punct + internal-punct (structural wildcards):
///   wildcard. They reflect tokenizer mechanics, not caller
///   text — failing-closed on `U.S.A.`'s internal `.` would
///   cripple normal English alignment.
///
/// Encodes the "WhisperX-style alphanumeric, fail-closed-on-
/// pronounced" behaviour whispery shipped before the
/// `whisperx-strict-tokenizer` Cargo feature was removed.
///
/// Pure caller-side helper. Callers wanting per-language /
/// per-deployment policy should write their own loop over
/// `events`.
#[must_use]
pub fn default_oov_decisions(events: &[OovEvent]) -> alloc::vec::Vec<ResolvedOov> {
  events
    .iter()
    .map(|ev| {
      let decision = match &ev.kind {
        OovKind::Symbol(c) => {
          if c.is_alphanumeric() || *c == '\'' || *c == '\u{2019}' {
            OovDecision::Wildcard
          } else {
            OovDecision::FailClosed
          }
        }
        // Structural wildcards: keep historical behaviour.
        OovKind::BoundaryPunct | OovKind::InternalPunct(_) => OovDecision::Wildcard,
      };
      ResolvedOov {
        event: ev.clone(),
        decision,
      }
    })
    .collect()
}

/// WhisperX-bit-equivalent: every OOV → wildcard. Replaces the
/// removed `whisperx-strict-tokenizer` Cargo feature for
/// callers that want WhisperX-1:1 outputs and accept the
/// silent-misalignment risk on pronounced symbols.
#[must_use]
pub fn wildcard_all_decisions(events: &[OovEvent]) -> alloc::vec::Vec<ResolvedOov> {
  events
    .iter()
    .map(|ev| ResolvedOov {
      event: ev.clone(),
      decision: OovDecision::Wildcard,
    })
    .collect()
}

/// Strictest: every OOV → fail-closed. Use for workflows where
/// even one wildcard alignment is too much (e.g. legal /
/// medical transcription pipelines that read PII aloud).
#[must_use]
pub fn fail_closed_all_decisions(events: &[OovEvent]) -> alloc::vec::Vec<ResolvedOov> {
  events
    .iter()
    .map(|ev| ResolvedOov {
      event: ev.clone(),
      decision: OovDecision::FailClosed,
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn ev(c: char) -> OovEvent {
    OovEvent {
      kind: OovKind::Symbol(c),
      char_index: 0,
      word_index: 0,
      language: Lang::En,
    }
  }

  fn boundary_ev() -> OovEvent {
    OovEvent {
      kind: OovKind::BoundaryPunct,
      char_index: 0,
      word_index: 0,
      language: Lang::En,
    }
  }

  fn internal_ev(c: char) -> OovEvent {
    OovEvent {
      kind: OovKind::InternalPunct(c),
      char_index: 0,
      word_index: 0,
      language: Lang::En,
    }
  }

  fn decisions_only(resolved: &[ResolvedOov]) -> alloc::vec::Vec<OovDecision> {
    resolved.iter().map(|r| r.decision).collect()
  }

  #[test]
  fn default_wildcards_alphanumeric() {
    let events = alloc::vec![ev('4'), ev('a'), ev('Z')];
    let resolved = default_oov_decisions(&events);
    assert_eq!(
      decisions_only(&resolved),
      alloc::vec![
        OovDecision::Wildcard,
        OovDecision::Wildcard,
        OovDecision::Wildcard,
      ]
    );
    // Identity binding: each ResolvedOov carries its own event.
    for (r, e) in resolved.iter().zip(events.iter()) {
      assert_eq!(&r.event, e);
    }
  }

  #[test]
  fn default_wildcards_apostrophes() {
    let events = alloc::vec![ev('\''), ev('\u{2019}')];
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      alloc::vec![OovDecision::Wildcard, OovDecision::Wildcard]
    );
  }

  #[test]
  fn default_fails_closed_on_pronounced_symbols() {
    let events = alloc::vec![ev('&'), ev('@'), ev('%'), ev(',')];
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      alloc::vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ]
    );
  }

  #[test]
  fn wildcard_all_does_what_it_says() {
    let events = alloc::vec![ev('a'), ev('&'), ev(',')];
    assert_eq!(
      decisions_only(&wildcard_all_decisions(&events)),
      alloc::vec![
        OovDecision::Wildcard,
        OovDecision::Wildcard,
        OovDecision::Wildcard,
      ]
    );
  }

  #[test]
  fn fail_closed_all_does_what_it_says() {
    let events = alloc::vec![ev('a'), ev('&'), ev(',')];
    assert_eq!(
      decisions_only(&fail_closed_all_decisions(&events)),
      alloc::vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ]
    );
  }

  /// Codex round-37 round-3 (parity loop) [high]: structural
  /// wildcards (boundary + internal-punct) get Wildcard
  /// under the default policy — matches historical behaviour.
  #[test]
  fn default_wildcards_structural_kinds() {
    let events = alloc::vec![boundary_ev(), internal_ev('.')];
    assert_eq!(
      decisions_only(&default_oov_decisions(&events)),
      alloc::vec![OovDecision::Wildcard, OovDecision::Wildcard],
    );
  }

  /// Strict policy applies to EVERY wildcard-generating
  /// position, including structural ones — that's the point
  /// of `fail_closed_all_decisions` for workflows where any
  /// wildcard alignment is unacceptable.
  #[test]
  fn fail_closed_all_includes_structural_wildcards() {
    let events = alloc::vec![ev('a'), boundary_ev(), internal_ev('.')];
    assert_eq!(
      decisions_only(&fail_closed_all_decisions(&events)),
      alloc::vec![
        OovDecision::FailClosed,
        OovDecision::FailClosed,
        OovDecision::FailClosed,
      ],
    );
  }

  #[test]
  fn empty_events_returns_empty_decisions() {
    assert!(default_oov_decisions(&[]).is_empty());
    assert!(wildcard_all_decisions(&[]).is_empty());
    assert!(fail_closed_all_decisions(&[]).is_empty());
  }
}
