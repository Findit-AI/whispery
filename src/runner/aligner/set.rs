//! `AlignmentSet` — registry of `Aligner`s keyed by `AlignerKey`.

use std::{collections::HashMap, sync::Mutex};

use crate::{
  runner::aligner::{
    aligner::Aligner,
    key::{AlignerKey, AlignmentFallback},
  },
  types::Lang,
};

/// The result of a registry lookup. Surfaces both the matched
/// aligner key (for diagnostics) and a borrow of the `Mutex<Aligner>`
/// the worker will lock; or, on miss, the configured fallback
/// policy.
///
/// Returned by [`AlignmentSet::lookup`].
pub enum AlignmentLookup<'a> {
  /// Hit on `AlignerKey::Lang(L)`. The worker locks the mutex
  /// and runs the language-specific aligner. Failure of this
  /// path does NOT silently fall through to `Any`
  /// (strict-lookup contract).
  Hit {
    /// The matched key (always `Lang(...)`).
    matched: AlignerKey,
    /// The mutex-wrapped aligner; lock to call `align()`.
    aligner: &'a Mutex<Aligner>,
  },
  /// Miss on `Lang(L)`, hit on `Any`. The multilingual fallback
  /// is consulted.
  AnyFallback {
    /// The mutex-wrapped multilingual aligner.
    aligner: &'a Mutex<Aligner>,
  },
  /// Miss on both `Lang(L)` and `Any`. The configured fallback
  /// policy decides what the worker emits (`SkipChunk` => empty
  /// `words`; `Error` => `LanguageUnsupportedForAlignment`).
  Miss {
    /// The fallback policy.
    fallback: AlignmentFallback,
  },
}

/// Registry of `Aligner`s. Owned by `ManagedTranscriber`; shared
/// with the alignment worker via `Arc<AlignmentSet>`.
///
/// Fields are private; construct via [`AlignmentSetBuilder`].
/// Lookup is `&self` so the worker can hold a long-lived borrow
/// without blocking other workers (the `Mutex<Aligner>` inside
/// is the per-language lock).
pub struct AlignmentSet {
  aligners: HashMap<AlignerKey, Mutex<Aligner>>,
  fallback: AlignmentFallback,
}

impl AlignmentSet {
  /// Crate-private constructor. Public callers go through
  /// `AlignmentSetBuilder` so the construction surface stays
  /// consistent with the `with_*` builder pattern used elsewhere
  /// in the crate.
  pub(super) const fn from_parts(
    aligners: HashMap<AlignerKey, Mutex<Aligner>>,
    fallback: AlignmentFallback,
  ) -> Self {
    Self { aligners, fallback }
  }

  /// Configured registry-miss policy.
  pub const fn fallback(&self) -> AlignmentFallback {
    self.fallback
  }

  /// Number of registered aligners (excluding `Any` if not registered).
  pub fn len(&self) -> usize {
    self.aligners.len()
  }

  /// Whether the registry has zero aligners. A pool with an
  /// `is_empty()` set is equivalent to `with_alignment(set)` not
  /// being called at all — the runner skips emitting
  /// `Command::Alignment` for every chunk.
  pub fn is_empty(&self) -> bool {
    self.aligners.is_empty()
  }

  /// Detect out-of-vocab characters in `text` against the
  /// aligner registered for `language` (or
  /// [`AlignerKey::Any`]'s aligner if no language-specific one
  /// is registered). Returns events in the order
  /// [`tokenize_with_word_map`](crate::runner::aligner::algorithm::tokenize::tokenize_with_word_map)
  /// would encounter them — caller-supplied `&[ResolvedOov]`
  /// on the resulting [`AlignWorkItem::oov_decisions`](crate::AlignWorkItem)
  /// must be in the same order.
  ///
  /// Returns `Ok(empty)` when no aligner matches (registry
  /// miss) — the alignment dispatch will then either skip the
  /// chunk (under [`AlignmentFallback::SkipChunk`]) or surface
  /// the language-unsupported error itself, so an empty
  /// decisions vec on the work item is the right shape.
  ///
  /// Sans-I/O OOV resolution entry point: callers feed the
  /// returned events into a policy helper
  /// (`crate::core::oov::default_oov_decisions` etc.) and
  /// pass the resulting decisions to
  /// `AlignWorkItem::from_run_alignment`.
  pub fn detect_oov(
    &self,
    text: &str,
    language: &Lang,
  ) -> Result<Vec<crate::core::OovEvent>, crate::types::WorkFailure> {
    let aligner_mu = match self.lookup(language) {
      AlignmentLookup::Hit { aligner, .. } | AlignmentLookup::AnyFallback { aligner } => aligner,
      AlignmentLookup::Miss { .. } => return Ok(Vec::new()),
    };
    let guard = aligner_mu.lock().unwrap_or_else(|p| p.into_inner());
    let mut events = guard.detect_oov(text)?;
    // // `Aligner::detect_oov` stamps every event with its OWN
    // construction language. When `lookup` falls back to
    // `AlignerKey::Any` (e.g. an English aligner registered
    // as the multilingual fallback for an unsupported
    // language), the caller's requested language is
    // overwritten with the fallback aligner's language.
    // Per-language policy (e.g. wildcard-en /
    // fail-closed-ko) then sees the wrong key. Patch the
    // event language back to the caller's request so the
    // policy decides on the run/chunk language, not the
    // aligner's construction-time tag.
    for ev in &mut events {
      ev.set_language(language.clone());
    }
    Ok(events)
  }

  /// Detect OOV chars per-run for a code-switched chunk's
  /// script-dispatched runs. Returns `events_per_run[i]`
  /// matching `runs[i]` order; empty when `runs` is empty.
  /// Each run's events are detected against its own
  /// language's aligner (`runs[i].language()`).
  ///
  /// Companion to [`Self::detect_oov`]. Use this when
  /// `Command::Alignment::runs` is non-empty (the typical
  /// `WhisperAsrSource` path); use [`Self::detect_oov`] for
  /// the whole-chunk path.
  ///
  /// Returns `Err` immediately on the first per-run detection
  /// failure (tokenizer-engine / normalisation error), so the
  /// caller can surface the failure to the chunk before
  /// alignment.
  ///
  /// introduced
  /// to thread caller policy through the per-run path —
  /// the dispatcher silently substituted
  /// `default_oov_decisions` regardless of caller intent.
  pub fn detect_oov_per_run(
    &self,
    runs: &[crate::align::Run],
  ) -> Result<Vec<Vec<crate::core::OovEvent>>, crate::types::WorkFailure> {
    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
      out.push(self.detect_oov(run.text(), run.language())?);
    }
    Ok(out)
  }

  /// Look up an aligner for `language`, applying the
  /// strict-lookup order.
  pub fn lookup<'a>(&'a self, language: &Lang) -> AlignmentLookup<'a> {
    let lang_key = AlignerKey::Lang(language.clone());
    if let Some(m) = self.aligners.get(&lang_key) {
      return AlignmentLookup::Hit {
        matched: lang_key,
        aligner: m,
      };
    }
    if let Some(m) = self.aligners.get(&AlignerKey::Any) {
      return AlignmentLookup::AnyFallback { aligner: m };
    }
    AlignmentLookup::Miss {
      fallback: self.fallback,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::runner::aligner::{normalizer::DynTextNormalizer, normalizers::EnglishNormalizer};

  // Direct AlignmentSet construction without a real Aligner is
  // not possible (Aligner has private fields and a from_paths
  // constructor that requires real ONNX). We assert the
  // miss-only path here, which doesn't need a populated
  // registry.

  #[test]
  fn empty_set_misses_with_default_fallback() {
    let set = AlignmentSet::from_parts(HashMap::new(), AlignmentFallback::SkipChunk);
    match set.lookup(&Lang::En) {
      AlignmentLookup::Miss { fallback } => {
        assert_eq!(fallback, AlignmentFallback::SkipChunk);
      }
      _ => panic!("expected Miss"),
    }
  }

  #[test]
  fn empty_set_misses_with_error_fallback() {
    let set = AlignmentSet::from_parts(HashMap::new(), AlignmentFallback::Error);
    match set.lookup(&Lang::Zh) {
      AlignmentLookup::Miss { fallback } => {
        assert_eq!(fallback, AlignmentFallback::Error);
      }
      _ => panic!("expected Miss"),
    }
  }

  #[test]
  fn is_empty_reports_correctly() {
    let set = AlignmentSet::from_parts(HashMap::new(), AlignmentFallback::SkipChunk);
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
  }

  // Suppress dead-code warning in the test module: pull in the
  // EN normaliser even though we don't construct an Aligner.
  #[test]
  fn normalizer_imports_compile() {
    let _: DynTextNormalizer = Box::new(EnglishNormalizer::new());
  }
}
