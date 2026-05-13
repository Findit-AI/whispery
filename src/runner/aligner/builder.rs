//! `AlignmentSetBuilder` — public construction for [`AlignmentSet`].

use std::{collections::HashMap, sync::Mutex};

use crate::runner::aligner::{
  aligner::Aligner,
  key::{AlignerKey, AlignmentFallback},
  set::AlignmentSet,
};

/// Builder for [`AlignmentSet`]. Mirrors the `with_*` builder
/// style used elsewhere in the crate.
///
/// Usage:
///
/// ```no_run
/// # #[cfg(feature = "alignment")]
/// # {
/// use std::path::Path;
/// use asry::{AlignmentSet, AlignmentSetBuilder, AlignerKey, Aligner};
/// use asry::{AlignmentFallback, EnglishNormalizer, Lang};
///
/// let aligner = Aligner::from_paths(
/// Lang::En,
/// Path::new("path/to/wav2vec2.onnx"),
/// Path::new("path/to/tokenizer.json"),
/// Box::new(EnglishNormalizer::new()),
/// )
/// .expect("aligner load");
///
/// let set = AlignmentSetBuilder::new()
/// .with_fallback(AlignmentFallback::SkipChunk)
/// .register(AlignerKey::Lang(Lang::En), aligner)
/// .build();
/// # }
/// ```
pub struct AlignmentSetBuilder {
  aligners: HashMap<AlignerKey, Mutex<Aligner>>,
  fallback: AlignmentFallback,
}

impl AlignmentSetBuilder {
  /// Construct an empty builder. Fallback defaults to
  /// [`AlignmentFallback::SkipChunk`].
  pub fn new() -> Self {
    Self {
      aligners: HashMap::new(),
      fallback: AlignmentFallback::SkipChunk,
    }
  }

  /// Override the registry-miss policy.
  pub const fn with_fallback(mut self, value: AlignmentFallback) -> Self {
    self.fallback = value;
    self
  }

  /// Set the fallback policy in place (mutator-style).
  pub const fn set_fallback(&mut self, value: AlignmentFallback) {
    self.fallback = value;
  }

  /// Register an aligner under `key`. Replaces any prior
  /// registration for the same key (last call wins).
  ///
  /// Wrapped in a `Mutex<Aligner>`.
  ///
  /// panics if `key` is
  /// [`AlignerKey::Lang`]`(L)` and `aligner.language() != L`.
  /// The registry accepted any aligner under any key,
  /// so a swapped registration (e.g. `AlignerKey::Lang(Lang::Zh)`
  /// pointing at an English aligner) silently routed CJK
  /// script-dispatch runs through the English normalizer /
  /// tokenizer / model, producing plausible-but-corrupt word
  /// timings tagged with the run's language. Fail fast at
  /// registration so the misconfiguration surfaces at build
  /// time. [`AlignerKey::Any`] remains the explicit escape
  /// hatch — it accepts any aligner regardless of language
  /// because it is the documented multilingual fallback.
  pub fn register(mut self, key: AlignerKey, aligner: Aligner) -> Self {
    if let AlignerKey::Lang(ref key_lang) = key {
      assert_eq!(
        aligner.language(),
        key_lang,
        "AlignerKey::Lang({key_lang:?}) registration cannot accept an aligner built \
 for {actual:?}; either register under AlignerKey::Lang({actual:?}), \
 AlignerKey::Any (explicit multilingual fallback), or rebuild the \
 aligner for the desired language",
        actual = aligner.language(),
      );
    }
    self.aligners.insert(key, Mutex::new(aligner));
    self
  }

  /// Number of currently-registered aligners (excludes `Any` if
  /// not registered).
  pub fn len(&self) -> usize {
    self.aligners.len()
  }

  /// Whether the builder has zero registered aligners.
  pub fn is_empty(&self) -> bool {
    self.aligners.is_empty()
  }

  /// Finalise the builder into an [`AlignmentSet`].
  pub fn build(self) -> AlignmentSet {
    AlignmentSet::from_parts(self.aligners, self.fallback)
  }
}

impl Default for AlignmentSetBuilder {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Lang;

  #[test]
  fn empty_builder_default_fallback() {
    let b = AlignmentSetBuilder::new();
    assert!(b.is_empty());
    assert_eq!(b.len(), 0);
  }

  #[test]
  fn with_fallback_round_trip() {
    let b = AlignmentSetBuilder::new().with_fallback(AlignmentFallback::Error);
    let set = b.build();
    assert_eq!(set.fallback(), AlignmentFallback::Error);
  }

  #[test]
  fn build_empty_produces_empty_set() {
    let set = AlignmentSetBuilder::new().build();
    assert!(set.is_empty());
    match set.lookup(&Lang::En) {
      crate::runner::aligner::set::AlignmentLookup::Miss { fallback } => {
        assert_eq!(fallback, AlignmentFallback::SkipChunk);
      }
      _ => panic!("expected Miss"),
    }
  }

  // The register-with-real-Aligner test path requires a real
  // ONNX file; covered by an end-to-end test.
}
