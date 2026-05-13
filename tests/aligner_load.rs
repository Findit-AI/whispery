//! Non-ignored regression for codex finding [high]: the upstream
//! `wav2vec2-base-960h/tokenizer.json` ships in an older HF format
//! that `tokenizers 0.20` rejects with `ModelUntagged`. `build.rs`
//! patches the fetched JSON to inject the `model.type` discriminator
//! so `Aligner::from_paths` can actually load it.
//!
//! This test is the direct counterpart to that build.rs path: when
//! the fixture is present (the `alignment` feature is on AND
//! `ASRY_OFFLINE` is unset), it asserts that `Aligner::from_paths`
//! returns `Ok` against the fetched + patched files. Without this
//! test the patch could silently regress and only `alignment_e2e`
//! (currently `#[ignore]`'d for unrelated drain-hang reasons) would
//! catch it.

#![cfg(feature = "alignment")]

use std::path::Path;

use asry::{Aligner, EnglishNormalizer, Lang};

const W2V_MODEL: Option<&str> = option_env!("ASRY_W2V_MODEL");
const W2V_TOKENIZER: Option<&str> = option_env!("ASRY_W2V_TOKENIZER");

#[test]
fn from_paths_loads_bundled_wav2vec2_fixtures() {
  let (Some(model_path), Some(tokenizer_path)) = (W2V_MODEL, W2V_TOKENIZER) else {
    // Build environment didn't fetch fixtures (offline / feature
    // off). Skip — alignment_e2e covers the full pipeline when
    // they are present.
    return;
  };

  let aligner = Aligner::from_paths(
    Lang::En,
    Path::new(model_path),
    Path::new(tokenizer_path),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("Aligner::from_paths must succeed against the patched bundled fixture");

  // Sanity: the build.rs patch injects the discriminator with
  // `unk_token = "<unk>"`, and detect_blank_token_id reads the
  // `<pad>` entry. Confirm both detections fired.
  assert_eq!(*aligner.language(), Lang::En);
  assert_eq!(aligner.sample_rate(), 16_000);
  assert_eq!(aligner.hop_samples(), 320);
  // wav2vec2-base-960h's `<pad>` lives at vocab id 0.
  assert_eq!(aligner.blank_token_id(), 0);
}
