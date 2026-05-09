//! Alignment-adjacent helpers that don't require a wav2vec2 model.
//!
//! Lives outside `runner` so non-runner callers can use the script
//! → language mapping and the [`script_dispatch::Run`] type without
//! pulling in the runner feature flag (and the `whispercpp`
//! dependency it gates). The runner-feature
//! [`script_dispatch::dispatch`] entry point — the one that
//! consumes real `whispercpp::Segment<'_>` — is gated on
//! `feature = "runner"`; everything else is always-on.

pub mod script;
pub mod script_dispatch;

pub use script::{CharClass, SegmentContext, is_latin_script_lang, script_to_lang};
pub use script_dispatch::{BoundsSource, Run, SegmentLike, dispatch_segments};

#[cfg(feature = "runner")]
pub use script_dispatch::dispatch;
