//! whispery — Sans-I/O cut/batch/whisper/align state machine for
//! speech-to-text indexing pipelines.
//!
//! See `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md`
//! for the full design.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
// Crate default: no unsafe. Modules that need `core::arch` SIMD
// intrinsics opt in locally via `#![allow(unsafe_code)]`. This is
// `deny` rather than `forbid` so an explicit per-module override is
// possible — the only such opt-in today is the aarch64 NEON kernel
// in `runner::aligner::algorithm::normalize::neon`.
#![deny(unsafe_code)]

extern crate alloc;

pub mod core;
pub mod time;
pub mod types;

// Re-exports of mediatime types that appear in whispery's public API
// (so consumers don't need to add a separate `mediatime` dependency
// just to name them; they may still do so to call methods like
// `rescale_to`).
//
// SemVer note: re-exporting mediatime types ties whispery's public
// API to mediatime's. A breaking change in mediatime (major-version
// bump) is automatically a breaking change for whispery, so the
// `mediatime` dependency is pinned to a single major in Cargo.toml.
pub use mediatime::{TimeRange, Timebase, Timestamp};

pub use types::{
  AlignmentFailureKind, AsrFailureKind, ChunkId, Lang, PushKind, TranscriberError, Transcript,
  VadSegment, Word, WorkFailure, WorkerKind,
};

pub use core::{
  AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, Event, LanguagePolicy,
  SamplingStrategy, Transcriber, TranscriberOptions,
};

#[cfg(feature = "runner")]
pub mod runner;

#[cfg(feature = "runner")]
pub use runner::{ManagedTranscriber, ManagedTranscriberBuilder, RunnerError, WhisperPoolOptions};

// Re-export whisper-rs types that appear on the runner's public
// API (so consumers don't need a direct whisper-rs dep just to name
// them; they may still add it to call non-re-exported methods).
//
// SemVer note: identical to the mediatime situation — re-exporting
// pins whispery's public API to whisper-rs's semver. We pin to a
// single major in Cargo.toml.
#[cfg(feature = "runner")]
pub use whisper_rs::{WhisperContext, WhisperContextParameters};

#[cfg(feature = "alignment")]
pub use runner::{
  Aligner, AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet, AlignmentSetBuilder,
  ChineseNormalizer, DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE, DynTextNormalizer,
  EnglishNormalizer, JapaneseNormalizer, NormalizationError, NormalizedText, TextNormalizer,
  bundled,
};

// Re-export ort types that appear on the alignment public API.
//
// SemVer note: re-exporting pins whispery's public API to ort's
// semver. Cargo.toml pins ort to =2.0.0-rc.12; bumping it requires
// a matching whispery-major bump.
#[cfg(feature = "alignment")]
pub use ort;

/// **Internal use only — gated on `feature = "bench-internals"`.**
///
/// Re-exports the alignment pipeline's `pub(crate)` SIMD/scalar
/// kernels and the `LogProbsTV` lattice-input struct so the
/// `aligner_simd_baseline` Criterion bench (an external binary)
/// can compare backends head-to-head. Not part of any
/// stability contract — symbols here can move or disappear
/// between commits without notice.
#[cfg(all(feature = "bench-internals", feature = "alignment"))]
#[doc(hidden)]
pub mod __bench {
  pub use crate::runner::aligner::algorithm::{
    encode::LogProbsTV,
    normalize::{scalar, zero_mean_unit_var_normalize},
    tokenize::{TokenizedText, tokenize_with_word_map},
    trellis_beam::{
      ALIGN_BEAM_WIDTH, PathPointPublic, WILDCARD_TOKEN_ID, WordSegment, align_to_word_segments,
      backtrack_beam, get_trellis,
    },
  };

  #[cfg(target_arch = "aarch64")]
  pub use crate::runner::aligner::algorithm::normalize::neon;

  #[cfg(target_arch = "x86_64")]
  pub use crate::runner::aligner::algorithm::normalize::{x86_avx2, x86_avx512, x86_sse41};
}
