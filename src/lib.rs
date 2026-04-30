//! whispery — Sans-I/O cut/batch/whisper/align state machine for
//! speech-to-text indexing pipelines.
//!
//! See `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md`
//! for the full design.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

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
  SamplingStrategy, Transcriber, TranscriberConfig,
};

#[cfg(feature = "runner")]
pub mod runner;

#[cfg(feature = "runner")]
pub use runner::{
    ManagedTranscriber, ManagedTranscriberBuilder, RunnerError, WhisperPoolConfig,
};

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
    ChineseNormalizer, DynTextNormalizer, EnglishNormalizer, JapaneseNormalizer,
    NormalizationError, NormalizedText, TextNormalizer,
};

// Re-export ort types that appear on the alignment public API.
//
// SemVer note: re-exporting pins whispery's public API to ort's
// semver. Cargo.toml pins ort to =2.0.0-rc.12; bumping it requires
// a matching whispery-major bump.
#[cfg(feature = "alignment")]
pub use ort;
