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

pub mod time;
pub mod types;
pub mod core;

// Re-exports of mediatime types that appear in whispery's public API
// (so consumers don't need to add a separate `mediatime` dependency
// just to name them; they may still do so to call methods like
// `rescale_to`).
//
// SemVer note: re-exporting mediatime types ties whispery's public
// API to mediatime's. A breaking change in mediatime (major-version
// bump) is automatically a breaking change for whispery, so the
// `mediatime` dependency is pinned to a single major in Cargo.toml.
pub use mediatime::{Timebase, TimeRange, Timestamp};

pub use types::{
    AlignmentFailureKind, AsrFailureKind, ChunkId, Lang, PushKind, Transcript,
    TranscriberError, VadSegment, Word, WorkFailure, WorkerKind,
};

pub use core::{
    AlignmentResult, AsrParams, AsrParamsOverride, AsrResult, Command, Event,
    LanguagePolicy, SamplingStrategy, Transcriber, TranscriberConfig,
};
