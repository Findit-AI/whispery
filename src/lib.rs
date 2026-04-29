//! whispery — Sans-I/O cut/batch/whisper/align state machine for
//! speech-to-text indexing pipelines.
//!
//! See `docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md`
//! for the full design. The crate is organised as a small public
//! type surface (this file's re-exports), a `core` module with the
//! Sans-I/O state machine (no ML deps), and — gated behind the
//! `runner` and `alignment` features in later milestones — a runner
//! module wrapping whisper-rs and an `ort`-based forced aligner.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod time;
pub mod types;
pub mod core;
