//! Runner — sync compute primitives + (transitionally) the
//! built-in thread-pool orchestration.
//!
//! ## Migration in progress
//!
//! Whispery is moving to a Sans-I/O posture (per its
//! `Cargo.toml` charter, matching whisperX's function-style
//! API): threading and lifecycle become the caller's
//! responsibility. The new public primitives are:
//!
//! - [`AsrSource`] / [`WhisperAsrSource`] — sync ASR
//!   compute, no internal threads, caller-driven cancellation
//!   via a shared `Arc<AtomicBool>`.
//! - [`crate::core::Transcriber`] — the existing Sans-I/O
//!   state machine; pull commands via `poll_command()`,
//!   dispatch them inline, push results back via
//!   `handle_asr` / `handle_alignment` /
//!   `handle_failure`.
//!
//! Sync users (CLI tools, batch indexers) drive the pump on
//! one thread. The full ASR + alignment loop, using
//! [`run_one_alignment`] so per-language script-dispatch
//! `runs` are honoured:
//!
//! ```ignore
//! use core::num::NonZeroU32;
//! use std::sync::{Arc, atomic::AtomicBool};
//! use mediatime::{TimeRange, Timebase};
//! use whispery::{AlignWorkItem, run_one_alignment};
//! use whispery::core::Command;
//! use whispery::ort::session::RunOptions;
//!
//! let abort_flag = Arc::new(AtomicBool::new(false));
//! // Allocate a FRESH `RunOptions` per alignment chunk
//! // ([high]). ORT termination is
//! // sticky — reusing a single handle means the first
//! // `terminate()` poisons every subsequent
//! // `Session::run`. Per-chunk allocation keeps each
//! // watchdog deadline independent: hand the new handle to
//! // the watchdog at chunk start, drop both at chunk end.
//! // Cancellation between chunks lives on `abort_flag`.
//!
//! while let Some(cmd) = transcriber.poll_command() {
//!   match cmd {
//!     Command::Asr { chunk_id, samples, params, .. } => {
//!       let result = asr_source.run_chunk(AsrChunkContext::new(
//!         &samples,
//!         &params,
//!         &abort_flag,
//!         chunk_id,
//!       ))?;
//!       transcriber.handle_asr(chunk_id, result)?;
//!     }
//!     Command::Alignment { chunk_id, samples, sub_segments: _,
//!                              text, language, runs } => {
//!       // Sans-I/O OOV resolution: per-run detect + decide.
//!       // Each run gets its own decisions vec sized + ordered
//!       // by the events `detect_oov` produces for that run's
//!       // text + language. Whole-chunk fallback (when `runs`
//!       // is empty) gets one inner vec.
//!       let oov_decisions: Vec<Vec<whispery::core::ResolvedOov>> =
//!         if runs.is_empty() {
//!           let events = alignment_set.detect_oov(&text, &language)?;
//!           vec![whispery::core::default_oov_decisions(&events)]
//!         } else {
//!           alignment_set.detect_oov_per_run(&runs)?
//!             .iter()
//!             .map(|ev| whispery::core::default_oov_decisions(ev))
//!             .collect()
//!         };
//!       // `AlignWorkItem::from_run_alignment` flips the
//!       // command's output-timebase `sub_segments` into
//!       // chunk-local 1/16000 (the form `Aligner::align`
//!       // requires) and pulls the chunk anchor + bridge from
//!       // `Transcriber`. Returns `None` only if the chunk
//!       // already drained — recoverable.
//!       let job = AlignWorkItem::from_run_alignment(
//!         &transcriber, chunk_id, samples, text, language,
//!         runs, abort_flag.clone(), oov_decisions,
//!       ).expect("chunk in flight");
//!       // Fresh `RunOptions` per chunk so a watchdog's
//!       // `terminate()` for chunk N does not poison chunk N+1.
//!       let run_options = RunOptions::new().unwrap();
//!       let aligned = run_one_alignment(&alignment_set, &job, &run_options)?;
//!       transcriber.handle_alignment(chunk_id, aligned)?;
//!     }
//!   }
//! }
//! while let Some(event) = transcriber.poll_event() { /* ... */ }
//! ```
//!
//! `run_one_alignment` honours `job.runs` for per-language
//! dispatch (Ja+Zh in one chunk, En+Ko, etc.), falling back to
//! whole-chunk alignment keyed on `job.language` when `runs`
//! is empty. Without it, code-switched chunks regress to
//! single-language alignment.
//!
//! Async users (tokio, smol) call `WhisperAsrSource::run_chunk`
//! from `spawn_blocking` and wire shutdown via their own
//! cancellation tokens flipping the supplied `abort_flag`.

mod asr_source;
mod errors;
pub(crate) mod whisper_pool;

// `pub(crate)` (rather than `mod`) so the crate-root
// `#[cfg(feature = "bench-internals")] pub mod __bench` can
// re-export this module's `pub(crate)` SIMD/scalar kernels.
// Outside the bench gate the module's items are only visible
// through the curated `pub use` re-exports below.
#[cfg(feature = "alignment")]
pub(crate) mod aligner;
#[cfg(feature = "alignment")]
mod alignment_pool;

#[cfg(feature = "runner")]
pub use asr_source::WhisperAsrSource;
pub use asr_source::{AsrChunkContext, AsrSource};
pub use errors::RunnerError;

#[cfg(feature = "alignment")]
pub use aligner::{
  Aligner, AlignerKey, AlignmentFallback, AlignmentLookup, AlignmentSet, AlignmentSetBuilder,
  ChineseNormalizer, DEFAULT_MAX_INTRA_SILENT_RUN, DEFAULT_MIN_SPEECH_COVERAGE, DynTextNormalizer,
  EnglishNormalizer, JapaneseNormalizer, KoreanNormalizer, LatinNormalizer, NormalizationError,
  NormalizedText, TextNormalizer, bundled, default_normalizer_for,
};

#[cfg(feature = "alignment")]
pub use alignment_pool::{AlignWorkItem, run_one_alignment};
