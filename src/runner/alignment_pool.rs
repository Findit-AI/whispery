//! Alignment worker pool. See spec §6.3.3.
//!
//! Single worker per spec §6.3.3 (v1). The pool consumes
//! `AlignWorkItem`s from a bounded crossbeam channel, looks up
//! the right `Aligner` in the shared `Arc<AlignmentSet>`, runs the
//! 8-step pipeline, and ships `AlignResultMsg` back to the runner
//! via a separate result channel.
//!
//! Mirrors Plan B's `WhisperPool` shape with three differences:
//! 1. **Single worker** by spec §6.3.3 (no per-language parallel).
//! 2. **Drop-hang fix from the start** — `mem::replace`s `work_tx`
//!    with a dummy disconnected channel before joining workers, so
//!    the worker's blocking `recv()` returns immediately.
//! 3. **Cancellable watchdog** — the per-job watchdog uses
//!    `recv_timeout` on a one-shot channel rather than
//!    `thread::sleep`, so the worker can cancel it instantly when
//!    inference completes.

use alloc::sync::Arc;
use std::sync::atomic::AtomicBool;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::core::AlignmentResult;
use crate::types::{ChunkId, Lang, WorkFailure};

/// One unit of alignment work shipped to the alignment worker.
/// Crate-private.
pub(super) struct AlignWorkItem {
    /// Identity of the chunk this alignment fulfils.
    pub chunk_id: ChunkId,
    /// Chunk audio (16 kHz f32 mono); shared via `Arc` with the
    /// core.
    pub samples: Arc<[f32]>,
    /// Sub-VAD-segments inside the chunk, in chunk-local 16 kHz
    /// sample-index space (encoded as TimeRanges with timebase
    /// 1/16000 so `start_pts() == start_sample`). The runner
    /// converts from output-timebase before enqueueing.
    pub sub_segments: alloc::vec::Vec<TimeRange>,
    /// Whisper's transcribed text for this chunk.
    pub text: SmolStr,
    /// Detected language for this chunk.
    pub language: Lang,
    /// Per-job timeout. The worker's watchdog flips abort_flag
    /// after this elapses.
    pub align_timeout: core::time::Duration,
    /// Watchdog flag. The worker checks this between pipeline
    /// stages; if true, it returns
    /// [`WorkFailure::WorkerHangTimeout`] without continuing.
    pub abort_flag: Arc<AtomicBool>,
    /// Chunk's first 16 kHz sample index in stream coordinates.
    /// Used by the aligner to map wav2vec2 frame indices back
    /// into stream sample space; the runner converts further into
    /// output-timebase via the `samples_to_output_range` closure.
    pub chunk_first_sample_in_stream: u64,
    /// Bridge from stream sample indices to output-timebase
    /// `TimeRange`s. Pre-bound by the runner to Plan A's
    /// `SampleBuffer::samples_to_output_range`.
    pub samples_to_output_range: Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync>,
}

/// Worker-emitted alignment result. Crate-private.
pub(super) type AlignResultMsg = (ChunkId, Result<AlignmentResult, WorkFailure>);
