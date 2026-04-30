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

use core::sync::atomic::Ordering;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::runner::RunnerError;
use crate::runner::aligner::{Aligner, AlignmentFallback, AlignmentLookup, AlignmentSet};
use crate::types::{AlignmentFailureKind, WorkerKind};

/// Single-thread alignment pool. See spec §6.3.3.
pub(super) struct AlignmentPool {
    workers: alloc::vec::Vec<JoinHandle<()>>,
    pub(super) work_tx: Sender<AlignWorkItem>,
    pub(super) result_rx: Receiver<AlignResultMsg>,
    pub(super) work_tx_capacity: usize,
}

impl AlignmentPool {
    /// Build the pool with a single alignment worker. Per spec
    /// §6.3.3, v1 ships exactly one worker; multi-worker is v2.
    pub(super) fn new(
        set: Arc<AlignmentSet>,
        max_queued_chunks: usize,
    ) -> Result<Self, RunnerError> {
        let (work_tx, work_rx) = bounded::<AlignWorkItem>(max_queued_chunks);
        let (result_tx, result_rx) = bounded::<AlignResultMsg>(max_queued_chunks + 16);

        let mut workers = alloc::vec::Vec::with_capacity(1);
        let handle = std::thread::Builder::new()
            .name("whispery-align-0".into())
            .spawn(move || {
                worker_loop(set, work_rx, result_tx);
            })
            .map_err(RunnerError::Io)?;
        workers.push(handle);

        Ok(Self {
            workers,
            work_tx,
            result_rx,
            work_tx_capacity: max_queued_chunks,
        })
    }
}

impl Drop for AlignmentPool {
    fn drop(&mut self) {
        // Plan B's drop-hang fix, applied from the start: replace
        // work_tx with a dummy bounded(1) sender and drop the
        // original. The worker's recv() then returns Err
        // immediately, exiting the loop. Joining is fast.
        let (dummy_tx, _) = bounded::<AlignWorkItem>(1);
        let original = core::mem::replace(&mut self.work_tx, dummy_tx);
        drop(original);

        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Alignment worker main loop. Single iteration per chunk; no
/// state recycling between jobs (the `Aligner` is stateless across
/// `align()` calls; ort::Session arenas are allocated lazily inside
/// the session and reused).
fn worker_loop(
    set: Arc<AlignmentSet>,
    work_rx: Receiver<AlignWorkItem>,
    result_tx: Sender<AlignResultMsg>,
) {
    while let Ok(job) = work_rx.recv() {
        let chunk_id = job.chunk_id;
        let outcome = run_one_alignment(&set, &job);
        let _ = result_tx.send((chunk_id, outcome));
    }
    // work_tx dropped: clean exit.
}

/// Drive one alignment from start to finish.
///
/// Looks up the language's aligner (or falls back to `Any` /
/// fallback policy), runs `Aligner::align` under the lock, and
/// returns the per-chunk result.
///
/// Strictness contract (spec §6.3.1): if the registered Lang(L)
/// aligner returns `WorkFailure::AlignmentFailed`, that failure is
/// returned as-is — `Any` is *not* consulted. The worker only
/// consults `Any` on registry miss.
fn run_one_alignment(
    set: &AlignmentSet,
    job: &AlignWorkItem,
) -> Result<AlignmentResult, WorkFailure> {
    // Spawn the cancellable watchdog. Uses recv_timeout on a
    // one-shot oneshot channel so the worker can cancel it by
    // dropping the sender once inference completes (Plan B's
    // watchdog sleep-blocks the join; this avoids that).
    let (cancel_tx, cancel_rx) = bounded::<()>(1);
    let abort_flag = job.abort_flag.clone();
    let timeout = job.align_timeout;
    let watchdog = std::thread::Builder::new()
        .name("whispery-align-watchdog".into())
        .spawn(move || match cancel_rx.recv_timeout(timeout) {
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                abort_flag.store(true, Ordering::Relaxed);
            }
            Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // Cancelled by the worker — clean exit.
            }
        })
        .expect("spawn watchdog");

    let started_at = Instant::now();

    // Lookup + dispatch. Strict on registered failure.
    let outcome = match set.lookup(&job.language) {
        AlignmentLookup::Hit { aligner, .. } => {
            // Registered aligner; failure does NOT consult Any.
            run_under_lock(aligner, job)
        }
        AlignmentLookup::AnyFallback { aligner } => {
            // Multilingual fallback; same call shape.
            run_under_lock(aligner, job)
        }
        AlignmentLookup::Miss { fallback } => match fallback {
            AlignmentFallback::SkipChunk => {
                // Empty result is a valid alignment outcome.
                Ok(AlignmentResult::new(alloc::vec::Vec::new()))
            }
            AlignmentFallback::Error => Err(WorkFailure::LanguageUnsupportedForAlignment {
                language: job.language.clone(),
            }),
        },
    };

    // Cancel the watchdog by dropping the sender. The watchdog's
    // recv_timeout returns Err(Disconnected) and exits.
    drop(cancel_tx);
    let _ = watchdog.join();

    // If abort_flag was flipped, surface as WorkerHangTimeout
    // regardless of what `run_under_lock` returned (the inference
    // may have completed concurrently with the timeout firing).
    if job.abort_flag.load(Ordering::Relaxed) {
        return Err(WorkFailure::WorkerHangTimeout {
            kind: WorkerKind::Alignment,
            elapsed: started_at.elapsed(),
        });
    }

    outcome
}

/// Lock the per-language `Mutex<Aligner>` and run the 8-step
/// pipeline. The mutex is uncontended in the v1 single-worker case
/// but exists for v2 multi-worker safety (spec §6.3.3).
fn run_under_lock(
    aligner: &Mutex<Aligner>,
    job: &AlignWorkItem,
) -> Result<AlignmentResult, WorkFailure> {
    let mut guard = match aligner.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            // A prior alignment panicked while holding the lock.
            // We recover the poisoned guard and proceed; the
            // session's internal state may be inconsistent but
            // the next `align` call will either succeed or
            // surface a `ModelInferenceFailed`. Do not propagate
            // panic across thread boundary.
            poisoned.into_inner()
        }
    };

    let bound = job.samples_to_output_range.clone();
    guard.align(
        &job.samples,
        &job.sub_segments,
        job.text.as_str(),
        job.chunk_first_sample_in_stream,
        move |a, b| (bound)(a, b),
    )
}

// Re-exports of the algorithm error kinds so the worker can
// surface them without re-importing the chain.
#[allow(dead_code)]
pub(super) const ALIGNMENT_FAILURE_KIND_REFERENCE: AlignmentFailureKind =
    AlignmentFailureKind::EmptyText;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn align_work_item_is_send() {
        assert_send::<AlignWorkItem>();
    }

    #[test]
    fn align_result_msg_is_send() {
        assert_send::<AlignResultMsg>();
    }

    #[test]
    fn alignment_pool_channel_halves_are_send() {
        assert_send::<crossbeam_channel::Sender<AlignWorkItem>>();
        assert_send::<crossbeam_channel::Receiver<AlignResultMsg>>();
    }
}
