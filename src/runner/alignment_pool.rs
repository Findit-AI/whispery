//! Alignment worker pool.
//!
//! Single worker (v1). The pool consumes `AlignWorkItem`s from a
//! bounded crossbeam channel, looks up the right `Aligner` in the
//! shared `Arc<AlignmentSet>`, runs the alignment pipeline, and
//! ships `AlignResultMsg` back to the runner via a separate result
//! channel.
//!
//! Mirrors `WhisperPool`'s shape with three differences:
//! 1. **Single worker** (no per-language parallel).
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

use crate::{
  core::AlignmentResult,
  types::{ChunkId, Lang, WorkFailure},
};

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
  /// `TimeRange`s. Pre-bound by the runner to the core's
  /// `SampleBuffer::samples_to_output_range`.
  pub samples_to_output_range: Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync>,
}

/// Worker-emitted alignment result. Crate-private.
pub(super) type AlignResultMsg = (ChunkId, Result<AlignmentResult, WorkFailure>);

use core::sync::atomic::Ordering;
use std::{sync::Mutex, thread::JoinHandle, time::Instant};

use crossbeam_channel::{Receiver, Sender, bounded};

use ort::session::RunOptions;

use crate::{
  runner::{
    RunnerError,
    aligner::{Aligner, AlignmentFallback, AlignmentLookup, AlignmentSet},
  },
  types::{AlignmentFailureKind, WorkerKind},
};

/// Single-thread alignment pool.
pub(super) struct AlignmentPool {
  workers: alloc::vec::Vec<JoinHandle<()>>,
  pub(super) work_tx: Sender<AlignWorkItem>,
  pub(super) result_rx: Receiver<AlignResultMsg>,
  pub(super) work_tx_capacity: usize,
}

impl AlignmentPool {
  /// Build the pool with a single alignment worker. v1 ships
  /// exactly one worker; multi-worker is v2.
  pub(super) fn new(set: Arc<AlignmentSet>, max_queued_chunks: usize) -> Result<Self, RunnerError> {
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
    // Replace work_tx with a dummy bounded(1) sender and drop
    // the original; idle workers' recv() then returns Err and
    // they exit cleanly. Critical to do this BEFORE joining /
    // detaching — Drop runs before field destructors, so the
    // worker would otherwise see the live `work_tx` here and
    // block on recv forever.
    let (dummy_tx, _) = bounded::<AlignWorkItem>(1);
    let original = core::mem::replace(&mut self.work_tx, dummy_tx);
    drop(original);

    // **Detach** rather than join. Even though the watchdog
    // calls `RunOptions::terminate()` on timeout — which lets
    // ORT itself exit `Session::run` cleanly — Drop fires
    // *before* any per-job watchdog timer is up. Joining here
    // would block Drop on whatever inference is currently in
    // flight. Detaching mirrors `WhisperPool::Drop` for the
    // same reason: hung Drop blocks unrelated cleanup
    // (process shutdown, test teardown). Workers finish
    // naturally on the next recv() once the in-flight job
    // completes; the OS reclaims them at process exit.
    self.workers.clear();
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
/// Strictness contract: if the registered Lang(L) aligner returns
/// `WorkFailure::AlignmentFailed`, that failure is returned as-is
/// — `Any` is *not* consulted. The worker only consults `Any` on
/// registry miss.
fn run_one_alignment(
  set: &AlignmentSet,
  job: &AlignWorkItem,
) -> Result<AlignmentResult, WorkFailure> {
  // Per-call ORT termination handle. The watchdog calls
  // `RunOptions::terminate()` on timeout, which forces
  // `Session::run_with_options` (inside `encode_log_softmax`)
  // to return an error from inside the graph rather than
  // blocking the worker until the model finishes naturally.
  // Without this, a stuck or pathologically slow inference would
  // strand the worker, and `drain` / `Drop` would wait
  // indefinitely.
  let run_options = match RunOptions::new() {
    Ok(opts) => Arc::new(opts),
    Err(e) => {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!("RunOptions::new failed: {e:?}"),
        language: job.language.clone(),
      });
    }
  };

  // Spawn the cancellable watchdog. Uses recv_timeout on a
  // one-shot channel so the worker can cancel it by dropping the
  // sender once inference completes (a sleep-based watchdog
  // would block the join until the timeout elapsed).
  let (cancel_tx, cancel_rx) = bounded::<()>(1);
  let abort_flag = job.abort_flag.clone();
  let timeout = job.align_timeout;
  let run_options_for_watchdog = run_options.clone();
  // Spawn the watchdog. Under thread / fd / memory exhaustion
  // the OS can refuse to spawn — we surface that as a fatal
  // in-band `WorkFailure` rather than panic the only alignment
  // worker. Running ORT without the watchdog risks the worker
  // getting stuck on a pathological input with no way to
  // cancel, so the right action is to fail this job fast and
  // let the caller see the resource pressure.
  let watchdog = match std::thread::Builder::new()
    .name("whispery-align-watchdog".into())
    .spawn(move || match cancel_rx.recv_timeout(timeout) {
      Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
        abort_flag.store(true, Ordering::Relaxed);
        // Tell ORT to bail out of any in-flight `Session::run`
        // for this job; the failure surfaces as
        // `Session::run_with_options` returning an error, which
        // the worker maps to `WorkerHangTimeout` below.
        let _ = run_options_for_watchdog.terminate();
      }
      Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
        // Cancelled by the worker — clean exit.
      }
    }) {
    Ok(handle) => handle,
    Err(e) => {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "failed to spawn alignment watchdog ({e}); \
           refusing to run inference without a cancellable timeout"
        ),
        language: job.language.clone(),
      });
    }
  };

  let started_at = Instant::now();

  // Lookup + dispatch. Strict on registered failure.
  let outcome = match set.lookup(&job.language) {
    AlignmentLookup::Hit { aligner, .. } => {
      // Registered aligner; failure does NOT consult Any.
      run_under_lock(aligner, job, &run_options)
    }
    AlignmentLookup::AnyFallback { aligner } => {
      // Multilingual fallback; same call shape.
      run_under_lock(aligner, job, &run_options)
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

  // An alignment-stage failure is NOT a reason to discard the
  // cached ASR transcript. Without this, a `NoAlignmentPath`
  // from a too-short chunk or a 32 M-cell budget overflow would
  // propagate to `inject_failure` upstream, turning the chunk
  // into `Event::Error` and dropping the (perfectly valid) ASR
  // text. Convert recoverable alignment-stage failures to an
  // empty `AlignmentResult` so the dispatch emits
  // `Transcript { text, words: [] }` instead — alignment is
  // best-effort, not destructive.
  //
  // `WorkerHangTimeout` and the abort-flag race above stay fatal
  // because they signal a worker liveness problem the runner
  // needs to know about. Configuration / setup failures
  // (`LanguageUnsupportedForAlignment` produced by
  // `AlignmentFallback::Error`) also stay fatal — those are
  // intentional opt-in errors from the registry policy, not
  // recoverable alignment-compute failures.
  match outcome {
    Ok(_) => outcome,
    Err(ref f) if alignment_failure_is_recoverable(f) => {
      Ok(AlignmentResult::new(alloc::vec::Vec::new()))
    }
    Err(_) => outcome,
  }
}

/// Classify an alignment worker error: best-effort
/// (recoverable, ASR text preserved) vs fatal (event surfaces as
/// `Event::Error`).
///
/// The classification is per-`AlignmentFailureKind`. Backend /
/// configuration failures must propagate so the caller learns
/// about a broken setup — silently emitting empty alignments
/// forever would mask a real problem.
///
/// Recoverable (return empty `AlignmentResult`, preserve ASR
/// text):
///
/// - `AlignmentFailed { kind: NoAlignmentPath, .. }` — viterbi
///   gave up because of a too-short chunk, lattice budget
///   overflow, or no finite path. Data-dependent.
/// - `AlignmentFailed { kind: EmptyText, .. }` — empty
///   normalisation. Already handled upstream in `Aligner::align`
///   via the `NormalizationError::EmptyText` short-circuit, so
///   this branch is defence in depth; if it ever fires we
///   still want the ASR text preserved.
///
/// Fatal (propagate as `Event::Error`):
///
/// - `AlignmentFailed { kind: ModelInferenceFailed, .. }` — ORT
///   error, non-finite samples, output shape mismatch, or
///   blank-id-out-of-vocab. These point at a broken backend or
///   model/tokenizer skew the caller needs to know about.
/// - `AlignmentFailed { kind: TokenizationFailed, .. }` —
///   tokenizer's `encode` errored, word_count mismatched the
///   normaliser, or a token id was out of model vocab. Indicates
///   a normaliser or tokenizer bug that won't go away on retry.
/// - `AlignmentFailed { kind: NormalizationFailed, .. }` —
///   `NormalizationError::RuleFailed` from the language
///   normaliser. Indicates a normaliser bug, not a per-chunk
///   miss.
/// - `WorkerHangTimeout` — liveness; worker thread or ORT graph
///   misbehaved.
/// - `LanguageUnsupportedForAlignment` — opt-in
///   `AlignmentFallback::Error` policy on registry miss.
/// - `AsrFailed` — logically impossible on the alignment path;
///   surface as a bug rather than swallow.
fn alignment_failure_is_recoverable(failure: &WorkFailure) -> bool {
  matches!(
    failure,
    WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::NoAlignmentPath | AlignmentFailureKind::EmptyText,
      ..
    }
  )
}

/// Lock the per-language `Mutex<Aligner>` and run the alignment
/// pipeline. The mutex is uncontended in the v1 single-worker
/// case but exists for v2 multi-worker safety.
fn run_under_lock(
  aligner: &Mutex<Aligner>,
  job: &AlignWorkItem,
  run_options: &RunOptions,
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
    &job.abort_flag,
    run_options,
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

  /// Only data-dependent alignment failures preserve the ASR
  /// transcript. Backend / config kinds (`ModelInferenceFailed` /
  /// `TokenizationFailed` / `NormalizationFailed`) propagate as
  /// `Event::Error` so the caller can detect a broken setup.
  #[test]
  fn data_dependent_failures_are_recoverable() {
    use crate::types::AlignmentFailureKind;
    let recoverable = [
      AlignmentFailureKind::NoAlignmentPath,
      AlignmentFailureKind::EmptyText,
    ];
    for kind in recoverable {
      let f = WorkFailure::AlignmentFailed {
        kind,
        message: alloc::string::String::new(),
        language: crate::types::Lang::En,
      };
      assert!(
        alignment_failure_is_recoverable(&f),
        "{kind:?} must preserve ASR text",
      );
    }
  }

  /// Backend / configuration alignment failures must stay
  /// fatal. Pre-fix these were being silently swallowed into
  /// `Ok(empty)`, masking broken backends.
  #[test]
  fn backend_alignment_failures_stay_fatal() {
    use crate::types::AlignmentFailureKind;
    let fatal = [
      AlignmentFailureKind::ModelInferenceFailed,
      AlignmentFailureKind::TokenizationFailed,
      AlignmentFailureKind::NormalizationFailed,
    ];
    for kind in fatal {
      let f = WorkFailure::AlignmentFailed {
        kind,
        message: alloc::string::String::new(),
        language: crate::types::Lang::En,
      };
      assert!(
        !alignment_failure_is_recoverable(&f),
        "{kind:?} signals a backend/config bug; must propagate",
      );
    }
  }

  /// Liveness / registry failures stay fatal. These signal a
  /// worker or registry problem, not a "couldn't compute
  /// alignment" outcome.
  #[test]
  fn liveness_and_registry_failures_stay_fatal() {
    use core::time::Duration;

    use crate::types::{AsrFailureKind, Lang, WorkerKind};

    assert!(!alignment_failure_is_recoverable(
      &WorkFailure::WorkerHangTimeout {
        kind: WorkerKind::Alignment,
        elapsed: Duration::from_secs(30),
      }
    ));
    assert!(!alignment_failure_is_recoverable(
      &WorkFailure::LanguageUnsupportedForAlignment { language: Lang::En }
    ));
    // Logically impossible on the alignment path, but if it
    // ever shows up we surface it rather than swallow it.
    assert!(!alignment_failure_is_recoverable(&WorkFailure::AsrFailed {
      kind: AsrFailureKind::AllTemperaturesFailed,
      message: alloc::string::String::new(),
    }));
  }
}
