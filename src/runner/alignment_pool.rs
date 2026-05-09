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
  /// Script-dispatcher per-language runs over the transcript,
  /// computed by the whisper worker just after `state.full(...)`.
  /// Empty when the dispatcher was not run (no segments, or a
  /// caller injecting `AsrResult` directly without populating
  /// `AsrResult::runs`); the worker then falls back to a single
  /// whole-chunk alignment keyed on [`Self::language`].
  pub runs: alloc::vec::Vec<crate::align::Run>,
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

  // Lookup + dispatch. Two paths:
  //
  // 1. **Per-run dispatch** (`!job.runs.is_empty()`): each
  //    [`crate::align::Run`] in `job.runs` is a single-language
  //    slice of the chunk's transcript carrying its own audio
  //    bounds. We walk the list, look up the per-language
  //    `Aligner`, and call `align_chunk` once per run; results are
  //    stitched into a single `AlignmentResult`.
  //
  // 2. **Legacy whole-chunk dispatch** (`job.runs.is_empty()`):
  //    the runner did not populate `runs` (no segments emitted, or
  //    a caller injecting `AsrResult` directly without filling
  //    `runs`). We fall back to the pre-script-dispatch behaviour:
  //    one alignment over the full chunk text, keyed on
  //    `job.language`.
  //
  // Strict-lookup contract: registered `Lang(L)` failure does NOT
  // silently fall through to `Any`. The per-run path applies the
  // same rule per-run.
  let outcome = if job.runs.is_empty() {
    match set.lookup(&job.language) {
      AlignmentLookup::Hit { aligner, .. } => run_under_lock(aligner, job, &run_options),
      AlignmentLookup::AnyFallback { aligner } => run_under_lock(aligner, job, &run_options),
      AlignmentLookup::Miss { fallback } => match fallback {
        AlignmentFallback::SkipChunk => Ok(AlignmentResult::new(alloc::vec::Vec::new())),
        AlignmentFallback::Error => Err(WorkFailure::LanguageUnsupportedForAlignment {
          language: job.language.clone(),
        }),
      },
    }
  } else {
    dispatch_runs(set, job, &run_options)
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

/// Per-chunk script-dispatch telemetry. Counts how the
/// dispatcher's [`crate::align::BoundsSource`] decisions
/// distributed across the chunk's runs, plus how many runs
/// landed on a [`Lang`] with no registered aligner.
///
/// The counters are accumulated once per chunk by
/// [`dispatch_runs`] and emitted to stderr with a
/// `script_dispatch chunk=...` prefix. Fields are private with
/// accessors per the project convention.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BoundsSourceCounters {
  runs_total: usize,
  runs_dtw: usize,
  runs_segment: usize,
  runs_wholeclip: usize,
  runs_unaligned: usize,
}

impl BoundsSourceCounters {
  /// Tally one run's [`crate::align::BoundsSource`].
  pub(super) fn observe_bounds(&mut self, source: crate::align::BoundsSource) {
    self.runs_total += 1;
    match source {
      crate::align::BoundsSource::Dtw => self.runs_dtw += 1,
      crate::align::BoundsSource::Segment => self.runs_segment += 1,
      crate::align::BoundsSource::Wholeclip => self.runs_wholeclip += 1,
    }
  }

  /// Increment the unaligned-language counter (run's `Lang` had
  /// no [`crate::Aligner`] registered AND no `Any` fallback).
  pub(super) const fn observe_unaligned(&mut self) {
    self.runs_unaligned += 1;
  }

  /// Total runs observed.
  pub(super) const fn runs_total(&self) -> usize {
    self.runs_total
  }

  /// Runs whose bounds came from per-token DTW timestamps.
  pub(super) const fn runs_dtw(&self) -> usize {
    self.runs_dtw
  }

  /// Runs whose bounds came from the parent segment envelope.
  pub(super) const fn runs_segment(&self) -> usize {
    self.runs_segment
  }

  /// Runs whose bounds came from the whole-clip sentinel
  /// fallback.
  pub(super) const fn runs_wholeclip(&self) -> usize {
    self.runs_wholeclip
  }

  /// Runs whose `Lang` had no registered aligner.
  pub(super) const fn runs_unaligned(&self) -> usize {
    self.runs_unaligned
  }
}

/// Per-run dispatch path: for each [`crate::align::Run`] in
/// `job.runs`, look up the matching [`crate::Aligner`] and run
/// `align_chunk` over the run's audio slice. Results are stitched
/// into a single [`AlignmentResult`].
///
/// **Audio slicing.** The dispatcher inherits each run's bounds
/// from the parent whisper segment (per the design spec — finer
/// per-token slicing is a follow-up). We translate
/// `(audio_t0_ms, audio_t1_ms)` to chunk-local sample indices
/// via the analysis sample rate (16 kHz). The whole-clip
/// sentinel ([`crate::align::BoundsSource::Wholeclip`])
/// degrades to running over the full chunk audio.
///
/// **Sub-segment intersection.** Sub-VAD segments are passed
/// through unchanged; the aligner's silence-mask handles the
/// case where they extend past the run window (out-of-range
/// positions get clamped inside `Aligner::align`).
///
/// **Fallback for unaligned languages.** When neither a
/// `Lang(L)` aligner nor an `Any` aligner is registered, AND
/// the configured fallback is `SkipChunk`, we synthesise a
/// single pseudo-[`crate::types::Word`] covering the run's
/// `(audio_t0_ms, audio_t1_ms)` with `score = 0.0` and the
/// run's verbatim text. This preserves the run's place in the
/// output stream (downstream consumers can render it as
/// non-aligned text) instead of dropping it. The
/// `AlignmentFallback::Error` policy still surfaces an error.
///
/// **Telemetry.** Logs one `script_dispatch chunk=...` line per
/// dispatched chunk to stderr with the
/// [`BoundsSourceCounters`] distribution.
fn dispatch_runs(
  set: &AlignmentSet,
  job: &AlignWorkItem,
  run_options: &RunOptions,
) -> Result<AlignmentResult, WorkFailure> {
  use crate::align::BoundsSource;

  let mut counters = BoundsSourceCounters::default();
  let mut all_words: alloc::vec::Vec<crate::types::Word> = alloc::vec::Vec::new();

  for run in job.runs.iter() {
    counters.observe_bounds(run.bounds_source());

    // Resolve the audio slice for this run. Bounds in ms get
    // converted to chunk-local sample indices at 16 kHz; the
    // wholeclip sentinel falls back to the full chunk.
    let (slice_lo, slice_hi) =
      run_audio_slice(run, job.samples.len(), job.chunk_first_sample_in_stream);

    let lookup = set.lookup(run.language());
    let aligner_lock = match lookup {
      AlignmentLookup::Hit { aligner, .. } => Some(aligner),
      AlignmentLookup::AnyFallback { aligner } => Some(aligner),
      AlignmentLookup::Miss { fallback } => match fallback {
        AlignmentFallback::SkipChunk => {
          counters.observe_unaligned();
          all_words.push(unsupported_language_pseudo_word(run));
          None
        }
        AlignmentFallback::Error => {
          emit_telemetry(job.chunk_id, &counters);
          return Err(WorkFailure::LanguageUnsupportedForAlignment {
            language: run.language().clone(),
          });
        }
      },
    };

    let Some(aligner) = aligner_lock else {
      continue;
    };

    // Slice sub_segments to those that overlap the run's audio
    // window. The aligner clamps out-of-range PTS internally,
    // but pre-filtering keeps the silence mask sharp.
    let run_subs = clip_sub_segments(&job.sub_segments, slice_lo, slice_hi);
    let run_samples = &job.samples[slice_lo..slice_hi];

    // Per-run `chunk_first_sample_in_stream`: the parent chunk's
    // first sample plus this run's offset inside the chunk. The
    // aligner uses this to convert frame indices back into
    // stream sample space, which downstream
    // `samples_to_output_range` then maps to caller timebase.
    let run_first_sample_in_stream = job
      .chunk_first_sample_in_stream
      .saturating_add(slice_lo as u64);

    let outcome = run_one_per_run(
      aligner,
      run,
      run_samples,
      &run_subs,
      run_first_sample_in_stream,
      job.samples_to_output_range.clone(),
      &job.abort_flag,
      run_options,
    );
    match outcome {
      Ok(result) => {
        for word in result.into_words() {
          all_words.push(word);
        }
      }
      Err(failure) => {
        if alignment_failure_is_recoverable(&failure) {
          // Recoverable: drop this run's words but keep going on
          // remaining runs so a single bad run doesn't poison
          // the whole chunk's alignment.
          continue;
        }
        emit_telemetry(job.chunk_id, &counters);
        return Err(failure);
      }
    }
    // Defensive: if we've consumed the full chunk on a
    // `Wholeclip`-bounded run there's no more audio left to
    // align — break to avoid replaying the same audio for a
    // second run.
    if matches!(run.bounds_source(), BoundsSource::Wholeclip) {
      break;
    }
  }

  emit_telemetry(job.chunk_id, &counters);
  Ok(AlignmentResult::new(all_words))
}

/// Translate a run's `(audio_t0_ms, audio_t1_ms)` into chunk-local
/// sample indices. The whole-clip sentinel
/// ([`crate::align::BoundsSource::Wholeclip`]) maps to the full
/// chunk (`0..samples_len`). Out-of-range or inverted bounds
/// degrade to the full chunk as well — the dispatcher should never
/// emit those, but we tolerate them defensively rather than panic
/// inside the alignment worker.
///
/// `_chunk_first_sample_in_stream` is currently unused; it is
/// reserved for a future refinement that lets the dispatcher emit
/// per-run audio bounds in stream coordinates rather than chunk-
/// local ms (the runner already passes the chunk-relative form
/// here).
fn run_audio_slice(
  run: &crate::align::Run,
  samples_len: usize,
  _chunk_first_sample_in_stream: u64,
) -> (usize, usize) {
  use crate::align::BoundsSource;
  if matches!(run.bounds_source(), BoundsSource::Wholeclip) {
    return (0, samples_len);
  }
  let t0 = run.audio_t0_ms();
  let t1 = run.audio_t1_ms();
  if t0 < 0 || t1 <= t0 {
    return (0, samples_len);
  }
  // 16 kHz sample rate: 1 ms = 16 samples.
  let lo = (t0 as u64)
    .saturating_mul(16)
    .min(samples_len as u64) as usize;
  let hi = (t1 as u64)
    .saturating_mul(16)
    .min(samples_len as u64) as usize;
  if hi <= lo {
    return (0, samples_len);
  }
  (lo, hi)
}

/// Clip and offset chunk-local sub-segments into a run's
/// audio window. Inputs are in chunk-local 1/16000 timebase
/// (start/end PTS == sample indices); outputs are in the
/// run's local 1/16000 timebase (start/end PTS == sample
/// indices relative to `slice_lo`).
fn clip_sub_segments(
  subs: &[TimeRange],
  slice_lo: usize,
  slice_hi: usize,
) -> alloc::vec::Vec<TimeRange> {
  use core::num::NonZeroU32;
  let tb = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
  let mut out = alloc::vec::Vec::with_capacity(subs.len());
  let lo_i = slice_lo as i64;
  let hi_i = slice_hi as i64;
  for sub in subs {
    let s = sub.start_pts().max(lo_i);
    let e = sub.end_pts().min(hi_i);
    if e > s {
      out.push(TimeRange::new(s - lo_i, e - lo_i, tb));
    }
  }
  out
}

/// Pseudo-word for runs whose [`Lang`] has no registered aligner
/// and the configured fallback is `SkipChunk`. Carries the run's
/// verbatim text and audio bounds with `score = 0.0` so
/// downstream consumers can render the un-aligned slice as
/// timed-but-unscored text. Word range is in milliseconds (1/1000
/// timebase, matching the dispatcher's `audio_t*_ms` fields).
fn unsupported_language_pseudo_word(run: &crate::align::Run) -> crate::types::Word {
  use core::num::NonZeroU32;
  let tb = mediatime::Timebase::new(1, NonZeroU32::new(1_000).unwrap());
  let t0 = run.audio_t0_ms().max(0);
  let t1 = run.audio_t1_ms().max(t0);
  let range = TimeRange::new(t0, t1, tb);
  crate::types::Word::new(SmolStr::new(run.text()), range, 0.0)
}

/// Lock + run for one per-run alignment call. Mirrors
/// [`run_under_lock`] but with the run's audio slice + sub-segment
/// intersection.
#[allow(clippy::too_many_arguments)]
fn run_one_per_run(
  aligner: &Mutex<Aligner>,
  run: &crate::align::Run,
  run_samples: &[f32],
  run_sub_segments: &[TimeRange],
  run_first_sample_in_stream: u64,
  samples_to_output_range: Arc<dyn Fn(u64, u64) -> TimeRange + Send + Sync>,
  abort_flag: &AtomicBool,
  run_options: &RunOptions,
) -> Result<AlignmentResult, WorkFailure> {
  let mut guard = match aligner.lock() {
    Ok(g) => g,
    Err(poisoned) => poisoned.into_inner(),
  };
  let bound = samples_to_output_range.clone();
  guard.align(
    run_samples,
    run_sub_segments,
    run.text(),
    run_first_sample_in_stream,
    move |a, b| (bound)(a, b),
    abort_flag,
    run_options,
  )
}

/// One-line telemetry per chunk. Format chosen to be greppable
/// from logs (`grep script_dispatch`) and to match the structured
/// shape from the spec:
/// `script_dispatch chunk=<id> runs=<total> dtw=<n> segment=<n>
/// wholeclip=<n> unaligned=<n>`.
fn emit_telemetry(chunk_id: ChunkId, c: &BoundsSourceCounters) {
  std::eprintln!(
    "script_dispatch chunk={} runs={} dtw={} segment={} wholeclip={} unaligned={}",
    chunk_id.as_u64(),
    c.runs_total(),
    c.runs_dtw(),
    c.runs_segment(),
    c.runs_wholeclip(),
    c.runs_unaligned(),
  );
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

  /// `BoundsSourceCounters` accumulates the dispatcher's
  /// `BoundsSource` distribution one observation at a time. The
  /// counters in script_dispatch chunk-level telemetry are derived
  /// solely from these increments, so a regression here would silently
  /// corrupt every line of operator-facing log output.
  #[test]
  fn bounds_source_counters_accumulate_distribution() {
    use crate::align::BoundsSource;
    let mut c = BoundsSourceCounters::default();
    c.observe_bounds(BoundsSource::Dtw);
    c.observe_bounds(BoundsSource::Dtw);
    c.observe_bounds(BoundsSource::Segment);
    c.observe_bounds(BoundsSource::Wholeclip);
    c.observe_unaligned();
    c.observe_unaligned();
    assert_eq!(c.runs_total(), 4);
    assert_eq!(c.runs_dtw(), 2);
    assert_eq!(c.runs_segment(), 1);
    assert_eq!(c.runs_wholeclip(), 1);
    assert_eq!(c.runs_unaligned(), 2);
  }

  /// Default-constructed counters are all-zero — used when a chunk
  /// dispatches the legacy whole-chunk path (empty `runs`).
  #[test]
  fn bounds_source_counters_default_is_zero() {
    let c = BoundsSourceCounters::default();
    assert_eq!(c.runs_total(), 0);
    assert_eq!(c.runs_dtw(), 0);
    assert_eq!(c.runs_segment(), 0);
    assert_eq!(c.runs_wholeclip(), 0);
    assert_eq!(c.runs_unaligned(), 0);
  }

  /// `run_audio_slice` translates the dispatcher's millisecond
  /// bounds into chunk-local sample indices at the analysis
  /// sample rate (16 kHz). Spot-check the standard segment-sourced
  /// case, the wholeclip sentinel, and the inverted-bounds
  /// defensive fallback.
  #[test]
  fn run_audio_slice_segment_bounds_clamp_to_chunk_length() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      100,
      300,
      0,
      BoundsSource::Segment,
    );
    let (lo, hi) = run_audio_slice(&r, 16_000, 0);
    assert_eq!(lo, 1_600);
    assert_eq!(hi, 4_800);
  }

  #[test]
  fn run_audio_slice_wholeclip_uses_full_chunk() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      i64::MIN,
      i64::MAX,
      0,
      BoundsSource::Wholeclip,
    );
    let (lo, hi) = run_audio_slice(&r, 16_000, 0);
    assert_eq!(lo, 0);
    assert_eq!(hi, 16_000);
  }

  #[test]
  fn run_audio_slice_inverted_bounds_fall_back_to_full_chunk() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      500,
      100,
      0,
      BoundsSource::Segment,
    );
    let (lo, hi) = run_audio_slice(&r, 16_000, 0);
    assert_eq!(lo, 0);
    assert_eq!(hi, 16_000);
  }

  /// `clip_sub_segments` keeps only the portion of each
  /// sub-segment that overlaps the run's audio window, and
  /// re-bases the timestamps so they remain chunk-local within
  /// the run's slice.
  #[test]
  fn clip_sub_segments_offsets_into_run_local_space() {
    use core::num::NonZeroU32;
    let tb = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let subs = alloc::vec![
      // Fully inside the run window.
      TimeRange::new(2_000, 3_000, tb),
      // Straddles the lower bound.
      TimeRange::new(800, 2_400, tb),
      // Outside the run entirely; dropped.
      TimeRange::new(8_000, 9_000, tb),
    ];
    let out = clip_sub_segments(&subs, 1_600, 4_800);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].start_pts(), 400);
    assert_eq!(out[0].end_pts(), 1_400);
    assert_eq!(out[1].start_pts(), 0);
    assert_eq!(out[1].end_pts(), 800);
  }

  /// Pseudo-word for unsupported-language runs carries the run's
  /// verbatim text and a millisecond-timebase range covering the
  /// run's audio bounds. `score = 0.0` flags it as un-aligned.
  #[test]
  fn unsupported_language_pseudo_word_carries_run_text_and_bounds() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    let r = Run::new(
      Lang::Other(SmolStr::new("xx")),
      SmolStr::new("untranslated"),
      120,
      450,
      0,
      BoundsSource::Segment,
    );
    let w = unsupported_language_pseudo_word(&r);
    assert_eq!(w.text(), "untranslated");
    assert_eq!(w.range().start_pts(), 120);
    assert_eq!(w.range().end_pts(), 450);
    assert!((w.score() - 0.0).abs() < f32::EPSILON);
  }
}
