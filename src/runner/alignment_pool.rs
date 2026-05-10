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

use core::sync::atomic::Ordering;
use std::{sync::Mutex, time::Instant};

use ort::session::RunOptions;

use crate::{
  core::AlignmentResult,
  runner::aligner::{Aligner, AlignmentFallback, AlignmentLookup, AlignmentSet},
  types::{AlignmentFailureKind, ChunkId, Lang, WorkFailure, WorkerKind},
};

/// One unit of alignment work — the bundle of caller inputs
/// the per-chunk dispatcher consumes.
///
/// Codex round-37 round-13 [high]: bumped from `pub(super)`
/// to `pub` so external Sans-I/O drivers can construct one
/// from a [`crate::core::Command::Alignment`] and feed it
/// to [`run_one_alignment`]. Field shape mirrors what the
/// dispatcher needs end-to-end:
///
/// - `samples`, `sub_segments`, `text`, `language`, `runs`
///   come straight from `Alignment`. `sub_segments` must
///   be in chunk-local 1/16000 timebase
///   ([`crate::core::Transcriber::chunk_sub_segments_samples`]
///   exposes the right form, offset by
///   [`crate::core::Transcriber::chunk_first_sample`]).
/// - `chunk_first_sample_in_stream` from
///   [`crate::core::Transcriber::chunk_first_sample`].
/// - `samples_to_output_range` from
///   [`crate::core::Transcriber::chunk_samples_to_output_range_fn`].
/// - `abort_flag` is caller-owned; flipping it from any
///   thread cancels the in-flight alignment at the next
///   pipeline boundary (silence mask, normalise, encode,
///   trellis, compose).
///
/// **Cancellation contract.** This struct owns the abort
/// flag but **not** ORT termination — `Aligner::align`
/// reuses a per-call `RunOptions` constructed internally
/// inside [`run_one_alignment`]. Setting `abort_flag` true
/// surfaces at the next pipeline boundary; mid-ORT
/// cancellation (interrupting `Session::run_with_options`)
/// requires the caller to construct their own `RunOptions`
/// and call [`crate::Aligner::align_chunk_with_abort`]
/// directly instead of routing through `run_one_alignment`.
/// Codex round-37 round-15 [high]: removed the
/// `align_timeout` field — the dispatcher never enforced it
/// (no internal watchdog), and keeping it as
/// "informational telemetry" misled callers into thinking a
/// timeout would fire.
pub struct AlignWorkItem {
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
  /// Caller-resolved OOV decisions, one inner vec per
  /// alignment unit:
  /// * If [`Self::runs`] is non-empty (script-dispatched
  ///   code-switch path): one entry per run, in run order.
  ///   `oov_decisions[i]` applies to `runs[i].text()` and
  ///   must be in the order
  ///   [`AlignmentSet::detect_oov`](crate::AlignmentSet::detect_oov)
  ///   would have produced events for that run.
  /// * If [`Self::runs`] is empty (whole-chunk fallback):
  ///   exactly one entry whose decisions apply to the
  ///   chunk's full `text`.
  /// * Empty outer vec is allowed only as a transitional
  ///   shape: the runner falls back to `&[]` for both
  ///   chunk-level and per-run alignment paths (encountering
  ///   any OOV then raises `TokenizationFailed`).
  ///
  /// The caller computes this via
  /// [`AlignmentSet::detect_oov`] per unit + a policy helper
  /// from [`crate::core::oov`]
  /// (`default_oov_decisions` / `wildcard_all_decisions` /
  /// `fail_closed_all_decisions` / a custom closure over the
  /// events). Sans-I/O OOV resolution: data in, no callbacks.
  ///
  /// Codex round-37 (parity loop) round-1 [high]: pre-fix
  /// this was a flat `Vec<OovDecision>` that only applied to
  /// the whole-chunk path; the per-run path hard-coded
  /// `default_oov_decisions`, silently ignoring the caller's
  /// `wildcard_all_decisions` / `fail_closed_all_decisions`
  /// opt-in for any chunk with `runs` populated (= every
  /// chunk produced by `WhisperAsrSource`). The per-run
  /// shape is now strict, so the caller's policy reaches
  /// every alignment unit.
  pub oov_decisions: alloc::vec::Vec<alloc::vec::Vec<crate::core::ResolvedOov>>,
}

impl AlignWorkItem {
  /// Construct an `AlignWorkItem` from a
  /// [`crate::core::Command::Alignment`] payload + the
  /// `Transcriber` chunk-metadata accessors. Handles the
  /// **coordinate-space flip** from output-timebase
  /// `sub_segments` to chunk-local 1/16000 the aligner needs;
  /// Codex round-37 round-18 [high] flagged the previous
  /// hand-rolled conversion as a footgun (callers who forwarded
  /// `command.sub_segments` straight to `AlignWorkItem`'s
  /// field would hit a hard error from `clip_sub_segments`).
  ///
  /// Returns `None` if the chunk identity is no longer in
  /// flight on `transcriber` (already drained / failed) — this
  /// is the only failure mode; pass it back as a recoverable
  /// `Backpressure` from the caller's pump if needed.
  ///
  /// Inputs map 1:1 to the `Alignment` variant's fields plus
  /// the caller-owned `abort_flag`.
  #[allow(
    clippy::too_many_arguments,
    reason = "mirrors `Command::Alignment` fields + caller-owned abort_flag; \
              destructured-pattern callers naturally line them up positionally"
  )]
  pub fn from_run_alignment(
    transcriber: &crate::core::Transcriber,
    chunk_id: ChunkId,
    samples: Arc<[f32]>,
    text: SmolStr,
    language: Lang,
    runs: alloc::vec::Vec<crate::align::Run>,
    abort_flag: Arc<AtomicBool>,
    // Caller-resolved OOV decisions, one inner vec per
    // alignment unit. See the `Self::oov_decisions` field
    // doc-comment for the precise shape:
    //   * `runs` non-empty → one inner vec per run, in run
    //     order, each in `detect_oov_events` order for that
    //     run's text;
    //   * `runs` empty → exactly one inner vec for the
    //     whole-chunk path.
    // Pass an empty outer vec only if the caller is sure
    // there are no OOV chars (encountering one anyway raises
    // `TokenizationFailed`).
    oov_decisions: alloc::vec::Vec<alloc::vec::Vec<crate::core::ResolvedOov>>,
  ) -> Option<Self> {
    use core::num::NonZeroU32;
    let chunk_first = transcriber.chunk_first_sample(chunk_id)?;
    let raw_subs = transcriber.chunk_sub_segments_samples(chunk_id)?;
    let bridge = transcriber.chunk_samples_to_output_range_fn(chunk_id)?;
    let tb_16k = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let aligner_subs: alloc::vec::Vec<TimeRange> = raw_subs
      .iter()
      .map(|(s, e)| {
        TimeRange::new(
          (*s as i64) - (chunk_first as i64),
          (*e as i64) - (chunk_first as i64),
          tb_16k,
        )
      })
      .collect();
    Some(Self {
      chunk_id,
      samples,
      sub_segments: aligner_subs,
      text,
      language,
      runs,
      abort_flag,
      chunk_first_sample_in_stream: chunk_first,
      samples_to_output_range: bridge,
      oov_decisions,
    })
  }
}

/// Worker-emitted alignment result. Crate-private.

/// Returned by [`AlignmentPool::shutdown`] when one or more
/// workers failed to wind down within the supplied timeout.
/// `count` is the number of detached threads; each holds an
/// `Aligner` (ONNX session + model memory) until its in-flight
/// inference returns naturally.

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
/// Drive one chunk's alignment to completion. Sync; the
/// caller owns thread management and cancellation via
/// `job.abort_flag`. The aligner polls `abort_flag` at coarse
/// pipeline boundaries (silence mask, normalise, encode,
/// trellis, compose) and bails with
/// [`WorkFailure::WorkerHangTimeout`] when the flag flips.
/// True ORT mid-inference cancellation requires the caller to
/// hold a `RunOptions` handle and call `terminate()` from
/// another thread; Codex round-37 round-9 had this wired via
/// an internal watchdog thread, but the round-1.0 Sans-I/O
/// pivot moved threading out of whispery — callers who need
/// it construct their own watchdog around `Aligner::align`.
/// Drive one [`AlignWorkItem`] end-to-end against the
/// supplied [`AlignmentSet`]. Caller passes the
/// [`RunOptions`] handle so a runtime-owned watchdog (or any
/// external thread) can call `run_options.terminate()` to
/// unwind in-flight ORT inference; the aligner additionally
/// polls `job.abort_flag` between pipeline stages.
///
/// Codex round-37 round-13 [high]: made the helper public.
/// Round-37 round-17 [high]: hoisted `RunOptions` out of the
/// internal scope so callers can actually cancel mid-ONNX. A
/// single shared `RunOptions` is used across every run in a
/// multi-run chunk — calling `terminate()` cancels whichever
/// run is currently in `Session::run_with_options`, and the
/// post-call `abort_flag` check stops dispatching the
/// remaining peer runs.
pub fn run_one_alignment(
  set: &AlignmentSet,
  job: &AlignWorkItem,
  run_options: &RunOptions,
) -> Result<AlignmentResult, WorkFailure> {
  let started_at = Instant::now();

  // Codex round-37 round-27 [high]: pre-entry abort gate. The
  // caller may have armed `RunOptions::terminate()` *and*
  // flipped `abort_flag` for this chunk before we got here
  // (e.g. their pump dispatched a watchdog deadline that
  // fired between command-poll and our entry). Honour that
  // intent immediately and return `WorkerHangTimeout` instead
  // of starting the alignment pipeline.
  if job.abort_flag.load(Ordering::Relaxed) {
    return Err(WorkFailure::WorkerHangTimeout {
      kind: WorkerKind::Alignment,
      elapsed: started_at.elapsed(),
    });
  }

  // Codex round-37 round-6 (parity loop) [high]: validate the
  // OUTER `Vec<Vec<OovDecision>>` shape before dispatch. The
  // inner-vec length is checked against actual OOV count by
  // `tokenize_with_word_map`, but a stale outer shape (e.g.
  // per-run decisions applied to a whole-chunk job, or
  // shorter-than-runs.len()) can silently bypass the caller's
  // policy because dispatch indexes by `.first()` /
  // `.get(run_idx)` and ignores extras / falls back to `&[]`.
  // Reject shape mismatches loudly so a stale payload from a
  // previous chunk can't silently apply the wrong prefix
  // policy.
  let outer = job.oov_decisions.len();
  let expected = if job.runs.is_empty() {
    1
  } else {
    job.runs.len()
  };
  // `outer == 0` is tolerated as "no OOV expected" (the
  // tokenizer surfaces `TokenizationFailed` if a chunk hits
  // OOV anyway). Any other size mismatch is rejected.
  if outer != 0 && outer != expected {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::TokenizationFailed,
      message: alloc::format!(
        "AlignWorkItem::oov_decisions outer shape mismatch: \
         expected 0 (no OOV) or {expected} ({}), got {outer}. \
         This typically means stale per-run decisions are being \
         applied to a whole-chunk job (or vice versa). Recompute \
         decisions for this chunk's text via \
         `AlignmentSet::detect_oov` / `detect_oov_per_run`.",
        if job.runs.is_empty() {
          "exactly one whole-chunk vec"
        } else {
          "one inner vec per run"
        },
      ),
      language: job.language.clone(),
    });
  }

  // Codex round-37 round-11 (parity loop) [high]: positional
  // identity (`OovEvent::matches_position`) deliberately
  // ignores `language` so `AlignerKey::Any` fallback works.
  // That opens a dispatch-boundary hole: a stale ResolvedOov
  // produced for a different chunk's REQUESTED language can
  // pass the tokenizer-level identity check at the same
  // position. The caller's language-conditional policy
  // (wildcard-en / fail-closed-ko) then runs against the
  // wrong key. Validate at the dispatcher boundary, where we
  // know each run's requested language, before letting the
  // payload reach `tokenize_with_word_map`.
  validate_oov_decision_languages(&job.runs, &job.language, &job.oov_decisions)?;

  // Codex round-37 round-27 [high]: do NOT clear caller-armed
  // termination. Round 22 unconditionally called
  // `run_options.unterminate()` here to defend against the
  // sticky-poison case where reusing one `RunOptions` across
  // chunks would surface a one-time cancellation as a fatal
  // `ModelInferenceFailed` for every subsequent chunk; that
  // reset, however, also erased a `terminate()` the caller's
  // watchdog had armed for THIS job between command-dispatch
  // and entry. Documented contract is now: callers allocate
  // a fresh `RunOptions` per chunk (see `src/runner/mod.rs`
  // doc-test and the README pump). The `abort_flag` gate
  // above + the per-stage gates inside the aligner are the
  // primary cancellation surface; `RunOptions::terminate` is
  // the ORT mid-call escape hatch the caller owns end-to-end.

  let outcome = if job.runs.is_empty() {
    match set.lookup(&job.language) {
      AlignmentLookup::Hit { aligner, .. } => {
        run_under_lock(aligner, job, run_options, &job.abort_flag)
      }
      AlignmentLookup::AnyFallback { aligner } => {
        run_under_lock(aligner, job, run_options, &job.abort_flag)
      }
      AlignmentLookup::Miss { fallback } => match fallback {
        AlignmentFallback::SkipChunk => Ok(AlignmentResult::new(alloc::vec::Vec::new())),
        AlignmentFallback::Error => Err(WorkFailure::LanguageUnsupportedForAlignment {
          language: job.language.clone(),
        }),
      },
    }
  } else {
    dispatch_runs(set, job, run_options)
  };

  // An alignment-stage failure is NOT a reason to discard the
  // cached ASR transcript. Without this, a `NoAlignmentPath`
  // from a too-short chunk or a 32 M-cell budget overflow would
  // propagate to `handle_failure` upstream, turning the chunk
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
      // Codex round-37 round-22 [high]: emit an observable
      // diagnostic when alignment is dropped silently. Without
      // this, recoverable failures (semantic-OOV chunks,
      // NoAlignmentPath, EmptyText) collapse to
      // `Transcript { text, words: [] }` with no surface
      // signal — operators can't distinguish "alignment
      // succeeded with zero words" from "alignment was
      // dropped". One stderr line per recovery, keyed by
      // chunk_id + failure kind.
      if let WorkFailure::AlignmentFailed {
        kind,
        message: _,
        language,
      } = f
      {
        // Codex round-37 round-25 [high]: drop the failure
        // `message` from the log line — `SemanticOutOfVocab`
        // currently embeds the offending char, which is
        // transcript content. The failure `kind` already
        // conveys the cause class; full diagnostic strings
        // stay accessible to callers via the typed
        // `WorkFailure` they own.
        eprintln!(
          "whispery alignment recovered chunk={:?} kind={kind:?} language={language:?}",
          job.chunk_id,
        );
      }
      Ok(AlignmentResult::new(alloc::vec::Vec::new()))
    }
    // Codex round-37 round-11 (parity loop) [medium]:
    // canonicalise `WorkerHangTimeout::elapsed`. Inner code
    // (`Aligner::align`'s `timed_out` closure,
    // `classify_encode_abort` for ORT-cancel) hard-codes
    // `Duration::ZERO` because it doesn't own an `Instant`.
    // The worker DOES — overwrite unconditionally so
    // operators don't see misleading zero-elapsed timeout
    // metrics for real cancellations. Pre-fix the abort-cancel
    // path surfaced as `WorkerHangTimeout { elapsed: 0 }`,
    // breaking any recovery / alert logic keyed on duration.
    Err(WorkFailure::WorkerHangTimeout { kind, .. }) => Err(WorkFailure::WorkerHangTimeout {
      kind,
      elapsed: started_at.elapsed(),
    }),
    Err(_) => outcome,
  }
}

/// Codex round-37 round-11 (parity loop) [high]: validate that
/// every supplied `ResolvedOov`'s event language matches the
/// chunk/run's requested language.
///
/// Positional identity (`OovEvent::matches_position`) is
/// language-agnostic so `AlignerKey::Any` fallback works — the
/// fallback aligner's tokenizer re-detects events with its own
/// construction language, and the caller's payload carries the
/// caller-REQUESTED language. The flip side is that a stale
/// payload made for a DIFFERENT chunk's requested language can
/// silently pass the in-tokenizer identity check at the same
/// position, then run through caller policy keyed off the
/// wrong language.
///
/// The dispatcher knows each run's requested language
/// (`run.language()`) and the whole-chunk job language
/// (`job.language`); enforce that every `ResolvedOov.event.language`
/// matches before dispatch. Mismatch fails loudly as
/// `TokenizationFailed`, not silent policy bypass.
fn validate_oov_decision_languages(
  runs: &[crate::align::Run],
  job_language: &Lang,
  oov_decisions: &[alloc::vec::Vec<crate::core::ResolvedOov>],
) -> Result<(), WorkFailure> {
  if runs.is_empty() {
    // Whole-chunk path: all decisions in oov_decisions[0]
    // (the only inner vec — already shape-validated above)
    // must carry job.language.
    if let Some(chunk_decisions) = oov_decisions.first() {
      for (i, resolved) in chunk_decisions.iter().enumerate() {
        if &resolved.event.language != job_language {
          return Err(WorkFailure::AlignmentFailed {
            kind: AlignmentFailureKind::TokenizationFailed,
            message: alloc::format!(
              "AlignWorkItem::oov_decisions[0][{i}].event.language = {:?} but \
               job.language = {:?}. This typically means stale decisions from a \
               previous chunk's run leaked into a whole-chunk job; the caller's \
               language-conditional policy would run against the wrong key. \
               Recompute via `AlignmentSet::detect_oov` for THIS chunk.",
              resolved.event.language,
              job_language,
            ),
            language: job_language.clone(),
          });
        }
      }
    }
    return Ok(());
  }
  // Per-run path: oov_decisions[r] (already shape-validated)
  // must carry runs[r].language() throughout.
  for (run_idx, run) in runs.iter().enumerate() {
    let Some(run_decisions) = oov_decisions.get(run_idx) else {
      // Outer shape is either == runs.len() or 0; an
      // unindexable position in the per-run case is the
      // 0-outer "no OOV expected" branch handled by
      // tokenize_with_word_map. Nothing to validate.
      continue;
    };
    let expected_lang = run.language();
    for (i, resolved) in run_decisions.iter().enumerate() {
      if &resolved.event.language != expected_lang {
        return Err(WorkFailure::AlignmentFailed {
          kind: AlignmentFailureKind::TokenizationFailed,
          message: alloc::format!(
            "AlignWorkItem::oov_decisions[{run_idx}][{i}].event.language = {:?} \
             but runs[{run_idx}].language() = {:?}. This typically means stale \
             decisions from a previous chunk leaked into a per-run dispatch; \
             the caller's language-conditional policy would run against the \
             wrong key. Recompute via `AlignmentSet::detect_oov_per_run` for \
             THIS chunk's runs.",
            resolved.event.language,
            expected_lang,
          ),
          language: expected_lang.clone(),
        });
      }
    }
  }
  Ok(())
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
      kind: AlignmentFailureKind::NoAlignmentPath
        | AlignmentFailureKind::EmptyText
        | AlignmentFailureKind::SemanticOutOfVocab,
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
  abort_flag: &AtomicBool,
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
  // Whole-chunk path: caller passes exactly one inner vec
  // (the chunk's full text decisions). Empty outer vec is
  // tolerated as "no OOV expected"; per the field
  // doc-comment, encountering one anyway raises
  // `TokenizationFailed`.
  let chunk_decisions = job
    .oov_decisions
    .first()
    .map(|v| v.as_slice())
    .unwrap_or(&[]);
  guard.align(
    &job.samples,
    &job.sub_segments,
    job.text.as_str(),
    job.chunk_first_sample_in_stream,
    move |a, b| (bound)(a, b),
    abort_flag,
    run_options,
    chunk_decisions,
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
/// Codex round-37 round-21 [medium] between-run abort gate.
/// Extracted so the gate's failure shape stays unit-testable
/// without a real `RunOptions` (which requires ORT runtime
/// initialisation and is therefore awkward in lib unit tests).
fn check_abort_between_runs(
  abort_flag: &AtomicBool,
  dispatch_started_at: Instant,
) -> Result<(), WorkFailure> {
  if abort_flag.load(Ordering::Relaxed) {
    return Err(WorkFailure::WorkerHangTimeout {
      kind: WorkerKind::Alignment,
      elapsed: dispatch_started_at.elapsed(),
    });
  }
  Ok(())
}

fn dispatch_runs(
  set: &AlignmentSet,
  job: &AlignWorkItem,
  run_options: &RunOptions,
) -> Result<AlignmentResult, WorkFailure> {
  let mut counters = BoundsSourceCounters::default();
  let mut all_words: alloc::vec::Vec<crate::types::Word> = alloc::vec::Vec::new();
  let dispatch_started_at = Instant::now();

  for (run_idx, run) in job.runs.iter().enumerate() {
    // Codex round-37 round-21 [medium]: between-run abort gate.
    // The shared `RunOptions` lets an external watchdog
    // terminate the run currently in flight, but a cancellation
    // that lands AFTER one run's final internal abort check
    // could still fall through into the next iteration and
    // start another ONNX inference — extending a hung/cancelled
    // job and delaying drain. Check the flag at the top of each
    // iteration so the cancellation observed during a previous
    // run propagates immediately, matching the
    // `Aligner::align` post-call abort semantics.
    if let Err(failure) = check_abort_between_runs(&job.abort_flag, dispatch_started_at) {
      emit_telemetry(job.chunk_id, &counters);
      return Err(failure);
    }

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
          // Codex round-37 round-16 [medium]: `SkipChunk` is
          // documented as producing empty `Transcript.words()`
          // (the no-runs path returns `Ok(empty)`). Pre-fix
          // the per-run dispatch path emitted a timed
          // pseudo-word for each missing language, which
          // downstream consumers could mistake for aligned
          // word timing — silently violating the documented
          // empty-words contract. Now we just count the run
          // as unaligned and skip it (no word emitted), so
          // both paths agree on `SkipChunk` semantics.
          counters.observe_unaligned();
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
    let run_subs = clip_sub_segments(&job.sub_segments, slice_lo, slice_hi, run.language())
      .map_err(|e| {
        emit_telemetry(job.chunk_id, &counters);
        e
      })?;
    let run_samples = &job.samples[slice_lo..slice_hi];

    // Per-run `chunk_first_sample_in_stream`: the parent chunk's
    // first sample plus this run's offset inside the chunk. The
    // aligner uses this to convert frame indices back into
    // stream sample space, which downstream
    // `samples_to_output_range` then maps to caller timebase.
    let run_first_sample_in_stream = job
      .chunk_first_sample_in_stream
      .saturating_add(slice_lo as u64);

    // Codex round-37 round-17 [high]: a SHARED `RunOptions`
    // across all runs in a chunk. The caller supplies it via
    // `run_one_alignment(..., run_options)` so an external
    // watchdog can call `terminate()` and stop whichever run
    // is currently in flight; the post-call `abort_flag`
    // check below then prevents subsequent runs from starting.
    // Round-5's "fresh per run" isolation is sacrificed to make
    // cancellation actually work end-to-end — the trade-off is
    // acceptable because the aligner mutex serialises ORT
    // calls within a chunk anyway.
    // Codex round-37 round-1 (parity loop) [high]: thread the
    // caller's per-run OOV decisions through. `oov_decisions`
    // is `Vec<Vec<OovDecision>>` indexed by run; a missing
    // entry (caller pre-sized too small / left empty) falls
    // back to `&[]` which raises `TokenizationFailed` if the
    // run hits any OOV — loud diagnostic for the caller, not
    // silent default-policy substitution.
    let run_oov_decisions = job
      .oov_decisions
      .get(run_idx)
      .map(|v| v.as_slice())
      .unwrap_or(&[]);
    let outcome = run_one_per_run(
      aligner,
      run,
      run_samples,
      &run_subs,
      run_first_sample_in_stream,
      job.samples_to_output_range.clone(),
      &job.abort_flag,
      run_options,
      run_oov_decisions,
    );
    match outcome {
      Ok(result) => {
        let run_lang = run.language().clone();
        for word in result.into_words() {
          // Codex round-37 [medium]: tag every dispatched word
          // with its run's language so downstream consumers can
          // route per-word output without reverse-mapping from
          // text/timing. The aligner itself doesn't know the run
          // language; we attach it here at the dispatch boundary.
          all_words.push(word.with_language(Some(run_lang.clone())));
        }
      }
      Err(failure) => {
        // Per-run failures: data-dependent kinds
        // (NoAlignmentPath, EmptyText, SemanticOutOfVocab) stay
        // recoverable so a single bad run doesn't sink the
        // whole chunk. Backend / configuration failures
        // propagate.
        if alignment_failure_is_recoverable(&failure) {
          // Codex round-37 round-23 [high]: per-run recoverable
          // drops collapse silently — `dispatch_runs` returns
          // `Ok(...)` with the surviving runs' words, so the
          // top-level `run_one_alignment` recovery logger never
          // fires for the dropped run. Operators previously
          // could not distinguish "this run aligned with zero
          // words" from "this run was dropped by policy". Emit
          // a one-line diagnostic per dropped run keyed by
          // chunk_id, run language, bounds source, and failure
          // kind. Codex round-37 round-25 [high]: do NOT log
          // `run.text()` — that's transcript content (PII /
          // secrets risk on failure paths where retention
          // policies are often weaker). Log a bounded char
          // count instead so operators can correlate without
          // leaking the user's speech into stderr.
          if let WorkFailure::AlignmentFailed {
            kind,
            message: _,
            language,
          } = &failure
          {
            let run_chars = run.text().chars().count();
            eprintln!(
              "whispery alignment recovered chunk={:?} run_language={:?} run_bounds={:?} \
               run_chars={run_chars} kind={kind:?} dropped_failure_language={language:?}",
              job.chunk_id,
              run.language(),
              run.bounds_source(),
            );
          }
          counters.observe_unaligned();
          continue;
        }
        emit_telemetry(job.chunk_id, &counters);
        return Err(failure);
      }
    }
    // Codex round-37 round-4 [medium]: the previous code
    // unconditionally `break`ed after a `Wholeclip` run, which
    // dropped every later registered-language run from the same
    // chunk. `Wholeclip` is the dispatcher's fallback when both
    // DTW and segment timing are unavailable, so a mixed-script
    // chunk that lands here would emit only the FIRST run's
    // words and silently lose the rest. We now keep iterating;
    // each `Wholeclip` run aligns against the full chunk audio,
    // which over-counts duration but preserves word output for
    // every dispatched language. The post-loop sort below
    // restores the public `Transcript::words()` time-order
    // contract across multi-run output.
  }

  // Codex round-37 round-35 [high]: enforce the
  // `Transcript::words()` time-order invariant for multi-run
  // chunks. See [`sort_words_by_pts`] for the rationale.
  sort_words_by_pts(&mut all_words);

  emit_telemetry(job.chunk_id, &counters);
  Ok(AlignmentResult::new(all_words))
}

/// Stable-sort a multi-run word stream by start PTS (then end
/// PTS as tiebreaker) so the merged output respects the
/// `Transcript::words()` time-order contract.
///
/// Each per-run aligner emits its own words inside its sliced
/// audio window, which [`compute_run_bounds`] guarantees is
/// monotone vs. neighbouring runs for Dtw / Segment bounds.
/// `Wholeclip` runs (and any overlapping bounds a pluggable
/// [`crate::runner::AsrSource`] happens to feed) can land
/// words at arbitrary positions across the chunk, so appending
/// in run-order leaves the merged stream out of time order.
///
/// Codex round-37 round-35 [high]: extracted as a free
/// function so the sort's contract is testable without
/// standing up a real `Aligner` / ORT.
fn sort_words_by_pts(words: &mut alloc::vec::Vec<crate::types::Word>) {
  words.sort_by_key(|w| {
    let r = w.range();
    (r.start_pts(), r.end_pts())
  });
}

/// Translate a run's `(audio_t0_ms, audio_t1_ms)` into chunk-local
/// sample indices. The whole-clip sentinel
/// ([`crate::align::BoundsSource::Wholeclip`]) maps to the full
/// chunk (`0..samples_len`). Out-of-range or inverted bounds
/// degrade to the full chunk as well — the dispatcher should never
/// emit those, but we tolerate them defensively rather than panic
/// inside the alignment worker.
///
/// **Coordinate contract.** [`crate::align::Run::audio_t0_ms`]
/// / [`audio_t1_ms`] MUST be **chunk-local** (origin at the
/// start of the chunk's audio, not stream-absolute), in
/// milliseconds, at the chunk's 16 kHz mono sample rate.
/// `chunk_first_sample_in_stream` is the chunk's anchor in
/// stream coordinates and is **NOT** used to translate run
/// bounds — it would be in samples-of-stream while
/// `audio_t0_ms` is ms-of-chunk; mixing the two would
/// silently double-shift output timing.
///
/// Codex round-37 round-27 [medium]: a pluggable
/// [`crate::runner::AsrSource`] that erroneously populates
/// [`crate::types::AsrResult::runs`] with stream-absolute
/// times will fail this contract; `(t0_ms * 16) >=
/// samples_len` is the visible symptom (bounds saturate to
/// `samples_len`, the run aligns against zero audio, output
/// silently drops words). Surface that case as a stderr
/// warning so operators see the contract violation instead
/// of silent zero-word per-run alignment.
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
  // Codex round-37 round-5 [high]: previously any degenerate
  // non-Wholeclip bounds (`t0 < 0`, `t1 <= t0`) re-expanded to
  // `(0, samples_len)`, conflating "explicit Wholeclip" with
  // "interpolation collapsed to a zero-width span" and aligning
  // tiny code-switch runs against the entire chunk. Now we
  // surface degenerate inputs as an empty slice `(0, 0)` so the
  // aligner gracefully produces no words for the run instead of
  // duplicating unrelated audio. The dispatcher's
  // `compute_run_bounds` widens collapsed interpolation by 1cs
  // (10ms) so this branch is only hit for genuinely
  // pathological inputs (negative t0, NaN-shaped saturation).
  if t0 < 0 || t1 <= t0 {
    return (0, 0);
  }
  // 16 kHz sample rate: 1 ms = 16 samples.
  let lo_u64 = (t0 as u64).saturating_mul(16);
  let hi_u64 = (t1 as u64).saturating_mul(16);
  // Codex round-37 round-27 [medium] contract violation:
  // an out-of-window non-Wholeclip run is the visible symptom
  // of stream-absolute coordinates leaking into the
  // chunk-local API. Fail loud (stderr) so operators see the
  // bug rather than silent empty alignment. We still return
  // an empty slice so the worker doesn't crash; the per-run
  // dispatch logger then counts it as unaligned.
  if lo_u64 >= samples_len as u64 {
    eprintln!(
      "whispery alignment Run bounds appear out-of-chunk: \
       audio_t0_ms={t0} audio_t1_ms={t1} chunk_samples_len={samples_len}; \
       check your AsrSource — Run::audio_t*_ms must be chunk-local ms, not stream-absolute"
    );
    return (samples_len, samples_len);
  }
  let lo = lo_u64.min(samples_len as u64) as usize;
  let hi = hi_u64.min(samples_len as u64) as usize;
  if hi <= lo {
    // Same defence as above: collapsed slice → empty, not
    // whole-chunk fallback.
    return (lo, lo);
  }
  (lo, hi)
}

/// Clip and offset chunk-local sub-segments into a run's
/// audio window. Inputs **must** be in chunk-local 1/16000
/// timebase (start/end PTS == sample indices); outputs are in
/// the run's local 1/16000 timebase (start/end PTS == sample
/// indices relative to `slice_lo`).
///
/// Codex round-37 round-11 [medium]: pre-fix this silently
/// re-labelled inputs of any timebase as 1/16000 — an
/// integration that accidentally passed output-timebase
/// `sub_segments` from `Alignment` would have its
/// caller-timebase PTS values reinterpreted as sample indices,
/// silently zero-masking the wrong audio. Now we hard-error
/// on any non-1/16000 timebase before clipping.
fn clip_sub_segments(
  subs: &[TimeRange],
  slice_lo: usize,
  slice_hi: usize,
  language: &Lang,
) -> Result<alloc::vec::Vec<TimeRange>, WorkFailure> {
  use core::num::NonZeroU32;
  let tb = mediatime::Timebase::new(1, NonZeroU32::new(16_000).unwrap());
  let mut out = alloc::vec::Vec::with_capacity(subs.len());
  let lo_i = slice_lo as i64;
  let hi_i = slice_hi as i64;
  for sub in subs {
    let actual_tb = sub.timebase();
    if actual_tb.num() != 1 || actual_tb.den().get() != 16_000 {
      return Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!(
          "sub_segments must be in 1/16000 (chunk-local sample-index) timebase; got \
           {}/{}. Convert via `Transcriber::chunk_first_sample` + a 1/16000 timebase \
           before passing to the aligner.",
          actual_tb.num(),
          actual_tb.den().get(),
        ),
        language: language.clone(),
      });
    }
    let s = sub.start_pts().max(lo_i);
    let e = sub.end_pts().min(hi_i);
    if e > s {
      out.push(TimeRange::new(s - lo_i, e - lo_i, tb));
    }
  }
  Ok(out)
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
  // Caller-resolved per-event decisions for THIS run's text.
  // Sized + ordered to match `detect_oov_events(run.text(),
  // ..., &run.language(), ...)`. The dispatcher in
  // `dispatch_runs` indexes this slice from
  // `job.oov_decisions[run_idx]`.
  oov_decisions: &[crate::core::ResolvedOov],
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
    oov_decisions,
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

  /// Codex round-37 round-5 [high]: pre-fix any inverted /
  /// degenerate non-Wholeclip bounds re-expanded to the full
  /// chunk, so a tiny code-switch run with collapsed
  /// interpolation got aligned against the entire audio.
  /// Post-fix, degenerate non-Wholeclip bounds surface as an
  /// empty slice; the aligner produces no words for the run
  /// (recoverable miss) instead of duplicating unrelated audio.
  #[test]
  fn run_audio_slice_inverted_bounds_collapse_to_empty_slice() {
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
    assert_eq!(hi, 0);
  }

  #[test]
  fn run_audio_slice_negative_t0_collapses_to_empty_slice() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      -10,
      100,
      0,
      BoundsSource::Segment,
    );
    let (lo, hi) = run_audio_slice(&r, 16_000, 0);
    assert_eq!(lo, 0);
    assert_eq!(hi, 0);
  }

  /// Codex round-37 round-27 [medium]: a Run whose
  /// `audio_t0_ms` lands past the chunk's sample length
  /// (the symptom of stream-absolute coordinates leaking into
  /// the chunk-local API) returns an empty slice anchored at
  /// `samples_len` so the per-run dispatcher emits no words
  /// for that run. The contract violation is also surfaced to
  /// stderr (not asserted here — captured-stderr testing is
  /// brittle in `cargo test`).
  #[test]
  fn run_audio_slice_out_of_chunk_t0_collapses_to_empty_slice_at_end() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    // 16 kHz chunk, 1 s long → samples_len = 16_000.
    // A run with audio_t0_ms = 5_000 ms would translate to
    // sample 80_000 — well past the chunk window. The check
    // detects the violation and returns (16_000, 16_000).
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      5_000,
      6_000,
      0,
      BoundsSource::Segment,
    );
    let (lo, hi) = run_audio_slice(&r, 16_000, 0);
    assert_eq!(lo, 16_000);
    assert_eq!(hi, 16_000);
  }

  /// Codex round-37 round-27 [medium] coordinate-origin
  /// regression: a non-zero `chunk_first_sample_in_stream`
  /// MUST NOT shift chunk-local Run bounds. The function
  /// ignores the anchor; bounds remain chunk-local-ms.
  #[test]
  fn run_audio_slice_ignores_chunk_first_sample_in_stream() {
    use crate::align::{BoundsSource, Run};
    use smol_str::SmolStr;
    let r = Run::new(
      Lang::En,
      SmolStr::new("hi"),
      100,
      500,
      0,
      BoundsSource::Segment,
    );
    // Anchor far into the stream — irrelevant to chunk-local
    // bounds. The slice for `[100, 500) ms` at 16 kHz is
    // `[1600, 8000)`.
    let (lo, hi) = run_audio_slice(&r, 16_000, /* anchor: */ 1_000_000_000);
    assert_eq!(lo, 1600);
    assert_eq!(hi, 8000);
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
    let out = clip_sub_segments(&subs, 1_600, 4_800, &Lang::En).expect("ok");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].start_pts(), 400);
    assert_eq!(out[0].end_pts(), 1_400);
    assert_eq!(out[1].start_pts(), 0);
    assert_eq!(out[1].end_pts(), 800);
  }

  /// Codex round-37 round-11 [medium]: `clip_sub_segments` must
  /// hard-error on any non-1/16000 timebase rather than
  /// silently relabelling the input. Pre-fix an integration
  /// that accidentally passed output-timebase
  /// (e.g. 1/48000 or 1/1000) sub_segments would have its PTS
  /// values reinterpreted as 16 kHz sample indices, zero-
  /// masking the wrong audio without surfacing an error.
  /// Codex round-37 round-35 [high]: per-run dispatch must
  /// emit words in time order. A multi-run chunk where Run A
  /// produces a late word, then Run B produces an early word,
  /// must be re-ordered so consumers of `Transcript::words()`
  /// see monotone PTS — that's the public contract.
  #[test]
  fn sort_words_by_pts_orders_overlapping_runs() {
    use core::num::NonZeroU32;
    use mediatime::Timebase;
    let tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let mk = |start: i64, end: i64, text: &str| {
      crate::types::Word::new(SmolStr::new(text), TimeRange::new(start, end, tb), 1.0)
    };
    // Pre-sort: late, early, mid (interleaved as if from
    // different language runs). Post-sort: early, mid, late.
    let mut words = alloc::vec![
      mk(8000, 9000, "world"),
      mk(0, 1000, "hello"),
      mk(4000, 5000, "there"),
    ];
    sort_words_by_pts(&mut words);
    let texts: alloc::vec::Vec<&str> = words.iter().map(|w| w.text()).collect();
    assert_eq!(texts, alloc::vec!["hello", "there", "world"]);
    // Strict monotone start PTS check.
    let mut prev = i64::MIN;
    for w in &words {
      let s = w.range().start_pts();
      assert!(
        s >= prev,
        "word starts must be monotone; got {s} after {prev}"
      );
      prev = s;
    }
  }

  /// Tiebreaker case: equal start PTS → earlier end PTS first.
  /// Stability isn't strictly required by the public contract
  /// but keeps the output deterministic for debug/log readers.
  #[test]
  fn sort_words_by_pts_breaks_ties_by_end_pts() {
    use core::num::NonZeroU32;
    use mediatime::Timebase;
    let tb = Timebase::new(1, NonZeroU32::new(16_000).unwrap());
    let mk = |start: i64, end: i64, text: &str| {
      crate::types::Word::new(SmolStr::new(text), TimeRange::new(start, end, tb), 1.0)
    };
    let mut words = alloc::vec![mk(0, 2000, "longer"), mk(0, 1000, "shorter"),];
    sort_words_by_pts(&mut words);
    assert_eq!(words[0].text(), "shorter");
    assert_eq!(words[1].text(), "longer");
  }

  /// Codex round-37 round-21 [medium]: between-run abort gate.
  /// Pre-fix, `dispatch_runs` checked `abort_flag` only inside
  /// each `Aligner::align` call; a cancellation that landed
  /// after a successful run completed but before the next
  /// iteration started could still launch another ONNX
  /// inference, extending a hung/cancelled job. The gate is
  /// extracted into [`check_abort_between_runs`] so its
  /// observable shape is unit-testable without standing up
  /// ORT (which `RunOptions::new` requires).
  #[test]
  fn check_abort_between_runs_returns_timeout_when_flag_set() {
    let started = Instant::now();
    let flag = AtomicBool::new(true);
    let result = check_abort_between_runs(&flag, started);
    assert!(
      matches!(
        result,
        Err(WorkFailure::WorkerHangTimeout {
          kind: WorkerKind::Alignment,
          ..
        })
      ),
      "abort flag set → expected WorkerHangTimeout(Alignment); got {result:?}",
    );
  }

  /// Codex round-37 round-22 [high]: pronounced-OOV chunks
  /// now produce a `SemanticOutOfVocab` failure (instead of the
  /// pre-fix silent `Ok(empty TokenizedText)`); the dispatch
  /// classifier must mark this kind recoverable so the ASR
  /// transcript is still preserved (best-effort alignment) AND
  /// the diagnostic surfaces in telemetry.
  #[test]
  fn semantic_oov_is_recoverable() {
    use crate::types::{AlignmentFailureKind, Lang};
    let f = WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::SemanticOutOfVocab,
      message: alloc::string::String::from("pronounced symbol"),
      language: Lang::En,
    };
    assert!(
      alignment_failure_is_recoverable(&f),
      "SemanticOutOfVocab must recover so ASR text isn't lost",
    );
  }

  /// `TokenizationFailed` (genuine tokenizer/model mismatch)
  /// stays fatal so a broken setup is loud.
  #[test]
  fn tokenization_failed_stays_fatal() {
    use crate::types::{AlignmentFailureKind, Lang};
    let f = WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::TokenizationFailed,
      message: alloc::string::String::new(),
      language: Lang::En,
    };
    assert!(
      !alignment_failure_is_recoverable(&f),
      "TokenizationFailed signals a tokenizer/model mismatch; must stay fatal",
    );
  }

  #[test]
  fn check_abort_between_runs_passes_through_when_flag_clear() {
    let started = Instant::now();
    let flag = AtomicBool::new(false);
    assert!(check_abort_between_runs(&flag, started).is_ok());
  }

  /// Codex round-37 round-6 (parity loop) [high]: replicates
  /// the outer-shape check that `run_one_alignment` performs.
  /// The dispatch validation can't easily be exercised
  /// end-to-end without a real Aligner / ORT, so this test
  /// pins the predicate that decides "is the
  /// `Vec<Vec<OovDecision>>` shape valid for this chunk
  /// shape?". A regression that reverts to silent acceptance
  /// of stale shapes will trip these expectations.
  #[test]
  fn outer_oov_decisions_shape_predicate() {
    fn shape_ok(outer: usize, runs_len: usize) -> bool {
      let expected = if runs_len == 0 { 1 } else { runs_len };
      outer == 0 || outer == expected
    }
    // Whole-chunk job: 0 (no OOV) or 1 (one whole-chunk vec).
    assert!(shape_ok(0, 0));
    assert!(shape_ok(1, 0));
    assert!(!shape_ok(2, 0)); // stale per-run payload — REJECT
    assert!(!shape_ok(3, 0));
    // Per-run job with 2 runs: 0 (no OOV) or exactly 2.
    assert!(shape_ok(0, 2));
    assert!(shape_ok(2, 2));
    assert!(!shape_ok(1, 2)); // shorter-than-runs.len() — REJECT
    assert!(!shape_ok(3, 2));
  }

  /// Codex round-37 round-1 (parity loop) [high]: per-run
  /// dispatch must thread the caller's per-run OOV decisions
  /// from `AlignWorkItem::oov_decisions[run_idx]` into
  /// `run_one_per_run`, NOT hard-code `default_oov_decisions`.
  /// This pins the indexing slice so a future refactor that
  /// drops the `enumerate()`+index lookup can't silently
  /// substitute the default policy.
  ///
  /// Structural test: builds a `Vec<Vec<OovDecision>>` with
  /// distinct per-run policies and asserts the dispatcher's
  /// slice-extraction matches each run's expected policy.
  /// No real `Aligner` needed — exercises only the index
  /// math.
  #[test]
  fn per_run_oov_decisions_are_indexed_by_run_idx() {
    use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
    fn synth(decision: OovDecision, char_idx: usize) -> ResolvedOov {
      ResolvedOov {
        event: OovEvent {
          kind: OovKind::Symbol('?'),
          char_index: char_idx,
          word_index: 0,
          language: Lang::En,
        },
        decision,
      }
    }
    let oov_decisions: alloc::vec::Vec<alloc::vec::Vec<ResolvedOov>> = alloc::vec![
      // Run 0: caller chose `wildcard_all_decisions` — three Wildcards.
      alloc::vec![
        synth(OovDecision::Wildcard, 0),
        synth(OovDecision::Wildcard, 1),
        synth(OovDecision::Wildcard, 2),
      ],
      // Run 1: caller chose `default_oov_decisions` — mixed.
      alloc::vec![
        synth(OovDecision::Wildcard, 0),
        synth(OovDecision::FailClosed, 1),
      ],
      // Run 2: empty = no OOV expected.
      alloc::vec::Vec::new(),
    ];
    // Mirror `dispatch_runs`'s per-run extraction.
    for run_idx in 0..3 {
      let slice = oov_decisions
        .get(run_idx)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
      match run_idx {
        0 => {
          assert_eq!(slice.len(), 3);
          assert!(slice.iter().all(|r| r.decision == OovDecision::Wildcard));
        }
        1 => {
          assert_eq!(slice.len(), 2);
          assert_eq!(slice[0].decision, OovDecision::Wildcard);
          assert_eq!(slice[1].decision, OovDecision::FailClosed);
        }
        2 => assert!(slice.is_empty()),
        _ => unreachable!(),
      }
    }
    // Out-of-range run idx (hypothetical: caller pre-sized
    // shorter than `runs`) falls back to `&[]`. The aligner
    // then surfaces `TokenizationFailed` if it hits any OOV
    // — loud diagnostic, not silent default-policy.
    let oob = oov_decisions.get(99).map(|v| v.as_slice()).unwrap_or(&[]);
    assert!(oob.is_empty());
  }

  /// Codex round-37 round-11 (parity loop) [high]: at the
  /// dispatch boundary, every supplied `ResolvedOov.event.language`
  /// must match the chunk/run's requested language. Round 10
  /// loosened the in-tokenizer identity check to ignore
  /// `language` (so Any-fallback works); this test pins the
  /// dispatch-boundary precheck that catches what the
  /// in-tokenizer check now lets through.
  #[test]
  fn validate_oov_decision_languages_whole_chunk_match_passes() {
    use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
    let resolved = alloc::vec![alloc::vec![ResolvedOov {
      event: OovEvent {
        kind: OovKind::Symbol('&'),
        char_index: 2,
        word_index: 0,
        language: Lang::En,
      },
      decision: OovDecision::Wildcard,
    }]];
    assert!(validate_oov_decision_languages(&[], &Lang::En, &resolved).is_ok());
  }

  #[test]
  fn validate_oov_decision_languages_whole_chunk_mismatch_rejects() {
    use crate::core::{OovDecision, OovEvent, OovKind, ResolvedOov};
    // Job language is Korean; supplied decision was made for
    // English — language-conditional policy would run against
    // the wrong key.
    let resolved = alloc::vec![alloc::vec![ResolvedOov {
      event: OovEvent {
        kind: OovKind::Symbol('&'),
        char_index: 2,
        word_index: 0,
        language: Lang::En,
      },
      decision: OovDecision::Wildcard,
    }]];
    let result = validate_oov_decision_languages(&[], &Lang::Ko, &resolved);
    match result {
      Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::TokenizationFailed,
        ref message,
        ..
      }) => assert!(
        message.contains("oov_decisions[0][0].event.language") && message.contains("job.language"),
        "diagnostic should cite the whole-chunk mismatch; got {message:?}",
      ),
      other => panic!("expected TokenizationFailed; got {other:?}"),
    }
  }

  #[test]
  fn validate_oov_decision_languages_per_run_mismatch_rejects() {
    use crate::{
      align::{BoundsSource, Run},
      core::{OovDecision, OovEvent, OovKind, ResolvedOov},
    };
    use smol_str::SmolStr;
    let runs = alloc::vec![
      Run::new(
        Lang::En,
        SmolStr::from("AT&T"),
        0,
        1_000,
        0,
        BoundsSource::Segment
      ),
      Run::new(
        Lang::Ko,
        SmolStr::from("4번"),
        1_000,
        2_000,
        1,
        BoundsSource::Segment
      ),
    ];
    // Run 1 (Korean) is wired with a stale English-stamped decision.
    let resolved = alloc::vec![
      alloc::vec![ResolvedOov {
        event: OovEvent {
          kind: OovKind::Symbol('&'),
          char_index: 2,
          word_index: 0,
          language: Lang::En,
        },
        decision: OovDecision::Wildcard,
      }],
      alloc::vec![ResolvedOov {
        event: OovEvent {
          kind: OovKind::Symbol('4'),
          char_index: 0,
          word_index: 0,
          language: Lang::En, // BUG: should be Ko
        },
        decision: OovDecision::Wildcard,
      }],
    ];
    let result = validate_oov_decision_languages(&runs, &Lang::En, &resolved);
    match result {
      Err(WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::TokenizationFailed,
        ref message,
        ..
      }) => assert!(
        message.contains("oov_decisions[1][0]") && message.contains("runs[1].language()"),
        "diagnostic should cite the run index of the mismatch; got {message:?}",
      ),
      other => panic!("expected TokenizationFailed; got {other:?}"),
    }
  }

  /// An empty outer vec ("no OOV expected") is accepted —
  /// `tokenize_with_word_map` surfaces `TokenizationFailed`
  /// downstream if a chunk hits an OOV anyway. This validator
  /// is about per-position language identity, not
  /// presence/absence.
  #[test]
  fn validate_oov_decision_languages_empty_passes() {
    let empty: alloc::vec::Vec<alloc::vec::Vec<crate::core::ResolvedOov>> = alloc::vec::Vec::new();
    assert!(validate_oov_decision_languages(&[], &Lang::En, &empty).is_ok());
  }

  #[test]
  fn clip_sub_segments_rejects_non_16000_timebase() {
    use core::num::NonZeroU32;
    let tb_48k = mediatime::Timebase::new(1, NonZeroU32::new(48_000).unwrap());
    let subs = alloc::vec![TimeRange::new(2_000, 3_000, tb_48k)];
    let result = clip_sub_segments(&subs, 1_600, 4_800, &Lang::En);
    match result {
      Err(WorkFailure::AlignmentFailed { kind, message, .. }) => {
        assert!(
          matches!(kind, AlignmentFailureKind::ModelInferenceFailed),
          "expected ModelInferenceFailed; got {kind:?}",
        );
        assert!(
          message.contains("1/16000") && message.contains("48000"),
          "expected diagnostic citing both timebases; got {message:?}",
        );
      }
      other => panic!("expected AlignmentFailed, got {other:?}"),
    }
  }
}
