//! Public error types.
//!
//! Two distinct error channels:
//!
//! - [`TranscriberError`] is for state-machine push/inject failures
//!   returned synchronously from `Transcriber::push_*` /
//!   `inject_*` / `handle_restart`.
//! - [`WorkFailure`] is for per-chunk inference failures surfaced
//!   asynchronously via `Event::Error { chunk_id, error: WorkFailure }`
//!   (drained by `poll_event`).

use core::time::Duration;

use crate::types::{ChunkId, Lang};

/// Push or inject failure on the state machine.
#[derive(Clone, Debug, thiserror::Error)]
pub enum TranscriberError {
  /// PTS regression: caller pushed samples or a VAD segment with a
  /// timestamp earlier than the current high-water mark. The
  /// check runs in output-PTS space (not 16 kHz space) to avoid
  /// spurious regressions on non-integer-ratio output timebases.
  #[error("PTS regression on {kind:?}: advance = {advance}")]
  PtsRegression {
    /// Which input kind regressed.
    kind: PushKind,
    /// Negative delta in output-timebase PTS units.
    advance: i64,
  },

  /// Forward gap exceeds the configured tolerance. Caller likely
  /// has a stream restart or a packet drop larger than expected.
  /// Recover via `Transcriber::handle_restart`.
  #[error("forward gap {gap_samples} samples exceeds tolerance {tolerance_samples}")]
  GapExceedsTolerance {
    /// Size of the forward gap in 16 kHz samples.
    gap_samples: u64,
    /// Currently configured tolerance.
    tolerance_samples: u64,
  },

  /// Sample buffer would exceed its configured cap. The runner has
  /// not drained completed chunks fast enough; the caller should
  /// pause and call `poll_event` until the buffer trims.
  #[error("sample buffer at capacity ({buffered}/{cap})")]
  Backpressure {
    /// Buffered sample count after this push attempt would have
    /// committed.
    buffered: usize,
    /// Configured cap.
    cap: usize,
  },

  /// `handle_vad_segment` was called before any `handle_samples`. The
  /// output timebase is not yet established.
  #[error("handle_vad_segment called before any handle_samples")]
  OutputTimebaseUnset,

  /// `handle_vad_segment` referenced audio that has not been
  /// buffered yet — `seg.end_sample()` is past the buffer's
  /// high-water mark. This typically indicates upstream
  /// VAD/sample skew (the caller pushed VAD ahead of the
  /// corresponding audio packets). Without this guard, a later
  /// `handle_eof` flush or chunk emission would `extract` past
  /// the buffer's tail and panic.
  #[error("VAD segment end {vad_end} is past buffered samples {buffered}")]
  VadAheadOfAudio {
    /// `seg.end_sample()` value the caller passed in.
    vad_end: u64,
    /// `buffer.absolute_sample_offset()` at the time of the push.
    buffered: u64,
  },

  /// `handle_samples` was called with a `Timestamp` whose timebase
  /// does not match the timebase recorded from the first push.
  #[error("inconsistent output timebase: expected {expected:?}, got {got:?}")]
  InconsistentTimebase {
    /// Expected output timebase (recorded on first push).
    expected: mediatime::Timebase,
    /// Caller-supplied timebase that did not match.
    got: mediatime::Timebase,
  },

  /// Caller passed a `Timestamp` whose timebase has a zero
  /// numerator. `mediatime::Timebase::new(0, _)` is constructible
  /// (the type only enforces non-zero denominator), and using it
  /// as the target of `Timebase::rescale_pts` would panic. The
  /// public boundary rejects it explicitly so a malformed caller
  /// timebase surfaces as `Err(_)` instead of an abort.
  #[error("timebase numerator must be non-zero (got {numerator})")]
  InvalidTimebase {
    /// The offending zero (or otherwise rejected) numerator.
    numerator: u32,
  },

  /// Caller `inject_*`-ed a chunk_id that does not match an in-flight
  /// chunk.
  #[error("unknown or already-resolved chunk_id {0}")]
  UnknownChunk(ChunkId),

  /// Caller called `handle_eof` and then attempted to push or
  /// `handle_restart`. Once a stream is ended it cannot be re-anchored;
  /// construct a fresh `Transcriber` instead.
  #[error("operation rejected after handle_eof")]
  AfterEof,
}

/// Per-chunk inference failure surfaced via `Event::Error`.
#[derive(Clone, Debug, thiserror::Error)]
pub enum WorkFailure {
  /// ASR (whisper) inference failed.
  #[error("ASR failed: {message}")]
  AsrFailed {
    /// Failure category.
    kind: AsrFailureKind,
    /// Human-readable detail (typically the backend's error text).
    message: alloc::string::String,
  },

  /// Word-level forced alignment failed.
  #[error("alignment failed for language {language:?}: {message}")]
  AlignmentFailed {
    /// Failure category.
    kind: AlignmentFailureKind,
    /// Human-readable detail.
    message: alloc::string::String,
    /// Language whose aligner failed.
    language: Lang,
  },

  /// No aligner registered for the chunk's language and the
  /// fallback policy is `Error`.
  #[error("no aligner registered for language {language:?}")]
  LanguageUnsupportedForAlignment {
    /// Detected language without a registered aligner.
    language: Lang,
  },

  /// Worker exceeded its per-job timeout.
  #[error("{kind:?} worker hung; elapsed {elapsed:?}")]
  WorkerHangTimeout {
    /// Which worker timed out.
    kind: WorkerKind,
    /// Time spent on the failed job.
    elapsed: Duration,
  },
}

/// Why an ASR inference failed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AsrFailureKind {
  /// All temperatures in the runner's retry ladder were tried and
  /// every result violated the log-prob or compression-ratio
  /// thresholds.
  AllTemperaturesFailed,
  /// Auto-detected language is not in whisper.cpp's supported set.
  UnsupportedLanguage,
  /// Backend returned an error during inference.
  BackendError,
}
// Note: there is no `EmptyOutput` variant. A whisper-rs result with
// zero segments is normal output — usually a silent chunk — and is
// represented as a `Transcript` with empty `text` and an elevated
// `no_speech_prob`. Treating empty output as a failure would convert
// every silent chunk into Event::Error and contradict the
// `no_speech_prob` field's semantics.

/// Why a word-level alignment failed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AlignmentFailureKind {
  /// wav2vec2 ONNX inference failed.
  ModelInferenceFailed,
  /// Tokenization of the normalised text against the wav2vec2
  /// vocab failed.
  TokenizationFailed,
  /// Text normalisation step failed.
  NormalizationFailed,
  /// CTC Viterbi found no valid alignment path.
  NoAlignmentPath,
  /// Whisper text was empty after normalisation.
  EmptyText,
  /// A pronounced symbol the model cannot honestly align (e.g.
  /// `&` in `AT&T`, `@`, `%`) appeared in the chunk and is not
  /// covered by the wildcard policy. The chunk's word-level
  /// alignment is dropped (the dispatch emits `Transcript {
  /// text, words: [] }`), but the failure is **observable** —
  /// callers can see this kind in telemetry rather than the
  /// pre-fix silent empty-`AlignmentResult` path. Distinct
  /// from `TokenizationFailed`, which is reserved for genuine
  /// tokenizer/model mismatches that are not recoverable.
  /// (Codex round-37 round-22 [high].)
  SemanticOutOfVocab,
}

/// Which input kind triggered a `PtsRegression`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PushKind {
  /// `handle_samples`.
  Samples,
  /// `handle_vad_segment`.
  VadSegment,
}

/// Which worker timed out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WorkerKind {
  /// ASR (whisper) worker.
  Asr,
  /// Alignment (wav2vec2) worker.
  Alignment,
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::string::ToString;

  #[test]
  fn pts_regression_displays_kind() {
    let e = TranscriberError::PtsRegression {
      kind: PushKind::Samples,
      advance: -100,
    };
    let s = e.to_string();
    assert!(s.contains("Samples"));
    assert!(s.contains("-100"));
  }

  #[test]
  fn work_failure_clones() {
    let f = WorkFailure::AsrFailed {
      kind: AsrFailureKind::AllTemperaturesFailed,
      message: "oops".into(),
    };
    let _ = f.clone();
  }
}
