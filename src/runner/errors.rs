//! Runner-level error type. See spec §4.5 / §9.

use core::time::Duration;

use crate::types::TranscriberError;

/// Runner-level structural failure.
///
/// Distinguished from [`crate::WorkFailure`], which is per-chunk
/// inference failure surfaced asynchronously via `Event::Error`.
/// `RunnerError` is returned synchronously from
/// [`crate::runner::ManagedTranscriber::process_packet`],
/// `signal_eof`, `drain`, and the builder's `build`.
///
/// See spec §4.5 / §9.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
  /// `WhisperContext::new_with_params` failed at builder time.
  /// No worker threads were spawned.
  #[error("failed to load whisper context: {message}")]
  WhisperContextLoad {
    /// Verbatim error from whisper-rs.
    message: alloc::string::String,
  },

  /// A worker channel is disconnected — typically because a worker
  /// thread panicked. Fatal; rebuild the `ManagedTranscriber`.
  #[error("whisper pool shutdown (worker channel disconnected)")]
  WhisperPoolShutdown,

  /// Worker queue is full and `WhisperPoolConfig::block_on_full_queue`
  /// is `false`. The caller must drain via `poll_transcript` /
  /// `poll_error` before pushing more audio.
  ///
  /// **Side-effect contract (spec §6.4.2):** when this is returned
  /// from `process_packet`, the input *was already consumed* — the
  /// caller must not retry the same call with the same arguments.
  #[error("backpressure: buffer at {buffered}/{cap} samples")]
  Backpressure {
    /// Currently buffered samples.
    buffered: usize,
    /// Configured `buffer_cap_samples`.
    cap: usize,
  },

  /// `drain()` exceeded the configured `drain_timeout` without
  /// reaching `core.is_idle()`. Typically indicates a hung worker
  /// (which should also surface a `WorkerHangTimeout` per chunk).
  #[error("drain exceeded {timeout:?} with {in_flight} chunks still in flight")]
  DrainTimeout {
    /// Configured drain timeout.
    timeout: Duration,
    /// Snapshot of chunks still awaiting results when the timeout fired.
    in_flight: usize,
  },

  /// I/O error while loading the model file.
  #[error("model I/O: {0}")]
  Io(#[from] std::io::Error),

  /// Wraps a [`TranscriberError`] from the underlying state machine
  /// so the runner's API exposes a single error type. `process_packet`
  /// converts every push/inject error from the core into this variant.
  #[error("transcriber: {0}")]
  Transcriber(#[from] TranscriberError),
}
