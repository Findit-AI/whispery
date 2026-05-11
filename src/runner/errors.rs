//! Runner-level error type.

use core::time::Duration;

use smol_str::SmolStr;

use crate::types::TranscriberError;

/// Runner-level structural failure.
///
/// Distinguished from [`crate::WorkFailure`], which is per-chunk
/// inference failure surfaced asynchronously via `Event::Error`.
/// `RunnerError` is returned synchronously from
/// [`crate::runner::ManagedTranscriber::process_packet`],
/// `handle_eof`, `drain`, the builder's `build`, and (with the
/// `alignment` feature) `Aligner::from_paths`.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
  /// `WhisperContext::new_with_params` failed at builder time.
  /// No worker threads were spawned.
  #[error("failed to load whisper context: {message}")]
  WhisperContextLoad {
    /// Verbatim error from whisper-rs.
    message: SmolStr,
  },

  /// `Aligner::from_paths` failed at builder time. The wav2vec2
  /// ONNX model or tokenizer.json could not be loaded; no
  /// alignment workers were spawned. The aligner-bearing builder
  /// (`ManagedTranscriberBuilder::with_alignment`) returns this
  /// from `build()`.
  ///
  /// Two common causes: (1) `model_path` does not exist or is
  /// not a valid ONNX graph; (2) `tokenizer_path` does not exist
  /// or is not a valid HuggingFace `tokenizer.json`. The verbatim
  /// upstream error string is in `message`.
  ///
  /// Gated on `feature = "alignment"`.
  #[cfg(feature = "alignment")]
  #[error("failed to load aligner: {message}")]
  AlignerLoad {
    /// Verbatim error from `ort` or `tokenizers`.
    message: SmolStr,
  },

  /// A worker channel is disconnected — typically because a worker
  /// thread panicked. Fatal; rebuild the `ManagedTranscriber`.
  #[error("whisper pool shutdown (worker channel disconnected)")]
  WhisperPoolShutdown,

  /// Worker queue is full and `WhisperPoolOptions::block_on_full_queue`
  /// is `false`. The caller must drain via `poll_transcript` /
  /// `poll_error` before pushing more audio.
  ///
  /// **Side-effect contract:** when this is returned from
  /// `process_packet`, the input *was already consumed* — the
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
