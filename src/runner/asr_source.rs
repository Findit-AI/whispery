//! [`AsrSource`] — pluggable, thread-free ASR backend trait.
//!
//! Asry is positioned as Sans-I/O: the library exposes
//! sync compute primitives, and threading / cancellation /
//! lifecycle is the caller's responsibility. `AsrSource` is
//! the canonical primitive for "give me a chunk of audio,
//! produce an [`AsrResult`]".
//!
//! Two intended consumers:
//!
//! - **Sync users** (CLI tools, batch pipelines) construct
//!   one [`WhisperAsrSource`] and call
//!   [`AsrSource::run_chunk`] inline as they pull commands
//!   from [`crate::core::Transcriber`].
//! - **Async users** (tokio, smol, etc.) own their runtime;
//!   they construct one `WhisperAsrSource` per concurrent
//!   inference slot, call `run_chunk` from `spawn_blocking`
//!   (or equivalent), and wire shutdown via their own
//!   cancellation tokens flipping the supplied
//!   `abort_flag`.
//!
//! Whisper.cpp's `set_abort_callback` already polls the flag
//! at progress-callback boundaries; flipping it from the
//! caller's runtime causes the in-flight inference to unwind
//! at the next callback. No asry-side threads are spawned.

use std::sync::{Arc, atomic::AtomicBool};

use crate::{
  core::{AsrParams, AsrResult},
  types::{AsrError, AsrFailure, ChunkId, WorkFailure},
};

#[cfg(feature = "runner")]
use smol_str::format_smolstr;
#[cfg(feature = "runner")]
use whispercpp::{Context as WhisperContext, State as WhisperState};

/// Pluggable ASR backend.
///
/// One instance is owned per worker / per concurrent
/// inference. The trait is `Send` so async users can move it
/// into a `spawn_blocking` task; it is **not** `Sync` because
/// most ASR engines (whisper.cpp, faster-whisper, parakeet)
/// own per-thread mutable state and callers must not share a
/// single instance across concurrent inferences.
///
/// Implementations are responsible for:
///
/// - Producing transcript text + per-segment metadata.
/// - Populating [`AsrResult::runs`](AsrResult) when the
///   backend has access to per-token DTW timestamps; an
///   empty `runs` falls back to whole-chunk alignment.
/// - Honouring `abort_flag` for cooperative cancellation —
///   the caller's runtime flips the flag, the backend
///   returns at the next safe point.
pub trait AsrSource: Send {
  /// Run ASR on one chunk of audio. Blocks until inference
  /// completes or `abort_flag` is flipped.
  fn run_chunk(&mut self, chunk: AsrChunkContext<'_>) -> Result<AsrResult, WorkFailure>;
}

/// Per-chunk inputs for [`AsrSource::run_chunk`]. Borrowed
/// references so the impl can hold them on the stack.
pub struct AsrChunkContext<'a> {
  /// 16 kHz f32 mono audio.
  samples: &'a [f32],
  /// ASR knobs (language hint, temperature ladder,
  /// thresholds, etc.).
  params: &'a AsrParams,
  /// Cooperative-cancellation flag the impl polls at coarse
  /// boundaries. Flipping this from the caller's runtime
  /// unwinds in-flight inference (whisper.cpp re-checks via
  /// the abort callback wired into `Params`).
  abort_flag: &'a Arc<AtomicBool>,
  /// Caller-supplied chunk identity — surfaces back into
  /// [`WorkFailure::Asr`](crate::types::WorkFailure::Asr) /
  /// [`WorkFailure::WorkerHang`](crate::types::WorkFailure::WorkerHang)
  /// telemetry. Asry does not assign chunk ids itself.
  chunk_id: ChunkId,
}

impl<'a> AsrChunkContext<'a> {
  /// Construct from the four per-chunk inputs.
  #[must_use]
  pub const fn new(
    samples: &'a [f32],
    params: &'a AsrParams,
    abort_flag: &'a Arc<AtomicBool>,
    chunk_id: ChunkId,
  ) -> Self {
    Self {
      samples,
      params,
      abort_flag,
      chunk_id,
    }
  }

  /// 16 kHz f32 mono audio.
  #[must_use]
  pub const fn samples(&self) -> &'a [f32] {
    self.samples
  }

  /// ASR knobs for this chunk.
  #[must_use]
  pub const fn params(&self) -> &'a AsrParams {
    self.params
  }

  /// Cooperative-cancellation flag for this chunk.
  #[must_use]
  pub const fn abort_flag(&self) -> &'a Arc<AtomicBool> {
    self.abort_flag
  }

  /// Caller-supplied chunk identity.
  #[must_use]
  pub const fn chunk_id(&self) -> ChunkId {
    self.chunk_id
  }
}

/// Default [`AsrSource`] impl backed by whisper.cpp via the
/// `whispercpp` crate. Owns one [`WhisperState`] per
/// instance; users construct N of these for N parallel
/// inference slots.
///
/// **No internal threads.** Asry wires whisper.cpp's
/// abort callback to the caller-supplied `abort_flag`; the
/// caller's runtime is responsible for flipping the flag
/// (timer, cancellation token, signal handler, etc.).
#[cfg(feature = "runner")]
pub struct WhisperAsrSource {
  ctx: Arc<WhisperContext>,
  state: WhisperState,
}

#[cfg(feature = "runner")]
impl WhisperAsrSource {
  /// Construct a `WhisperAsrSource` from a shared
  /// [`WhisperContext`]. Allocates a fresh
  /// [`WhisperState`] tied to the context.
  ///
  /// Returns [`crate::runner::RunnerError::WhisperContextLoad`]
  /// when state allocation fails (poisoned context, OOM).
  pub fn new(ctx: Arc<WhisperContext>) -> Result<Self, crate::runner::RunnerError> {
    let state = ctx
      .create_state()
      .map_err(|e| crate::runner::RunnerError::WhisperContextLoad {
        message: format_smolstr!("create_state failed: {e:?}"),
      })?;
    Ok(Self { ctx, state })
  }

  /// Borrow the underlying [`WhisperContext`]. Useful for
  /// callers that want to share the model across multiple
  /// `WhisperAsrSource` instances or query model metadata.
  pub fn context(&self) -> &Arc<WhisperContext> {
    &self.ctx
  }
}

#[cfg(feature = "runner")]
impl AsrSource for WhisperAsrSource {
  fn run_chunk(&mut self, chunk: AsrChunkContext<'_>) -> Result<AsrResult, WorkFailure> {
    use crate::runner::whisper_pool::{
      AsrWorkItem, run_with_temperature_ladder, validate_for_whisper_ffi,
    };

    validate_for_whisper_ffi(chunk.params())?;
    // scan samples for
    // finiteness before they reach `state.full`. Without this
    // a single NaN/Inf in public audio poisons the encoder's
    // float math and surfaces as an opaque whisper.cpp backend
    // failure (or worse, contaminates downstream attention
    // matrices into producing plausible-looking nonsense).
    // The alignment path has the equivalent guard
    // (`Aligner::align_chunk_with_abort`); ASR was the only
    // public ingestion path missing it. Reject as
    // `::BackendError` with the failing index;
    // the audio sample value itself is not logged because
    // floats are not user-content but the index is enough to
    // localise the bug.
    if let Some((idx, val)) = chunk
      .samples()
      .iter()
      .copied()
      .enumerate()
      .find(|(_, s)| !s.is_finite())
    {
      return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
        format_smolstr!(
          "non-finite ASR sample at index {idx} (value {val:?}); upstream audio corruption — \
           refuse to encode rather than poison whisper.cpp's float math"
        ),
      ))));
    }
    let job = AsrWorkItem {
      chunk_id: chunk.chunk_id(),
      samples: Arc::<[f32]>::from(chunk.samples().to_vec()),
      params: chunk.params().clone(),
      // No internal watchdog — caller drives cancellation via
      // `abort_flag`. The legacy worker watchdog used this
      // field to prefer `WorkerHangTimeout` over `BackendError`
      // when its own timer fired; here the user owns that
      // distinction. Stamp a sentinel duration; downstream
      // code only consults this when the legacy worker
      // wraps it (and that path is going away).
      abort_flag: chunk.abort_flag().clone(),
    };
    let started_at = std::time::Instant::now();
    run_with_temperature_ladder(&mut self.state, &job, started_at)
  }
}

#[cfg(test)]
#[cfg(feature = "runner")]
mod tests {
  use super::*;

  fn assert_send<T: Send>() {}

  #[test]
  fn whisper_asr_source_is_send() {
    assert_send::<WhisperAsrSource>();
  }

  #[test]
  fn asr_source_is_object_safe() {
    // Compile-time check: `AsrSource` is dyn-safe so async
    // users can hold `Box<dyn AsrSource>` per worker without
    // monomorphising the runtime over backend choice.
    fn _take(_: &mut dyn AsrSource) {}
  }
}
