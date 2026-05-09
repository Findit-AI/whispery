//! Whisper worker pool.
//!
//! This file used to wrap `whisper-rs`. It was migrated to the
//! in-house `whisper-cpp` bindings crate (under `crates/whisper-cpp/`)
//! after we reproduced two soundness/leak bugs in `whisper-rs 0.16`:
//!
//! 1. `set_abort_callback_safe` UB (closure-type vs.
//!    `Box<dyn FnMut>` mismatch in the trampoline) → manifests as
//!    `whisper_full_with_state: failed to encode` on every
//!    decode. Structurally absent in `whisper-cpp` —
//!    `Params::set_abort_callback` types the trampoline as
//!    `*mut Box<dyn FnMut() -> bool>` end-to-end.
//! 2. `set_language` / `set_initial_prompt` `CString::into_raw`
//!    leak (no `Drop`). Structurally absent — `whisper-cpp`'s
//!    `Params` owns every `CString` it hands to whisper.cpp and
//!    drops them with the struct.
//!
//! Because both bug classes are gone, this file no longer needs
//! the `FullParamsCache`, the `attach_abort_callback` workaround,
//! or any `#![allow(unsafe_code)]` exemption. The `unsafe`
//! surface lives entirely inside the `whisper-cpp` crate.

use alloc::sync::Arc;
use core::{sync::atomic::Ordering, time::Duration};
use std::{
  path::{Path, PathBuf},
  sync::atomic::AtomicBool,
  thread::JoinHandle,
};

use crossbeam_channel::{Receiver, Sender, bounded};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use whispercpp::{
  Context as WhisperContext, ContextParams as WhisperContextParameters, Params as FullParams,
  SamplingStrategy as WhisperStrategy, State as WhisperState,
};

use smol_str::SmolStr;

use crate::{
  core::{AsrParams, AsrResult, SamplingStrategy},
  runner::RunnerError,
  types::{AsrFailureKind, ChunkId, Lang, WorkFailure, WorkerKind},
};

use std::{
  collections::HashMap,
  sync::{LazyLock, Mutex},
};

/// Process-wide intern table for `&'static str` representations of
/// language codes passed to `whisper-rs`'s `FullParams::set_language`
/// (which requires the borrow's lifetime to match `FullParams`'s, so
/// our `FullParams<'static, 'static>` return type forces a `'static`
/// `&str`).
///
/// A naive `Box::leak(Box::<str>::from(s))` in `full_params_from`
/// would allocate a fresh leak on every chunk attempt — bounded
/// *per call* but unbounded over a long-running stream. With
/// auto-lock, a 24-hour transcription can leak tens of thousands
/// of identical `"en"` strings.
///
/// This intern table allocates+leaks at most **once per distinct
/// language code** the process ever sees. Named [`Lang`] variants
/// (≈ 100) cap the working set; `Lang::Other(...)` adds at most
/// the cardinality of unknown ISO codes a caller actually feeds in.
///
/// `Mutex` over `HashMap` is fine — the lock is held for at most a
/// hash lookup + a single allocation on first sight; ASR latency
/// dwarfs that.
static INTERNED_LANG_STRS: LazyLock<Mutex<HashMap<String, &'static str>>> =
  LazyLock::new(|| Mutex::new(HashMap::new()));

/// Return a `&'static str` for `s`, allocating + leaking once per
/// distinct value. Subsequent calls with the same `s` return the
/// same pointer — bounded leak, see [`INTERNED_LANG_STRS`].
///
/// **Caller contract**: pre-validate `s` via
/// [`validate_language_code`]. Without that, `Lang::Other(SmolStr)`
/// could feed unique adversarial strings here and grow the
/// intern table without bound (one leak per unique hint).
fn intern_lang_str(s: &str) -> &'static str {
  let mut map = INTERNED_LANG_STRS
    .lock()
    .expect("INTERNED_LANG_STRS mutex poisoned");
  if let Some(&interned) = map.get(s) {
    return interned;
  }
  let leaked: &'static str = Box::leak(Box::<str>::from(s));
  map.insert(s.to_string(), leaked);
  leaked
}

/// Maximum byte length accepted for a language hint. Whisper.cpp's
/// recognized set is 2-3 letter ISO codes; 8 bytes covers every
/// real code with comfortable headroom for future regional
/// variants while still bounding the intern table.
const MAX_LANGUAGE_CODE_LEN: usize = 8;

/// Validate a language hint before it's interned and shipped into
/// `whisper-rs::FullParams::set_language`.
///
/// Returns `Err(reason)` for an empty string, anything longer
/// than [`MAX_LANGUAGE_CODE_LEN`], or anything containing a byte
/// outside `[a-z]`. The reason is a `'static` slogan suitable
/// for inclusion in an in-band [`WorkFailure::AsrFailed`]
/// message; intentionally does NOT echo the offending bytes
/// back to the caller because the language hint can be set
/// from public input.
///
/// Codex round-32: prevents `Lang::Other(...)` from feeding
/// unique adversarial strings into [`intern_lang_str`], whose
/// leak-on-first-sight design has no eviction policy.
fn validate_language_code(s: &str) -> Result<(), &'static str> {
  if s.is_empty() {
    return Err("language code is empty");
  }
  if s.len() > MAX_LANGUAGE_CODE_LEN {
    return Err("language code longer than 8 bytes (whisper.cpp codes are 2–3 ASCII letters)");
  }
  if !s.bytes().all(|b| b.is_ascii_lowercase()) {
    return Err("language code must be lowercase ASCII letters [a-z] only");
  }
  Ok(())
}

/// Configuration for the runner's whisper worker pool.
///
/// Fields are private; use [`WhisperPoolOptions::new`] (or
/// [`Default::default`]) and the `set_*` / `with_*` accessors. Most
/// accessors are `const fn` and run in const contexts. Path-typed
/// fields (`model_path`) cannot be `const fn` because [`PathBuf`]
/// does not currently expose const accessors.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WhisperPoolOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_worker_count"))]
  worker_count: usize,
  // No `default = ...` because there is no sane default model path —
  // a deserialised config must specify one. `PathBuf` itself is
  // serde-friendly (`String` round-trip).
  model_path: PathBuf,
  #[cfg_attr(feature = "serde", serde(default))]
  use_gpu: bool,
  #[cfg_attr(feature = "serde", serde(default))]
  gpu_device: i32,
  #[cfg_attr(feature = "serde", serde(default))]
  flash_attn: bool,
  // `default_max_queued_chunks` mirrors `Self::new`'s
  // `worker_count + 4` heuristic without needing a reference to
  // `worker_count` (serde defaults can't see other fields). We
  // resolve the dependency conservatively at default-time using
  // the same `default_worker_count()`, so a partial config with a
  // bumped `worker_count` but no `max_queued_chunks` matches the
  // `Self::new` shape.
  #[cfg_attr(feature = "serde", serde(default = "default_max_queued_chunks"))]
  max_queued_chunks: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_block_on_full_queue"))]
  block_on_full_queue: bool,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_dispatch_idle_poll", with = "humantime_serde")
  )]
  dispatch_idle_poll: Duration,
  #[cfg_attr(feature = "serde", serde(default = "default_timeout_streak_threshold"))]
  timeout_streak_threshold: u32,
}

#[cfg(feature = "serde")]
fn default_max_queued_chunks() -> usize {
  default_worker_count() + 4
}
#[cfg(feature = "serde")]
const fn default_block_on_full_queue() -> bool {
  true
}
#[cfg(feature = "serde")]
const fn default_dispatch_idle_poll() -> Duration {
  Duration::from_millis(10)
}

impl WhisperPoolOptions {
  /// Construct a CPU-flavored config with the given model path.
  ///
  /// Defaults: `worker_count = max(1, physical_cores/2)`,
  /// `timeout_streak_threshold = 1`, `use_gpu = false`. For
  /// GPU-flavored defaults (single worker, looser timeout streak),
  /// use [`Self::new_for_gpu`] instead.
  pub fn new(model_path: impl Into<PathBuf>) -> Self {
    let worker_count = default_worker_count();
    Self {
      worker_count,
      model_path: model_path.into(),
      use_gpu: false,
      gpu_device: 0,
      flash_attn: false,
      max_queued_chunks: worker_count + 4,
      block_on_full_queue: true,
      dispatch_idle_poll: Duration::from_millis(10),
      timeout_streak_threshold: default_timeout_streak_threshold(),
    }
  }

  /// Construct a GPU-flavored config with the given model path.
  ///
  /// Defaults: `worker_count = 1` (whisper.cpp serialises on a
  /// single GPU regardless of concurrent `WhisperState`s, so
  /// extra workers oversubscribe rather than parallelise),
  /// `timeout_streak_threshold = 3` (GPU launches have higher
  /// per-call variance), `use_gpu = true`.
  ///
  /// Codex round-34: previously `WhisperPoolOptions::new` tried
  /// to detect the GPU backend via private cargo features that
  /// were never wired into `Cargo.toml`, so GPU users silently
  /// received CPU defaults (oversubscribing one GPU with multiple
  /// workers). Make the GPU intent explicit at construction
  /// instead of guessing from compile-time cfg.
  pub fn new_for_gpu(model_path: impl Into<PathBuf>) -> Self {
    let worker_count = gpu_worker_count();
    Self {
      worker_count,
      model_path: model_path.into(),
      use_gpu: true,
      gpu_device: 0,
      flash_attn: false,
      max_queued_chunks: worker_count + 4,
      block_on_full_queue: true,
      dispatch_idle_poll: Duration::from_millis(10),
      timeout_streak_threshold: gpu_timeout_streak_threshold(),
    }
  }

  /// Worker thread count. Default
  /// `max(1, num_cpus::get_physical() / 2)` from
  /// [`Self::new`]; `1` from [`Self::new_for_gpu`].
  pub const fn worker_count(&self) -> usize {
    self.worker_count
  }

  /// Path to the GGML/GGUF whisper model file.
  pub fn model_path(&self) -> &Path {
    &self.model_path
  }

  /// Forwarded to `WhisperContextParameters::use_gpu`. Default `false`.
  pub const fn use_gpu(&self) -> bool {
    self.use_gpu
  }

  /// Forwarded to `WhisperContextParameters::gpu_device`. Default `0`.
  pub const fn gpu_device(&self) -> i32 {
    self.gpu_device
  }

  /// Forwarded to `WhisperContextParameters::flash_attn`. Default `false`.
  /// Mutually exclusive with DTW (which is not enabled in v1).
  pub const fn flash_attn(&self) -> bool {
    self.flash_attn
  }

  /// Cap on the work_tx channel before saturation kicks in.
  /// Default `worker_count + 4`.
  pub const fn max_queued_chunks(&self) -> usize {
    self.max_queued_chunks
  }

  /// When `true` (default), `process_packet` blocks when the work
  /// channel is full. When `false`, surfaces
  /// [`crate::RunnerError::Backpressure`] for caller-side pacing.
  pub const fn block_on_full_queue(&self) -> bool {
    self.block_on_full_queue
  }

  /// Maximum time the saturation wait blocks on
  /// `Select::ready_timeout` before spinning. Default 10 ms.
  pub const fn dispatch_idle_poll(&self) -> Duration {
    self.dispatch_idle_poll
  }

  /// Recycle a worker's `WhisperState` after this many consecutive
  /// `WorkerHangTimeout`s. Default 1 on CPU (cheap recycle), 3 on
  /// GPU.
  pub const fn timeout_streak_threshold(&self) -> u32 {
    self.timeout_streak_threshold
  }

  // --- Mutating setters ----------------------------------------

  /// Set [`Self::worker_count`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. Codex round-24 flagged that a
  /// pool built with zero workers spawns no worker threads
  /// but still accepts work via the channel — chunks enter
  /// `in_flight` with no receiver capable of producing
  /// results, stalling the pump/drain loop until the
  /// configured timeout fires. Failing fast at the setter
  /// catches the explicit programmer-error case; the
  /// `serde`-deserialised path is caught at
  /// [`WhisperPool::new`].
  pub const fn set_worker_count(&mut self, value: usize) {
    assert!(
      value > 0,
      "worker_count must be > 0; a zero-worker pool cannot complete work"
    );
    self.worker_count = value;
  }

  /// Set [`Self::model_path`].
  pub fn set_model_path(&mut self, value: impl Into<PathBuf>) {
    self.model_path = value.into();
  }

  /// Set [`Self::use_gpu`].
  pub const fn set_use_gpu(&mut self, value: bool) {
    self.use_gpu = value;
  }

  /// Set [`Self::gpu_device`].
  pub const fn set_gpu_device(&mut self, value: i32) {
    self.gpu_device = value;
  }

  /// Set [`Self::flash_attn`].
  pub const fn set_flash_attn(&mut self, value: bool) {
    self.flash_attn = value;
  }

  /// Set [`Self::max_queued_chunks`].
  pub const fn set_max_queued_chunks(&mut self, value: usize) {
    self.max_queued_chunks = value;
  }

  /// Set [`Self::block_on_full_queue`].
  pub const fn set_block_on_full_queue(&mut self, value: bool) {
    self.block_on_full_queue = value;
  }

  /// Set [`Self::dispatch_idle_poll`].
  pub const fn set_dispatch_idle_poll(&mut self, value: Duration) {
    self.dispatch_idle_poll = value;
  }

  /// Set [`Self::timeout_streak_threshold`].
  pub const fn set_timeout_streak_threshold(&mut self, value: u32) {
    self.timeout_streak_threshold = value;
  }

  // --- Builder-style (consuming) -------------------------------

  /// Builder-style override for [`Self::worker_count`].
  ///
  /// # Panics
  ///
  /// Panics if `value == 0`. See [`Self::set_worker_count`].
  pub const fn with_worker_count(mut self, value: usize) -> Self {
    assert!(
      value > 0,
      "worker_count must be > 0; a zero-worker pool cannot complete work"
    );
    self.worker_count = value;
    self
  }

  /// Builder-style override for [`Self::model_path`].
  pub fn with_model_path(mut self, value: impl Into<PathBuf>) -> Self {
    self.model_path = value.into();
    self
  }

  /// Builder-style override for [`Self::use_gpu`].
  pub const fn with_use_gpu(mut self, value: bool) -> Self {
    self.use_gpu = value;
    self
  }

  /// Builder-style override for [`Self::gpu_device`].
  pub const fn with_gpu_device(mut self, value: i32) -> Self {
    self.gpu_device = value;
    self
  }

  /// Builder-style override for [`Self::flash_attn`].
  pub const fn with_flash_attn(mut self, value: bool) -> Self {
    self.flash_attn = value;
    self
  }

  /// Builder-style override for [`Self::max_queued_chunks`].
  pub const fn with_max_queued_chunks(mut self, value: usize) -> Self {
    self.max_queued_chunks = value;
    self
  }

  /// Builder-style override for [`Self::block_on_full_queue`].
  pub const fn with_block_on_full_queue(mut self, value: bool) -> Self {
    self.block_on_full_queue = value;
    self
  }

  /// Builder-style override for [`Self::dispatch_idle_poll`].
  pub const fn with_dispatch_idle_poll(mut self, value: Duration) -> Self {
    self.dispatch_idle_poll = value;
    self
  }

  /// Builder-style override for [`Self::timeout_streak_threshold`].
  pub const fn with_timeout_streak_threshold(mut self, value: u32) -> Self {
    self.timeout_streak_threshold = value;
    self
  }
}

/// CPU-flavored default worker count: half the physical cores
/// (min 1). Used by [`WhisperPoolOptions::new`] and serde defaults.
///
/// Codex round-34: an earlier `cfg!(any(feature = "_whisper_cuda", ...))`
/// branch was dead — the listed feature names never existed in this
/// crate's `Cargo.toml`, and consumer-side feature unification of
/// `whisper-rs`'s `cuda` / `metal` / etc. doesn't propagate through
/// `cfg!(feature = "...")` here. The result was that GPU users
/// silently got CPU defaults: half their physical cores worth of
/// `WhisperState`s contending for one GPU, increasing
/// timeout/resource-failure risk. The dead branch is gone; GPU
/// users should construct via [`WhisperPoolOptions::new_for_gpu`]
/// (which picks the single-worker / longer-streak defaults
/// appropriate for serialised GPU inference) or set
/// `worker_count` / `timeout_streak_threshold` explicitly.
fn default_worker_count() -> usize {
  let physical = num_cpus::get_physical();
  core::cmp::max(1, physical / 2)
}

/// CPU-flavored timeout-streak threshold: a single timeout
/// recycles the worker state. GPU inference is more variance-
/// tolerant; [`WhisperPoolOptions::new_for_gpu`] picks `3`.
const fn default_timeout_streak_threshold() -> u32 {
  1
}

/// GPU-flavored worker count. whisper.cpp serialises on a single
/// GPU regardless of concurrent `WhisperState`s, so multiple
/// workers oversubscribe instead of parallelising.
const fn gpu_worker_count() -> usize {
  1
}

/// GPU-flavored timeout-streak threshold. GPU launches have higher
/// per-call variance (kernel queueing, driver warmup); a single
/// slow chunk shouldn't recycle the state.
const fn gpu_timeout_streak_threshold() -> u32 {
  3
}

/// One unit of ASR work shipped to a worker thread. Crate-private.
pub(super) struct AsrWorkItem {
  /// Identity of the chunk this inference fulfils.
  pub chunk_id: ChunkId,
  /// Chunk audio (16 kHz f32 mono); shared via `Arc` with the core.
  pub samples: Arc<[f32]>,
  /// ASR knobs (per-call overrides already merged in by the runner).
  pub params: AsrParams,
  /// Per-job timeout. Stamped at dispatch time so each in-flight
  /// chunk carries its own budget; the worker's watchdog feeds
  /// this into the abort_flag check.
  pub asr_timeout: core::time::Duration,
  /// Watchdog flag. The worker installs this into `FullParams` via
  /// `set_abort_callback_safe`; a separate watchdog thread flips
  /// it true if the per-job timeout elapses.
  pub abort_flag: Arc<AtomicBool>,
}

/// Worker-emitted result for one chunk. Crate-private.
pub(super) type AsrResultMsg = (
  ChunkId,
  Result<crate::core::AsrResult, crate::types::WorkFailure>,
);

/// Validate caller-supplied `AsrParams` before any FFI work.
/// Pulled out of `full_params_from` so the `FullParamsCache`
/// (round 38 follow-up) can run validation on every chunk while
/// reusing a cached `FullParams` template on cache hits.
///
/// Returns:
/// - `Ok(())` — params are FFI-safe.
/// - `Err(WorkFailure::AsrFailed { kind: BackendError, .. })` —
///   for any of: NUL / non-`[a-z]{1,8}` language hint, NUL prompt,
///   `n_threads < 1`. The runner converts this to a chunk-level
///   failure rather than letting whisper.cpp panic / abort.
pub(super) fn validate_for_whisper_ffi(params: &AsrParams) -> Result<(), WorkFailure> {
  // Codex round-31/32 — language hint shape + NUL.
  if let Some(lang) = params.language_hint()
    && let Err(reason) = validate_language_code(lang.as_str())
  {
    return Err(WorkFailure::AsrFailed {
      kind: AsrFailureKind::BackendError,
      message: alloc::format!(
        "language hint rejected: {reason}. whisper-rs/whisper.cpp would either panic in \
         CString::new or fall back to detect; either way refusing to leak an unbounded \
         intern entry."
      ),
    });
  }
  // Codex round-31 — NUL byte in initial_prompt.
  if let Some(prompt) = params.initial_prompt()
    && prompt.as_str().contains('\0')
  {
    return Err(WorkFailure::AsrFailed {
      kind: AsrFailureKind::BackendError,
      message: alloc::format!(
        "initial_prompt of len {} contains an interior NUL byte; whisper-rs's set_initial_prompt \
         would panic. Reject before FFI.",
        prompt.as_str().len()
      ),
    });
  }
  // Codex round-34 — n_threads.
  if params.n_threads() < 1 {
    return Err(WorkFailure::AsrFailed {
      kind: AsrFailureKind::BackendError,
      message: alloc::format!(
        "n_threads must be >= 1 (got {}); whisper.cpp's std::vector<std::thread>({} - 1) \
         would underflow / abort. Reject before FFI.",
        params.n_threads(),
        params.n_threads(),
      ),
    });
  }
  Ok(())
}

/// Build a fresh `FullParams` template carrying the
/// allocate-on-set fields: strategy (via `FullParams::new`),
/// `set_language`, `set_initial_prompt`. Caller wires the
/// per-chunk fields on top via [`finalize_chunk`].
///
/// **Note (post whisper-cpp migration).** This used to be the
/// allocate-prone half of a two-tier `FullParams` system whose
/// raison d'être was to bound the `whisper-rs` CString leak in
/// `set_language` / `set_initial_prompt`. With the in-house
/// `whisper-cpp` bindings, `Params` owns and drops every CString
/// it stores — building fresh per attempt is leak-free. The
/// build/finalize split is retained because it cleanly separates
/// "decode-config" from "per-chunk knobs" for readers, not
/// because of a leak constraint.
fn build_template(params: &AsrParams) -> FullParams {
  let strategy = match params.strategy() {
    SamplingStrategy::Greedy { best_of } => WhisperStrategy::Greedy { best_of },
    SamplingStrategy::BeamSearch {
      beam_size,
      patience,
    } => WhisperStrategy::BeamSearch {
      beam_size,
      patience,
    },
  };
  let mut p = FullParams::new(strategy);
  if let Some(lang) = params.language_hint() {
    let static_lang: &'static str = intern_lang_str(lang.as_str());
    // `set_language` is fallible (interior NUL). Whispery's lang
    // codes come from a closed set (`Lang::as_str` for known
    // ISO-639-1 codes plus `Lang::Other(SmolStr)` whose validation
    // happens elsewhere); a NUL byte slipping through is a
    // programming bug we'd rather catch later than silently swallow.
    // Drop the error here only because the next call (`set_initial_prompt`)
    // and the caller's flow expect a `&mut FullParams`. The
    // `validate_for_whisper_ffi` precondition + `intern_lang_str`'s
    // closed alphabet make this path unreachable on valid inputs.
    let _ = p.set_language(static_lang);
  } else {
    p.set_detect_language(true);
  }
  if let Some(prompt) = params.initial_prompt() {
    // Same fallibility note as `set_language`. Prompt content is
    // user-supplied so we are slightly more cautious — if a NUL
    // shows up, drop the prompt rather than the chunk; whisper.cpp
    // tolerates an unset prompt and the alternative is failing the
    // entire ASR job over a string-encoding nit.
    let _ = p.set_initial_prompt(prompt.as_str());
  }
  p
}

/// Set every per-chunk field on a freshly cloned template:
/// `n_threads`, suppression bools, print toggles, no_speech
/// threshold, temperature_inc (pinned at 0), and the watchdog
/// abort callback. None of these allocate CString memory.
/// Per-attempt callers then `Clone` the result and only update
/// `set_temperature` per attempt.
fn finalize_chunk(
  mut full: FullParams,
  params: &AsrParams,
  abort_flag: Arc<AtomicBool>,
) -> FullParams {
  full.set_n_threads(params.n_threads());
  full.set_no_context(params.no_context());
  full.set_suppress_blank(params.suppress_blank());
  full.set_suppress_nst(params.suppress_non_speech_tokens());
  full.silence_print_toggles();
  full.set_no_speech_thold(params.no_speech_threshold());
  // Pin temperature_inc; whisper.cpp's internal ladder runs
  // exactly once at the runner-supplied temperature.
  full.set_temperature_inc(0.0);
  // Worker-hang watchdog. whisper-cpp's `Params::set_abort_callback`
  // is properly typed end-to-end (Box<dyn FnMut> matches the C
  // trampoline) and owns the closure for the Params lifetime —
  // both the whisper-rs `set_abort_callback_safe` UB and the
  // `FullParamsCache`-mediated leak workaround are obsolete.
  full.set_abort_callback(move || abort_flag.load(Ordering::Relaxed));
  full
}

/// Build a `FullParams` for one decoding attempt. The runner's outer
/// retry ladder calls this once per attempt with `attempt_temperature`
/// set to the next ladder step.
///
/// Disables whisper.cpp's internal temperature ladder via
/// `set_temperature_inc(0.0)`; each `state.full()` call is exactly
/// one decoding attempt at exactly `attempt_temperature`. A
/// `set_max_decoding_failures(...)` belt-and-braces secondary
/// safeguard is omitted here because whisper-rs 0.13.x does not
/// expose that setter; with `temperature_inc = 0.0` the internal
/// ladder iterates exactly once regardless.
///
/// Wires the worker-hang watchdog via the whisper-cpp safe
/// abort callback. The closure reads `abort_flag` on every
/// whisper.cpp progress callback; when the watchdog flips it
/// true, whisper.cpp returns mid-inference.
pub(super) fn full_params_from(
  params: &AsrParams,
  attempt_temperature: f32,
  abort_flag: Arc<AtomicBool>,
) -> Result<FullParams, WorkFailure> {
  validate_for_whisper_ffi(params)?;
  let template = build_template(params);
  let mut full = finalize_chunk(template, params, abort_flag);
  full.set_temperature(attempt_temperature);
  Ok(full)
}

/// Mean of per-segment `avg_logprob` across the just-decoded chunk.
/// Returns `f32::MIN` when the state has no segments — that signals
/// a truly empty result and trips the retry ladder via the
/// log_prob_threshold check.
///
/// whisper-rs does not expose a per-segment `avg_logprob` accessor.
/// We reconstruct it faithfully: per segment, average
/// `WhisperTokenData::plog` (the per-token log-probability returned
/// by whisper.cpp) across all tokens in that segment; then average
/// those segment means. This matches whisper.cpp's own internal
/// computation of the value it gates `logprob_thold` against.
pub(super) fn compute_avg_logprob(state: &WhisperState) -> f32 {
  let n = state.n_segments();
  if n <= 0 {
    return f32::MIN;
  }
  let mut seg_sum = 0.0f64;
  let mut seg_count = 0i32;
  for i in 0..n {
    let Some(segment) = state.segment(i) else {
      continue;
    };
    let n_tok = segment.n_tokens();
    if n_tok <= 0 {
      continue;
    }
    let mut tok_sum = 0.0f64;
    let mut tok_count = 0i32;
    for j in 0..n_tok {
      if let Some(token) = segment.token(j) {
        tok_sum += token.plog() as f64;
        tok_count += 1;
      }
    }
    if tok_count == 0 {
      continue;
    }
    seg_sum += tok_sum / tok_count as f64;
    seg_count += 1;
  }
  if seg_count == 0 {
    f32::MIN
  } else {
    (seg_sum / seg_count as f64) as f32
  }
}

/// Mean of per-segment `no_speech_probability()` across the
/// just-decoded chunk. Returns `0.0` for an empty state — the
/// runner's no-segment short-circuit already handles that case
/// before this is consulted.
///
/// Codex round-38: the runner uses this to honor the
/// documented `no_speech_threshold` knob, which previously was
/// a public no-op (only `avg_logprob` and `compression_ratio`
/// gated acceptance).
pub(super) fn compute_avg_no_speech_prob(state: &WhisperState) -> f32 {
  let n = state.n_segments();
  if n <= 0 {
    return 0.0;
  }
  let mut sum = 0.0_f32;
  let mut count = 0_i32;
  for i in 0..n {
    if let Some(segment) = state.segment(i) {
      sum += segment.no_speech_prob();
      count += 1;
    }
  }
  if count == 0 {
    return 0.0;
  }
  sum / count as f32
}

/// Concatenate all segments' text and compute whisperx's
/// "compression ratio" = `text.len() / zlib_compress(text).len()`.
///
/// A high ratio (whisperx default threshold 2.4) means the model
/// emitted long repeated runs that compressed disproportionately —
/// a strong hallucination signal.
///
/// whisperx's exact zlib choice is a heuristic, not a spec
/// requirement. To avoid pulling a `flate2` dep, we adopt an
/// equally-discriminative proxy: ratio of `text.len()` to the count
/// of unique 4-byte shingles. This catches the "yes yes yes yes ..."
/// failure mode the threshold was designed for.
pub(super) fn compute_compression_ratio(state: &WhisperState) -> f32 {
  use std::collections::HashSet;

  let n = state.n_segments();
  if n <= 0 {
    return 0.0;
  }
  let mut text = String::new();
  for i in 0..n {
    if let Some(segment) = state.segment(i) {
      if let Ok(s) = segment.text() {
        text.push_str(s);
      }
    }
  }
  let raw = text.len();
  if raw < 4 {
    return 0.0;
  }
  let bytes = text.as_bytes();
  let mut shingles: HashSet<[u8; 4]> = HashSet::with_capacity(raw);
  for window in bytes.windows(4) {
    let mut s = [0u8; 4];
    s.copy_from_slice(window);
    shingles.insert(s);
  }
  let unique = shingles.len();
  if unique == 0 {
    return 0.0;
  }
  raw as f32 / unique as f32
}

/// Run one chunk's ASR through the runner's temperature retry ladder.
///
/// Each attempt is exactly one `state.full()` call. The runner — not
/// whisper.cpp's internal loop — picks the temperature for each
/// attempt; `full_params_from` pins `temperature_inc=0.0` so
/// whisper.cpp's internal `for t = initial; t <= 1.0; t += inc` loop
/// runs exactly once at the runner-supplied temperature.
///
/// Success criteria:
/// - `avg_logprob >= params.log_prob_threshold` (default −1.0)
/// - `compression_ratio <= params.compression_ratio_threshold` (default 2.4)
///
/// Failure: `WorkFailure::AsrFailed { kind: AllTemperaturesFailed, .. }`
/// after all `max_attempts` failed; `WorkFailure::AsrFailed { kind:
/// BackendError, .. }` if `state.full()` itself returned an error;
/// `WorkFailure::WorkerHangTimeout { kind: Asr, .. }` if the abort
/// flag was flipped (the watchdog detected timeout).
pub(super) fn run_with_temperature_ladder(
  state: &mut WhisperState,
  job: &AsrWorkItem,
  started_at: std::time::Instant,
) -> Result<AsrResult, WorkFailure> {
  let p = &job.params;
  let mut temperature = p.initial_temperature();
  let max = p.max_attempts() as usize;

  // Post whisper-cpp migration: build a fresh `Params` per
  // attempt. The old `FullParamsCache` + `Clone` + `set_temperature`
  // dance only existed to bound the whisper-rs CString leak; the
  // in-house bindings own + drop every `CString`, so per-attempt
  // construction is cheap AND leak-free. `Params` is intentionally
  // not `Clone` (the boxed abort closure can't be safely
  // duplicated), which makes "rebuild fresh" the only sound path
  // anyway.
  validate_for_whisper_ffi(p)?;

  for _attempt in 0..max {
    let template = build_template(p);
    let mut full = finalize_chunk(template, p, job.abort_flag.clone());
    full.set_temperature(temperature);
    let outcome = state.full(&full, job.samples.as_ref());
    if let Err(e) = outcome {
      // Distinguish abort (watchdog timeout) from a real backend
      // error. whisper-rs returns `Err(_)` from full() when the
      // abort callback flipped to true, but the error variant is
      // not always distinct; we double-check via abort_flag.
      if job.abort_flag.load(Ordering::Relaxed) {
        return Err(WorkFailure::WorkerHangTimeout {
          kind: WorkerKind::Asr,
          elapsed: started_at.elapsed(),
        });
      }
      return Err(WorkFailure::AsrFailed {
        kind: AsrFailureKind::BackendError,
        message: format!("{e:?}"),
      });
    }

    // Zero-segment short-circuit: whisper.cpp's contract is that
    // an utterance with no speech (silent input, VAD false
    // positives, very quiet chunks) returns 0 segments, NOT an
    // error. `compute_avg_logprob` returns `f32::MIN` in that
    // case, which would fail the threshold gate and force a
    // temperature retry — every retry would yield the same 0
    // segments, the loop would exhaust, and the chunk would
    // surface as `AllTemperaturesFailed`. Detect the empty
    // outcome here and return an empty `AsrResult` instead.
    if state.n_segments() == 0 {
      return build_asr_result(state, temperature, p);
    }

    let logprob = compute_avg_logprob(state);
    let cratio = compute_compression_ratio(state);
    let nsp = compute_avg_no_speech_prob(state);

    let logprob_ok = logprob >= p.log_prob_threshold();
    let cratio_ok = cratio <= p.compression_ratio_threshold();

    // Codex round-38: implement the documented
    // `no_speech_threshold` knob. Mirrors WhisperX / OpenAI
    // Whisper's "silent chunk" detection: when the mean
    // no_speech probability across segments exceeds the
    // configured threshold AND the average logprob is too low
    // to trust the transcript, the chunk is treated as silence
    // — empty `AsrResult`, no temperature retries (temperature
    // can't conjure speech the model already evaluated as
    // absent). The `avg_logprob` conjunct mirrors WhisperX:
    // a high no_speech_prob alone with confident text isn't
    // enough to discard.
    if nsp > p.no_speech_threshold() && !logprob_ok {
      return Ok(AsrResult::new(
        SmolStr::new(""),
        p.language_hint()
          .cloned()
          .unwrap_or(Lang::Other(SmolStr::new(""))),
        logprob,
        nsp,
        temperature,
      ));
    }

    if logprob_ok && cratio_ok {
      return build_asr_result(state, temperature, p);
    }
    temperature += p.temperature_increment();
  }

  Err(WorkFailure::AsrFailed {
    kind: AsrFailureKind::AllTemperaturesFailed,
    message: format!(
      "all {} temperature attempts failed for chunk {:?}",
      max, job.chunk_id,
    ),
  })
}

/// Compose an [`AsrResult`] from a successful `WhisperState::full` call.
///
/// `full_lang_id_from_state` returns a bare `c_int`; a negative id
/// means "no detection" and we fall back to the language hint (or
/// `Lang::Other("")`).
///
/// `no_speech_prob` is averaged across segments via
/// `WhisperSegment::no_speech_probability()`; the value defaults to
/// `0.0` when there are no segments. Downstream `AsrResult::no_speech_prob`
/// consumers tolerate `0.0`, and the runner's retry ladder gates on
/// `avg_logprob` / `compression_ratio` rather than `no_speech_prob`.
fn build_asr_result(
  state: &WhisperState,
  final_temperature: f32,
  params: &AsrParams,
) -> Result<AsrResult, WorkFailure> {
  let n = state.n_segments();
  let mut text = String::new();
  let mut nsp_sum = 0.0f32;
  let mut nsp_count: i32 = 0;
  for i in 0..n {
    if let Some(segment) = state.segment(i) {
      if let Ok(s) = segment.text() {
        text.push_str(s);
      }
      nsp_sum += segment.no_speech_prob();
      nsp_count += 1;
    }
  }

  let avg_logprob = compute_avg_logprob(state);
  let no_speech_prob: f32 = if nsp_count > 0 {
    nsp_sum / nsp_count as f32
  } else {
    0.0
  };

  // `detected_lang` returns the typed `Lang` directly. Falls
  // back to the configured language hint (or `Lang::Other("")`
  // if none) when whisper.cpp didn't detect or assert a language
  // for this chunk.
  let language = state.detected_lang().unwrap_or_else(|| {
    params
      .language_hint()
      .cloned()
      .unwrap_or(Lang::Other(SmolStr::new("")))
  });

  // Script-dispatch: walk the just-decoded segments and split each
  // into per-language [`Run`]s. Materialising into a Vec here is
  // necessary because `dispatch` takes `&[Segment<'_>]` and the
  // segment iterator is consumed lazily; the live borrow on
  // `state` is OK because the dispatcher only reads each
  // segment's text + tokens + DTW timestamps and produces an
  // owned `Vec<Run>` whose data outlives the state borrow.
  //
  // `state_lang` is the just-detected language (or the hint),
  // used by `script_to_lang` for Latin / ambiguous-script
  // disambiguation. Passing `Some(language.clone())` rather than
  // `params.language_hint()` lets the dispatcher benefit from
  // whisper.cpp's own auto-detection in the auto-detect path
  // (`language_hint = None` originally).
  let segments: alloc::vec::Vec<whispercpp::Segment<'_>> = state.segments_iter().collect();
  let runs = crate::align::dispatch(&segments, Some(language.clone()));

  Ok(
    AsrResult::new(
      SmolStr::new(text.trim()),
      language,
      avg_logprob,
      no_speech_prob,
      final_temperature,
    )
    .with_runs(runs),
  )
}

/// Worker pool. Shared `Arc<WhisperContext>` per the v1 architecture
/// decision (verified: whisper-rs 0.13.x marks the context
/// thread-safe; states are owned per worker).
pub(super) struct WhisperPool {
  pub(super) ctx: Arc<WhisperContext>,
  workers: Vec<JoinHandle<()>>,
  pub(super) work_tx: Sender<AsrWorkItem>,
  pub(super) result_rx: Receiver<AsrResultMsg>,
  pub(super) work_tx_capacity: usize,
}

impl WhisperPool {
  /// Build the pool. Caller-supplied `WhisperContext` controls
  /// flash_attn / GPU device / model path explicitly (the public
  /// `ManagedTranscriberBuilder::build` hands one in).
  pub(super) fn new(ctx: WhisperContext, config: &WhisperPoolOptions) -> Result<Self, RunnerError> {
    // Defensive: the setter / builder panic on `worker_count == 0`,
    // but a config deserialised via serde bypasses both. A zero-
    // worker pool spawns no worker threads while still accepting
    // work via the channel — chunks enter `in_flight` with no
    // receiver capable of producing results, stalling the
    // pump/drain loop until timeout. Codex round-24 flagged this.
    if config.worker_count() == 0 {
      return Err(RunnerError::WhisperContextLoad {
        message: alloc::string::String::from(
          "WhisperPoolOptions.worker_count must be > 0; a zero-worker \
           pool cannot complete work and would stall the pump loop.",
        ),
      });
    }
    let ctx = Arc::new(ctx);
    let (work_tx, work_rx) = bounded::<AsrWorkItem>(config.max_queued_chunks());
    let (result_tx, result_rx) = bounded::<AsrResultMsg>(config.max_queued_chunks() + 16);

    let mut workers = Vec::with_capacity(config.worker_count());
    for worker_idx in 0..config.worker_count() {
      let ctx_for_worker = ctx.clone();
      let work_rx = work_rx.clone();
      let result_tx = result_tx.clone();
      let timeout_streak_threshold = config.timeout_streak_threshold();
      let handle = std::thread::Builder::new()
        .name(format!("whispery-asr-{}", worker_idx))
        .spawn(move || {
          worker_loop(ctx_for_worker, work_rx, result_tx, timeout_streak_threshold);
        })
        .map_err(RunnerError::Io)?;
      workers.push(handle);
    }
    // Drop the local references so the pool owns the only senders/receivers
    // routed externally; the cloned ones live on each worker.
    drop(work_rx);
    drop(result_tx);

    Ok(Self {
      ctx,
      workers,
      work_tx,
      result_rx,
      work_tx_capacity: config.max_queued_chunks(),
    })
  }

  /// Build a pool from a model path + config. Shorthand for
  /// `WhisperContext::new(...)?` then `Self::new`.
  pub(super) fn from_path(config: &WhisperPoolOptions) -> Result<Self, RunnerError> {
    let ctx_params = WhisperContextParameters::new()
      .with_use_gpu(config.use_gpu())
      .with_gpu_device(config.gpu_device())
      .with_flash_attn(config.flash_attn());
    let ctx = WhisperContext::new(config.model_path(), ctx_params).map_err(|e| {
      RunnerError::WhisperContextLoad {
        message: format!("{e:?}"),
      }
    })?;
    Self::new(ctx, config)
  }
}

impl Drop for WhisperPool {
  fn drop(&mut self) {
    // Closing work_tx is what lets idle workers exit — their
    // `recv()` returns `Err(_)` once the last Sender is dropped.
    // The naive "drop the field automatically" approach is
    // wrong: Drop runs BEFORE struct fields are dropped, so a
    // join() call below would block forever on an idle worker
    // because it still saw a live `work_tx` in `self`. Replace
    // it with a dummy channel up front so the original drops
    // here, signaling disconnect.
    let (dummy_tx, _dummy_rx) = bounded::<AsrWorkItem>(1);
    let _live_tx = core::mem::replace(&mut self.work_tx, dummy_tx);
    drop(_live_tx);

    // **Detach** rather than join. whisper.cpp's `state.full`
    // is uninterruptible from outside the worker thread once it
    // has started (the abort_callback is checked inside the C
    // loop only at coarse boundaries; for a long chunk it can
    // run for many seconds with no abort opportunity). Joining
    // here would block Drop indefinitely on an in-flight call,
    // which the existing `#[ignore]`'d real-model regression
    // tests document. Dropping the JoinHandles detaches the
    // workers; they continue running until the in-flight job
    // finishes (and pick up the dummy_tx disconnect at the
    // top of the next loop iteration), the `Arc<WhisperContext>`
    // they carry keeps model memory alive until they exit, and
    // the OS reclaims everything at process termination. The
    // trade-off is "Drop is fast" vs. "transient thread leak
    // until the stuck call finishes naturally" — the right
    // call for v1, since hung Drop blocks unrelated cleanup
    // (test teardown, daemon shutdown, etc.).
    self.workers.clear();
  }
}

/// True iff a `recv_timeout` on the watchdog's cancel channel
/// indicates a real per-job timeout (vs. a clean cancellation
/// from the worker dropping `cancel_tx`).
///
/// Codex round-36 introduced this helper to fix a critical
/// regression in the previous `recv_timeout(timeout).is_err()`
/// shortcut: `Err(Disconnected)` (clean cancel) ALSO matches
/// `is_err()`, so every fast successful job tripped the post-
/// watchdog check and got rewritten to `WorkerHangTimeout`. Only
/// `RecvTimeoutError::Timeout` indicates an actual hang.
fn watchdog_should_signal_timeout(result: Result<(), crossbeam_channel::RecvTimeoutError>) -> bool {
  matches!(result, Err(crossbeam_channel::RecvTimeoutError::Timeout))
}

/// Worker thread main loop. Implements per-worker timeout-streak
/// recycling.
fn worker_loop(
  ctx: Arc<WhisperContext>,
  work_rx: Receiver<AsrWorkItem>,
  result_tx: Sender<AsrResultMsg>,
  timeout_streak_threshold: u32,
) {
  let mut state_opt: Option<WhisperState> = None;
  let mut timeout_streak: u32 = 0;

  while let Ok(job) = work_rx.recv() {
    // Lazy-create the state on first job, recreate after threshold.
    if state_opt.is_none() {
      match ctx.create_state() {
        Ok(s) => state_opt = Some(s),
        Err(e) => {
          let _ = result_tx.send((
            job.chunk_id,
            Err(WorkFailure::AsrFailed {
              kind: AsrFailureKind::BackendError,
              message: format!("create_state failed: {e:?}"),
            }),
          ));
          continue;
        }
      }
    }
    let state = state_opt.as_mut().expect("state present");

    // Spawn a watchdog that flips abort_flag if asr_timeout
    // elapses. We use a one-shot Sender to cancel the watchdog
    // early once inference completes — a `thread::sleep` would
    // wait the full timeout regardless, blocking `watchdog.join()`
    // and adding `asr_timeout` of latency per chunk.
    //
    // Under thread/fd/memory exhaustion `Builder::spawn` can
    // fail. Surface it as an in-band `WorkFailure::AsrFailed`
    // for this chunk and continue serving subsequent jobs;
    // panicking would either kill the only worker (single-
    // worker pool) or strand the chunk on the result channel
    // until drain timeout (multi-worker pool, blocking
    // in-order emission).
    let abort_flag = job.abort_flag.clone();
    let timeout = job.asr_timeout;
    let (cancel_tx, cancel_rx) = bounded::<()>(1);
    let watchdog = match std::thread::Builder::new()
      .name("whispery-asr-watchdog".into())
      .spawn(move || {
        // Block on the cancel channel for up to `timeout`.
        //
        // Codex round-36 fix: only flip `abort_flag` on a real
        // `RecvTimeoutError::Timeout`. Clean cancellation
        // (`Ok(())` from a sender, or `Err(Disconnected)` from
        // the worker dropping `cancel_tx` after fast inference)
        // must NOT flip the flag — the round-32 post-watchdog
        // check would otherwise rewrite every successful
        // `AsrResult` to `WorkerHangTimeout`. The shared
        // `watchdog_should_signal_timeout` helper makes the
        // discrimination explicit and unit-testable.
        if watchdog_should_signal_timeout(cancel_rx.recv_timeout(timeout)) {
          abort_flag.store(true, Ordering::Relaxed);
        }
      }) {
      Ok(handle) => handle,
      Err(e) => {
        let _ = result_tx.send((
          job.chunk_id,
          Err(WorkFailure::AsrFailed {
            kind: AsrFailureKind::BackendError,
            message: format!(
              "failed to spawn ASR watchdog ({e}); refusing to run inference \
               without a cancellable timeout"
            ),
          }),
        ));
        continue;
      }
    };

    let started_at = std::time::Instant::now();
    let outcome = run_with_temperature_ladder(state, &job, started_at);

    // Cancel the watchdog by dropping cancel_tx; the watchdog's
    // recv_timeout returns Err(Disconnected) and exits cleanly.
    drop(cancel_tx);
    let _ = watchdog.join();

    // Codex round-32: post-watchdog abort-flag check.
    // `run_with_temperature_ladder` only consults `abort_flag`
    // when `state.full(...)` returned `Err`. If the watchdog
    // fired DURING inference but whisper.cpp finished before
    // observing the next abort-callback poll, we get an
    // apparently-successful `AsrResult` that violates the
    // configured per-job timeout. Rewrite it to
    // `WorkerHangTimeout` here, matching the alignment worker's
    // post-run check (`alignment_pool.rs` does the same).
    //
    // This also makes timeout-streak recycling honest:
    // `was_timeout` below is now derived from the rewritten
    // outcome, so a long sequence of "succeeded after timeout"
    // results still trips the streak threshold and forces
    // state recreation.
    let outcome = if job.abort_flag.load(Ordering::Relaxed) {
      Err(WorkFailure::WorkerHangTimeout {
        kind: WorkerKind::Asr,
        elapsed: started_at.elapsed(),
      })
    } else {
      outcome
    };

    let was_timeout = matches!(outcome, Err(WorkFailure::WorkerHangTimeout { .. }));

    let _ = result_tx.send((job.chunk_id, outcome));

    if was_timeout {
      timeout_streak += 1;
      if timeout_streak >= timeout_streak_threshold {
        // Drop the state; next iteration recreates it.
        state_opt = None;
        timeout_streak = 0;
      }
    } else {
      timeout_streak = 0;
    }
  }
  // work_tx dropped: clean exit.
}

#[cfg(test)]
mod tests {
  use super::*;

  // --- Codex round-36: ASR watchdog cancellation discrimination ---

  /// `Err(Timeout)` is the only case that should flip the
  /// abort_flag. This is the actual hang signal whisper.cpp's
  /// abort_callback picks up.
  #[test]
  fn watchdog_signals_timeout_on_real_timeout() {
    assert!(watchdog_should_signal_timeout(Err(
      crossbeam_channel::RecvTimeoutError::Timeout
    )));
  }

  /// `Err(Disconnected)` is what the worker's `drop(cancel_tx)`
  /// produces after a fast successful inference. It MUST NOT
  /// be treated as a timeout — otherwise round-32's post-
  /// watchdog check rewrites every successful `AsrResult` into
  /// `WorkerHangTimeout`. This is the regression Codex round-36
  /// caught.
  #[test]
  fn watchdog_does_not_signal_timeout_on_clean_disconnect() {
    assert!(!watchdog_should_signal_timeout(Err(
      crossbeam_channel::RecvTimeoutError::Disconnected
    )));
  }

  /// `Ok(())` (sender explicitly sent) is also a clean exit.
  /// We don't use the send path today, but the helper has to
  /// handle it for completeness; treating it as a timeout
  /// would be a footgun for any future "early-cancel via send"
  /// path.
  #[test]
  fn watchdog_does_not_signal_timeout_on_clean_send() {
    assert!(!watchdog_should_signal_timeout(Ok(())));
  }

  /// End-to-end watchdog behaviour with a real channel: a fast
  /// "drop tx" never flips the abort flag.
  #[test]
  fn watchdog_loop_drop_tx_does_not_flip_flag() {
    use core::sync::atomic::AtomicBool;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let timeout = Duration::from_secs(5); // long
    let (cancel_tx, cancel_rx) = bounded::<()>(1);
    let abort_clone = abort_flag.clone();
    let handle = std::thread::spawn(move || {
      if watchdog_should_signal_timeout(cancel_rx.recv_timeout(timeout)) {
        abort_clone.store(true, Ordering::Relaxed);
      }
    });
    drop(cancel_tx);
    handle.join().unwrap();
    assert!(
      !abort_flag.load(Ordering::Relaxed),
      "clean drop(cancel_tx) must NOT flip abort_flag"
    );
  }

  /// End-to-end watchdog behaviour: a real timeout DOES flip
  /// the abort flag.
  #[test]
  fn watchdog_loop_timeout_flips_flag() {
    use core::sync::atomic::AtomicBool;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let timeout = Duration::from_millis(40);
    let (cancel_tx, cancel_rx) = bounded::<()>(1);
    let abort_clone = abort_flag.clone();
    let handle = std::thread::spawn(move || {
      if watchdog_should_signal_timeout(cancel_rx.recv_timeout(timeout)) {
        abort_clone.store(true, Ordering::Relaxed);
      }
    });
    // Sleep past the watchdog timeout before dropping. The
    // watchdog's recv_timeout fires Err(Timeout) and flips the
    // flag; the subsequent drop is a no-op.
    std::thread::sleep(Duration::from_millis(120));
    drop(cancel_tx);
    handle.join().unwrap();
    assert!(
      abort_flag.load(Ordering::Relaxed),
      "real timeout MUST flip abort_flag"
    );
  }

  /// Language-code interning must allocate at most once per
  /// distinct value. Calling `intern_lang_str` twice with the
  /// same string returns the same `&'static str` pointer;
  /// calling with a different string returns a different
  /// pointer. (Bounded-leak invariant — per-attempt `Box::leak`
  /// would be unbounded.)
  #[test]
  fn intern_lang_str_returns_stable_pointer_per_value() {
    let a1 = intern_lang_str("en");
    let a2 = intern_lang_str("en");
    let b = intern_lang_str("zh");
    assert_eq!(a1, "en");
    assert_eq!(a2, "en");
    assert!(
      core::ptr::eq(a1, a2),
      "same input must return the same `&'static str`; got distinct pointers"
    );
    assert_ne!(a1, b, "distinct inputs must intern to distinct strings");
  }

  #[test]
  fn defaults_round_trip() {
    let cfg = WhisperPoolOptions::new("/tmp/model.bin");
    assert_eq!(cfg.use_gpu(), false);
    assert_eq!(cfg.gpu_device(), 0);
    assert_eq!(cfg.flash_attn(), false);
    assert_eq!(cfg.block_on_full_queue(), true);
    assert_eq!(cfg.dispatch_idle_poll(), Duration::from_millis(10));
    assert!(cfg.worker_count() >= 1);
    assert_eq!(cfg.max_queued_chunks(), cfg.worker_count() + 4);
    assert!(cfg.timeout_streak_threshold() >= 1);
    assert_eq!(cfg.model_path(), Path::new("/tmp/model.bin"));
  }

  /// Codex round-34: `new_for_gpu` MUST pick GPU-flavored
  /// defaults — single worker (whisper.cpp serialises on one
  /// GPU), longer timeout-streak threshold (GPU variance),
  /// `use_gpu = true`. Pre-fix the only path to GPU-flavored
  /// defaults was a dead `cfg!(...)` check.
  #[test]
  fn new_for_gpu_picks_gpu_defaults() {
    let cfg = WhisperPoolOptions::new_for_gpu("/tmp/model.bin");
    assert!(cfg.use_gpu(), "new_for_gpu must enable use_gpu");
    assert_eq!(
      cfg.worker_count(),
      1,
      "GPU default is 1 worker (single-GPU serialisation)"
    );
    assert_eq!(
      cfg.timeout_streak_threshold(),
      3,
      "GPU default tolerates 3 consecutive timeouts before recycling"
    );
    assert_eq!(cfg.max_queued_chunks(), 5); // worker_count + 4
    assert_eq!(cfg.model_path(), Path::new("/tmp/model.bin"));
  }

  #[test]
  fn with_setters_round_trip() {
    let cfg = WhisperPoolOptions::new("/tmp/model.bin")
      .with_worker_count(2)
      .with_use_gpu(true)
      .with_gpu_device(7)
      .with_flash_attn(true)
      .with_max_queued_chunks(20)
      .with_block_on_full_queue(false)
      .with_dispatch_idle_poll(Duration::from_millis(25))
      .with_timeout_streak_threshold(5);
    assert_eq!(cfg.worker_count(), 2);
    assert!(cfg.use_gpu());
    assert_eq!(cfg.gpu_device(), 7);
    assert!(cfg.flash_attn());
    assert_eq!(cfg.max_queued_chunks(), 20);
    assert!(!cfg.block_on_full_queue());
    assert_eq!(cfg.dispatch_idle_poll(), Duration::from_millis(25));
    assert_eq!(cfg.timeout_streak_threshold(), 5);
  }

  #[test]
  fn set_setters_round_trip() {
    let mut cfg = WhisperPoolOptions::new("/tmp/model.bin");
    cfg.set_worker_count(3);
    cfg.set_model_path("/var/cache/model.gguf");
    assert_eq!(cfg.worker_count(), 3);
    assert_eq!(cfg.model_path(), Path::new("/var/cache/model.gguf"));
  }

  #[test]
  fn asr_work_item_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<AsrWorkItem>();
    assert_send::<AsrResultMsg>();
  }

  use crate::{
    core::{AsrParams, SamplingStrategy},
    types::Lang,
  };
  use core::sync::atomic::AtomicBool;

  #[test]
  fn full_params_from_greedy_is_finite() {
    let p = AsrParams::default().with_strategy(SamplingStrategy::Greedy { best_of: 1 });
    let flag = Arc::new(AtomicBool::new(false));
    let _full = full_params_from(&p, 0.4, flag).expect("valid params");
    // FullParams' fields aren't all readable; the assertion is that
    // the build does not panic and the abort closure compiles.
    // Recording-mock tests verify temperature_inc=0.0 and the
    // explicit set_temperature(t) call sequence.
  }

  #[test]
  fn full_params_from_with_language_hint_does_not_panic() {
    let p = AsrParams::default().with_language_hint(Some(Lang::En));
    let flag = Arc::new(AtomicBool::new(false));
    let _full = full_params_from(&p, 0.0, flag).expect("valid params");
  }

  /// Codex round-31 regression: an interior NUL in the language
  /// hint must be rejected as an in-band `WorkFailure::AsrFailed`
  /// rather than panicking inside `whisper-rs`'s `set_language`
  /// (which uses `CString::new(...).expect("...")`). Reaches here
  /// via `Lang::Other("xx\0yy")` which the public surface accepts
  /// without validation.
  ///
  /// Round-32 narrowed the validation to lowercase-ASCII-only
  /// (Codex flagged the unbounded intern leak). NUL bytes are
  /// non-ASCII-letter and therefore still rejected, but via the
  /// charset check rather than a NUL-specific check.
  #[test]
  fn full_params_from_rejects_interior_nul_in_language_hint() {
    let p = AsrParams::default().with_language_hint(Some(Lang::Other(SmolStr::from("xx\0yy"))));
    let flag = Arc::new(AtomicBool::new(false));
    let res = full_params_from(&p, 0.0, flag);
    match res {
      Err(WorkFailure::AsrFailed {
        kind: AsrFailureKind::BackendError,
        message,
      }) => {
        assert!(
          message.contains("language hint") && message.contains("lowercase ASCII"),
          "expected charset-violation diagnostic; got {message:?}"
        );
      }
      other => panic!("expected AsrFailed/BackendError; got {other:?}"),
    }
  }

  // --- Codex round-32: language hint shape validation ---

  #[test]
  fn validate_language_code_accepts_iso_shapes() {
    assert!(validate_language_code("en").is_ok());
    assert!(validate_language_code("es").is_ok());
    assert!(validate_language_code("zh").is_ok());
    assert!(validate_language_code("yue").is_ok()); // Cantonese
    assert!(validate_language_code("haw").is_ok()); // Hawaiian
    assert!(validate_language_code("a").is_ok()); // single letter (degenerate but bounded)
    assert!(validate_language_code("abcdefgh").is_ok()); // 8 chars
  }

  #[test]
  fn validate_language_code_rejects_empty() {
    let err = validate_language_code("").unwrap_err();
    assert!(err.contains("empty"), "got {err}");
  }

  #[test]
  fn validate_language_code_rejects_overlong() {
    let err = validate_language_code("abcdefghi").unwrap_err(); // 9 chars
    assert!(err.contains("longer than"), "got {err}");
    let err2 = validate_language_code(&"x".repeat(64)).unwrap_err();
    assert!(err2.contains("longer than"), "got {err2}");
  }

  #[test]
  fn validate_language_code_rejects_uppercase() {
    let err = validate_language_code("EN").unwrap_err();
    assert!(err.contains("lowercase ASCII"), "got {err}");
  }

  #[test]
  fn validate_language_code_rejects_dash_or_digits() {
    // Even regional variants like "zh-tw" are rejected: whisper.cpp
    // doesn't recognize them and this keeps the intern table truly
    // bounded to ~26^8 worst case (in practice ≪50 named codes).
    assert!(validate_language_code("zh-tw").is_err());
    assert!(validate_language_code("zh1").is_err());
  }

  #[test]
  fn validate_language_code_rejects_non_ascii() {
    // UTF-8 multibyte: "français" — definitely a language name,
    // but not a code. Reject; the user should pass `Lang::Fr`.
    assert!(validate_language_code("français").is_err());
    // Control chars including NUL.
    assert!(validate_language_code("a\0b").is_err());
    assert!(validate_language_code("a\nb").is_err());
  }

  /// Codex round-32 regression: a high-cardinality
  /// `Lang::Other(SmolStr)` from a malicious or buggy caller
  /// must NOT reach `intern_lang_str` (which leaks one
  /// `&'static str` per distinct value forever). Validation
  /// rejects the request as an in-band chunk failure long
  /// before the intern table sees it.
  #[test]
  fn full_params_from_rejects_high_cardinality_language_hint() {
    let p = AsrParams::default().with_language_hint(Some(Lang::Other(SmolStr::from(
      "very-long-attacker-string",
    ))));
    let flag = Arc::new(AtomicBool::new(false));
    let res = full_params_from(&p, 0.0, flag);
    match res {
      Err(WorkFailure::AsrFailed {
        kind: AsrFailureKind::BackendError,
        ..
      }) => {}
      other => panic!("expected AsrFailed/BackendError; got {other:?}"),
    }
  }

  /// Codex round-31 regression: an interior NUL in `initial_prompt`
  /// must be rejected as an in-band `WorkFailure::AsrFailed`
  /// rather than panicking inside `whisper-rs`'s
  /// `set_initial_prompt`. Reaches here via the public
  /// `AsrParamsOverride::with_initial_prompt(Some(SmolStr::new(...)))`
  /// which accepts arbitrary content.
  #[test]
  fn full_params_from_rejects_interior_nul_in_initial_prompt() {
    let p = AsrParams::default().with_initial_prompt(Some(SmolStr::from("hint\0poison")));
    let flag = Arc::new(AtomicBool::new(false));
    let res = full_params_from(&p, 0.0, flag);
    match res {
      Err(WorkFailure::AsrFailed {
        kind: AsrFailureKind::BackendError,
        message,
      }) => {
        assert!(
          message.contains("initial_prompt") && message.contains("NUL"),
          "expected NUL diagnostic; got {message:?}"
        );
      }
      other => panic!("expected AsrFailed/BackendError; got {other:?}"),
    }
  }

  /// Internal-only variant testable without a live `WhisperState`.
  /// Mirrors `compute_compression_ratio`'s algorithm so the unit
  /// test can pin the algorithm against canned inputs.
  fn compression_ratio_of_text(text: &str) -> f32 {
    use std::collections::HashSet;
    let raw = text.len();
    if raw < 4 {
      return 0.0;
    }
    let bytes = text.as_bytes();
    let mut shingles: HashSet<[u8; 4]> = HashSet::with_capacity(raw);
    for w in bytes.windows(4) {
      let mut s = [0u8; 4];
      s.copy_from_slice(w);
      shingles.insert(s);
    }
    if shingles.is_empty() {
      0.0
    } else {
      raw as f32 / shingles.len() as f32
    }
  }

  #[test]
  fn compression_ratio_low_for_diverse_text() {
    let r = compression_ratio_of_text("the quick brown fox jumps over the lazy dog");
    assert!(r < 1.5, "diverse text ratio = {}", r);
  }

  #[test]
  fn compression_ratio_high_for_repeated_text() {
    let r = compression_ratio_of_text("yes yes yes yes yes yes yes yes yes yes ");
    assert!(
      r >= 2.4,
      "repeated text ratio should trip the 2.4 default; got {}",
      r
    );
  }

  #[test]
  fn compression_ratio_short_input_returns_zero() {
    assert_eq!(compression_ratio_of_text(""), 0.0);
    assert_eq!(compression_ratio_of_text("ab"), 0.0);
  }

  /// Sanity check: confirm `run_with_temperature_ladder` is callable
  /// with the expected signature (and is pinned as `Send` so it can
  /// run on a worker thread). End-to-end ladder behaviour and
  /// layered-ladder suppression are exercised by other tests.
  /// Here we only assert the type signature compiles.

  // FullParamsCache regression tests deleted alongside the cache
  // itself in the whisper-cpp migration. The cache existed only
  // to bound the whisper-rs `set_language` / `set_initial_prompt`
  // CString leak; whisper-cpp's `Params` owns + drops every
  // CString, so the test invariants ("differing only in language
  // produces a new entry", etc.) no longer have a structure to
  // pin down. Equivalent coverage now lives implicitly in
  // `Params::Drop` (no leaks) and the integration parity runs.

  #[test]
  fn run_with_temperature_ladder_signature_compiles() {
    // Coerce the function to its expected signature; if the actual
    // signature drifts, this fails to compile.
    let _f: fn(
      &mut WhisperState,
      &AsrWorkItem,
      std::time::Instant,
    ) -> Result<crate::core::AsrResult, crate::types::WorkFailure> = run_with_temperature_ladder;
  }

  #[test]
  fn whisper_pool_handles_are_send() {
    fn assert_send<T: Send>() {}
    // The Sender / Receiver halves crossbeam exposes are Send + Sync;
    // this assertion fails to compile if a future refactor introduces
    // a non-Send field.
    assert_send::<crossbeam_channel::Sender<AsrWorkItem>>();
    assert_send::<crossbeam_channel::Receiver<AsrResultMsg>>();
  }

  #[test]
  #[should_panic(expected = "worker_count must be > 0")]
  fn set_worker_count_zero_panics() {
    let mut opts = WhisperPoolOptions::new("/tmp/model.bin");
    opts.set_worker_count(0);
  }

  #[test]
  #[should_panic(expected = "worker_count must be > 0")]
  fn with_worker_count_zero_panics() {
    let _ = WhisperPoolOptions::new("/tmp/model.bin").with_worker_count(0);
  }
}
