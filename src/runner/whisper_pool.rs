//! Whisper worker pool. See spec §6.2.

use core::time::Duration;
use std::path::{Path, PathBuf};

/// Configuration for the runner's whisper worker pool.
///
/// Fields are private; use [`WhisperPoolConfig::new`] (or
/// [`Default::default`]) and the `set_*` / `with_*` accessors. Most
/// accessors are `const fn` and run in const contexts. Path-typed
/// fields (`model_path`) cannot be `const fn` because [`PathBuf`]
/// does not currently expose const accessors.
#[derive(Clone, Debug)]
pub struct WhisperPoolConfig {
    worker_count: usize,
    model_path: PathBuf,
    use_gpu: bool,
    gpu_device: i32,
    flash_attn: bool,
    max_queued_chunks: usize,
    block_on_full_queue: bool,
    dispatch_idle_poll: Duration,
    timeout_streak_threshold: u32,
}

impl WhisperPoolConfig {
    /// Construct a config with all defaults except `model_path`.
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

    /// Worker thread count. Default
    /// `max(1, num_cpus::get_physical() / 2)` on CPU backends, `1`
    /// on GPU backends (cuda / metal / vulkan / hipblas / coreml
    /// active).
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
    /// See spec §6.4.2 for the side-effect contract.
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
    /// GPU. See spec §6.4.3.
    pub const fn timeout_streak_threshold(&self) -> u32 {
        self.timeout_streak_threshold
    }

    // --- Mutating setters ----------------------------------------

    /// Set [`Self::worker_count`].
    pub const fn set_worker_count(&mut self, value: usize) {
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
    pub const fn with_worker_count(mut self, value: usize) -> Self {
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

/// Detect the active backend via Cargo features. CPU-only builds get
/// half the physical cores (min 1); GPU builds default to 1 worker
/// because whisper.cpp serialises on a single GPU regardless of
/// concurrent `WhisperState`s.
fn default_worker_count() -> usize {
    if is_gpu_backend_active() {
        1
    } else {
        let physical = num_cpus::get_physical();
        core::cmp::max(1, physical / 2)
    }
}

/// Default threshold per spec §6.4.3: 1 on CPU, 3 on GPU.
const fn default_timeout_streak_threshold() -> u32 {
    if is_gpu_backend_active_const() { 3 } else { 1 }
}

/// `cfg!(...)` form that the `default_worker_count` runtime helper uses.
fn is_gpu_backend_active() -> bool {
    cfg!(any(
        feature = "_whisper_cuda",
        feature = "_whisper_metal",
        feature = "_whisper_vulkan",
        feature = "_whisper_hipblas",
        feature = "_whisper_coreml",
    ))
}

/// `const fn` mirror for the threshold default. Each `feature = ".."`
/// branch is independently `cfg!`-able.
const fn is_gpu_backend_active_const() -> bool {
    cfg!(any(
        feature = "_whisper_cuda",
        feature = "_whisper_metal",
        feature = "_whisper_vulkan",
        feature = "_whisper_hipblas",
        feature = "_whisper_coreml",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip() {
        let cfg = WhisperPoolConfig::new("/tmp/model.bin");
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

    #[test]
    fn with_setters_round_trip() {
        let cfg = WhisperPoolConfig::new("/tmp/model.bin")
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
        let mut cfg = WhisperPoolConfig::new("/tmp/model.bin");
        cfg.set_worker_count(3);
        cfg.set_model_path("/var/cache/model.gguf");
        assert_eq!(cfg.worker_count(), 3);
        assert_eq!(cfg.model_path(), Path::new("/var/cache/model.gguf"));
    }
}
