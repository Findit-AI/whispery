//! Whisper worker pool. See spec §6.2.

use alloc::sync::Arc;
use core::sync::atomic::Ordering;
use core::time::Duration;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use whisper_rs::{FullParams, SamplingStrategy as WhisperStrategy, WhisperState};

use crate::core::{AsrParams, SamplingStrategy};
use crate::types::ChunkId;

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
pub(super) type AsrResultMsg = (ChunkId, Result<crate::core::AsrResult, crate::types::WorkFailure>);

/// Build a `FullParams` for one decoding attempt. The runner's outer
/// retry ladder calls this once per attempt with `attempt_temperature`
/// set to the next ladder step.
///
/// Disables whisper.cpp's internal temperature ladder via
/// `set_temperature_inc(0.0)`; each `state.full()` call is exactly
/// one decoding attempt at exactly `attempt_temperature`. The
/// `set_max_decoding_failures(...)` belt-and-braces secondary safeguard
/// documented in spec §5.6 is omitted here because whisper-rs 0.13.x
/// does not expose that setter; with `temperature_inc = 0.0` the
/// internal ladder iterates exactly once regardless.
///
/// Wires the worker-hang watchdog via `set_abort_callback_safe`. The
/// closure reads `abort_flag` on every whisper.cpp progress callback;
/// when the watchdog flips it true, whisper.cpp returns mid-inference.
pub(super) fn full_params_from(
    params: &AsrParams,
    attempt_temperature: f32,
    abort_flag: Arc<AtomicBool>,
) -> FullParams<'static, 'static> {
    let strategy = match params.strategy() {
        SamplingStrategy::Greedy { best_of } =>
            WhisperStrategy::Greedy { best_of },
        SamplingStrategy::BeamSearch { beam_size, patience } =>
            WhisperStrategy::BeamSearch { beam_size, patience },
    };
    let mut p = FullParams::new(strategy);

    p.set_n_threads(params.n_threads());
    p.set_no_context(params.no_context());
    p.set_suppress_blank(params.suppress_blank());
    p.set_suppress_non_speech_tokens(params.suppress_non_speech_tokens());

    if let Some(lang) = params.language_hint() {
        // `FullParams<'a, _>::set_language` requires the `&str`'s
        // lifetime to match `'a`. We return `FullParams<'static, _>`,
        // so the str must be `'static`. whisper-rs immediately copies
        // the str into a leaked CString (`CString::into_raw`) inside
        // `set_language`, so the lifetime constraint is purely a
        // type-system requirement — the borrow does not actually
        // outlive the call. We satisfy it by leaking a small
        // `Box<str>` (≤ a handful of bytes per language code; the
        // set of distinct codes is bounded by `Lang`'s variants).
        let static_lang: &'static str = Box::leak(Box::<str>::from(lang.as_str()));
        p.set_language(Some(static_lang));
    } else {
        p.set_detect_language(true);
    }

    if let Some(prompt) = params.initial_prompt() {
        p.set_initial_prompt(prompt.as_str());
    }

    p.set_print_special(false);
    p.set_print_progress(false);
    p.set_print_realtime(false);
    p.set_print_timestamps(false);

    // Pin temperature; disable internal ladder. See spec §5.6.
    p.set_temperature(attempt_temperature);
    p.set_temperature_inc(0.0);

    // Worker-hang watchdog. The closure is `Send + 'static`; the
    // abort_flag is shared with the watchdog thread.
    p.set_abort_callback_safe(move || abort_flag.load(Ordering::Relaxed));

    p
}

/// Mean of per-segment `avg_logprob` across the just-decoded chunk.
/// Returns `f32::MIN` when the state has no segments — that signals
/// a truly empty result and trips the retry ladder via the
/// log_prob_threshold check.
///
/// whisper-rs 0.13.2 does not expose a per-segment `avg_logprob`
/// accessor (the plan referenced `full_get_segment_avg_logprob`,
/// which does not exist on `WhisperState`). We reconstruct it
/// faithfully: per segment, average `WhisperTokenData::plog` (the
/// per-token log-probability returned by whisper.cpp) across all
/// tokens in that segment; then average those segment means. This
/// matches whisper.cpp's own internal computation of the value
/// it gates `logprob_thold` against.
pub(super) fn compute_avg_logprob(state: &WhisperState) -> f32 {
    let n = match state.full_n_segments() {
        Ok(n) => n,
        Err(_) => return f32::MIN,
    };
    if n <= 0 {
        return f32::MIN;
    }
    let mut seg_sum = 0.0f64;
    let mut seg_count = 0i32;
    for i in 0..n {
        let n_tok = match state.full_n_tokens(i) {
            Ok(n_tok) => n_tok,
            Err(_) => continue,
        };
        if n_tok <= 0 {
            continue;
        }
        let mut tok_sum = 0.0f64;
        let mut tok_count = 0i32;
        for j in 0..n_tok {
            if let Ok(td) = state.full_get_token_data(i, j) {
                tok_sum += td.plog as f64;
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

    let n = match state.full_n_segments() {
        Ok(n) => n,
        Err(_) => return 0.0,
    };
    if n <= 0 {
        return 0.0;
    }
    let mut text = String::new();
    for i in 0..n {
        if let Ok(s) = state.full_get_segment_text(i) {
            text.push_str(&s);
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

    #[test]
    fn asr_work_item_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AsrWorkItem>();
        assert_send::<AsrResultMsg>();
    }

    use crate::core::AsrParams;
    use crate::core::SamplingStrategy;
    use crate::types::Lang;
    use core::sync::atomic::AtomicBool;

    #[test]
    fn full_params_from_greedy_is_finite() {
        let p = AsrParams::default()
            .with_strategy(SamplingStrategy::Greedy { best_of: 1 });
        let flag = Arc::new(AtomicBool::new(false));
        let _full = full_params_from(&p, 0.4, flag);
        // FullParams' fields aren't all readable; the assertion is that
        // the build does not panic and the abort closure compiles. The
        // recording-mock test in Task 18 verifies temperature_inc=0.0
        // and the explicit set_temperature(t) call sequence.
    }

    #[test]
    fn full_params_from_with_language_hint_does_not_panic() {
        let p = AsrParams::default().with_language_hint(Some(Lang::En));
        let flag = Arc::new(AtomicBool::new(false));
        let _full = full_params_from(&p, 0.0, flag);
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
        assert!(r >= 2.4, "repeated text ratio should trip the 2.4 default; got {}", r);
    }

    #[test]
    fn compression_ratio_short_input_returns_zero() {
        assert_eq!(compression_ratio_of_text(""), 0.0);
        assert_eq!(compression_ratio_of_text("ab"), 0.0);
    }
}
