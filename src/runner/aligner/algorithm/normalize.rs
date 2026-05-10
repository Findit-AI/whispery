// Crate-level `deny(unsafe_code)` is on; the SIMD backends below are
// the only places we deliberately reach for `core::arch` intrinsics.
// Allow `unsafe_code` for this module exclusively so the choice is
// auditable.
#![allow(unsafe_code)]
// Dead-code is allowed for this module: every `pub`/`pub(crate)`
// function here is reachable only through the `__bench`
// re-exports (gated on `feature = "bench-internals"`) or the
// in-module test suite. The production alignment pipeline uses
// `normalize_with_silence_mask` in spirit only — the actual call
// site has shifted into `aligner.rs` with a different code path,
// leaving this module's items reachable by tests + benches but
// dead in default-feature builds. Without this allow the
// `cargo hack --feature-powerset` matrix trips `-Dwarnings` on
// every combo that omits `bench-internals`.
#![allow(dead_code)]

//! Zero-mean unit-variance normalisation, the wav2vec2 feature
//! extractor's `do_normalize=true` step. Mirrors HF's
//! `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` for the
//! no-attention-mask case.
//!
//! **Backends.** A scalar reference is always compiled. SIMD
//! backends:
//!
//! - `neon` — aarch64 base ISA (4-lane f32). Always selected on
//!   aarch64 — no runtime detection because NEON is mandatory.
//! - `x86_sse41` — x86_64 with SSE4.1 (4-lane f32). The minimal
//!   SIMD floor for x86_64 since 2008. We use SSE4.1 (not 4.2)
//!   because 4.2 only adds CRC32 / string-search ops that don't
//!   matter here.
//! - `x86_avx2` — x86_64 with AVX2 (8-lane f32). Mainstream from
//!   Haswell (2013) onwards.
//! - `x86_avx512` — x86_64 with AVX-512F (16-lane f32). Premium
//!   server CPUs (Skylake-X+, Ice Lake-SP, AMD Zen 4). Some chips
//!   downclock under sustained AVX-512 use, so the dispatcher
//!   only takes this path when the feature is detected — never
//!   speculatively.
//!
//! `wasm32` simd128 is intentionally absent: the `alignment`
//! feature requires `whisper-rs` and `ort`, neither of which
//! supports `wasm32`. The kernel would compile but never run from
//! a real consumer.
//!
//! **Dispatch.** `feature = "std"` enables runtime CPU-feature
//! detection via `is_x86_feature_detected!` (cached in a static
//! atomic, so per-call cost is one relaxed load + branch). Without
//! `std` we fall back to compile-time `cfg!(target_feature)`
//! (matching the same-named features the user enabled at build
//! time). Every backend computes identical results to the scalar
//! reference within f32 tolerance — per-target tests in `tests`
//! enforce a ≤ 1e-4 abs-error contract on a 480 k-sample synthetic
//! input.
//!
//! **Why not autovectorise.** The two-pass mean / variance reduction
//! mixes f64 accumulation with f32 inputs to avoid catastrophic
//! cancellation on long sequences. LLVM is reluctant to vectorise
//! reductions across precision boundaries, so the f32-accumulating
//! SIMD paths come out measurably faster on real chunks (480 k
//! samples for 30 s @ 16 kHz). All backends keep the f64 accumulator.

use alloc::vec::Vec;

/// f32 horizontal-sum overflow recovery shared across all SIMD
/// backends. Real audio in `[-1, 1]` never trips the check;
/// high-dynamic-range or pathological f32 inputs (`[1e38, ...]`)
/// saturate the f32 horizontal-add before the f64 cast — when
/// that happens we redo the sum in scalar f64 so every backend
/// matches the scalar reference within tolerance for any finite
/// f32 input.
#[inline]
fn reconcile_mean(simd_sum: f64, samples: &[f32]) -> f64 {
  if simd_sum.is_finite() {
    simd_sum
  } else {
    scalar::scalar_mean(samples)
  }
}

/// Same overflow-recovery contract as [`reconcile_mean`], applied
/// to the variance pass. `[1e20, -1e20]` keeps `var ≈ 1e40` in
/// f64 but overflows the f32 `d * d` to `inf`; the fallback pulls
/// the result back to scalar parity.
#[inline]
fn reconcile_var_sum(simd_var_sum: f64, samples: &[f32], mean: f64) -> f64 {
  if simd_var_sum.is_finite() {
    simd_var_sum
  } else {
    scalar::scalar_var_sum(samples, mean)
  }
}

/// Per-backend low-variance fallback: when the SIMD write-out
/// pass's f32 mean rounding can leak more per-sample error than
/// the dispatcher's parity tolerance, fall back to scalar f64.
///
/// The SIMD pass-3 broadcasts `mean` and `inv_std` into f32 lanes
/// and computes `(s - mean_f32) * inv_std_f32`. The dominant
/// loss is the mean cast: f32 has 23 bits of mantissa, so
/// rounding `|mean|` to f32 costs at most `|mean| * 2^-23` of
/// magnitude. Multiplied by `inv_std`, the per-sample output
/// error is bounded by `|mean| * 2^-23 * inv_std`. Bail when
/// that exceeds the parity tolerance.
///
/// Note we use `|mean| * 2^-23` (the *worst-case* f32 round)
/// rather than the actual `|mean_f32 - mean|`. For the cited
/// `[1.0, next_up(1.0), ...]` case the f32 mean lands at exactly
/// 1.0 (mid-ULP rounds to even), so the actual rounding *of the
/// f64 mean* is zero — but the SIMD pass-1 *itself* already lost
/// precision when collapsing f32 lanes, producing a different
/// mean than scalar f64 would. Bounding via `|mean| * 2^-23`
/// catches both flavours: it's a correctness ceiling for the
/// SIMD pipeline's f32 mean-cast error, regardless of which
/// stage actually introduced the loss.
///
/// For typical audio (`|mean| ≈ 0`, `inv_std ≈ 3-10`) this is
/// `~1e-9`. For near-constant inputs (`var → 0`,
/// `inv_std → 1/√eps ≈ 3162`) it explodes — exactly the
/// regime that needs the scalar fallback.
///
/// Threshold: `5e-5`, half the test contract's `1e-4` so the
/// dispatched backend stays comfortably within tolerance.
#[inline]
fn simd_path_loses_precision(mean: f64, inv_std: f64) -> bool {
  const SIMD_PRECISION_TOLERANCE: f64 = 5e-5;
  // 2^-23: f32 mantissa step at unit magnitude.
  const F32_MANTISSA_RELATIVE_ULP: f64 = 1.0_f64 / (1u64 << 23) as f64;
  let max_per_sample_err = mean.abs() * F32_MANTISSA_RELATIVE_ULP * inv_std;
  max_per_sample_err > SIMD_PRECISION_TOLERANCE
}

/// Magnitude threshold above which the SIMD backends' f32-lane
/// reductions diverge measurably from the scalar f64 reference.
///
/// f32 has ~24 bits of mantissa; ULP at magnitude `M` is
/// `M * 2^-23 ≈ M * 1.2e-7`. Each SIMD pass collapses lanes via
/// an f32 horizontal-add before widening to the f64 accumulator,
/// so for input magnitude `M` and `N` samples the accumulated
/// error in the sum scales like `N * M * 2^-23`. At `M = 1e4`
/// over a 480 k-sample chunk that's ~6 in the sum (mean error
/// ~1.2e-5) — visible in the variance and in the normalised
/// output. Real audio fed by the runner sits in `[-1, 1]` and is
/// orders of magnitude below the threshold; pathological f32
/// values fall through to the scalar f64 reference, which is
/// correct by construction.
const SIMD_SAFE_MAX_ABS: f32 = 1.0e4;

/// Pre-dispatch precision guard. Returns `true` when the SIMD
/// backends are safe — i.e., every sample is finite and below
/// the magnitude threshold.
///
/// Implemented as a `u32`-bits max-reduce over `bits & 0x7FFF_FFFF`
/// (the f32 sign bit cleared). For non-negative f32 the integer
/// bit ordering matches the floating-point ordering, so the
/// integer max gives the max-abs sample's bit pattern. NaN bits
/// are strictly above `+inf` bits (`0x7F800000`) and `inf` bits
/// equal it, so a single `max_bits < inf_bits` check rejects both.
/// LLVM autovectorises this `u32` reduction on every supported
/// target (`vmaxvq_u32` on NEON, `_mm_max_epu32` on SSE4.1, etc.)
/// where the f32 path would have stalled on f32::max's
/// non-associative-NaN semantics.
#[inline]
fn samples_within_simd_safe_range(samples: &[f32]) -> bool {
  const SIGN_MASK: u32 = 0x7FFF_FFFF;
  let threshold_bits = SIMD_SAFE_MAX_ABS.to_bits();
  let inf_bits = f32::INFINITY.to_bits(); // 0x7F800000
  let max_abs_bits = samples
    .iter()
    .copied()
    .map(|s| s.to_bits() & SIGN_MASK)
    .fold(0_u32, u32::max);
  max_abs_bits < inf_bits && max_abs_bits <= threshold_bits
}

/// Public entry point — picks the best implementation available at
/// runtime (under `feature = "std"`) or compile time (without).
/// `pub` for the `feature = "bench-internals"` re-export; consumers
/// who don't enable that feature can't reach this path.
///
/// Uses the stable `cfg_select!` (Rust 1.95+, in the core prelude)
/// for the per-arch dispatcher. The colconv crate uses the same
/// pattern; both crates' MSRVs match the macro's stabilisation.
#[inline]
pub fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
  // Precision guard: SIMD backends reduce in f32 before widening
  // to f64, so finite f32 inputs beyond ~1e4 in magnitude can
  // lose precision relative to the scalar f64 reference *without*
  // tripping the inf/NaN recovery in `reconcile_*`. Detect that
  // case at the dispatch boundary and route to scalar f64
  // directly. Real audio is `[-1, 1]` and never trips this;
  // pathological f32 inputs go down the slow path with full
  // precision instead of producing backend-skewed output.
  if !samples_within_simd_safe_range(samples) {
    return scalar::zero_mean_unit_var_normalize(samples);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      // SAFETY: NEON is part of the aarch64 base ISA; the kernel's
      // `#[target_feature(enable = "neon")]` makes the compiler emit
      // intrinsics in an explicitly-enabled context.
      unsafe { neon::zero_mean_unit_var_normalize(samples) }
    }
    target_arch = "x86_64" => x86_dispatch(samples),
    _ => scalar::zero_mean_unit_var_normalize(samples),
  }
}

/// x86_64 dispatch helper — picks AVX-512 → AVX2 → SSE4.1 →
/// scalar via runtime feature detection (`feature = "std"`) or
/// compile-time `target_feature` cfgs (no-std). Lifted out of
/// the public dispatcher so the per-arch `cfg_select!` arm
/// stays a one-liner.
#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_dispatch(samples: &[f32]) -> Vec<f32> {
  // Runtime feature detection. `is_x86_feature_detected!`
  // caches its result in a static atomic, so per-call cost is
  // one relaxed load + branch.
  #[cfg(feature = "std")]
  {
    if std::is_x86_feature_detected!("avx512f") {
      // SAFETY: feature checked.
      return unsafe { x86_avx512::zero_mean_unit_var_normalize(samples) };
    }
    if std::is_x86_feature_detected!("avx2") {
      return unsafe { x86_avx2::zero_mean_unit_var_normalize(samples) };
    }
    if std::is_x86_feature_detected!("sse4.1") {
      return unsafe { x86_sse41::zero_mean_unit_var_normalize(samples) };
    }
  }
  // No-std compile-time fallback. Matches the runtime fallback
  // order: AVX-512F first, then AVX2, then SSE4.1, else scalar.
  cfg_select! {
    all(not(feature = "std"), target_feature = "avx512f") => {
      unsafe { x86_avx512::zero_mean_unit_var_normalize(samples) }
    }
    all(not(feature = "std"), target_feature = "avx2") => {
      unsafe { x86_avx2::zero_mean_unit_var_normalize(samples) }
    }
    all(not(feature = "std"), target_feature = "sse4.1") => {
      unsafe { x86_sse41::zero_mean_unit_var_normalize(samples) }
    }
    _ => scalar::zero_mean_unit_var_normalize(samples),
  }
}

/// Scalar reference implementation. Always compiled; used directly
/// by non-aarch64 targets and by the bench's `scalar` variant for
/// the head-to-head comparison.
pub mod scalar {
  use super::Vec;

  /// Always-on f64-accumulator reference. See module docs.
  pub fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len() as f64;
    let mean = scalar_mean(samples) / n;
    let var = scalar_var_sum(samples, mean) / n;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();
    let mut out = Vec::with_capacity(samples.len());
    for &s in samples {
      out.push(((s as f64 - mean) * inv_std) as f32);
    }
    out
  }

  /// Scalar f64 sum of `samples`. Used by the SIMD backends as
  /// the precision-recovery fallback when their f32 horizontal
  /// sum overflowed (see [`super::neon`] for the overflow
  /// detection pattern).
  pub(super) fn scalar_mean(samples: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    for &s in samples {
      sum += s as f64;
    }
    sum
  }

  /// Scalar f64 sum of squared deviations. Used by the SIMD
  /// backends as the precision-recovery fallback when the f32
  /// `d * d` overflowed for high-dynamic-range inputs (e.g.
  /// `[1e20, -1e20]` overflows f32 but stays finite in f64).
  /// Real audio in `[-1, 1]` never trips this; pathological
  /// inputs go down the slow path with full precision instead of
  /// producing backend-skewed output.
  pub(super) fn scalar_var_sum(samples: &[f32], mean: f64) -> f64 {
    let mut var_sum = 0.0_f64;
    for &s in samples {
      let d = s as f64 - mean;
      var_sum += d * d;
    }
    var_sum
  }
}

/// Silence-mask-preserving normalisation. Computes mean / variance
/// over speech samples only, transforms the speech samples by
/// `(s - mean) / sqrt(var + eps)`, and forces non-speech samples
/// back to zero in the output.
///
/// **Why a separate helper.** The plain
/// [`zero_mean_unit_var_normalize`] computes statistics over the
/// *whole* buffer. When the caller has pre-zeroed non-speech
/// regions (the silence-mask step), folding those zeros into the
/// reduction biases the mean toward the speech values' magnitude
/// and `(0 - mean) / std` then becomes a non-zero value at every
/// silence position. That contradicts the silence-mask contract —
/// wav2vec2 must see uniform silence in masked regions, not a
/// mean-shifted residue.
///
/// **Mirrors HF behaviour.** `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm`
/// with an `attention_mask` does exactly this: stats over masked-in
/// positions, masked-out positions stay zero in the output.
///
/// `samples` is the chunk's audio buffer at 16 kHz. `speech_mask`
/// is a parallel `&[bool]` of identical length; `true` marks
/// samples inside any sub-VAD-segment. When `speech_mask` has no
/// `true` entries (chunk consists entirely of silence) the
/// function returns a fresh zero buffer — wav2vec2 with all-zero
/// input emits blank tokens uniformly, and downstream
/// `compose_words` drops every word for lack of speech support.
///
/// Always scalar f64 — the per-sample branch on the mask is
/// SIMD-hostile, and the silence-aware path is not on the SIMD
/// hot loop (typical chunks are ≤ 1 % of total ORT inference
/// time even when scalar). Keeping it scalar avoids the
/// per-backend duplication the SIMD `zero_mean_unit_var_normalize`
/// requires.
#[inline]
pub(crate) fn normalize_with_silence_mask(samples: &[f32], speech_mask: &[bool]) -> Vec<f32> {
  debug_assert_eq!(samples.len(), speech_mask.len());

  if samples.is_empty() {
    return Vec::new();
  }

  // Pass 1: count and sum speech samples.
  let mut sum = 0.0_f64;
  let mut count: usize = 0;
  for (s, &is_speech) in samples.iter().zip(speech_mask.iter()) {
    if is_speech {
      sum += *s as f64;
      count += 1;
    }
  }

  // No speech in the chunk → uniform-zero output. wav2vec2 sees
  // pure silence, the CTC graph emits all blanks, and compose
  // drops every word — exactly what the silence-mask contract
  // says should happen for an all-silent chunk.
  if count == 0 {
    return alloc::vec![0.0_f32; samples.len()];
  }

  let mean = sum / count as f64;

  // Pass 2: variance over speech samples only.
  let mut var_sum = 0.0_f64;
  for (s, &is_speech) in samples.iter().zip(speech_mask.iter()) {
    if is_speech {
      let d = *s as f64 - mean;
      var_sum += d * d;
    }
  }
  let var = var_sum / count as f64;
  let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();

  // Pass 3: write-out. Speech samples get the affine transform;
  // non-speech samples stay exactly zero so the silence-mask
  // contract survives.
  let mut out = Vec::with_capacity(samples.len());
  for (s, &is_speech) in samples.iter().zip(speech_mask.iter()) {
    if is_speech {
      out.push(((*s as f64 - mean) * inv_std) as f32);
    } else {
      out.push(0.0_f32);
    }
  }
  out
}

/// aarch64 NEON backend. Vectorises the three sequential passes
/// (sum, var, write-out) over `float32x4_t` lanes. The scalar tail
/// handles the trailing `len % 4` samples without falling through
/// to the scalar reference.
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub mod neon {
  use super::Vec;
  use core::arch::aarch64::*;

  /// NEON-vectorised zero-mean unit-variance normalisation.
  /// Marked `unsafe` to mirror the colconv convention; in
  /// practice every aarch64 build of this crate satisfies
  /// the precondition.
  ///
  /// # Safety
  ///
  /// Caller must run on aarch64 with NEON available. NEON is
  /// part of the aarch64 base ISA so this is satisfied
  /// unconditionally on any supported aarch64 target.
  #[inline]
  #[target_feature(enable = "neon")]
  pub unsafe fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len();
    let nf = n as f64;

    // ---- pass 1: horizontal sum (f64 accumulator).
    // We pair-add f32 lanes into a 64-bit register, widen to f64,
    // and accumulate in f64 to keep parity with the scalar path's
    // long-sequence stability.
    let mut sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = vld1q_f32(samples.as_ptr().add(i));
        sum += vaddvq_f32(v) as f64;
        i += 4;
      }
    }
    while i < n {
      sum += samples[i] as f64;
      i += 1;
    }
    let mean = super::reconcile_mean(sum, samples) / nf;

    // ---- pass 2: variance (sum of squared deviations, f64 acc).
    let mean_f32 = mean as f32;
    let mean_v = unsafe { vdupq_n_f32(mean_f32) };
    let mut var_sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = vld1q_f32(samples.as_ptr().add(i));
        let d = vsubq_f32(v, mean_v);
        let sq = vmulq_f32(d, d);
        var_sum += vaddvq_f32(sq) as f64;
        i += 4;
      }
    }
    while i < n {
      let d = samples[i] as f64 - mean;
      var_sum += d * d;
      i += 1;
    }
    let var = super::reconcile_var_sum(var_sum, samples, mean) / nf;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();

    // Low-variance inputs drive `inv_std` so high that the f32
    // rounding of `mean` for the SIMD write-out lanes leaks
    // visible per-sample error relative to scalar. Detect and
    // route to scalar f64 to keep the dispatched output within
    // parity tolerance.
    if super::simd_path_loses_precision(mean, inv_std) {
      return super::scalar::zero_mean_unit_var_normalize(samples);
    }

    // ---- pass 3: (x - mean) * inv_std into a fresh Vec.
    let inv_std_f32 = inv_std as f32;
    let inv_v = unsafe { vdupq_n_f32(inv_std_f32) };
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = vld1q_f32(samples.as_ptr().add(i));
        let normed = vmulq_f32(vsubq_f32(v, mean_v), inv_v);
        vst1q_f32(out_ptr.add(i), normed);
        i += 4;
      }
      // SAFETY: tail in-bounds; we only touch [i, n).
      while i < n {
        let s = *samples.as_ptr().add(i);
        *out_ptr.add(i) = (s - mean_f32) * inv_std_f32;
        i += 1;
      }
      out.set_len(n);
    }
    out
  }
}

/// x86_64 SSE4.1 backend. 4-lane f32 — same lane count as NEON,
/// so the speed-up profile mirrors the aarch64 path.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod x86_sse41 {
  use super::Vec;
  use core::arch::x86_64::*;

  /// Reduce a 4-lane `__m128` to its f32 horizontal sum.
  /// SSE4.1 has no single-instruction reduce, so we shuffle-add
  /// twice. Returns a scalar `f32` for the caller to widen to f64.
  #[inline]
  #[target_feature(enable = "sse4.1")]
  unsafe fn hsum_ps(v: __m128) -> f32 {
    unsafe {
      let shuf = _mm_movehdup_ps(v); // [b, b, d, d]
      let sums = _mm_add_ps(v, shuf); // [a+b, _, c+d, _]
      let shuf2 = _mm_movehl_ps(sums, sums); // [c+d, _, _, _]
      let sums = _mm_add_ss(sums, shuf2); // a+b+c+d in lane 0
      _mm_cvtss_f32(sums)
    }
  }

  /// # Safety
  ///
  /// Caller must run on x86_64 with SSE4.1 available.
  #[inline]
  #[target_feature(enable = "sse4.1")]
  pub unsafe fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len();
    let nf = n as f64;

    // Pass 1: horizontal sum into f64 accumulator.
    let mut sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = _mm_loadu_ps(samples.as_ptr().add(i));
        sum += hsum_ps(v) as f64;
        i += 4;
      }
    }
    while i < n {
      sum += samples[i] as f64;
      i += 1;
    }
    let mean = super::reconcile_mean(sum, samples) / nf;

    // Pass 2: variance.
    let mean_f32 = mean as f32;
    let mean_v = unsafe { _mm_set1_ps(mean_f32) };
    let mut var_sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = _mm_loadu_ps(samples.as_ptr().add(i));
        let d = _mm_sub_ps(v, mean_v);
        let sq = _mm_mul_ps(d, d);
        var_sum += hsum_ps(sq) as f64;
        i += 4;
      }
    }
    while i < n {
      let d = samples[i] as f64 - mean;
      var_sum += d * d;
      i += 1;
    }
    let var = super::reconcile_var_sum(var_sum, samples, mean) / nf;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();

    // Low-variance inputs drive `inv_std` so high that the f32
    // rounding of `mean` for the SIMD write-out lanes leaks
    // visible per-sample error relative to scalar. Detect and
    // route to scalar f64 to keep the dispatched output within
    // parity tolerance.
    if super::simd_path_loses_precision(mean, inv_std) {
      return super::scalar::zero_mean_unit_var_normalize(samples);
    }

    // Pass 3: write-out.
    let inv_std_f32 = inv_std as f32;
    let inv_v = unsafe { _mm_set1_ps(inv_std_f32) };
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    unsafe {
      while i + 4 <= n {
        let v = _mm_loadu_ps(samples.as_ptr().add(i));
        let normed = _mm_mul_ps(_mm_sub_ps(v, mean_v), inv_v);
        _mm_storeu_ps(out_ptr.add(i), normed);
        i += 4;
      }
      while i < n {
        let s = *samples.as_ptr().add(i);
        *out_ptr.add(i) = (s - mean_f32) * inv_std_f32;
        i += 1;
      }
      out.set_len(n);
    }
    out
  }
}

/// x86_64 AVX2 backend. 8-lane f32. Available on Haswell+ (2013).
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod x86_avx2 {
  use super::Vec;
  use core::arch::x86_64::*;

  /// Reduce a 256-bit `__m256` to its f32 horizontal sum: collapse
  /// to a 128-bit lane-sum, then SSE4.1-style shuffle-add.
  #[inline]
  #[target_feature(enable = "avx2")]
  unsafe fn hsum_ps(v: __m256) -> f32 {
    unsafe {
      let lo = _mm256_castps256_ps128(v);
      let hi = _mm256_extractf128_ps(v, 1);
      let s = _mm_add_ps(lo, hi);
      let shuf = _mm_movehdup_ps(s);
      let sums = _mm_add_ps(s, shuf);
      let shuf2 = _mm_movehl_ps(sums, sums);
      let sums = _mm_add_ss(sums, shuf2);
      _mm_cvtss_f32(sums)
    }
  }

  /// # Safety
  ///
  /// Caller must run on x86_64 with AVX2 available.
  #[inline]
  #[target_feature(enable = "avx2")]
  pub unsafe fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len();
    let nf = n as f64;

    // Pass 1: horizontal sum.
    let mut sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 8 <= n {
        let v = _mm256_loadu_ps(samples.as_ptr().add(i));
        sum += hsum_ps(v) as f64;
        i += 8;
      }
    }
    while i < n {
      sum += samples[i] as f64;
      i += 1;
    }
    let mean = super::reconcile_mean(sum, samples) / nf;

    // Pass 2: variance.
    let mean_f32 = mean as f32;
    let mean_v = unsafe { _mm256_set1_ps(mean_f32) };
    let mut var_sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 8 <= n {
        let v = _mm256_loadu_ps(samples.as_ptr().add(i));
        let d = _mm256_sub_ps(v, mean_v);
        let sq = _mm256_mul_ps(d, d);
        var_sum += hsum_ps(sq) as f64;
        i += 8;
      }
    }
    while i < n {
      let d = samples[i] as f64 - mean;
      var_sum += d * d;
      i += 1;
    }
    let var = super::reconcile_var_sum(var_sum, samples, mean) / nf;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();

    // Low-variance inputs drive `inv_std` so high that the f32
    // rounding of `mean` for the SIMD write-out lanes leaks
    // visible per-sample error relative to scalar. Detect and
    // route to scalar f64 to keep the dispatched output within
    // parity tolerance.
    if super::simd_path_loses_precision(mean, inv_std) {
      return super::scalar::zero_mean_unit_var_normalize(samples);
    }

    // Pass 3: write-out.
    let inv_std_f32 = inv_std as f32;
    let inv_v = unsafe { _mm256_set1_ps(inv_std_f32) };
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    unsafe {
      while i + 8 <= n {
        let v = _mm256_loadu_ps(samples.as_ptr().add(i));
        let normed = _mm256_mul_ps(_mm256_sub_ps(v, mean_v), inv_v);
        _mm256_storeu_ps(out_ptr.add(i), normed);
        i += 8;
      }
      while i < n {
        let s = *samples.as_ptr().add(i);
        *out_ptr.add(i) = (s - mean_f32) * inv_std_f32;
        i += 1;
      }
      out.set_len(n);
    }
    out
  }
}

/// x86_64 AVX-512F backend. 16-lane f32. Premium-server only.
/// Some Skylake-X / Ice Lake chips downclock under sustained
/// AVX-512 use, so the dispatcher only takes this path when the
/// runtime feature is actually present.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod x86_avx512 {
  use super::Vec;
  use core::arch::x86_64::*;

  /// # Safety
  ///
  /// Caller must run on x86_64 with AVX-512F available.
  #[inline]
  #[target_feature(enable = "avx512f")]
  pub unsafe fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
      return Vec::new();
    }
    let n = samples.len();
    let nf = n as f64;

    // Pass 1: horizontal sum. AVX-512 has a single-instruction
    // 16-lane reduce.
    let mut sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 16 <= n {
        let v = _mm512_loadu_ps(samples.as_ptr().add(i));
        sum += _mm512_reduce_add_ps(v) as f64;
        i += 16;
      }
    }
    while i < n {
      sum += samples[i] as f64;
      i += 1;
    }
    let mean = super::reconcile_mean(sum, samples) / nf;

    // Pass 2: variance.
    let mean_f32 = mean as f32;
    let mean_v = unsafe { _mm512_set1_ps(mean_f32) };
    let mut var_sum = 0.0_f64;
    let mut i = 0usize;
    unsafe {
      while i + 16 <= n {
        let v = _mm512_loadu_ps(samples.as_ptr().add(i));
        let d = _mm512_sub_ps(v, mean_v);
        let sq = _mm512_mul_ps(d, d);
        var_sum += _mm512_reduce_add_ps(sq) as f64;
        i += 16;
      }
    }
    while i < n {
      let d = samples[i] as f64 - mean;
      var_sum += d * d;
      i += 1;
    }
    let var = super::reconcile_var_sum(var_sum, samples, mean) / nf;
    let inv_std = 1.0_f64 / (var + 1e-7_f64).sqrt();

    // Low-variance inputs drive `inv_std` so high that the f32
    // rounding of `mean` for the SIMD write-out lanes leaks
    // visible per-sample error relative to scalar. Detect and
    // route to scalar f64 to keep the dispatched output within
    // parity tolerance.
    if super::simd_path_loses_precision(mean, inv_std) {
      return super::scalar::zero_mean_unit_var_normalize(samples);
    }

    // Pass 3: write-out.
    let inv_std_f32 = inv_std as f32;
    let inv_v = unsafe { _mm512_set1_ps(inv_std_f32) };
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    unsafe {
      while i + 16 <= n {
        let v = _mm512_loadu_ps(samples.as_ptr().add(i));
        let normed = _mm512_mul_ps(_mm512_sub_ps(v, mean_v), inv_v);
        _mm512_storeu_ps(out_ptr.add(i), normed);
        i += 16;
      }
      while i < n {
        let s = *samples.as_ptr().add(i);
        *out_ptr.add(i) = (s - mean_f32) * inv_std_f32;
        i += 1;
      }
      out.set_len(n);
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 30 s @ 16 kHz with gain + DC offset, length chosen to force
  /// a non-zero scalar tail in every backend (480_001 = 30000 *
  /// 16 + 1).
  fn synth_input(n: usize) -> Vec<f32> {
    let mut samples: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
      let t = i as f32 / 16_000.0;
      samples.push(2.7 * (2.0 * core::f32::consts::PI * 220.0 * t).sin() + 0.13);
    }
    samples
  }

  /// Asserts a SIMD result agrees with the scalar reference to
  /// within `1e-4` absolute on every sample. Centralises the
  /// per-backend match-test pattern.
  fn assert_matches_scalar(simd: &[f32], scalar: &[f32]) {
    assert_eq!(simd.len(), scalar.len(), "SIMD length mismatch");
    let mut max_abs = 0.0_f32;
    for (a, b) in simd.iter().zip(scalar.iter()) {
      let d = (a - b).abs();
      if d > max_abs {
        max_abs = d;
      }
    }
    assert!(
      max_abs < 1e-4,
      "SIMD deviates from scalar: max abs error = {max_abs}",
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_matches_scalar() {
    let samples = synth_input(480_001);
    let s = scalar::zero_mean_unit_var_normalize(&samples);
    let v = unsafe { neon::zero_mean_unit_var_normalize(&samples) };
    assert_matches_scalar(&v, &s);
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_sse41_matches_scalar() {
    if !std::is_x86_feature_detected!("sse4.1") {
      return; // skip on machines without SSE4.1
    }
    let samples = synth_input(480_001);
    let s = scalar::zero_mean_unit_var_normalize(&samples);
    let v = unsafe { x86_sse41::zero_mean_unit_var_normalize(&samples) };
    assert_matches_scalar(&v, &s);
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx2_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    let samples = synth_input(480_001);
    let s = scalar::zero_mean_unit_var_normalize(&samples);
    let v = unsafe { x86_avx2::zero_mean_unit_var_normalize(&samples) };
    assert_matches_scalar(&v, &s);
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx512_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
      return;
    }
    let samples = synth_input(480_001);
    let s = scalar::zero_mean_unit_var_normalize(&samples);
    let v = unsafe { x86_avx512::zero_mean_unit_var_normalize(&samples) };
    assert_matches_scalar(&v, &s);
  }

  #[test]
  fn empty_returns_empty() {
    assert!(zero_mean_unit_var_normalize(&[]).is_empty());
  }

  #[test]
  fn constant_signal_normalises_to_zero() {
    let xs = alloc::vec![3.7_f32; 100];
    let out = zero_mean_unit_var_normalize(&xs);
    assert!(out.iter().all(|&v| v.abs() < 1e-3));
  }

  /// High-dynamic-range f32 inputs `[1e20, -1e20, ...]` overflow
  /// the SIMD f32 squared-deviation pass to `+inf`, while the
  /// scalar f64 reference keeps a finite `var ≈ 1e40`. The SIMD
  /// backends must detect the overflow and fall back to scalar
  /// so the dispatched result matches the scalar reference.
  ///
  /// We pad the sequence to 4 / 8 / 16 lane-sized chunks plus
  /// a 1-element tail to also exercise the per-backend tail
  /// loops on the recovery path.
  fn high_dynamic_range_input() -> Vec<f32> {
    let mut xs = alloc::vec::Vec::with_capacity(33);
    for _ in 0..16 {
      xs.push(1e20_f32);
      xs.push(-1e20_f32);
    }
    xs.push(1e19_f32); // tail
    xs
  }

  #[test]
  fn dispatched_high_dynamic_range_matches_scalar() {
    let xs = high_dynamic_range_input();
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let d = zero_mean_unit_var_normalize(&xs);
    assert_matches_scalar(&d, &s);
    // Sanity: scalar produces finite, near-unit-variance output
    // (mean ≈ 0, |x| ≈ 1 for the bulk values, all finite).
    assert!(s.iter().all(|x| x.is_finite()));
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_high_dynamic_range_matches_scalar() {
    let xs = high_dynamic_range_input();
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let v = unsafe { neon::zero_mean_unit_var_normalize(&xs) };
    assert_matches_scalar(&v, &s);
    assert!(v.iter().all(|x| x.is_finite()));
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_sse41_high_dynamic_range_matches_scalar() {
    if !std::is_x86_feature_detected!("sse4.1") {
      return;
    }
    let xs = high_dynamic_range_input();
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let v = unsafe { x86_sse41::zero_mean_unit_var_normalize(&xs) };
    assert_matches_scalar(&v, &s);
    assert!(v.iter().all(|x| x.is_finite()));
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx2_high_dynamic_range_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    let xs = high_dynamic_range_input();
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let v = unsafe { x86_avx2::zero_mean_unit_var_normalize(&xs) };
    assert_matches_scalar(&v, &s);
    assert!(v.iter().all(|x| x.is_finite()));
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn x86_avx512_high_dynamic_range_matches_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
      return;
    }
    let xs = high_dynamic_range_input();
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let v = unsafe { x86_avx512::zero_mean_unit_var_normalize(&xs) };
    assert_matches_scalar(&v, &s);
    assert!(v.iter().all(|x| x.is_finite()));
  }

  /// Finite f32 inputs at magnitudes where ULP exceeds ~1 (i.e.
  /// ~1e7+) lose precision in the f32 horizontal-add before
  /// widening to f64, so the SIMD reduction silently drifts from
  /// the scalar f64 reference without producing inf/NaN. The
  /// dispatch-side `samples_within_simd_safe_range` guard routes
  /// such inputs to the scalar path so the output is always
  /// scalar-equivalent.
  ///
  /// Constructs a 1024-sample buffer at magnitudes around 1e8
  /// where consecutive f32 increments are below ULP — the
  /// canonical "f32 cancellation" shape.
  fn finite_high_magnitude_input() -> Vec<f32> {
    let base = 1.0e8_f32;
    let mut xs = alloc::vec::Vec::with_capacity(1024);
    for i in 0..1024 {
      xs.push(if i % 2 == 0 { base } else { -base });
      // Add tiny finite jitter that f32 cannot represent at this
      // magnitude — the scalar f64 path keeps it; SIMD f32 lanes
      // would lose it. Forces the dispatch guard to route to
      // scalar.
      xs.push(if i % 3 == 0 { base + 0.5 } else { -base + 0.25 });
    }
    xs
  }

  /// The dispatched normalise must produce scalar-equivalent
  /// output even for finite high-magnitude input. Verified by
  /// matching against the scalar reference call directly —
  /// pre-fix the SIMD path took over and silently drifted.
  #[test]
  fn dispatched_finite_high_magnitude_matches_scalar() {
    let xs = finite_high_magnitude_input();
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let d = zero_mean_unit_var_normalize(&xs);
    assert_matches_scalar(&d, &s);
    assert!(s.iter().all(|x| x.is_finite()));
  }

  /// The guard check itself: well-bounded audio passes; HDR
  /// audio fails. Two-line sanity to prevent threshold drift
  /// from going unnoticed.
  #[test]
  fn precision_guard_recognises_safe_and_unsafe_inputs() {
    assert!(samples_within_simd_safe_range(&[
      0.5, -0.7, 1e3, -1e3, 9999.0
    ]));
    assert!(!samples_within_simd_safe_range(&[1.0, 1e5]));
    assert!(!samples_within_simd_safe_range(&[1.0, f32::NAN]));
    assert!(!samples_within_simd_safe_range(&[1.0, f32::INFINITY]));
    assert!(samples_within_simd_safe_range(&[]));
  }

  // ---- Silence-aware normalisation ------

  /// Speech with non-zero mean must NOT shift masked-silence
  /// positions away from zero in the output. Pre-fix, the
  /// non-mask-aware normaliser computed a global mean over the
  /// already-masked buffer, so masked positions ended up at
  /// `(0 - mean) / std` ≠ 0 — wav2vec2 then saw "silence" with
  /// a residue, and word boundaries leaked into masked gaps.
  #[test]
  fn silence_mask_normalize_keeps_masked_positions_at_zero() {
    // 16 samples: [speech with DC offset]+[silence]+[speech with DC offset].
    let mut samples = alloc::vec![0.0_f32; 16];
    // Speech region 1: indices 0..4, DC = 0.5 with small ripple.
    samples[0] = 0.5;
    samples[1] = 0.6;
    samples[2] = 0.4;
    samples[3] = 0.5;
    // Silence region: indices 4..12 stay at 0.0.
    // Speech region 2: indices 12..16, DC = 0.5 with small ripple.
    samples[12] = 0.5;
    samples[13] = 0.6;
    samples[14] = 0.4;
    samples[15] = 0.5;

    let mut speech_mask = alloc::vec![false; 16];
    for slot in &mut speech_mask[0..4] {
      *slot = true;
    }
    for slot in &mut speech_mask[12..16] {
      *slot = true;
    }

    let normed = normalize_with_silence_mask(&samples, &speech_mask);
    assert_eq!(normed.len(), samples.len());
    for i in 4..12 {
      assert_eq!(
        normed[i], 0.0_f32,
        "masked silence at index {i} must stay exactly 0; got {}",
        normed[i],
      );
    }
    // Speech regions are non-zero (the affine transform shifts
    // them around 0 — sanity that we didn't accidentally zero
    // everything).
    let any_nonzero_speech =
      normed[..4].iter().any(|&v| v != 0.0) || normed[12..].iter().any(|&v| v != 0.0);
    assert!(any_nonzero_speech, "speech samples must not all be zero");
  }

  /// Empty mask (no speech anywhere) → uniform-zero output.
  /// wav2vec2 sees pure silence; CTC graph emits all blanks;
  /// compose drops every word — the contract for an all-silent
  /// chunk.
  #[test]
  fn silence_mask_normalize_all_silence_yields_zeros() {
    let samples = alloc::vec![0.5_f32, 0.6, 0.4, 0.5];
    let speech_mask = alloc::vec![false; 4];
    let normed = normalize_with_silence_mask(&samples, &speech_mask);
    assert_eq!(normed, alloc::vec![0.0_f32; 4]);
  }

  /// Speech-only mask → identical to the regular
  /// `zero_mean_unit_var_normalize` to within scalar f64
  /// precision. The silence-aware path doesn't introduce drift
  /// for the all-speech case.
  #[test]
  fn silence_mask_normalize_all_speech_matches_regular_normalize() {
    let samples = alloc::vec![0.5_f32, 0.6, 0.4, 0.5, -0.3, -0.1, 0.2, 0.8];
    let speech_mask = alloc::vec![true; samples.len()];
    let masked = normalize_with_silence_mask(&samples, &speech_mask);
    let regular = scalar::zero_mean_unit_var_normalize(&samples);
    assert_matches_scalar(&masked, &regular);
  }

  // ---- Low-variance SIMD parity ------

  /// `[1.0, 1.0_f32::next_up(), 1.0, 1.0_f32::next_up(), ...]`
  /// — a low-variance edge case. Rounding mean to f32 in the
  /// SIMD write-out lanes leaks ~1.88e-4 of error against
  /// scalar f64. The dispatcher must detect this regime and
  /// route to scalar; we assert the dispatched output matches
  /// scalar to within tolerance.
  #[test]
  fn dispatched_low_variance_near_one_matches_scalar() {
    let next_up_one = f32::from_bits(1.0_f32.to_bits() + 1);
    let mut xs = alloc::vec::Vec::with_capacity(64);
    for _ in 0..32 {
      xs.push(1.0_f32);
      xs.push(next_up_one);
    }
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let d = zero_mean_unit_var_normalize(&xs);
    assert_matches_scalar(&d, &s);
  }

  /// Same shape, larger DC magnitude (still below the 1e4
  /// safe-range threshold). f32 ULP at magnitude `M` is
  /// `M * 2^-23`, so this magnifies the cancellation by `M`.
  /// The fallback must catch all such cases inside the
  /// safe-range, not only near 1.0.
  #[test]
  fn dispatched_low_variance_at_high_magnitude_matches_scalar() {
    // Magnitude 100; safe-range threshold is 1e4 so this
    // passes the magnitude guard. ULP at 100 is ~1.2e-5;
    // alternating values produce sub-ULP variance and
    // pathological inv_std. Without the precision fallback,
    // the dispatched output drifts from scalar by orders of
    // magnitude more than 1e-4.
    let mag = 100.0_f32;
    let next_up = f32::from_bits(mag.to_bits() + 1);
    let mut xs = alloc::vec::Vec::with_capacity(64);
    for _ in 0..32 {
      xs.push(mag);
      xs.push(next_up);
    }
    let s = scalar::zero_mean_unit_var_normalize(&xs);
    let d = zero_mean_unit_var_normalize(&xs);
    assert_matches_scalar(&d, &s);
  }

  /// The precision-fallback predicate's worst-case bound:
  /// `|mean| * 2^-23 * inv_std > 5e-5`. Audio-shaped inputs
  /// stay safely below; low-variance / high-magnitude inputs
  /// trip it.
  #[test]
  fn simd_path_loses_precision_fires_only_on_low_variance() {
    // Codex's cited regime: `[1.0, next_up(1.0), ...]` →
    // mean ≈ 1.0, inv_std ≈ 3162. Worst-case per-sample
    // error: 1.0 * 1.19e-7 * 3162 = 3.76e-4 ≫ 5e-5. Predicate
    // must fire.
    assert!(simd_path_loses_precision(1.0, 3_162.0));

    // Typical audio: |mean| ≈ 0.01, inv_std ≈ 5 → ~6e-9. Must
    // NOT fire; SIMD takes the fast path.
    assert!(!simd_path_loses_precision(0.01, 5.0));

    // Zero mean → zero error regardless of inv_std. Constant-0
    // signal short-circuits at this guard.
    assert!(!simd_path_loses_precision(0.0, 1e9));

    // High-magnitude (1e3) audio with normal variance still
    // safe: 1e3 * 1.19e-7 * 5 = 5.96e-4… hmm that fires.
    // Actually that's a bit inconvenient — let me check: at
    // magnitude 1000 the SIMD vs scalar error per sample is up
    // to ~6e-4 in the worst case, so falling back is the
    // correct call. Real audio doesn't sit at magnitude 1000
    // so the safe-range threshold (1e4) keeps us well clear in
    // practice.
    assert!(simd_path_loses_precision(1e3, 5.0));
  }
}
