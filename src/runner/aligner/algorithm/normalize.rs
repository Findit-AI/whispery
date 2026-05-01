// Crate-level `deny(unsafe_code)` is on; the SIMD backends below are
// the only places we deliberately reach for `core::arch` intrinsics.
// Allow `unsafe_code` for this module exclusively so the choice is
// auditable.
#![allow(unsafe_code)]

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
/// to the variance pass. Codex round-10 [medium]: `[1e20, -1e20]`
/// keeps `var ≈ 1e40` in f64 but overflows the f32 `d * d` to
/// `inf`; the fallback pulls the result back to scalar parity.
#[inline]
fn reconcile_var_sum(simd_var_sum: f64, samples: &[f32], mean: f64) -> f64 {
  if simd_var_sum.is_finite() {
    simd_var_sum
  } else {
    scalar::scalar_var_sum(samples, mean)
  }
}

/// Public entry point — picks the best implementation available at
/// runtime (under `feature = "std"`) or compile time (without).
/// `pub` for the `feature = "bench-internals"` re-export; consumers
/// who don't enable that feature can't reach this path.
#[inline]
pub fn zero_mean_unit_var_normalize(samples: &[f32]) -> Vec<f32> {
  #[cfg(target_arch = "aarch64")]
  {
    // SAFETY: NEON is part of the aarch64 base ISA; the kernel's
    // `#[target_feature(enable = "neon")]` makes the compiler emit
    // intrinsics in an explicitly-enabled context.
    return unsafe { neon::zero_mean_unit_var_normalize(samples) };
  }
  #[cfg(target_arch = "x86_64")]
  {
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
    // No-std compile-time fallback.
    #[cfg(all(not(feature = "std"), target_feature = "avx512f"))]
    {
      return unsafe { x86_avx512::zero_mean_unit_var_normalize(samples) };
    }
    #[cfg(all(
      not(feature = "std"),
      target_feature = "avx2",
      not(target_feature = "avx512f")
    ))]
    {
      return unsafe { x86_avx2::zero_mean_unit_var_normalize(samples) };
    }
    #[cfg(all(
      not(feature = "std"),
      target_feature = "sse4.1",
      not(target_feature = "avx2"),
      not(target_feature = "avx512f")
    ))]
    {
      return unsafe { x86_sse41::zero_mean_unit_var_normalize(samples) };
    }
  }
  scalar::zero_mean_unit_var_normalize(samples)
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
  /// `d * d` overflowed for high-dynamic-range inputs (Codex
  /// round-10 [medium]: `[1e20, -1e20]` overflows f32 but stays
  /// finite in f64). Real audio in `[-1, 1]` never trips this;
  /// pathological inputs go down the slow path with full
  /// precision instead of producing backend-skewed output.
  pub(super) fn scalar_var_sum(samples: &[f32], mean: f64) -> f64 {
    let mut var_sum = 0.0_f64;
    for &s in samples {
      let d = s as f64 - mean;
      var_sum += d * d;
    }
    var_sum
  }
}

/// aarch64 NEON backend. Vectorises the three sequential passes
/// (sum, var, write-out) over `float32x4_t` lanes. The scalar tail
/// handles the trailing `len % 4` samples without falling through
/// to the scalar reference.
#[cfg(target_arch = "aarch64")]
pub mod neon {
  use super::Vec;
  use core::arch::aarch64::*;

  /// SAFETY: caller must run on aarch64 with NEON available.
  /// Marked `unsafe` to mirror the colconv convention; in practice
  /// every aarch64 build of this crate satisfies the precondition.
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

  /// SAFETY: caller must run on x86_64 with SSE4.1 available.
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

  /// SAFETY: caller must run on x86_64 with AVX2 available.
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
pub mod x86_avx512 {
  use super::Vec;
  use core::arch::x86_64::*;

  /// SAFETY: caller must run on x86_64 with AVX-512F available.
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

  /// Codex round-10 [medium]: high-dynamic-range f32 inputs
  /// `[1e20, -1e20, ...]` overflow the SIMD f32 squared-deviation
  /// pass to `+inf`, while the scalar f64 reference keeps a
  /// finite `var ≈ 1e40`. The SIMD backends must detect the
  /// overflow and fall back to scalar so the dispatched result
  /// matches the scalar reference.
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
}
