//! `core::arch` baseline for the alignment hot paths.
//!
//! Each bench either pits two backends head-to-head (`scalar` vs.
//! `neon`) on adjacent lines so the speed-up is readable directly
//! from the Criterion report, or — for stages that don't have a
//! viable SIMD restructuring — captures the scalar number alone as
//! a baseline for future work.
//!
//! Bench inputs target the production envelope: 30 s of 16 kHz
//! mono audio (480 000 samples) for the per-utterance normaliser
//! and the wav2vec2-base-960h output dimensions (T = 1500 frames,
//! V = 32 vocab) for the trellis/beam stage.
//!
//! Run with:
//!
//! ```sh
//! cargo bench --features bench-internals --bench aligner_simd_baseline
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[cfg(target_arch = "aarch64")]
use asry::__bench::neon;
#[cfg(target_arch = "x86_64")]
use asry::__bench::{x86_avx2, x86_avx512, x86_sse41};
use asry::{
  __bench::{LogProbsTV, get_trellis, scalar, zero_mean_unit_var_normalize},
  Lang,
};

/// Deterministic synthetic audio with gain + DC offset so the
/// normaliser actually has work to do (a constant-zero signal would
/// have mean = 0, var = 0 and short-circuit through the scalar
/// path's `0/sqrt(eps) = 0` branch, hiding any SIMD-backed gain).
fn synth_audio(n: usize, seed: u32) -> Vec<f32> {
  let mut state = seed;
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let noise = ((state >> 8) as i32) as f32 / 16_777_216.0;
    let t = i as f32 / 16_000.0;
    out.push(2.7 * (2.0 * core::f32::consts::PI * 220.0 * t).sin() + 0.13 + noise);
  }
  out
}

fn bench_normalize(c: &mut Criterion) {
  // 10 s, 30 s @ 16 kHz. Larger sizes amplify the SIMD gain;
  // 30 s is the production chunk size in the spec.
  const SIZES: &[usize] = &[160_000, 480_000];

  let mut group = c.benchmark_group("zero_mean_unit_var_normalize");

  for &n in SIZES {
    let samples = synth_audio(n, 0xc0ffee);

    // Throughput in input bytes (f32 = 4 bytes/sample) so the
    // Criterion `MB/s` line is comparable across sizes.
    group.throughput(Throughput::Bytes((n * 4) as u64));

    // Scalar reference — always-compiled f64-accumulator path.
    group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
      b.iter(|| {
        let out = scalar::zero_mean_unit_var_normalize(black_box(&samples));
        black_box(out);
      });
    });

    // aarch64 NEON — `core::arch::aarch64::*`. Same f64
    // accumulator semantics for the long-sequence stability guard.
    #[cfg(target_arch = "aarch64")]
    group.bench_with_input(BenchmarkId::new("neon", n), &n, |b, &_n| {
      b.iter(|| {
        // SAFETY: NEON is part of the aarch64 base ISA.
        let out = unsafe { neon::zero_mean_unit_var_normalize(black_box(&samples)) };
        black_box(out);
      });
    });

    // x86_64 backends. Each is gated on a runtime feature check
    // so a host without that ISA quietly skips its bench rather
    // than crashing with SIGILL. The host's `is_x86_feature_detected!`
    // result is constant for the run, so the runtime check
    // amortises to zero across Criterion's iterations.
    #[cfg(target_arch = "x86_64")]
    {
      if std::is_x86_feature_detected!("sse4.1") {
        group.bench_with_input(BenchmarkId::new("x86_sse41", n), &n, |b, &_n| {
          b.iter(|| {
            // SAFETY: feature checked above.
            let out = unsafe { x86_sse41::zero_mean_unit_var_normalize(black_box(&samples)) };
            black_box(out);
          });
        });
      }
      if std::is_x86_feature_detected!("avx2") {
        group.bench_with_input(BenchmarkId::new("x86_avx2", n), &n, |b, &_n| {
          b.iter(|| {
            let out = unsafe { x86_avx2::zero_mean_unit_var_normalize(black_box(&samples)) };
            black_box(out);
          });
        });
      }
      if std::is_x86_feature_detected!("avx512f") {
        group.bench_with_input(BenchmarkId::new("x86_avx512", n), &n, |b, &_n| {
          b.iter(|| {
            let out = unsafe { x86_avx512::zero_mean_unit_var_normalize(black_box(&samples)) };
            black_box(out);
          });
        });
      }
    }

    // Public dispatcher — what production hits. On aarch64 this
    // is identical to `neon` above; we still bench it so a future
    // dispatcher change (runtime feature detection, x86 SIMD)
    // shows up here.
    group.bench_with_input(BenchmarkId::new("dispatched", n), &n, |b, &_n| {
      b.iter(|| {
        let out = zero_mean_unit_var_normalize(black_box(&samples));
        black_box(out);
      });
    });
  }
  group.finish();
}

/// Synthesise a `(T, V)` log-probability lattice with one strong
/// peak per frame at the assigned token. Mirrors the structure of
/// real wav2vec2 output well enough to exercise the DP without
/// needing ORT.
fn synth_log_probs(t: usize, v: usize) -> LogProbsTV {
  let mut data = vec![-100.0_f32; t * v];
  for ti in 0..t {
    let target = ti % v;
    data[ti * v + target] = -0.1;
  }
  LogProbsTV::new(t, v, data)
}

fn bench_ctc_viterbi(c: &mut Criterion) {
  // wav2vec2-base-960h's typical output: ~50 frames/sec, V = 32
  // (vocab size including specials). 30 s ≈ 1500 frames.
  // Tokens for 30 s of speech: roughly 100-300 chars depending on
  // language and density. Bench at the lower end (100, distinct
  // ids cycling through V-1 to keep adjacent-equal rare).
  //
  // `get_trellis` is the trellis-build half of the round-25
  // bit-exact port (formerly `ctc_viterbi`'s DP fill). It owns the
  // `(T+1, num_tokens)` tensor walk that's the hottest scalar
  // routine in the alignment pipeline, so it's the natural target
  // for SIMD restructuring once we have the bit-exact baseline.
  // We don't follow the full `align_to_word_segments` path here
  // because it requires per-token word indices — synthetic tokens
  // would need a fake mapping that adds setup noise without
  // changing what we're measuring.
  const T_FRAMES: usize = 1500;
  const VOCAB: usize = 32;
  const TOKENS_LEN: usize = 100;

  let log_probs = synth_log_probs(T_FRAMES, VOCAB);
  // `get_trellis` takes `&[i32]` (wildcard sentinel `-1` is
  // allowed). Tokens here are real vocab ids in `[1, V-1]`.
  let tokens: Vec<i32> = (0..TOKENS_LEN as i32)
    .map(|i| 1 + i % (VOCAB as i32 - 1))
    .collect();

  let mut group = c.benchmark_group("ctc_viterbi");
  // Throughput: report as "frames per second" via the `Elements`
  // dimension so we can compare future SIMD-restructured trellis
  // attempts directly.
  group.throughput(Throughput::Elements(T_FRAMES as u64));
  group.bench_function(
    BenchmarkId::new("scalar", format!("T={T_FRAMES}_M={TOKENS_LEN}_V={VOCAB}")),
    |b| {
      // Bench has no abort path — just point at a never-set flag.
      static NEVER: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
      b.iter(|| {
        // Discard through `black_box` so the optimiser can't
        // elide the call — we don't care about the result, just
        // that the DP runs.
        let _ = black_box(get_trellis(
          black_box(&log_probs),
          black_box(&tokens),
          /* blank_id: */ 0,
          &NEVER,
          black_box(&Lang::En),
        ));
      });
    },
  );
  group.finish();
}

/// Standalone scalar log-softmax over a synthetic `(T, V)` raw
/// logit buffer. Mirrors the inline body of `encode_log_softmax`'s
/// post-ORT reduction (`max → sum exp(x-max) → subtract log_z`)
/// without taking ORT's `Session::run` cost into account, so we
/// can size up whether the reduction itself is worth a SIMD
/// pass. Returns the same f64-accumulator shape the real code
/// uses.
fn scalar_log_softmax(raw: &[f32], t: usize, v: usize) -> Vec<f32> {
  let mut data = Vec::with_capacity(t * v);
  for t_idx in 0..t {
    let row = &raw[t_idx * v..(t_idx + 1) * v];
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for &x in row {
      sum += ((x - max) as f64).exp();
    }
    let log_z = max + (sum.ln() as f32);
    for &x in row {
      data.push(x - log_z);
    }
  }
  data
}

fn bench_log_softmax(c: &mut Criterion) {
  // wav2vec2-base-960h's typical post-encode shape: T=1500
  // (50 frames/sec × 30 s), V=32 (vocab + specials).
  const T_FRAMES: usize = 1500;
  const VOCAB: usize = 32;

  // Synthetic logits: deterministic noise + a strong peak per
  // frame so the softmax actually has work to do (uniform input
  // would short-circuit).
  let mut raw: Vec<f32> = Vec::with_capacity(T_FRAMES * VOCAB);
  let mut state: u32 = 0xc0ffee;
  for ti in 0..T_FRAMES {
    let target = ti % VOCAB;
    for vi in 0..VOCAB {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let noise = ((state >> 8) as i32) as f32 / 16_777_216.0;
      let base = if vi == target { -0.1 } else { -3.0 };
      raw.push(base + noise);
    }
  }

  let mut group = c.benchmark_group("log_softmax");
  group.throughput(Throughput::Elements((T_FRAMES * VOCAB) as u64));
  group.bench_function(
    BenchmarkId::new("scalar", format!("T={T_FRAMES}_V={VOCAB}")),
    |b| {
      b.iter(|| {
        let out = scalar_log_softmax(black_box(&raw), T_FRAMES, VOCAB);
        black_box(out);
      });
    },
  );
  group.finish();
}

criterion_group!(
  benches,
  bench_normalize,
  bench_ctc_viterbi,
  bench_log_softmax
);
criterion_main!(benches);
