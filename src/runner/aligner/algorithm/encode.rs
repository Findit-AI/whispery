//! Step 3-4 of the alignment algorithm: ONNX encode + log-softmax.

use alloc::{string::String, vec::Vec};

use ort::{
  session::{RunOptions, Session},
  value::{Shape, Tensor},
};

use crate::types::{AlignmentFailureKind, Lang, WorkFailure};

// NOTE on the (1, T) reshape: the plan's literal pseudocode uses
// `ndarray::Array2::from_shape_vec((1, T), …)`, but whispery declares
// `ndarray = "0.16"` while `ort 2.0.0-rc.12` re-exports `ndarray
// 0.17` internally — `Tensor::from_array(Array<T, D>)` only resolves
// for ort's own ndarray version, so the two `ndarray` crates collide
// at the trait-bound layer. We therefore use ort's
// version-agnostic `OwnedTensorArrayData for (D, Vec<T>)` impl
// (`Tensor::from_array((shape, v))`), which is exactly the
// constructor the ort docs use in their session-input examples. This
// keeps the (1, T) reshape semantically identical without forcing a
// cross-version ndarray bridge.

/// Output of `encode_log_softmax`. `pub` for the
/// `feature = "bench-internals"` re-export at the crate root —
/// the only way external code can reach this type. Out-of-tree
/// consumers do not see it.
pub struct LogProbsTV {
  /// Time dimension (number of wav2vec2 output frames).
  pub t: usize,
  /// Vocab dimension.
  pub v: usize,
  /// Flat row-major `(T, V)` log-probabilities. Index with
  /// `[t * v_dim + v_idx]`.
  pub data: Vec<f32>,
}

impl LogProbsTV {
  /// Read the log-probability of vocab index `v_idx` at frame `t_idx`.
  pub fn at(&self, t_idx: usize, v_idx: usize) -> f32 {
    self.data[t_idx * self.v + v_idx]
  }
}

// Per-utterance zero-mean unit-variance normalisation lives in
// `super::normalize` so the bench can pit the scalar and NEON
// backends against each other directly. The dispatcher there picks
// the NEON path on aarch64 and the scalar path elsewhere.
use super::normalize::zero_mean_unit_var_normalize;

/// Run wav2vec2 over `samples_for_aligner` and return per-frame
/// log-probabilities.
///
/// The model is expected to take an input named `"input_values"` of
/// shape `(1, T_samples)` and return logits of shape `(1, T_frames,
/// V)`. wav2vec2-base-960h follows this convention; if a different
/// variant uses a different I/O name, parameterise via
/// `Aligner::with_input_name(...)` (not in v1 scope).
///
/// `run_options` carries ONNX Runtime's per-call termination flag;
/// the alignment worker's watchdog calls `RunOptions::terminate()`
/// on timeout, which causes `Session::run_with_options` to surface
/// an error from inside the graph rather than blocking until the
/// model finishes naturally. This is the only way to interrupt a
/// stuck or pathological inference; the `abort_flag` checked at
/// stage boundaries can't help once we are inside `run`.
///
/// Returns `WorkFailure::AlignmentFailed { kind:
/// ModelInferenceFailed, .. }` on any ort error (including a
/// terminate-induced one — the watchdog's
/// `WorkerHangTimeout` is surfaced by the alignment pool wrapper).
pub(crate) fn encode_log_softmax(
  session: &mut Session,
  samples_for_aligner: &[f32],
  run_options: &RunOptions,
  language: &Lang,
) -> Result<LogProbsTV, WorkFailure> {
  let t_samples = samples_for_aligner.len();
  if t_samples == 0 {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: String::from("samples_for_aligner is empty"),
      language: language.clone(),
    });
  }

  // Codex round-13 [medium]: reject non-finite samples up front
  // with a typed in-band failure. See [`reject_non_finite_input`]
  // for the rationale.
  reject_non_finite_input(samples_for_aligner, language)?;

  // wav2vec2-base-960h's preprocessor sets `do_normalize=true`,
  // i.e., the model expects zero-mean unit-variance audio. Real
  // recordings carry gain and DC offset — feeding raw f32 PCM
  // would degrade the logits and pull Viterbi off the correct
  // path. Apply the same zero-mean / unit-var normalisation HF's
  // `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` applies.
  // The mean/var are computed over the (silence-masked)
  // samples we feed; pure-silence regions contribute 0 to both,
  // a small bias relative to HF's `attention_mask`-aware
  // computation but vastly closer than no normalisation.
  let normalized_samples = zero_mean_unit_var_normalize(samples_for_aligner);

  // Build a (1, T) f32 input via ort's `(shape, Vec<T>)` tensor
  // constructor — see the module-level NOTE for why we don't go
  // through `ndarray::Array2`.
  let input_shape: [i64; 2] = [1, t_samples as i64];
  let input_tensor = Tensor::from_array((input_shape, normalized_samples)).map_err(|e| {
    WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("Tensor::from_array failed: {e:?}"),
      language: language.clone(),
    }
  })?;

  // Most wav2vec2 ONNX exports use the input name "input_values".
  // If the export uses a different name, surface a clear error.
  // `run_with_options` is identical to `run` except it observes
  // the per-call termination flag in `run_options`, so the
  // alignment worker's watchdog can interrupt a stuck graph.
  let outputs = session
    .run_with_options(ort::inputs![input_tensor], run_options)
    .map_err(|e| WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("Session::run_with_options failed: {e:?}"),
      language: language.clone(),
    })?;

  // Take the first (only) output. wav2vec2 has a single logits
  // output; we pull index 0 by name-agnostic iteration.
  let mut iter = outputs.into_iter();
  let (_, output_value) = iter.next().ok_or_else(|| WorkFailure::AlignmentFailed {
    kind: AlignmentFailureKind::ModelInferenceFailed,
    message: String::from("Session::run returned no outputs"),
    language: language.clone(),
  })?;

  let (shape, raw): (&Shape, &[f32]) =
    output_value
      .try_extract_tensor::<f32>()
      .map_err(|e| WorkFailure::AlignmentFailed {
        kind: AlignmentFailureKind::ModelInferenceFailed,
        message: alloc::format!("try_extract_tensor::<f32> failed: {e:?}"),
        language: language.clone(),
      })?;

  if shape.len() != 3 || shape[0] != 1 {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("expected output shape (1, T, V); got {shape:?}"),
      language: language.clone(),
    });
  }
  let t = shape[1] as usize;
  let v = shape[2] as usize;
  if t == 0 || v == 0 {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!("output has zero T={t} or V={v}"),
      language: language.clone(),
    });
  }

  // Log-softmax over V.
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

  Ok(LogProbsTV { t, v, data })
}

/// Reject non-finite (NaN / ±inf) samples before any audio
/// processing runs.
///
/// Without this guard, a single bad sample propagates through
/// `zero_mean_unit_var_normalize`'s mean/variance reductions
/// (NaN poisons every downstream f64 op) and ends up in the
/// tensor we hand to ORT. The model then returns either NaN
/// logits (every word gets a NaN score) or the chunk fails
/// downstream as `NoAlignmentPath` with no clue why — exactly
/// the failure mode Codex round-13 [medium] flagged.
///
/// Pulled out as a helper so the unit tests can exercise the
/// rejection path without spinning up a `Session` (the public
/// `encode_log_softmax` consumes one). The `Aligner::align`
/// integration tests cover the full encode path against the
/// real ORT fixture.
pub(crate) fn reject_non_finite_input(samples: &[f32], language: &Lang) -> Result<(), WorkFailure> {
  if let Some(bad_idx) = samples.iter().position(|s| !s.is_finite()) {
    return Err(WorkFailure::AlignmentFailed {
      kind: AlignmentFailureKind::ModelInferenceFailed,
      message: alloc::format!(
        "samples_for_aligner contains non-finite value at index {bad_idx}: {}",
        samples[bad_idx]
      ),
      language: language.clone(),
    });
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Pure log-softmax math sanity check. Doesn't touch ort.
  #[test]
  fn log_softmax_sums_to_zero_in_log_space() {
    let row = [1.0f32, 2.0, 3.0];
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for &x in &row {
      sum += ((x - max) as f64).exp();
    }
    let log_z = max + (sum.ln() as f32);
    let lp: Vec<f32> = row.iter().map(|x| x - log_z).collect();
    let exp_sum: f32 = lp.iter().map(|x| x.exp()).sum();
    assert!((exp_sum - 1.0).abs() < 1e-5, "softmax must sum to 1");
    for &v in &lp {
      assert!(v <= 0.0, "log-prob must be <= 0");
    }
  }

  /// Codex round-13 [medium] regression: NaN / ±inf input
  /// must fail in-band with `ModelInferenceFailed` before the
  /// scalar normaliser runs. The error message names the
  /// offending index so a downstream operator has a hook for
  /// debugging upstream audio pipelines.
  #[test]
  fn reject_non_finite_input_flags_nan() {
    use crate::types::Lang;
    let samples = alloc::vec![0.1_f32, 0.2, f32::NAN, 0.4];
    let err = reject_non_finite_input(&samples, &Lang::En).unwrap_err();
    match err {
      WorkFailure::AlignmentFailed { kind, message, .. } => {
        assert!(matches!(kind, AlignmentFailureKind::ModelInferenceFailed));
        assert!(
          message.contains("index 2"),
          "message must name index; got {message:?}"
        );
      }
      other => panic!("expected AlignmentFailed; got {other:?}"),
    }
  }

  #[test]
  fn reject_non_finite_input_flags_positive_infinity() {
    use crate::types::Lang;
    let samples = alloc::vec![0.0_f32, f32::INFINITY];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_err());
  }

  #[test]
  fn reject_non_finite_input_flags_negative_infinity() {
    use crate::types::Lang;
    let samples = alloc::vec![f32::NEG_INFINITY, 0.0_f32];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_err());
  }

  #[test]
  fn reject_non_finite_input_passes_finite_audio() {
    use crate::types::Lang;
    // Both ordinary [-1, 1] audio and high-magnitude finite
    // inputs are accepted at this layer — magnitude precision
    // is the SIMD-precision-guard's job, not this guard's.
    let samples = alloc::vec![-1.0_f32, 0.0, 1.0, 1e10, -1e10];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_ok());
  }

  #[test]
  fn at_indexes_correctly() {
    let lp = LogProbsTV {
      t: 2,
      v: 3,
      data: alloc::vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    };
    assert_eq!(lp.at(0, 0), -1.0);
    assert_eq!(lp.at(0, 2), -3.0);
    assert_eq!(lp.at(1, 0), -4.0);
    assert_eq!(lp.at(1, 2), -6.0);
  }

  /// Adversarial regression for the wav2vec2 preprocessing
  /// finding: feeding raw audio with gain / DC offset to the
  /// model degrades logits. After normalisation the output
  /// should have mean ≈ 0 and variance ≈ 1.
  #[test]
  fn normalize_centres_and_scales_to_unit_variance() {
    // Synthetic sinusoid + DC offset + gain. After normalisation
    // the absolute amplitude / offset must wash out.
    let n = 1600;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
      let t = i as f32 / 16_000.0;
      // 100 Hz tone, gain 5.0, DC offset 0.3.
      samples.push(5.0 * (2.0 * core::f32::consts::PI * 100.0 * t).sin() + 0.3);
    }
    let normed = zero_mean_unit_var_normalize(&samples);
    let mean: f64 = normed.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var: f64 = normed
      .iter()
      .map(|&x| (x as f64 - mean).powi(2))
      .sum::<f64>()
      / n as f64;
    assert!(mean.abs() < 1e-5, "mean must be ~0; got {mean}");
    assert!((var - 1.0).abs() < 1e-3, "variance must be ~1; got {var}");
  }

  #[test]
  fn normalize_handles_empty_and_constant_signals() {
    // Empty input is a degenerate case but not a panic.
    assert!(zero_mean_unit_var_normalize(&[]).is_empty());

    // All-zero input: mean = 0, var = 0 → output stays at 0
    // (within the eps regularisation, the result is 0/sqrt(eps)
    // which is exactly 0). This matters for pure-silence
    // chunks after silence-masking.
    let zeros = alloc::vec![0.0_f32; 100];
    let normed = zero_mean_unit_var_normalize(&zeros);
    assert_eq!(normed.len(), 100);
    assert!(normed.iter().all(|&x| x == 0.0));

    // All-constant input: mean = c, var = 0 → output ~ 0.
    let const_signal = alloc::vec![3.7_f32; 100];
    let normed = zero_mean_unit_var_normalize(&const_signal);
    assert!(normed.iter().all(|&x| x.abs() < 1e-3));
  }
}
