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

/// Run wav2vec2 over `samples_for_aligner` and return per-frame
/// log-probabilities.
///
/// **`samples_for_aligner` must be pre-normalised.** The
/// silence-aware
/// [`crate::runner::aligner::algorithm::normalize::normalize_with_silence_mask`]
/// runs in `Aligner::align` before this function so the silence
/// mask is preserved through preprocessing — Codex round-14
/// [high]'s fix moved normalisation up the call stack so masked
/// regions stay exactly zero in the tensor we feed to ORT.
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

  // Build a (1, T) f32 input via ort's `(shape, Vec<T>)` tensor
  // constructor — see the module-level NOTE for why we don't go
  // through `ndarray::Array2`. Caller is responsible for the
  // zero-mean / unit-var normalisation (silence-aware variant in
  // `Aligner::align`); the input here goes straight to ORT.
  let input_shape: [i64; 2] = [1, t_samples as i64];
  let input_tensor =
    Tensor::from_array((input_shape, samples_for_aligner.to_vec())).map_err(|e| {
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

  // Note: the centring / scale and empty-input behaviour tests
  // moved to `super::normalize::tests` after Codex round-14
  // pulled normalisation up the call stack into `Aligner::align`.
  // `encode_log_softmax` no longer normalises, so its tests
  // here cover only the reductions and the input-validation
  // boundary it does still own.
}
