//! ONNX encode + log-softmax stage of the alignment algorithm.

use ort::{
  session::{RunOptions, Session},
  value::{Shape, Tensor},
};
use smol_str::{SmolStr, format_smolstr};

use crate::types::{AlignmentError, AlignmentFailure, Lang, WorkFailure};

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
  t: usize,
  /// Vocab dimension.
  v: usize,
  /// Flat row-major `(T, V)` log-probabilities. Index with
  /// `[t * v_dim + v_idx]`.
  data: Vec<f32>,
}

impl LogProbsTV {
  /// Construct from explicit dimensions + flat row-major buffer.
  ///
  /// `data.len()` must equal `t * v`. The constructor itself
  /// doesn't validate; callers (the encoder and the bench
  /// harness) construct from already-validated shapes.
  #[must_use]
  pub const fn new(t: usize, v: usize, data: Vec<f32>) -> Self {
    Self { t, v, data }
  }

  /// Time dimension (number of wav2vec2 output frames).
  #[must_use]
  pub const fn t(&self) -> usize {
    self.t
  }

  /// Vocab dimension.
  #[must_use]
  pub const fn v(&self) -> usize {
    self.v
  }

  /// Borrow the flat row-major `(T, V)` log-probability buffer.
  #[must_use]
  pub fn data(&self) -> &[f32] {
    &self.data
  }

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
/// mask is preserved through preprocessing. Normalisation lives
/// up the call stack so masked regions stay exactly zero in the
/// tensor we feed to ORT.
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
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        SmolStr::from("samples_for_aligner is empty"),
        language.clone(),
      ),
    )));
  }

  // Reject non-finite samples up front with a typed in-band
  // failure. See [`reject_non_finite_input`] for the rationale.
  reject_non_finite_input(samples_for_aligner, language)?;

  // Build a (1, T) f32 input via ort's `(shape, Vec<T>)` tensor
  // constructor — see the module-level NOTE for why we don't go
  // through `ndarray::Array2`. Caller is responsible for the
  // zero-mean / unit-var normalisation (silence-aware variant in
  // `Aligner::align`); the input here goes straight to ORT.
  let input_shape: [i64; 2] = [1, t_samples as i64];
  let input_tensor =
    Tensor::from_array((input_shape, samples_for_aligner.to_vec())).map_err(|e| {
      WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
        format_smolstr!("Tensor::from_array failed: {e:?}"),
        language.clone(),
      )))
    })?;

  // Most wav2vec2 ONNX exports use the input name "input_values".
  // If the export uses a different name, surface a clear error.
  // `run_with_options` is identical to `run` except it observes
  // the per-call termination flag in `run_options`, so the
  // alignment worker's watchdog can interrupt a stuck graph.
  let outputs = session
    .run_with_options(ort::inputs![input_tensor], run_options)
    .map_err(|e| {
      WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
        format_smolstr!("Session::run_with_options failed: {e:?}"),
        language.clone(),
      )))
    })?;

  // Take the first (only) output. wav2vec2 has a single logits
  // output; we pull index 0 by name-agnostic iteration.
  let mut iter = outputs.into_iter();
  let (_, output_value) = iter.next().ok_or_else(|| {
    WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
      SmolStr::from("Session::run returned no outputs"),
      language.clone(),
    )))
  })?;

  let (shape, raw): (&Shape, &[f32]) = output_value.try_extract_tensor::<f32>().map_err(|e| {
    WorkFailure::Alignment(AlignmentError::ModelInference(AlignmentFailure::new(
      format_smolstr!("try_extract_tensor::<f32> failed: {e:?}"),
      language.clone(),
    )))
  })?;

  if shape.len() != 3 || shape[0] != 1 {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!("expected output shape (1, T, V); got {shape:?}"),
        language.clone(),
      ),
    )));
  }

  // Validate the shape integers and their product against the
  // raw buffer length BEFORE we cast to usize / allocate /
  // slice. A model export bug that emits a negative dimension
  // would otherwise wrap to a huge usize and OOM the worker,
  // and a buffer-vs-shape mismatch would panic on the row-slice
  // in `log_softmax_with_finite_guard`.
  let (t, v) = validate_output_dims(shape[1], shape[2], raw.len(), language)?;

  // Validate logits + log-softmax for finiteness. See
  // `log_softmax_with_finite_guard`.
  let data = log_softmax_with_finite_guard(raw, t, v, language)?;
  Ok(LogProbsTV { t, v, data })
}

/// Validate the `(T, V)` output dimensions before allocation /
/// slicing.
///
/// Without these checks, a malformed ORT output (negative dim
/// from a buggy export, overflow on `t * v`, or `raw.len()` not
/// matching the declared shape) would either panic the alignment
/// worker on the row-slice, OOM the process on a
/// `Vec::with_capacity` for a wrapped-huge size, or — worst case
/// — silently read into adjacent memory.
///
/// Failure classification:
/// - **Fatal** (`ModelInferenceFailed`) for impossible shapes
/// the operator should hear about: `V <= 0`, negative `T`,
/// `T * V` overflow, or shape-vs-buffer-length mismatch.
/// - **Recoverable** (`NoAlignmentPath`) for `T == 0` with an
/// empty buffer — a chunk shorter than the model's stride
/// produces zero encoder frames; the ASR transcript should
/// surface with `words: []` rather than fail the chunk. A
/// blanket "non-positive dim" rule would turn a data-dependent
/// short-chunk miss into fatal transcript loss.
///
/// Pulled out as a helper so unit tests can drive each branch
/// without an ORT session.
pub(crate) fn validate_output_dims(
  raw_t: i64,
  raw_v: i64,
  raw_len: usize,
  language: &Lang,
) -> Result<(usize, usize), WorkFailure> {
  // V == 0 means the model declared no vocabulary axis — never
  // legitimate. Always fatal.
  if raw_v <= 0 {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!("ORT output has non-positive vocab dim: V={raw_v}"),
        language.clone(),
      ),
    )));
  }
  // Negative T is always a backend bug (truncated /
  // sign-flipped shape descriptor).
  if raw_t < 0 {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!("ORT output has negative time dim: T={raw_t}"),
        language.clone(),
      ),
    )));
  }
  // T == 0 — the model returned a well-formed empty output.
  // With an empty buffer that's a legitimate "chunk too short
  // for any encoder frame" outcome and we surface as
  // recoverable `NoAlignmentPath`. With a non-empty buffer the
  // shape declaration disagrees with the data length — fatal
  // model bug.
  if raw_t == 0 {
    if raw_len != 0 {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "ORT output declared T=0 but buffer has {raw_len} elements; shape/data mismatch"
          ),
          language.clone(),
        ),
      )));
    }
    return Err(WorkFailure::Alignment(AlignmentError::NoAlignmentPath(
      AlignmentFailure::new(
        SmolStr::from(
          "ORT output has zero encoder frames (chunk too short to align); \
 transcript will surface with words: []",
        ),
        language.clone(),
      ),
    )));
  }
  // i64 → usize is safe after the >0 check on 64-bit; on 32-bit
  // we still want the explicit overflow guard.
  let t = match usize::try_from(raw_t) {
    Ok(v) => v,
    Err(_) => {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!("ORT output T={raw_t} doesn't fit in usize"),
          language.clone(),
        ),
      )));
    }
  };
  let v = match usize::try_from(raw_v) {
    Ok(v) => v,
    Err(_) => {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!("ORT output V={raw_v} doesn't fit in usize"),
          language.clone(),
        ),
      )));
    }
  };
  let total = match t.checked_mul(v) {
    Some(p) => p,
    None => {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!("ORT output dimensions overflow: T={t} * V={v} doesn't fit in usize"),
          language.clone(),
        ),
      )));
    }
  };
  if total != raw_len {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "ORT output buffer length {raw_len} doesn't match declared T={t} × V={v} = {total}"
        ),
        language.clone(),
      ),
    )));
  }
  Ok((t, v))
}

/// Validate the encoder's frame count against the input audio
/// length. wav2vec2's CNN downsamples by `hop_samples`, so the
/// encoded "time" `T * hop_samples` should lie within
/// `chunk_extent ± 2*hop_samples` (a couple of frames of
/// receptive-field slack on each side).
///
/// Two-sided check — both bounds matter:
///
/// - **Upper bound** (`T * hop > chunk + 2*hop`): the model
/// reports more frames than the input could plausibly support.
/// Either the export uses a smaller stride than `hop_samples`
/// or the configured `hop_samples` is too small. `compose_words`
/// would otherwise emit ranges past the chunk's audio
/// boundary.
/// - **Lower bound** (`T * hop < chunk - 2*hop`): the model
/// reports far fewer frames than the input should produce.
/// Either the export uses a *larger* stride than
/// `hop_samples` or `hop_samples` is too large. `compose_words`
/// would otherwise emit ranges that compress every word into
/// the first portion of the chunk — plausible-looking
/// timestamps that all sit in (e.g.) the first half of the
/// audio.
///
/// `chunk_extent.saturating_sub(slack)` lets very short chunks
/// (where the slack is comparable to `chunk_extent`) pass without
/// false positives — the lower bound clamps to 0. T == 0 cases
/// are already routed to recoverable `NoAlignmentPath` by
/// [`validate_output_dims`].
pub(crate) fn validate_stride_extent(
  t: usize,
  hop_samples: u32,
  chunk_extent: usize,
  language: &Lang,
) -> Result<(), WorkFailure> {
  let frame_extent = (t as u64).saturating_mul(hop_samples as u64);
  let chunk_extent_u64 = chunk_extent as u64;
  let slack = 2u64.saturating_mul(hop_samples as u64);
  let upper_bound = chunk_extent_u64.saturating_add(slack);
  let lower_bound = chunk_extent_u64.saturating_sub(slack);
  if frame_extent > upper_bound {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "ORT output stride mismatch: T={t} × hop={hop_samples} = {frame_extent} \
 sample-equivalents exceeds chunk ({chunk_extent} samples) + 2-frame slack \
 ({upper_bound}); model export uses a smaller stride than `hop_samples` \
 or `hop_samples` is misconfigured"
        ),
        language.clone(),
      ),
    )));
  }
  if frame_extent < lower_bound {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "ORT output stride mismatch: T={t} × hop={hop_samples} = {frame_extent} \
 sample-equivalents below chunk ({chunk_extent} samples) − 2-frame slack \
 ({lower_bound}); model export uses a larger stride than `hop_samples` \
 or `hop_samples` is misconfigured"
        ),
        language.clone(),
      ),
    )));
  }
  Ok(())
}

/// Validate the model output's vocab dimension against the
/// tokenizer's vocab size. A wrong ONNX export (e.g. a hidden-
/// states tensor with a much larger trailing dim, or a CTC head
/// trained on a different alphabet) would otherwise pass the
/// per-token id check in `ctc_viterbi` whenever the chunk's
/// in-vocab token ids happen to fit, then read posteriors from
/// columns the tokenizer thinks correspond to the wrong tokens
/// — emitting believable but corrupt timings.
///
/// Strict equality matches the wav2vec2 ASR family (model output
/// dim == tokenizer vocab size, including special tokens like
/// `<pad>` / `<s>` / `</s>` / `<unk>` / `|`). If a future
/// downstream model legitimately has a different output width,
/// this helper would need a configured override; for the
/// supported family, mismatch is always a model/tokenizer
/// pairing bug.
pub(crate) fn validate_vocab_dim(
  v: usize,
  expected_v: usize,
  language: &Lang,
) -> Result<(), WorkFailure> {
  if v != expected_v {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "ORT output vocab dim V={v} doesn't match tokenizer vocab size {expected_v}; \
 model and tokenizer are paired incorrectly — Viterbi would otherwise read \
 posteriors from columns that don't correspond to the tokenizer's tokens"
        ),
        language.clone(),
      ),
    )));
  }
  Ok(())
}

/// Compute row-major `(T, V)` log-softmax of `raw` with a fatal
/// finiteness guard.
///
/// Without this guard a NaN / ±inf in any logit produced a NaN
/// row in the output; Viterbi then computed a non-finite final
/// `dp_prev` and surfaced `NoAlignmentPath`, which the alignment
/// pool classifies as recoverable — silently swallowing a backend
/// numeric failure (model export bug, GPU / ORT regression, NaN
/// propagation from upstream) as "no words". This helper checks
/// each row's input and the resulting `log_z`; either non-finite
/// returns `ModelInferenceFailed` so the runner emits
/// `Event::Error` and the operator learns about the broken
/// backend.
///
/// Pulled out of the public `encode_log_softmax` body so unit
/// tests can exercise the rejection paths without a `Session`.
pub(crate) fn log_softmax_with_finite_guard(
  raw: &[f32],
  t: usize,
  v: usize,
  language: &Lang,
) -> Result<Vec<f32>, WorkFailure> {
  let mut data = Vec::with_capacity(t * v);
  for t_idx in 0..t {
    let row = &raw[t_idx * v..(t_idx + 1) * v];
    if let Some(bad_v) = row.iter().position(|x| !x.is_finite()) {
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "ORT returned non-finite logit at frame {t_idx}, vocab {bad_v}: {}",
            row[bad_v]
          ),
          language.clone(),
        ),
      )));
    }
    // Shifted log-sum-exp computed in f64. We do NOT add `max`
    // back into f32 to form `log_z` and then subtract it again,
    // because for a row with a large common offset (e.g.
    // `[1e20, 1e20]`) `sum.ln()` rounds away when added to `max`
    // in f32 — `max + sum.ln() as f32 = 1e20 + 0.69 ≈ 1e20` in
    // f32 — and every `lp = x - log_z` then collapses to `0.0`
    // instead of the correct `-ln(2)`. The output passes the
    // finiteness checks but is no longer a log-probability,
    // hiding the backend numeric skew as plausible-looking
    // alignment input. Flagged this; the fix is
    // to keep the subtraction of `max` in shifted f64 space and
    // only cast the final `lp` to f32 (where `lp = (x - max) -
    // sum.ln()` is bounded between `-inf..=0` and never needs
    // `max` to fold in).
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_f64 = max as f64;
    let mut sum = 0.0_f64;
    for &x in row {
      sum += ((x as f64) - max_f64).exp();
    }
    let log_z_shifted = sum.ln();
    if !log_z_shifted.is_finite() {
      // `sum == 0.0` (whole row was -inf, or every shifted exp
      // underflowed) → ln(0) = -inf. `sum < 0` is impossible
      // here. `sum.ln() == NaN` is impossible from non-negative
      // f64 input. So this branch fires only on the all-(-inf)
      // case, which the existing all-`NEG_INFINITY` regression
      // also covers.
      return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
        AlignmentFailure::new(
          format_smolstr!(
            "log-softmax shifted normaliser non-finite at frame {t_idx}: \
 sum.ln()={log_z_shifted}, max={max}"
          ),
          language.clone(),
        ),
      )));
    }
    // Per-output log-probability. `lp_f64 = (x - max) - sum.ln()`
    // is bounded in `(-∞, 0]` for any finite input row (since
    // `(x - max) <= 0` and `sum.ln() >= 0` whenever any row
    // element equals `max`). We still keep the per-element
    // finiteness check because pathological inputs like
    // `[f32::MAX, -f32::MAX]` can underflow `lp_f64 as f32` to
    // `-inf`; surfacing that as `ModelInferenceFailed` keeps the
    // backend-numeric-failure path typed (a `-inf` slipping into
    // `data` would be visible to Viterbi as a valid-but-very-low
    // log-prob, masking the bug as `NoAlignmentPath` /
    // `words: []`).
    for &x in row {
      let lp_f64 = ((x as f64) - max_f64) - log_z_shifted;
      let lp = lp_f64 as f32;
      if !lp.is_finite() {
        return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
          AlignmentFailure::new(
            format_smolstr!(
              "log-softmax output non-finite at frame {t_idx}: \
 x={x}, max={max}, sum_ln={log_z_shifted}, lp={lp}"
            ),
            language.clone(),
          ),
        )));
      }
      data.push(lp);
    }
  }
  Ok(data)
}

/// Reject non-finite (NaN / ±inf) samples before any audio
/// processing runs.
///
/// Without this guard, a single bad sample propagates through
/// `zero_mean_unit_var_normalize`'s mean/variance reductions
/// (NaN poisons every downstream f64 op) and ends up in the
/// tensor we hand to ORT. The model then returns either NaN
/// logits (every word gets a NaN score) or the chunk fails
/// downstream as `NoAlignmentPath` with no clue why.
///
/// Pulled out as a helper so the unit tests can exercise the
/// rejection path without spinning up a `Session` (the public
/// `encode_log_softmax` consumes one). The `Aligner::align`
/// integration tests cover the full encode path against the
/// real ORT fixture.
pub(crate) fn reject_non_finite_input(samples: &[f32], language: &Lang) -> Result<(), WorkFailure> {
  if let Some(bad_idx) = samples.iter().position(|s| !s.is_finite()) {
    return Err(WorkFailure::Alignment(AlignmentError::ModelInference(
      AlignmentFailure::new(
        format_smolstr!(
          "samples_for_aligner contains non-finite value at index {bad_idx}: {}",
          samples[bad_idx]
        ),
        language.clone(),
      ),
    )));
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

  /// Regression: NaN / ±inf input must fail in-band with
  /// `ModelInferenceFailed` before the scalar normaliser runs.
  /// The error message names the offending index so a
  /// downstream operator has a hook for debugging upstream audio
  /// pipelines.
  #[test]
  fn reject_non_finite_input_flags_nan() {
    use crate::types::Lang;
    let samples = vec![0.1_f32, 0.2, f32::NAN, 0.4];
    let err = reject_non_finite_input(&samples, &Lang::En).unwrap_err();
    match err {
      WorkFailure::Alignment(AlignmentError::ModelInference(payload)) => {
        assert!(
          payload.message().contains("index 2"),
          "message must name index; got {message}",
          message = payload.message()
        );
      }
      other => panic!("expected AlignmentFailed; got {other:?}"),
    }
  }

  #[test]
  fn reject_non_finite_input_flags_positive_infinity() {
    use crate::types::Lang;
    let samples = vec![0.0_f32, f32::INFINITY];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_err());
  }

  #[test]
  fn reject_non_finite_input_flags_negative_infinity() {
    use crate::types::Lang;
    let samples = vec![f32::NEG_INFINITY, 0.0_f32];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_err());
  }

  #[test]
  fn reject_non_finite_input_passes_finite_audio() {
    use crate::types::Lang;
    // Both ordinary [-1, 1] audio and high-magnitude finite
    // inputs are accepted at this layer — magnitude precision
    // is the SIMD-precision-guard's job, not this guard's.
    let samples = vec![-1.0_f32, 0.0, 1.0, 1e10, -1e10];
    assert!(reject_non_finite_input(&samples, &Lang::En).is_ok());
  }

  #[test]
  fn at_indexes_correctly() {
    let lp = LogProbsTV {
      t: 2,
      v: 3,
      data: vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    };
    assert_eq!(lp.at(0, 0), -1.0);
    assert_eq!(lp.at(0, 2), -3.0);
    assert_eq!(lp.at(1, 0), -4.0);
    assert_eq!(lp.at(1, 2), -6.0);
  }

  /// NaN logits from a broken backend must surface as fatal
  /// `ModelInferenceFailed`, not get swallowed into NaN
  /// log-probs that Viterbi later classifies as
  /// `NoAlignmentPath` (the recoverable bucket).
  #[test]
  fn log_softmax_rejects_nan_logits_with_model_inference_failed() {
    use crate::types::Lang;
    let raw = vec![0.0_f32, f32::NAN, 0.0]; // 1×3
    let err = log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).unwrap_err();
    match err {
      WorkFailure::Alignment(AlignmentError::ModelInference(payload)) => {
        assert!(
          payload.message().contains("non-finite logit"),
          "message must call out the non-finite logit; got {message}",
          message = payload.message()
        );
        assert!(payload.message().contains("frame 0"));
        assert!(payload.message().contains("vocab 1"));
      }
      other => panic!("expected AlignmentFailed; got {other:?}"),
    }
  }

  #[test]
  fn log_softmax_rejects_positive_infinity_logits() {
    use crate::types::Lang;
    let raw = vec![0.0_f32, f32::INFINITY, 0.0];
    assert!(log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).is_err());
  }

  #[test]
  fn log_softmax_rejects_negative_infinity_logits() {
    use crate::types::Lang;
    let raw = vec![f32::NEG_INFINITY, 0.0, 0.0];
    assert!(log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).is_err());
  }

  /// All-`-inf` row passes the per-element finiteness check
  /// (each element IS finite under f32::is_finite for normal
  /// numbers, but NEG_INFINITY is not finite — let me re-check).
  /// `f32::NEG_INFINITY.is_finite()` returns false, so this
  /// fails at the per-element check. Document that path:
  /// even pathological all-inf rows surface as
  /// `ModelInferenceFailed`.
  #[test]
  fn log_softmax_rejects_all_neg_infinity_row() {
    use crate::types::Lang;
    let raw = vec![f32::NEG_INFINITY; 3];
    assert!(log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).is_err());
  }

  /// Finite extreme logits can produce a non-finite output
  /// log-prob: `[f32::MAX, -f32::MAX]` has finite input,
  /// finite max, finite log_z (= f32::MAX), but the second
  /// element's `x - log_z = -f32::MAX - f32::MAX = -inf`.
  /// The `-inf` was stored in `data`; Viterbi would
  /// later return `NoAlignmentPath` (recoverable) hiding a
  /// real backend numeric failure as `words: []`. The
  /// per-element finite check now surfaces it as fatal
  /// `ModelInferenceFailed`.
  #[test]
  fn log_softmax_rejects_finite_extremes_that_overflow_lp() {
    use crate::types::Lang;
    let raw = vec![f32::MAX, -f32::MAX];
    let err = log_softmax_with_finite_guard(&raw, 1, 2, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::ModelInference(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(
      message.contains("log-softmax output non-finite"),
      "diagnostic must call out the per-element finite check; got {message}",
      message = message
    );
  }

  /// Sanity: a finite, well-behaved row produces a finite
  /// log-softmax row that sums to 1 in linear space.
  #[test]
  fn log_softmax_finite_input_roundtrips() {
    use crate::types::Lang;
    let raw = vec![1.0_f32, 2.0, 3.0];
    let out = log_softmax_with_finite_guard(&raw, 1, 3, &Lang::En).expect("ok");
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|x| x.is_finite()));
    let sum: f32 = out.iter().map(|x| x.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
  }

  #[test]
  fn log_softmax_large_common_offset_normalises_to_unit_exp_sum() {
    // regression: the previous implementation
    // computed `log_z = max + sum.ln() as f32`. For a row with
    // a large common offset like `[1e20, 1e20]`, `sum.ln() =
    // ln(2) ≈ 0.69` rounds away when added to `max = 1e20` in
    // f32 — `1e20 + 0.69 ≈ 1e20` — so each `lp = x - log_z`
    // collapsed to `0.0` instead of the correct `-ln(2)`. The
    // outputs passed the finiteness check but were no longer
    // log-probabilities, hiding a backend numeric failure as
    // plausible alignment output. The fix keeps the
    // subtraction of `max` in f64 (`lp_f64 = (x as f64 -
    // max_f64) - sum.ln()`) so the shifted log-prob is correct
    // regardless of `max`'s magnitude.
    use crate::types::Lang;
    let raw = vec![1.0e20_f32, 1.0e20_f32];
    let out = log_softmax_with_finite_guard(&raw, 1, 2, &Lang::En).expect("ok");
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|x| x.is_finite()));
    // Exp-sum should be 1.0 (i.e. softmax probabilities sum to 1).
    let exp_sum: f32 = out.iter().map(|x| x.exp()).sum();
    assert!(
      (exp_sum - 1.0).abs() < 1e-5,
      "exp(lp) sum must equal 1, got {exp_sum}; lps = {out:?}"
    );
    // Each lp should be log(0.5) = -ln(2) ≈ -0.6931.
    for lp in &out {
      assert!(
        (lp - (-(2.0_f32).ln())).abs() < 1e-4,
        "expected ~{}, got {lp}",
        -(2.0_f32).ln()
      );
    }
  }

  // --- ORT output dims validation ---

  #[test]
  fn validate_output_dims_rejects_negative_t() {
    use crate::types::Lang;
    let err = validate_output_dims(-1, 32, 32, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::ModelInference(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(message.contains("negative time dim"));
  }

  #[test]
  fn validate_output_dims_rejects_zero_v() {
    use crate::types::Lang;
    let err = validate_output_dims(100, 0, 0, &Lang::En).unwrap_err();
    // V=0 is always fatal — model has no vocab axis.
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::ModelInference(_))
    ));
  }

  /// A chunk too short to produce any encoder frame must surface
  /// as recoverable `NoAlignmentPath`, not fatal
  /// `ModelInferenceFailed`. The ASR transcript stays alive with
  /// `words: []`.
  #[test]
  fn validate_output_dims_zero_t_with_empty_buffer_is_recoverable_no_alignment_path() {
    use crate::types::Lang;
    let err = validate_output_dims(0, 32, 0, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::NoAlignmentPath(payload)) = &err else {
      panic!("expected NoAlignmentPath; got {err:?}");
    };
    let message = payload.message();
    assert!(
      message.contains("zero encoder frames"),
      "diagnostic must explain the short-chunk cause; got {message}",
      message = message
    );
  }

  /// T=0 with a non-empty buffer means the model declared zero
  /// frames but returned data anyway — a shape/data
  /// inconsistency that should stay fatal.
  #[test]
  fn validate_output_dims_zero_t_with_nonempty_buffer_stays_fatal() {
    use crate::types::Lang;
    let err = validate_output_dims(0, 32, 5, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::ModelInference(payload)) = &err else {
      panic!("T=0 with non-empty buffer must stay fatal; got {err:?}");
    };
    let message = payload.message();
    assert!(
      message.contains("shape/data mismatch") || message.contains("buffer has"),
      "diagnostic must call out the shape/data inconsistency; got {message}",
      message = message
    );
  }

  // -------- stride / vocab-dim guards --------

  /// Stride in range — e.g., 16 000-sample chunk at hop=320
  /// gives T=49 (`49 × 320 = 15 680`, 320-sample slack from the
  /// chunk extent). Within the ±2-frame band, accepted.
  #[test]
  fn validate_stride_extent_accepts_typical_under_extent() {
    use crate::types::Lang;
    assert!(validate_stride_extent(49, 320, 16_000, &Lang::En).is_ok());
    // Exact integer match
    assert!(validate_stride_extent(50, 320, 16_000, &Lang::En).is_ok());
    // 1-frame over (within 2-frame slack)
    assert!(validate_stride_extent(51, 320, 16_000, &Lang::En).is_ok());
  }

  /// Stride too small (T overshoots): the model emits more
  /// frames than the chunk could produce, e.g. claimed stride
  /// is 320 but real stride is 160 → T is roughly 2× expected.
  /// Rejected as fatal `ModelInferenceFailed`.
  #[test]
  fn validate_stride_extent_rejects_t_too_large() {
    use crate::types::Lang;
    // 100 frames × 320 = 32 000 sample-equivalents for a
    // 16 000-sample chunk. Way past the upper bound (16 640).
    let err = validate_stride_extent(100, 320, 16_000, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::ModelInference(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(
      message.contains("smaller stride"),
      "diagnostic must call out the smaller-stride case; got {message}",
      message = message
    );
  }

  /// Stride too large (T undershoots): the model emits far
  /// fewer frames than the input audio supports, e.g. claimed
  /// stride is 320 but real stride is 640. Without this check,
  /// `compose_words` would compress every word into the first
  /// half of the chunk's audio. Rejected as fatal
  /// `ModelInferenceFailed`.
  #[test]
  fn validate_stride_extent_rejects_t_too_small() {
    use crate::types::Lang;
    // 25 frames × 320 = 8 000 sample-equivalents for a 16 000-
    // sample chunk. Half the expected — far below the lower
    // bound (15 360 = 16 000 − 640).
    let err = validate_stride_extent(25, 320, 16_000, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::ModelInference(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(
      message.contains("larger stride"),
      "diagnostic must call out the larger-stride case; got {message}",
      message = message
    );
  }

  /// Very short chunks where the slack is comparable to the
  /// chunk extent — the lower bound saturates to 0 and small
  /// `T` values pass. (T=0 itself is routed to recoverable
  /// `NoAlignmentPath` upstream by `validate_output_dims`.)
  #[test]
  fn validate_stride_extent_accepts_very_short_chunk_with_small_t() {
    use crate::types::Lang;
    // 200-sample chunk, hop=320 → slack=640, lower=0.
    // T=1 → frame_extent=320, within [0, 840]. Accepted.
    assert!(validate_stride_extent(1, 320, 200, &Lang::En).is_ok());
  }

  /// Vocab-dim equality: model output V matches tokenizer
  /// vocab size → accepted.
  #[test]
  fn validate_vocab_dim_accepts_exact_match() {
    use crate::types::Lang;
    assert!(validate_vocab_dim(32, 32, &Lang::En).is_ok());
  }

  /// Vocab-dim mismatch: model output V is larger than the
  /// tokenizer's vocab. Rejected as fatal — Viterbi would
  /// otherwise read posteriors from columns the tokenizer
  /// thinks correspond to the wrong tokens.
  #[test]
  fn validate_vocab_dim_rejects_oversized_model_output() {
    use crate::types::Lang;
    let err = validate_vocab_dim(1024, 32, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(AlignmentError::ModelInference(payload)) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.message();
    assert!(
      message.contains("doesn't match tokenizer vocab"),
      "diagnostic must call out the vocab mismatch; got {message}",
      message = message
    );
  }

  /// Vocab-dim mismatch: model output V is smaller than the
  /// tokenizer's vocab. Same rejection.
  #[test]
  fn validate_vocab_dim_rejects_undersized_model_output() {
    use crate::types::Lang;
    let err = validate_vocab_dim(16, 32, &Lang::En).unwrap_err();
    assert!(matches!(
      err,
      WorkFailure::Alignment(AlignmentError::ModelInference(_))
    ));
  }

  // -------- end stride / vocab-dim guards --------

  #[test]
  fn validate_output_dims_rejects_buffer_length_mismatch() {
    use crate::types::Lang;
    // Declared T=10, V=4 → 40 elements; provided buffer = 39.
    let err = validate_output_dims(10, 4, 39, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(payload) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.to_string();
    assert!(
      message.contains("doesn't match"),
      "must call out length mismatch; got {message}",
      message = message
    );
  }

  /// 32-bit-only path: usize is 32 bits, so `i64` of 1 << 33
  /// can't fit. We test the overflow logic indirectly by
  /// asking for `T * V` larger than usize::MAX; on aarch64
  /// (64-bit) we'd need an astronomical product, so this
  /// targets the `checked_mul` branch with two values whose
  /// product overflows. usize::MAX ≈ 1.8e19 on 64-bit; we use
  /// √max + 1 each.
  #[test]
  fn validate_output_dims_rejects_t_v_product_overflow() {
    use crate::types::Lang;
    // Two large values whose product overflows usize on any
    // platform. 2^32 × 2^32 = 2^64 > usize::MAX on 64-bit
    // (overflow); same on 32-bit (overflow much earlier).
    let big = i64::from(u32::MAX) + 1; // 2^32
    let err = validate_output_dims(big, big, 0, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(payload) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.to_string();
    assert!(
      message.contains("overflow") || message.contains("doesn't fit"),
      "must call out overflow; got {message}",
      message = message
    );
  }

  #[test]
  fn validate_output_dims_accepts_well_formed_shape() {
    use crate::types::Lang;
    let (t, v) = validate_output_dims(1500, 32, 1500 * 32, &Lang::En).expect("ok");
    assert_eq!(t, 1500);
    assert_eq!(v, 32);
  }

  /// Multi-frame: a NaN in frame 2 surfaces with `frame 2` in
  /// the message. Locks in that the frame index is precise
  /// (helpful for debugging upstream backend issues).
  #[test]
  fn log_softmax_locates_nan_to_specific_frame() {
    use crate::types::Lang;
    // 3 frames × 2 vocab; frame 2's first element is NaN.
    let raw = vec![0.0_f32, 0.1, 0.0, 0.1, f32::NAN, 0.1];
    let err = log_softmax_with_finite_guard(&raw, 3, 2, &Lang::En).unwrap_err();
    let WorkFailure::Alignment(payload) = err else {
      panic!("expected AlignmentFailed");
    };
    let message = payload.to_string();
    assert!(
      message.contains("frame 2"),
      "must locate the bad frame; got {message}",
      message = message
    );
  }

  // Note: the centring / scale and empty-input behaviour tests
  // moved to `super::normalize::tests` after normalisation was
  // pulled up the call stack into `Aligner::align`.
  // `encode_log_softmax` no longer normalises, so its tests
  // here cover only the reductions and the input-validation
  // boundary it does still own.
}
