//! Whisper worker pool.
//!
//! This file used to wrap `whisper-rs`. It was migrated to the
//! in-house `whisper-cpp` bindings crate (under `crates/whisper-cpp/`)
//! after we reproduced two soundness/leak bugs in `whisper-rs 0.16`:
//!
//! 1. `set_abort_callback_safe` UB (closure-type vs.
//! `Box<dyn FnMut>` mismatch in the trampoline) → manifests as
//! `whisper_full_with_state: failed to encode` on every
//! decode. Structurally absent in `whisper-cpp` —
//! `Params::set_abort_callback` types the trampoline as
//! `*mut Box<dyn FnMut() -> bool>` end-to-end.
//! 2. `set_language` / `set_initial_prompt` `CString::into_raw`
//! leak (no `Drop`). Structurally absent — `whisper-cpp`'s
//! `Params` owns every `CString` it hands to whisper.cpp and
//! drops them with the struct.
//!
//! Because both bug classes are gone, this file no longer needs
//! the `FullParamsCache`, the `attach_abort_callback` workaround,
//! or any `#![allow(unsafe_code)]` exemption. The `unsafe`
//! surface lives entirely inside the `whisper-cpp` crate.

use std::sync::{
  Arc,
  atomic::{AtomicBool, Ordering},
};

use whispercpp::{
  Params as FullParams, SamplingStrategy as WhisperStrategy, State as WhisperState,
};

use smol_str::{SmolStr, format_smolstr};

use crate::{
  core::{AsrParams, AsrResult, SamplingStrategy},
  types::{AsrError, AsrFailure, ChunkId, Lang, WorkFailure, WorkerHangTimeout, WorkerKind},
};

/// Maximum byte length accepted for a language hint. Whisper.cpp's
/// recognised set is 2–3 letter ISO codes; 8 bytes covers every
/// real code with headroom. Defence-in-depth: whispercpp's
/// `Params::set_language` already caps at 32 bytes and validates
/// against the `g_lang` table, but this earlier rejection lets
/// us surface a typed `WorkFailure::AsrFailed` with a stable
/// diagnostic instead of letting a malformed hint reach FFI.
const MAX_LANGUAGE_CODE_LEN: usize = 8;

/// Cheap shape check before a language hint reaches whispercpp.
///
/// Returns `Err(reason)` for an empty string, anything longer
/// than [`MAX_LANGUAGE_CODE_LEN`], or anything containing a byte
/// outside `[a-z]`. The reason is a `'static` slogan suitable
/// for inclusion in an in-band [`WorkFailure::AsrFailed`]
/// message; intentionally does NOT echo the offending bytes
/// back to the caller because the language hint can be set
/// from public input.
///
/// Cheap pre-FFI shape rejection. Whispercpp's own validation
/// against the `g_lang` table is more thorough but produces a
/// different error type; running this first means callers see
/// a stable `WorkFailure::AsrFailed { kind: BackendError }`
/// diagnostic.
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
pub(in crate::runner) struct AsrWorkItem {
  /// Identity of the chunk this inference fulfils.
  pub chunk_id: ChunkId,
  /// Chunk audio (16 kHz f32 mono); shared via `Arc` with the core.
  pub samples: Arc<[f32]>,
  /// ASR knobs (per-call overrides already merged in by the runner).
  pub params: AsrParams,
  /// Caller-owned cancellation flag. The worker installs this
  /// into `FullParams` via `set_abort_callback`; whisper.cpp
  /// polls it at progress-callback boundaries and unwinds the
  /// in-flight `state.full` when set. The worker also
  /// re-checks the flag before/after every attempt to surface
  /// post-success cancellation as `WorkerHangTimeout`.
  pub abort_flag: Arc<AtomicBool>,
}

/// Pre-FFI validation. Reject malformed params with a typed
/// `WorkFailure::AsrFailed` instead of letting whisper.cpp
/// panic / abort.
pub(in crate::runner) fn validate_for_whisper_ffi(params: &AsrParams) -> Result<(), WorkFailure> {
  // Language hint shape.
  if let Some(lang) = params.language_hint()
    && let Err(reason) = validate_language_code(lang.as_str())
  {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "language hint rejected: {reason}. Reject before FFI so callers see a stable \
 diagnostic; whispercpp's `Params::set_language` would otherwise return \
 `InputTooLong` / `UnknownLanguage` for the same input."
      ),
    ))));
  }
  // NUL byte in initial_prompt.
  if let Some(prompt) = params.initial_prompt()
    && prompt.as_str().contains('\0')
  {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "initial_prompt of len {} contains an interior NUL byte; whisper-rs's set_initial_prompt \
 would panic. Reject before FFI.",
        prompt.as_str().len()
      ),
    ))));
  }
  // n_threads.
  if params.n_threads() < 1 {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "n_threads must be >= 1 (got {}); whisper.cpp's std::vector<std::thread>({} - 1) \
 would underflow / abort. Reject before FFI.",
        params.n_threads(),
        params.n_threads(),
      ),
    ))));
  }
  // Sampling-strategy fields are caller-supplied (public Rust
  // + serde). `best_of <= 0`, `beam_size <= 0`, and
  // non-finite/negative `patience` either abort the C++
  // decoder or produce nonsensical decode behaviour. Reject
  // before `FullParams::new` so callers see a typed
  // `WorkFailure` instead of a backend error.
  match params.strategy() {
    SamplingStrategy::Greedy { best_of } => {
      if best_of < 1 {
        return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
          format_smolstr!(
            "SamplingStrategy::Greedy.best_of must be >= 1 (got {best_of}); \
 whisper.cpp would either abort or fall back to greedy=1 silently. \
 Reject before FFI."
          ),
        ))));
      }
    }
    SamplingStrategy::BeamSearch {
      beam_size,
      patience,
    } => {
      if beam_size < 1 {
        return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
          format_smolstr!(
            "SamplingStrategy::BeamSearch.beam_size must be >= 1 (got {beam_size}); \
 whisper.cpp's beam-search loop would underflow on a non-positive size. \
 Reject before FFI."
          ),
        ))));
      }
      // `-1.0` is whisper.cpp's documented "use default patience"
      // sentinel and matches `SamplingStrategy::default()`. Any
      // other negative or non-finite value produces nonsensical
      // pruning thresholds.
      if !patience.is_finite() || (patience <= 0.0 && (patience - -1.0).abs() > f32::EPSILON) {
        return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
          format_smolstr!(
            "SamplingStrategy::BeamSearch.patience must be a finite positive value \
 (or the documented `-1.0` whisper.cpp default sentinel); got {patience}. \
 Reject before FFI."
          ),
        ))));
      }
    }
  }
  // Every public `f32` knob is caller-supplied. NaN / Inf
  // temperatures get plumbed into `set_temperature`
  // and produce undefined sampling; non-finite thresholds make
  // the post-decode logprob / cratio / no_speech comparisons
  // arbitrary (NaN comparisons are always false), so the
  // ladder either rejects a valid decode or accepts a
  // hallucination depending on which side of the comparison
  // the NaN landed. Reject before `FullParams::new`.
  for (name, value) in [
    ("initial_temperature", params.initial_temperature()),
    ("temperature_increment", params.temperature_increment()),
    ("log_prob_threshold", params.log_prob_threshold()),
    (
      "compression_ratio_threshold",
      params.compression_ratio_threshold(),
    ),
    ("no_speech_threshold", params.no_speech_threshold()),
  ] {
    if !value.is_finite() {
      return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
        format_smolstr!(
          "AsrParams::{name} must be finite (got {value}); non-finite values are \
 either undefined to whisper.cpp's sampler or make the post-decode \
 gates degenerate. Reject before FFI."
        ),
      ))));
    }
  }
  // domain checks on the
  // already-finite knobs. WhisperX/OpenAI Whisper temperatures
  // are sampling probabilities in `[0, 1]`; thresholds have
  // their own meaningful domains. a finite-but-
  // out-of-range value (e.g. `initial_temperature = -2.0`)
  // would silently produce non-parity decodes instead of a
  // typed failure.
  let init_t = params.initial_temperature();
  if !(0.0..=1.0).contains(&init_t) {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "AsrParams::initial_temperature must be in [0.0, 1.0] (got {init_t}); \
 WhisperX sampling temperatures are probabilities."
      ),
    ))));
  }
  let step = params.temperature_increment();
  if !(0.0..=1.0).contains(&step) {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "AsrParams::temperature_increment must be in [0.0, 1.0] (got {step}); \
 a step outside this range either skips meaningful retries or wraps \
 past the [0, 1] sampler domain."
      ),
    ))));
  }
  let nsp = params.no_speech_threshold();
  if !(0.0..=1.0).contains(&nsp) {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "AsrParams::no_speech_threshold must be in [0.0, 1.0] (got {nsp}); \
 it gates against per-segment no-speech probabilities."
      ),
    ))));
  }
  let cratio = params.compression_ratio_threshold();
  if cratio <= 0.0 {
    return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
      format_smolstr!(
        "AsrParams::compression_ratio_threshold must be > 0 (got {cratio}); \
 WhisperX's gate compares `len(text) / len(zlib(text))` to a positive \
 ceiling — a non-positive ceiling rejects every transcript."
      ),
    ))));
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
fn build_template(params: &AsrParams) -> Result<FullParams, WorkFailure> {
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
    // `whispercpp::Params::set_language(&str)` copies into an
    // owned `CString` for the Params lifetime — no `'static`
    // borrow required, no leak. Whispercpp itself validates
    // against whisper.cpp's `g_lang` table and returns
    // `UnknownLanguage` for codes outside the recognised set.
    //
    // previously the Result was
    // dropped here, so a `Lang::Other("zzzz")` that passed the
    // pre-FFI shape check would silently be ignored and ASR
    // would proceed with a stale (or default) language tag. Now
    // we surface the error as `WorkFailure::AsrFailed/BackendError`
    // so the caller's retry / fallback logic sees the failure.
    p.set_language(lang.as_str()).map_err(|e| {
      WorkFailure::Asr(AsrError::Backend(AsrFailure::new(format_smolstr!(
        "set_language({}) rejected by whisper.cpp: {e}",
        lang.as_str()
      ))))
    })?;
  } else {
    p.set_detect_language(true);
  }
  if let Some(prompt) = params.initial_prompt() {
    // Prompt content is user-supplied so we are more cautious —
    // if whispercpp rejects it (NUL, oversize), drop the prompt
    // and continue rather than fail the chunk. Whisper.cpp
    // tolerates an unset prompt; the alternative would be
    // failing the entire ASR job over a string-encoding nit.
    //
    // surface the rejection
    // via a structured stderr line so callers can correlate
    // changed-language-biasing or glossary-regression behaviour
    // back to the dropped prompt instead of silently running
    // ASR without it. The prompt itself isn't logged — it is
    // user-supplied content.
    if let Err(e) = p.set_initial_prompt(prompt.as_str()) {
      eprintln!(
        "asry asr initial_prompt rejected by whisper.cpp; \
 continuing without prompt prompt_chars={} error={e:?}",
        prompt.chars().count(),
      );
    }
  }
  Ok(p)
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
  let template = build_template(params)?;
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
/// : the runner uses this to honor the
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
///
/// short-text guard: the shingle
/// proxy degenerates on tiny transcripts where the unique-shingle
/// count is dominated by the windowing arithmetic, not the model's
/// repetition pattern. A non-pathological 4-byte transcript like
/// `"test"` produces exactly one 4-byte shingle, yielding ratio
/// `4/1 = 4.0` — comfortably above the default 2.4 threshold and
/// rejected as a "hallucination". Real zlib would compress these
/// well below 1.0 because of fixed deflate header overhead. We
/// bypass the proxy below `MIN_RATIO_BYTES` so single-word /
/// short-utterance chunks are not falsely rejected; the threshold
/// only meaningfully fires on long repetitive runaways anyway.
pub(super) fn compute_compression_ratio(state: &WhisperState) -> f32 {
  let n = state.n_segments();
  if n <= 0 {
    return 0.0;
  }
  let mut text = String::new();
  for i in 0..n {
    if let Some(segment) = state.segment(i)
      && let Ok(s) = segment.text()
    {
      text.push_str(s);
    }
  }
  compression_ratio_for_text(&text)
}

/// Pure compression-ratio kernel split out of
/// [`compute_compression_ratio`] so the short-text guard is
/// testable without setting up a real `WhisperState`.
///
/// matches WhisperX / OpenAI
/// Whisper's `zlib.compress(text.encode("utf-8"))` byte length
/// 1:1 via [`miniz_oxide`]. The earlier shingle proxy
/// mis-rejected legitimately repetitive transcripts (e.g.
/// `"thank you "×4` scored 3.9 vs zlib's 1.86, exhausting the
/// temperature ladder and surfacing as `AllTemperaturesFailed`
/// for valid ASR text).
///
/// `MIN_RATIO_BYTES` short-text guard is
/// preserved so single-word transcripts that zlib still
/// compresses near 1.0 (deflate header overhead dominates) skip
/// the gate cleanly, even though the real ratio would already
/// pass — keeping the existing observable behaviour for tiny
/// outputs.
pub(super) fn compression_ratio_for_text(text: &str) -> f32 {
  use miniz_oxide::deflate::compress_to_vec_zlib;

  // 32 bytes ≈ 8 short "word " tokens — long enough that a
  // runaway "yes yes yes yes ..." style hallucination has
  // already started repeating, but short enough that ordinary
  // one-clause replies are exempt.
  const MIN_RATIO_BYTES: usize = 32;

  let raw = text.len();
  if raw < MIN_RATIO_BYTES {
    return 0.0;
  }
  // WhisperX uses Python's default zlib level (6). miniz_oxide's
  // level-6 zlib output matches in byte-length on the inputs the
  // threshold cares about (long repetitive runs).
  let compressed_len = compress_to_vec_zlib(text.as_bytes(), 6).len();
  if compressed_len == 0 {
    return 0.0;
  }
  raw as f32 / compressed_len as f32
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
/// Failure:
/// - `WorkFailure::Asr(AsrError::AllTemperaturesExhausted(_))`
///   after all `max_attempts` failed.
/// - `WorkFailure::Asr(AsrError::Backend(_))` if `state.full()`
///   itself returned an error.
/// - `WorkFailure::WorkerHang(_)` (kind `WorkerKind::Asr`) if the
///   abort flag was flipped.
pub(in crate::runner) fn run_with_temperature_ladder(
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
    // pre-attempt abort check.
    // The caller may have flipped `abort_flag` between
    // attempts (e.g. their cancellation token fired during
    // the previous temperature step's Ok path); without this
    // we'd start a fresh `state.full()` on cancelled work.
    if job.abort_flag.load(Ordering::Relaxed) {
      return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
        WorkerKind::Asr,
        started_at.elapsed(),
      )));
    }

    // re-validate the
    // per-attempt temperature. `validate_for_whisper_ffi`
    // checks `initial_temperature` and
    // `temperature_increment` are finite, but a finite
    // `initial + N * increment` can still saturate to `inf`
    // on later iterations (e.g. both fields = `f32::MAX`,
    // `max_attempts >= 2`). The ladder happily
    // forwarded that to `set_temperature` and into
    // whisper.cpp's sampler — exactly the FFI hardening the
    // earlier rounds were meant to prevent.
    if !temperature.is_finite() {
      return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
        format_smolstr!(
          "ladder temperature became non-finite ({temperature}) after starting from \
 initial={} step={}; refuse to forward into whisper.cpp's sampler",
          p.initial_temperature(),
          p.temperature_increment(),
        ),
      ))));
    }
    // stop the ladder at the
    // [0, 1] domain boundary. Even with valid initial /
    // increment values (e.g. `initial=0.9`, `increment=0.2`,
    // `max_attempts=2`), the derived second-attempt
    // temperature is `1.1` — outside WhisperX/OpenAI Whisper's
    // sampling-temperature domain. WhisperX itself stops at
    // `1.0`; mirror that contract by treating an
    // out-of-domain attempt as ladder exhaustion, surfacing
    // `AllTemperaturesFailed` rather than forwarding `1.1`
    // into `set_temperature`.
    if !(0.0..=1.0).contains(&temperature) {
      return Err(WorkFailure::Asr(AsrError::AllTemperaturesExhausted(
        AsrFailure::new(format_smolstr!(
          "temperature ladder exhausted at attempt-derived value {temperature} \
 outside [0.0, 1.0] (initial={}, step={}); WhisperX caps the \
 fallback ladder at 1.0",
          p.initial_temperature(),
          p.temperature_increment(),
        )),
      )));
    }
    let template = build_template(p)?;
    let mut full = finalize_chunk(template, p, job.abort_flag.clone());
    full.set_temperature(temperature);
    let outcome = state.full(&full, job.samples.as_ref());

    // post-call abort check
    // regardless of `outcome.is_ok()`. The caller can flip
    // `abort_flag` between whisper.cpp's last abort callback
    // and `state.full` returning, OR the backend can return
    // `Ok` despite the flag flipping near the end. Either way
    // we must NOT continue into segment-count / threshold
    // checks — that would hand the caller a transcript for
    // work they tried to cancel. Surface as
    // `WorkerHangTimeout` for both cases (the flag-flipped
    // semantics matches the legacy watchdog path).
    if job.abort_flag.load(Ordering::Relaxed) {
      return Err(WorkFailure::WorkerHang(WorkerHangTimeout::new(
        WorkerKind::Asr,
        started_at.elapsed(),
      )));
    }

    if let Err(e) = outcome {
      // Genuine backend error (abort_flag was not set; the
      // pre-Err check above already covered the abort case).
      return Err(WorkFailure::Asr(AsrError::Backend(AsrFailure::new(
        format_smolstr!("{e}"),
      ))));
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

    // : implement the documented
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

  Err(WorkFailure::Asr(AsrError::AllTemperaturesExhausted(
    AsrFailure::new(format_smolstr!(
      "all {} temperature attempts failed for chunk {:?}",
      max,
      job.chunk_id,
    )),
  )))
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
  let segments: Vec<whispercpp::Segment<'_>> = state.segments_iter().collect();
  // ctx for token-id → bytes resolution (per : the
  // dispatcher needs per-token byte offsets to slice DTW per run).
  let ctx_for_dispatch = state.context();
  let runs = crate::align::dispatch(ctx_for_dispatch, &segments, Some(language.clone()));

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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn asr_work_item_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<AsrWorkItem>();
  }

  /// Short transcripts must not trip the compression-ratio
  /// threshold. The shingle proxy would yield `4/1 = 4.0` for
  /// `"test"` and `5/2 = 2.5` for `"hello"` — both above the
  /// default 2.4 gate — and cause every temperature attempt to
  /// fail as `AllTemperaturesFailed`. Anything shorter than
  /// `MIN_RATIO_BYTES` therefore returns 0.0 (always passes).
  #[test]
  fn compression_ratio_short_transcripts_pass_default_threshold() {
    let threshold = AsrParams::default().compression_ratio_threshold();
    for sample in ["test", "hello", "yes", "Hi.", "A.", "transcript."] {
      let r = compression_ratio_for_text(sample);
      assert!(
        r <= threshold,
        "short transcript {sample:?} ({} bytes) ratio={r} above default threshold {threshold}",
        sample.len(),
      );
    }
  }

  /// The threshold still fires on long, repetitive runaway
  /// hallucinations — the failure mode the gate was designed
  /// for. WhisperX's threshold (2.4) is calibrated against the
  /// "yes yes yes ..." failure mode running on for ~100+ tokens;
  /// at 64 repetitions of `"yes "` (256 bytes) zlib's level-6
  /// output stays around 25 bytes for a ratio over 10. Tests
  /// against a fairly long runaway so the real zlib header
  /// overhead is amortised away.
  ///
  /// the previous shingle-proxy
  /// version of this test used `"yes "×16` (64 bytes); under
  /// real zlib that input scores ~2.37, just under the default
  /// 2.4. The point of the test is that the gate STILL fires
  /// on genuinely runaway output — `"yes "×64` clears any
  /// reasonable threshold.
  #[test]
  fn compression_ratio_runaway_repetition_above_threshold() {
    let threshold = AsrParams::default().compression_ratio_threshold();
    let runaway = "yes ".repeat(64);
    let r = compression_ratio_for_text(&runaway);
    assert!(
      r > threshold,
      "runaway repetition ratio {r} should exceed threshold {threshold}",
    );
  }

  /// regression: legitimately
  /// repetitive transcripts must NOT be rejected. WhisperX's
  /// real zlib gives `"thank you "×4` (40 bytes) a ratio
  /// around 1.86, well under the 2.4 default. The  /// shingle proxy returned 3.9, exhausting the temperature
  /// ladder and surfacing as `AllTemperaturesFailed` for
  /// otherwise valid ASR text.
  #[test]
  fn compression_ratio_legit_repetition_under_threshold() {
    let threshold = AsrParams::default().compression_ratio_threshold();
    let legit = "thank you ".repeat(4);
    let r = compression_ratio_for_text(&legit);
    assert!(
      r <= threshold,
      "legit repetitive transcript {legit:?} ratio={r} must not exceed threshold {threshold}",
    );
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

  /// a `Lang::Other` whose
  /// content passes the lowercase-ASCII shape check but is NOT
  /// in whisper.cpp's `g_lang` table must surface as
  /// `WorkFailure::AsrFailed/BackendError` rather than being
  /// silently dropped (`let _ = p.set_language(...)`).
  /// Without this propagation, callers could see transcripts
  /// in the wrong language while believing their hint was
  /// honoured.
  #[test]
  fn full_params_from_propagates_unknown_language_from_whispercpp() {
    // "zzzz" is lowercase ASCII (passes shape check) but isn't in
    // whisper.cpp's `g_lang` table, so `set_language` returns
    // `WhisperError::UnknownLanguage`.
    let p = AsrParams::default().with_language_hint(Some(Lang::Other(SmolStr::from("zzzz"))));
    let flag = Arc::new(AtomicBool::new(false));
    let res = full_params_from(&p, 0.0, flag);
    match res {
      Err(WorkFailure::Asr(AsrError::Backend(payload))) => {
        assert!(
          payload.message().contains("set_language") && payload.message().contains("zzzz"),
          "expected set_language diagnostic mentioning the offending code; got {message}",
          message = payload.message()
        );
      }
      other => panic!("expected AsrFailed/BackendError for unknown language; got {other:?}"),
    }
  }

  /// regression: an interior NUL in the language
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
      Err(WorkFailure::Asr(AsrError::Backend(payload))) => {
        assert!(
          payload.message().contains("language hint")
            && payload.message().contains("lowercase ASCII"),
          "expected charset-violation diagnostic; got {message}",
          message = payload.message()
        );
      }
      other => panic!("expected AsrFailed/BackendError; got {other:?}"),
    }
  }

  // --- : language hint shape validation ---

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

  /// A high-cardinality `Lang::Other(SmolStr)` from a malicious
  /// or buggy caller must be rejected as an in-band chunk
  /// failure rather than reaching FFI. The validator is cheap
  /// pre-FFI rejection so callers see a stable diagnostic
  /// instead of whispercpp's `InputTooLong` / `UnknownLanguage`
  /// variants for the same input shape.
  #[test]
  fn full_params_from_rejects_high_cardinality_language_hint() {
    let p = AsrParams::default().with_language_hint(Some(Lang::Other(SmolStr::from(
      "very-long-attacker-string",
    ))));
    let flag = Arc::new(AtomicBool::new(false));
    let res = full_params_from(&p, 0.0, flag);
    match res {
      Err(WorkFailure::Asr(AsrError::Backend(_))) => {}
      other => panic!("expected AsrFailed/BackendError; got {other:?}"),
    }
  }

  /// regression: an interior NUL in `initial_prompt`
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
      Err(WorkFailure::Asr(AsrError::Backend(payload))) => {
        assert!(
          payload.message().contains("initial_prompt") && payload.message().contains("NUL"),
          "expected NUL diagnostic; got {message}",
          message = payload.message()
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

  // Sanity check: confirm `run_with_temperature_ladder` is callable
  // with the expected signature (and is pinned as `Send` so it can
  // run on a worker thread). End-to-end ladder behaviour and
  // layered-ladder suppression are exercised by other tests.
  // Here we only assert the type signature compiles.
  // FullParamsCache regression tests deleted alongside the cache
  // itself in the whisper-cpp migration. The cache existed only
  // to bound the whisper-rs `set_language` / `set_initial_prompt`
  // CString leak; whisper-cpp's `Params` owns + drops every
  // CString, so the test invariants ("differing only in language
  // produces a new entry", etc.) no longer have a structure to
  // pin down. Equivalent coverage now lives implicitly in
  // `Params::Drop` (no leaks) and the integration parity runs.

  /// sampling-strategy
  /// fields must be validated before they reach whisper.cpp.
  /// `best_of=0` would either abort the C++ decoder or fall
  /// back silently; `beam_size=0` underflows the beam loop;
  /// non-finite `patience` produces nonsensical pruning.
  #[test]
  fn validate_for_whisper_ffi_rejects_zero_best_of() {
    let p = AsrParams::default().with_strategy(SamplingStrategy::Greedy { best_of: 0 });
    let err = validate_for_whisper_ffi(&p).unwrap_err();
    match err {
      WorkFailure::Asr(AsrError::Backend(payload)) => {
        assert!(
          payload.message().contains("best_of"),
          "got {message}",
          message = payload.message()
        );
      }
      other => panic!("expected AsrError::Backend, got {other:?}"),
    }
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_zero_beam_size() {
    let p = AsrParams::default().with_strategy(SamplingStrategy::BeamSearch {
      beam_size: 0,
      patience: 1.0,
    });
    let err = validate_for_whisper_ffi(&p).unwrap_err();
    match err {
      WorkFailure::Asr(asr) => {
        let message = asr.to_string();
        assert!(message.contains("beam_size"), "got {message}");
      }
      other => panic!("expected AsrFailed, got {other:?}"),
    }
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_non_finite_patience() {
    // `-1.0` is the documented whisper.cpp default sentinel and
    // is the value `SamplingStrategy::default()` carries — must
    // be accepted. Other invalid values must fail.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -2.0] {
      let p = AsrParams::default().with_strategy(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: bad,
      });
      let err = validate_for_whisper_ffi(&p).unwrap_err();
      assert!(
        matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("patience")),
        "patience={bad} should reject; got {err:?}",
      );
    }
  }

  #[test]
  fn validate_for_whisper_ffi_accepts_negative_one_sentinel_patience() {
    let p = AsrParams::default().with_strategy(SamplingStrategy::BeamSearch {
      beam_size: 5,
      patience: -1.0,
    });
    assert!(
      validate_for_whisper_ffi(&p).is_ok(),
      "-1.0 is whisper.cpp's `use default patience` sentinel and matches \
 SamplingStrategy::default(); must be accepted",
    );
  }

  /// every public f32 ASR
  /// knob must be finite. NaN/Inf temperatures get plumbed
  /// into `set_temperature`; NaN/Inf thresholds make the
  /// post-decode comparisons degenerate (NaN compares always
  /// false, so e.g. `compression_ratio_threshold = NaN` would
  /// never reject a runaway).
  #[test]
  fn validate_for_whisper_ffi_rejects_non_finite_temperature() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let p = AsrParams::default().with_initial_temperature(bad);
      let err = validate_for_whisper_ffi(&p).unwrap_err();
      assert!(
        matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("initial_temperature")),
        "initial_temperature={bad} should reject; got {err:?}",
      );
    }
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_non_finite_thresholds() {
    let bad = f32::NAN;
    let cases: [(&str, AsrParams); 3] = [
      (
        "log_prob_threshold",
        AsrParams::default().with_log_prob_threshold(bad),
      ),
      (
        "compression_ratio_threshold",
        AsrParams::default().with_compression_ratio_threshold(bad),
      ),
      (
        "no_speech_threshold",
        AsrParams::default().with_no_speech_threshold(bad),
      ),
    ];
    for (knob, p) in cases {
      let err = validate_for_whisper_ffi(&p).unwrap_err();
      assert!(
        matches!(&err, WorkFailure::Asr(e) if e.to_string().contains(knob)),
        "{knob}=NaN should reject; got {err:?}",
      );
    }
  }

  /// domain-check the
  /// finite f32 knobs. WhisperX temperatures are sampling
  /// probabilities in `[0, 1]`; out-of-range values produce
  /// non-parity decodes silently.
  #[test]
  fn validate_for_whisper_ffi_rejects_negative_initial_temperature() {
    let p = AsrParams::default().with_initial_temperature(-0.1);
    let err = validate_for_whisper_ffi(&p).unwrap_err();
    assert!(
      matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("initial_temperature")),
      "got {err:?}",
    );
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_above_one_initial_temperature() {
    let p = AsrParams::default().with_initial_temperature(1.5);
    let err = validate_for_whisper_ffi(&p).unwrap_err();
    assert!(
      matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("initial_temperature")),
      "got {err:?}",
    );
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_out_of_range_temperature_increment() {
    for bad in [-0.1_f32, 1.5_f32] {
      let p = AsrParams::default().with_temperature_increment(bad);
      let err = validate_for_whisper_ffi(&p).unwrap_err();
      assert!(
        matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("temperature_increment")),
        "increment={bad} should reject; got {err:?}",
      );
    }
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_out_of_range_no_speech_threshold() {
    for bad in [-0.1_f32, 1.5_f32] {
      let p = AsrParams::default().with_no_speech_threshold(bad);
      let err = validate_for_whisper_ffi(&p).unwrap_err();
      assert!(
        matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("no_speech_threshold")),
        "no_speech={bad} should reject; got {err:?}",
      );
    }
  }

  #[test]
  fn validate_for_whisper_ffi_rejects_non_positive_compression_ratio_threshold() {
    for bad in [0.0_f32, -2.4_f32] {
      let p = AsrParams::default().with_compression_ratio_threshold(bad);
      let err = validate_for_whisper_ffi(&p).unwrap_err();
      assert!(
        matches!(&err, WorkFailure::Asr(e) if e.to_string().contains("compression_ratio_threshold")),
        "cratio={bad} should reject; got {err:?}",
      );
    }
  }

  #[test]
  fn validate_for_whisper_ffi_accepts_default_threshold_values() {
    let p = AsrParams::default();
    assert!(
      validate_for_whisper_ffi(&p).is_ok(),
      "AsrParams::default() must round-trip through validation",
    );
  }

  #[test]
  fn validate_for_whisper_ffi_accepts_valid_strategies() {
    let g = AsrParams::default().with_strategy(SamplingStrategy::Greedy { best_of: 1 });
    assert!(validate_for_whisper_ffi(&g).is_ok());
    let b = AsrParams::default().with_strategy(SamplingStrategy::BeamSearch {
      beam_size: 5,
      patience: 1.0,
    });
    assert!(validate_for_whisper_ffi(&b).is_ok());
  }

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
}
