//! `Command` enum and its result-side companions.
//!
//! These types are deliberately backend-agnostic — they don't name
//! `whisper-rs` types and don't include whisper.cpp-specific fields.
//! The runner's `whisper_pool` translates `AsrParams` into
//! `FullParams` (Plan B); a future swap to candle-whisper or a
//! CTranslate2 binding would change only the runner.
//!
//! See spec §3.4 (backend invariant) and §5.6.

use alloc::sync::Arc;
use alloc::vec::Vec;

use mediatime::TimeRange;
use smol_str::SmolStr;

use crate::types::{ChunkId, Lang};

/// Universal ASR knobs. Each field corresponds to either a knob
/// exposed by whisper-rs's `FullParams` or a parameter the runner's
/// own temperature retry loop consumes; nothing aspirational lives
/// here.
#[derive(Clone, Debug)]
pub struct AsrParams {
    /// Language hint passed to `FullParams::set_language`. `None`
    /// means auto-detect.
    pub language_hint: Option<Lang>,

    /// Sampling strategy. The runner constructs a fresh `FullParams`
    /// per chunk via `FullParams::new(strategy.into_whisper_rs())`.
    pub strategy: SamplingStrategy,

    /// Initial decoding temperature; first attempt of the runner's
    /// retry ladder.
    pub initial_temperature: f32,

    /// Increment applied to temperature on each retry attempt.
    /// Default 0.2 (matches WhisperX).
    pub temperature_increment: f32,

    /// Maximum total attempts (initial + retries). Default 6.
    pub max_attempts: u8,

    /// Triggers temperature retry when avg_logprob falls below this.
    /// Default -1.0.
    pub log_prob_threshold: f32,

    /// Triggers temperature retry when output compression ratio
    /// exceeds this. Default 2.4.
    pub compression_ratio_threshold: f32,

    /// Threshold above which a chunk is reported as silence
    /// (`Transcript.no_speech_prob`).
    pub no_speech_threshold: f32,

    /// Forwarded to `FullParams::set_no_context`. **Polarity matches
    /// whisper-rs**: `true` = do not use past transcription as
    /// initial prompt. Default `true` (matches the WhisperX-default
    /// behaviour of `condition_on_previous_text = false`).
    pub no_context: bool,

    /// Forwarded to `FullParams::set_suppress_blank`. Default `true`.
    pub suppress_blank: bool,

    /// Forwarded to `FullParams::set_suppress_nst`. Default `false`.
    pub suppress_non_speech_tokens: bool,

    /// Forwarded to `FullParams::set_initial_prompt`.
    pub initial_prompt: Option<SmolStr>,

    /// Forwarded to `FullParams::set_n_threads`. Default 1; the
    /// runner's parallelism comes from multiple `WhisperState`s
    /// running concurrently, not from over-subscribing in-call
    /// threads. Type matches whisper-rs's setter exactly
    /// (`std::os::raw::c_int`).
    pub n_threads: i32,
}

impl Default for AsrParams {
    fn default() -> Self {
        Self {
            language_hint: None,
            strategy: SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 },
            initial_temperature: 0.0,
            temperature_increment: 0.2,
            max_attempts: 6,
            log_prob_threshold: -1.0,
            compression_ratio_threshold: 2.4,
            no_speech_threshold: 0.6,
            no_context: true,
            suppress_blank: true,
            suppress_non_speech_tokens: false,
            initial_prompt: None,
            n_threads: 1,
        }
    }
}

/// Decoder sampling strategy.
#[derive(Copy, Clone, Debug)]
pub enum SamplingStrategy {
    /// Greedy decoding: pick the token with highest probability
    /// after considering `best_of` candidates.
    Greedy {
        /// Candidates considered per token.
        best_of: i32,
    },
    /// Beam search.
    BeamSearch {
        /// Maximum beam width.
        beam_size: i32,
        /// Patience factor (whisper.cpp ignores this as of v1.7.6;
        /// keep `-1.0` to match whisper-rs default).
        patience: f32,
    },
}

/// Result of one chunk's ASR inference.
#[derive(Clone, Debug)]
pub struct AsrResult {
    /// Transcribed text, verbatim from whisper.
    pub text: SmolStr,
    /// Detected (or hint-confirmed) language.
    pub language: Lang,
    /// Mean log-probability over emitted tokens.
    pub avg_logprob: f32,
    /// No-speech probability.
    pub no_speech_prob: f32,
    /// Final temperature used after fallback retries.
    pub temperature: f32,
}

/// Result of one chunk's word-level alignment. Empty `words` is a
/// valid result (e.g., when whisper text was empty or normalisation
/// produced an empty string).
#[derive(Clone, Debug)]
#[cfg(feature = "alignment")]
pub struct AlignmentResult {
    /// Per-word alignment entries.
    pub words: Vec<crate::types::Word>,
}

/// Stub when alignment feature is off so other code paths can refer
/// to the type without a feature gate.
#[derive(Clone, Debug)]
#[cfg(not(feature = "alignment"))]
pub struct AlignmentResult {
    /// Always empty without the alignment feature.
    pub words: Vec<crate::types::Word>,
}

/// A directive the runner consumes.
#[derive(Debug)]
pub enum Command {
    /// Run ASR on the chunk's audio. The runner ships the result
    /// back via `Transcriber::inject_asr_result`.
    RunAsr {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Chunk audio (16 kHz f32 mono).
        samples: Arc<[f32]>,
        /// Sample rate of the audio. Always
        /// [`crate::time::SAMPLE_RATE_HZ`] in v1; the field exists
        /// for forward compatibility.
        sample_rate: u32,
        /// ASR knobs for this chunk.
        params: AsrParams,
    },

    /// Run word-level alignment on the chunk's audio + transcribed
    /// text. Only emitted when the runner was configured with
    /// `with_alignment(...)`. The runner ships the result back via
    /// `Transcriber::inject_alignment_result`.
    RunAlignment {
        /// Chunk identity.
        chunk_id: ChunkId,
        /// Chunk audio (16 kHz f32 mono).
        samples: Arc<[f32]>,
        /// Sub-VAD-segments inside the chunk, in the caller's
        /// output timebase. Used by the aligner to zero-mask
        /// non-speech regions before running wav2vec2.
        sub_segments: Vec<TimeRange>,
        /// Whisper's transcribed text.
        text: SmolStr,
        /// Detected language.
        language: Lang,
    },
}

/// Compact override applied per-packet. Each `Some` field replaces
/// the corresponding default from the runner's `AsrParams` for chunks
/// produced from the packet.
#[derive(Clone, Debug, Default)]
pub struct AsrParamsOverride {
    /// Override the language hint.
    pub language_hint: Option<Option<Lang>>,
    /// Override the sampling strategy.
    pub strategy: Option<SamplingStrategy>,
    /// Override the initial temperature.
    pub initial_temperature: Option<f32>,
    /// Override the initial prompt.
    pub initial_prompt: Option<Option<SmolStr>>,
}

/// Used by the dispatch state machine to refer to a chunk's audio
/// + sub-segments without copying.
pub(crate) type ChunkAudio = Arc<[f32]>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asr_params_defaults_match_spec() {
        let p = AsrParams::default();
        match p.strategy {
            SamplingStrategy::BeamSearch { beam_size, patience } => {
                assert_eq!(beam_size, 5);
                assert!((patience - -1.0).abs() < 1e-9);
            }
            _ => panic!("default should be BeamSearch"),
        }
        assert!((p.initial_temperature - 0.0).abs() < 1e-9);
        assert!((p.temperature_increment - 0.2).abs() < 1e-9);
        assert_eq!(p.max_attempts, 6);
        assert!(p.no_context);
    }
}
