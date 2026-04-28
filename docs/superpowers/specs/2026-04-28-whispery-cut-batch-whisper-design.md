# whispery — Cut, Batch, Whisper, Align

**Status:** Draft, awaiting review.
**Date:** 2026-04-28.
**Repository:** `findit-studio/whispery`.

A Rust crate that takes raw 16 kHz mono PCM and pre-computed VAD speech segments, cuts them into Whisper-friendly windows, batches whisper-rs inference, and emits per-chunk transcripts with word-level timestamps obtained via wav2vec2 forced alignment.

The design references WhisperX's cut-and-batch pipeline and is built around a Sans-I/O state-machine core with a feature-gated reference runner.

---

## 1. Background and goals

### 1.1 Pipeline context

The findit-studio media indexing pipeline is roughly:

```
ffmpeg → audio packets ──┬──► silero (VAD)        ──┐
                         │                          ├──► whispery (cut + batch + whisper + align)
                         │                          │
                         └──► soundevents (CED) ────┴──► [parallel branch into the index]

whispery output ──► consumed by the indexer for:
                    - text columns (BM25 / FTS)
                    - time-range references that the indexer uses to slice 48 kHz audio
                      for textclap embedding (textclap does NOT consume Whisper text)
```

soundevents (CED) runs in parallel with whispery and is independent of it.

silero (VAD) is upstream of whispery; whispery accepts silero-shaped speech segments as input but does not depend on the silero crate at runtime — the contract is the segment shape, not the implementation.

### 1.2 Reference: WhisperX

WhisperX's published architecture has three notable contributions over a naive Whisper invocation:

1. **VAD-based intelligent chunking.** Run VAD over the input, then greedily merge consecutive speech segments into chunks bounded by Whisper's 30 s encoder window. Silence gaps inside a merged chunk are preserved in the audio slice, so the model still sees the original audio context.
2. **Batched inference.** Stack the merged chunks into a batched mel-spectrogram tensor `(B, n_mels, n_frames)`, run the encoder once per batch.
3. **Forced word-level alignment.** Run a per-language wav2vec2 phoneme model over each chunk's audio, then CTC-align the transcribed text against the phoneme logits to recover sample-accurate word timestamps. This sits *after* Whisper, not inside it.

whispery adopts (1) and (3) directly. It adopts the *architectural intent* of (2) — concurrent inference across N chunks — but realises it differently because whisper-rs (which wraps whisper.cpp) lacks true batched encode/decode kernels. See §1.4.

### 1.3 Goals

- Provide a streaming, packet-by-packet entry point suitable for an indexing engine that processes hours of audio without buffering whole files in memory.
- Emit per-chunk `Transcript`s with word-level timestamps, language tags, and provenance back to the original VAD segments.
- Keep the cut and dispatch logic free of ML dependencies so it can be tested deterministically and embedded in alternative runtimes.
- Default to ergonomic single-call usage (`ManagedTranscriber`) for the existing indexer, with a Sans-I/O `Transcriber` exposed for tests and for users who need to plug their own runtime.
- Use mediatime types throughout for sample-accurate cross-timebase output.

### 1.4 whisper-rs vs faster-whisper

WhisperX's batched-inference speedup comes from running `batch_size=8` in one CTranslate2 GPU encode call. whisper.cpp has no equivalent — its concurrency story is "one shared `WhisperContext` (the model) plus N `WhisperState` instances (per-decoder state) running on N threads."

For an indexing workload (throughput-bound, latency-tolerant), this is acceptable: N-way state concurrency gives near-linear CPU scaling and meaningful GPU scaling up to memory limits. The bottleneck is usually the number of available cores or the memory ceiling for parallel states, not the lack of GPU-batched ops.

The design is **backend-agnostic in the core**: swapping whisper-rs for candle-whisper (which does support tensor-stacked batched inference) or for a future Rust CTranslate2 binding requires only changing the runner, not the cut/dispatch state machine or the public types.

### 1.5 Non-goals (v1)

- **Speaker diarization.** Speaker labels are not produced by whispery. See §1.6 for the integration model.
- **Multi-channel audio.** Input is mono f32 16 kHz. Caller mixes down.
- **Resampling.** Caller delivers 16 kHz; we do not own ffmpeg or libsamplerate logic.
- **Async runner.** No `tokio` dependency in v1. The runner exposes a sync push + sync poll API.
- **Live captioning latency profile.** We optimise for indexing throughput; v1 cuts at ≤ ~30 s chunks for max Whisper quality. Tuning a sub-second-latency profile is deferred to v1.x.
- **DTW token timestamps.** Word-level timestamps come from forced alignment only; whisper.cpp's DTW path is not enabled.
- **Auto-downloading wav2vec2 models.** Callers register paths explicitly. Auto-fetch is a v2 add-on that doesn't change the API.
- **Bundled model files.** v1 ships with no checked-in GGUF or wav2vec2 weights. Callers point at on-disk model paths.

### 1.6 Integration with diarization (speaker-agnostic by design)

whispery is speaker-agnostic. Diarization runs as a sibling of whispery on the same source audio, not upstream or downstream of it; the indexer joins the two outputs by time-range overlap.

**Join contract (verified against `dia` v0.1.0).** The findit-studio diarization crate `dia` (in `/Users/user/Develop/findit-studio/dia`) emits `DiarizedSpan` values with a `range: mediatime::TimeRange` at the **same 1/16000 timebase** as whispery's `Word.range`. The minimal join is therefore an interval-overlap test on the shared timebase — no rescaling, no anchor types, no whispery-side speaker awareness:

```rust
// indexer-side pseudo-code
for word in transcript.words() {
    for span in dia_spans_overlapping(word.range()) {
        index.insert(WordRow {
            chunk_id: transcript.chunk_id(),
            word_text: word.text(),
            range: word.range(),
            speaker_id: Some(span.speaker_id()),
            speaker_first_seen: span.is_new_speaker(),
        });
    }
}
```

Concretely from `dia`:

```rust
pub struct DiarizedSpan {
    range: mediatime::TimeRange,         // 1/16000 timebase
    speaker_id: u64,                      // session-local
    is_new_speaker: bool,
    average_activation: f32,
    activity_count: u32,
    clean_mask_fraction: f32,
}
```

`Transcript.vad_segments()` is the secondary anchor when overlap is ambiguous (e.g., a word that straddles silence inside the merged chunk): the indexer should weight overlap with VAD segments more heavily than overlap with the chunk's outer `range`.

**What whispery does not need to do.**

- whispery does not consume `DiarizedSpan`s. It does not depend on the `dia` crate.
- whispery does not assign or track speaker IDs. `dia`'s speaker IDs are session-local `u64`s; cross-file speaker stability is the indexer's concern (a global cluster table keyed by embeddings, owned outside both whispery and dia).
- whispery does not need to expose phonetic-level timestamps or per-VAD-segment confidence. The 1/16000 sample-accurate `Word.range` is sufficient for `dia`'s overlap-based join.

This section pins the join contract; if a future `dia` revision introduces a different output shape, this spec must be revisited.

### 1.7 Crate-only deployment (decided)

whispery is a **library crate** consistent with its siblings (silero, soundevents, textclap, dia — all library crates with sync APIs). Callers link it directly into the indexer and drive it in-process. There is no `findit-whispery-service` binary in v1.

If a wrapper service is later required for IPC isolation, it is a separately-shipped binary that calls `ManagedTranscriber` and exposes the runner's API over a queue or RPC surface; whispery the crate stays unchanged. This spec does not preclude that addition; it only declines to do it in v1.

---

## 2. Design principles

1. **Sans-I/O core.** The cut and dispatch logic is a pure state machine: no threads, no I/O, no ML deps, no async. The caller (or our own runner) drives it via push and poll.
2. **One crate, two layers.** A default-on `runner` feature wraps the core with a whisper-rs worker pool and an ort-based aligner. With the feature off, the crate has zero ML dependencies.
3. **Composable, not all-in-one.** whispery does cut + batch + whisper + align. silero (VAD) and ffmpeg (decoding) are caller's concerns, not ours.
4. **Indexing-first.** Throughput over latency. Default chunk size is 30 s for Whisper quality; default worker count is `num_cpus` or a reasonable cap.
5. **mediatime everywhere.** All emitted ranges are `mediatime::TimeRange` at a sample-rate-derived timebase. Downstream code that prefers ms or NTSC frame rate `rescale_to` it.
6. **Backend-agnostic core.** The state machine emits commands ("please run whisper on these samples") and consumes results; it does not name `whisper-rs` or `ort`.

---

## 3. Architecture overview

### 3.1 Crate layout

```
whispery/
├── Cargo.toml
├── src/
│   ├── lib.rs               re-exports
│   ├── types.rs             Transcript, Word, Lang, ChunkId, errors
│   ├── time.rs              timebase constants, helpers
│   │
│   ├── core/                Sans-I/O. NO whisper-rs, NO ort.
│   │   ├── mod.rs
│   │   ├── transcriber.rs   Transcriber: push / inject / poll
│   │   ├── cut.rs           merge_chunks state machine
│   │   ├── dispatch.rs      per-chunk lifecycle (whisper → align → emit)
│   │   ├── buffer.rs        bounded sample ring buffer
│   │   ├── command.rs       Command enum
│   │   └── event.rs         Event enum
│   │
│   └── runner/              cfg(feature = "runner")
│       ├── mod.rs
│       ├── managed.rs       ManagedTranscriber
│       ├── whisper_pool.rs  N WhisperState worker threads
│       ├── aligner.rs       cfg(feature = "alignment")
│       └── aligner_set.rs   cfg(feature = "alignment")
│
├── examples/
│   ├── core_only.rs
│   └── managed_runner.rs
│
├── benches/
│   ├── cut.rs                  cut state machine throughput
│   └── dispatch.rs             dispatch state machine with mocked inference
│
└── tests/
    ├── core_cut.rs
    ├── core_dispatch.rs
    └── runner_e2e.rs        cfg(feature = "runner")
```

### 3.2 Cargo features

| Feature        | Default | Pulls                                        | Notes |
|----------------|:-:|---------------------------------------------|-------|
| `std`          | yes | `alloc` + `std`                              | Core compiles to no_std + alloc with this off; runner is std-only. |
| `runner`       | yes | `whisper-rs ^0.13`, `crossbeam-channel ^0.5` | The bundled production runner. Whisper-rs version pinned to a specific compatible major; revisited per release. |
| `alignment`    | no  | `ort = "2.0.0-rc.12"`, `tokenizers ^0.23`, `ndarray ^0.16` | **Opt-in.** Forced-alignment pieces in the runner. Pulls the heaviest deps in the tree (~150 MB of build artefacts). Without this feature, `Transcript.words` is always empty. ort version matches silero/soundevents/textclap. |
| `serde`        | no  | `serde ^1`                                   | Derive Serialize/Deserialize on public types. |
| `arbitrary`    | no  | `arbitrary ^1`                               | Fuzz harnesses. |
| `quickcheck`   | no  | `quickcheck ^1`                              | Property tests. |

Permanent (non-feature-gated) dependencies: `mediatime`, `smol_str`, `thiserror`, `smallvec`. None of these can be turned off; they are part of the public type surface or its error contract. (`smallvec` is used by `AsrParams` for the inline-sized collection of temperature schedule and suppress tokens.)

`alignment` requires `runner`. The Cargo manifest enforces this with a `required-features` constraint on the alignment module. Default feature set (`default = ["std", "runner"]`) gives an indexer-ready build *without* alignment; users opting into word-level alignment add `whispery = { version = "...", features = ["alignment"] }`.

### 3.3 Public surface

```rust
// Always public:
pub mod types;
pub mod core;
pub use types::{
    Transcript, Word, Lang, ChunkId, VadSegment,
    TranscriberError, WorkFailure, AsrFailureKind, AlignmentFailureKind,
    PushKind, WorkerKind,
};
pub use core::{
    Transcriber, TranscriberConfig, LanguagePolicy,
    Command, Event,
    AsrParams, AsrResult, AsrTokenHint,
};

#[cfg(feature = "runner")]
pub mod runner;

#[cfg(feature = "runner")]
pub use runner::{
    ManagedTranscriber, ManagedTranscriberBuilder,
    WhisperPoolConfig, Device,
    AsrParamsOverride, RunnerError,
};

#[cfg(feature = "alignment")]
pub use runner::{
    AlignmentSet, AlignmentSetBuilder,
    Aligner, AlignerKey, AlignmentFallback,
    TextNormalizer, NormalizedText, NormalizationError,
    AlignmentResult,
};
```

### 3.4 Layering rule (backend invariant)

The runner depends on the core; the core does not name anything in the runner. Enforced at module level: `core/` modules `use` only `crate::types`, `crate::time`, and standard alloc/core types. `runner/` modules may freely call into `core/`.

**Backend invariant.** The core's `AsrParams`, `AsrResult`, and `AsrTokenHint` types contain only universal ASR knobs (language hint, beam size, suppress tokens, temperature schedule, no_speech_threshold). They must not name `whisper-rs` types directly, must not include whisper.cpp-specific config fields, and must not require the runner to extend them with whisper-only options. Any whisper-rs-specific tuning lives in the runner (`WhisperPoolConfig`) and is consumed by the runner's worker thread, not shipped through the state machine. This invariant is what makes a future swap to candle-whisper or a CTranslate2 binding a runner-only change.

---

## 4. Public types

### 4.1 Time

All emitted time ranges use a 16 kHz timebase, so PTS values are sample indices.

```rust
pub const SAMPLE_RATE_HZ: u32 = 16_000;

const SAMPLE_RATE_NZ: NonZeroU32 = match NonZeroU32::new(SAMPLE_RATE_HZ) {
    Some(n) => n,
    None => panic!("SAMPLE_RATE_HZ is nonzero"),
};

/// 1 / 16_000 — PTS values are sample counts.
pub const TIMEBASE: mediatime::Timebase = mediatime::Timebase::new(1, SAMPLE_RATE_NZ);
```

The `match` form for `NonZeroU32::new` keeps the constant evaluable on stable Rust without depending on `Option::unwrap`'s const stabilisation timeline. `mediatime::Timebase::new` is `const fn`; if a future mediatime release relaxes that, this becomes a `pub fn timebase() -> Timebase` accessor.

**Round-trip constraint with downstream consumers.** textclap operates on 48 kHz audio and slices using the same time anchors. 48 000 / 16 000 = 3 exactly, so any `TimeRange` whispery emits at the 1/16000 timebase round-trips losslessly to a 1/48000 timebase via `mediatime::TimeRange::rescale_to`. Introducing a non-integer-ratio intermediate sample rate downstream would break this; documented here so future changes don't silently lose sample alignment.

Downstream consumers call `.rescale_to(other_tb)` to express the same instants in their preferred timebase (ms, NTSC frame rate, 48 kHz, etc.).

### 4.2 Transcript

The per-chunk emission unit. One merged chunk produces exactly one `Transcript`. Fields are private; access is via getters per the findit-studio convention (silero, soundevents).

```rust
pub struct Transcript {
    range: mediatime::TimeRange,
    language: Lang,
    text: smol_str::SmolStr,
    words: Vec<Word>,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
    vad_segments: Vec<mediatime::TimeRange>,
    chunk_id: ChunkId,
}

impl Transcript {
    /// Bounds of the merged chunk in source-audio sample space (1/16000).
    pub fn range(&self) -> mediatime::TimeRange;

    /// Detected (after AutoLockAfter) or hint-supplied language for this chunk.
    pub fn language(&self) -> &Lang;

    /// Verbatim Whisper output for this chunk: includes punctuation, casing,
    /// and any model-emitted special characters. This is the canonical text
    /// surface; downstream BM25/FTS indexes this directly. Word-by-word
    /// surface forms (typically lowercased and punctuation-stripped) live in
    /// `words[].text()` and are derived from this via the alignment
    /// normalisation pipeline (§6.3).
    pub fn text(&self) -> &str;

    /// Word-level alignment results, in time order. Empty when:
    ///   - the `alignment` feature is disabled, or
    ///   - the runner was built without `with_alignment(...)`, or
    ///   - the chunk's language has no aligner registered and the
    ///     fallback is `SkipChunk`, or
    ///   - alignment failed for this chunk and the failure was tolerated.
    /// On other alignment failures, the chunk is emitted as `Event::Error`
    /// instead of `Event::Transcript` and no `Transcript` is produced.
    pub fn words(&self) -> &[Word];

    /// Whisper's mean log-probability over emitted tokens for this chunk.
    pub fn avg_logprob(&self) -> f32;

    /// Whisper's no-speech probability for this chunk. Useful for the
    /// indexer to filter borderline-silent chunks.
    pub fn no_speech_prob(&self) -> f32;

    /// Final decoding temperature after fallback retries. Equal to the
    /// first temperature in the schedule when no retry was needed.
    pub fn temperature(&self) -> f32;

    /// Sub-VAD-segments that composed this merged chunk, in source-audio
    /// sample space. The union of these is the speech-only subset of
    /// `range`; the silence between them is preserved in the audio fed
    /// to whisper but is not part of `vad_segments`. Used as the
    /// canonical anchor for speaker-overlap joins (§1.6).
    pub fn vad_segments(&self) -> &[mediatime::TimeRange];

    /// Monotonic chunk identity within a single Transcriber lifetime.
    /// Increases by 1 per emitted chunk (including chunks that produce
    /// `Event::Error`). Suitable as a lancedb primary key.
    pub fn chunk_id(&self) -> ChunkId;
}
```

`Transcript` is not `Copy`. It does not derive `Clone` by default — callers move it through the indexer. With the `serde` feature it derives `Serialize`/`Deserialize`. Construction is internal (the dispatch state machine builds it; there is no public builder); tests use a `pub(crate) fn for_test(...)` helper.

### 4.3 Word

```rust
pub struct Word {
    text: smol_str::SmolStr,
    range: mediatime::TimeRange,
    score: f32,
}

impl Word {
    /// Surface form of the word as it appears in the alignment input
    /// (typically lowercased and punctuation-stripped). The full
    /// punctuated/cased text for the chunk lives on
    /// `Transcript::text()`.
    pub fn text(&self) -> &str;

    /// Sample-accurate range of the word in source-audio sample space
    /// (1/16000 timebase). Half-open.
    pub fn range(&self) -> mediatime::TimeRange;

    /// Alignment confidence in [0, 1], NaN-free. Defined as
    /// `exp(mean(log_p_t))` where `log_p_t` is the per-frame
    /// log-probability of the chosen vocab item along the Viterbi
    /// path for the frames spanning this word. Equivalent to the
    /// geometric mean of the per-frame probabilities along the
    /// alignment path.
    pub fn score(&self) -> f32;
}
```

### 4.4 Language

A newtype around `SmolStr`. Whisper.cpp returns ISO 639-1 strings; we accept and emit the same. There is **no sentinel value**; `Lang` only ever holds a real ISO code. The "match any registered language" concept lives in the type system as `AlignerKey::Any` (§6.3), not as a magic `Lang` value.

```rust
pub struct Lang(smol_str::SmolStr);

impl Lang {
    pub const EN: Self = Self(SmolStr::new_inline("en"));
    pub const ZH: Self = Self(SmolStr::new_inline("zh"));
    pub const JA: Self = Self(SmolStr::new_inline("ja"));
    // …a curated list of constants for common languages

    pub fn as_str(&self) -> &str;
    pub fn from_iso639_1(s: &str) -> Result<Self, InvalidLang>;
}

impl Display for Lang { /* writes the ISO code */ }
```

The `from_iso639_1` constructor validates against Whisper.cpp's supported language set (lookup table); unknown codes return `InvalidLang`. Callers that want to accept arbitrary tags use a private constructor.

### 4.5 Errors

```rust
pub enum TranscriberError {
    /// PTS regression: caller pushed samples or a VAD segment with a
    /// timestamp earlier than the current high-water mark. Forward
    /// gaps are tolerated up to `gap_tolerance_samples` (§5.4).
    PtsRegression { kind: PushKind, advance: i64 },

    /// Forward gap exceeds the configured tolerance. Caller likely
    /// has a stream restart or a packet drop larger than expected.
    /// State machine refuses to silently zero-fill an arbitrarily
    /// large hole.
    GapExceedsTolerance { gap_samples: u64, tolerance_samples: u64 },

    /// Sample buffer would exceed its configured cap. The runner has
    /// not drained completed chunks fast enough; the caller should
    /// pause and call `poll_event` / `poll_transcript` until the
    /// buffer trims.
    Backpressure { buffered: usize, cap: usize },

    /// Caller `inject_*`ed a chunk_id that does not match an
    /// in-flight chunk.
    UnknownChunk(ChunkId),

    /// Caller called `signal_eof` and then pushed more samples or
    /// VAD segments.
    AfterEof,
}

pub enum WorkFailure {
    AsrFailed { kind: AsrFailureKind, message: String },
    AlignmentFailed { kind: AlignmentFailureKind, message: String, language: Lang },
    LanguageUnsupportedForAlignment { language: Lang },
    WorkerHangTimeout { kind: WorkerKind, elapsed: Duration },
}

pub enum AsrFailureKind {
    /// Whisper produced no tokens (silent or unintelligible chunk).
    EmptyOutput,
    /// All temperatures in the fallback schedule were tried and
    /// every result violated `compression_ratio_threshold` or
    /// `log_prob_threshold`.
    AllTemperaturesFailed,
    /// Auto-detected language is not in Whisper's supported set.
    /// (Should not happen in practice; defensive variant.)
    UnsupportedLanguage,
    /// Backend (whisper-rs) returned an error during inference.
    BackendError,
}

pub enum AlignmentFailureKind {
    /// Wav2vec2 ONNX inference failed.
    ModelInferenceFailed,
    /// Tokenization of the normalised text against the wav2vec2
    /// vocab failed (unknown characters, malformed input).
    TokenizationFailed,
    /// Text normalisation step failed (e.g., language-specific
    /// rules unavailable).
    NormalizationFailed,
    /// CTC Viterbi found no valid alignment path within the chunk.
    /// Symptomatic of severe Whisper/audio mismatch.
    NoAlignmentPath,
    /// Whisper text was empty after normalisation.
    EmptyText,
}

pub enum PushKind { Samples, VadSegment }
pub enum WorkerKind { Asr, Alignment }

#[cfg(feature = "runner")]
pub enum RunnerError {
    WhisperContextLoad { source: /* whisper-rs error */ },
    WhisperPoolShutdown,
    #[cfg(feature = "alignment")]
    AlignerLoad { language: Lang, source: /* ort error */ },
    #[cfg(feature = "alignment")]
    TokenizerLoad { language: Lang, source: /* tokenizers error */ },
    Io(std::io::Error),
    /// Wraps `TranscriberError` so the runner's API is single-error.
    Transcriber(TranscriberError),
}
```

All public error types `impl Error + Display + Debug` via `thiserror`.

---

## 5. Sans-I/O core

The core is a single struct, `Transcriber`, wrapping the cut state machine, the dispatch state machine, and a sample buffer. Its public API has six push/inject methods and two poll methods.

### 5.1 Transcriber surface

```rust
pub struct Transcriber {
    config: TranscriberConfig,
    buffer: SampleBuffer,
    cut: Cut,
    dispatch: Dispatch,
    next_chunk_id: u64,
    eof_signaled: bool,
}

impl Transcriber {
    pub fn new(config: TranscriberConfig) -> Self;

    // ── Push side ───────────────────────────────────────────────
    pub fn push_samples(
        &mut self,
        starts_at: mediatime::Timestamp,
        samples: &[f32],
    ) -> Result<(), TranscriberError>;

    pub fn push_vad_segment(
        &mut self,
        seg: VadSegment,
    ) -> Result<(), TranscriberError>;

    pub fn signal_eof(&mut self) -> Result<(), TranscriberError>;

    // ── Inject side ─────────────────────────────────────────────
    pub fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        out: AsrResult,
    ) -> Result<(), TranscriberError>;

    pub fn inject_alignment_result(
        &mut self,
        chunk_id: ChunkId,
        out: AlignmentResult,
    ) -> Result<(), TranscriberError>;

    pub fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError>;

    // ── Poll side ───────────────────────────────────────────────
    pub fn poll_command(&mut self) -> Option<Command>;
    pub fn poll_event(&mut self) -> Option<Event>;

    pub fn is_idle(&self) -> bool;       // no pending work, no buffered samples
    pub fn buffered_samples(&self) -> usize;
}
```

`VadSegment` is whispery's own type, structurally identical to silero's `SpeechSegment`. We accept it as input but do not depend on the silero crate; the runner's `examples/managed_runner.rs` shows how to convert.

```rust
pub struct VadSegment {
    pub range: mediatime::TimeRange,
}
```

### 5.2 TranscriberConfig

```rust
pub struct TranscriberConfig {
    /// Maximum duration of a merged chunk. Default 30 s.
    /// Also serves as the hard-split threshold for individual VAD
    /// segments longer than this (§5.3).
    pub chunk_size: Duration,

    /// Max samples kept in the internal buffer before the buffer
    /// returns Backpressure. Default 60 s × 16 kHz = 960_000 samples.
    pub buffer_cap_samples: usize,

    /// Maximum forward-gap (silence hole) between consecutive
    /// `push_samples` calls that the buffer will silently zero-fill.
    /// Real ffmpeg streams have small PTS gaps from container
    /// offsets, packet drops, and resample boundaries; rejecting
    /// any forward gap traps callers who behave correctly. Larger
    /// gaps are likely stream restarts and return
    /// `GapExceedsTolerance`. Default 200 ms × 16 kHz = 3 200 samples.
    pub gap_tolerance_samples: u64,

    /// Whether to emit Command::RunAlignment after each ASR
    /// completion. The runner's AlignmentSet is configured
    /// separately; this flag is set by the runner builder when
    /// `with_alignment(...)` was called.
    pub word_alignment: bool,

    /// Maximum chunks in flight (extracted samples shipped to
    /// the runner via Command::RunAsr but not yet event-emitted).
    /// Together with `buffer_cap_samples`, the dual ceiling on
    /// memory: `buffer_cap_samples` bounds buffered raw audio,
    /// `max_in_flight` bounds extracted-and-ref-counted audio
    /// owned by inference workers.
    pub max_in_flight: usize,

    /// Language detection / locking strategy. See `LanguagePolicy`.
    /// Default `AutoLockAfter(1)`.
    pub language_policy: LanguagePolicy,
}

pub enum LanguagePolicy {
    /// Each chunk independently auto-detects language. Cheapest at
    /// init but susceptible to drift on non-trivial first chunks.
    Auto,
    /// Caller supplies the language; whisper is given a hard
    /// language hint and never auto-detects. Use when the audio
    /// source is known to be a single language.
    Lock { hint: Lang },
    /// Auto-detect on the first `n` chunks that emit non-empty
    /// text, then lock the most-frequent detected language for
    /// the remainder of the session. WhisperX-equivalent default.
    AutoLockAfter(usize),
}

impl Default for TranscriberConfig { /* sensible defaults */ }
```

Locking rationale: Whisper's language detection on a single < 30 s window is unreliable when the chunk starts with non-speech, numbers, exclamations, or laughter. Auto-locking after the first non-trivial chunk keeps the rest of the session consistent and is what WhisperX does in practice. `Auto` is exposed for callers who deliberately want per-chunk re-detection (mixed-language sources where the language genuinely changes mid-stream); they accept the drift risk.

### 5.3 Cut state machine (`core/cut.rs`)

A direct port of WhisperX's `merge_chunks`, restated as an incremental state machine, with one extension: hard-splitting any single VAD segment longer than `chunk_size`.

State:

```rust
struct Cut {
    chunk_size: Duration,                // mediatime::Duration

    // The chunk currently accumulating (None when between chunks).
    current_start: Option<mediatime::Timestamp>,
    current_end: mediatime::Timestamp,
    current_subs: Vec<mediatime::TimeRange>,
}

/// Output of the cut state machine. Crate-private; the public
/// surface only sees the resulting Transcript.
pub(crate) struct MergedChunk {
    pub start: mediatime::Timestamp,
    pub end: mediatime::Timestamp,
    pub subs: Vec<mediatime::TimeRange>,
}
```

Transitions:

- `push_segment(seg)`:
  1. **Pre-split overlong segments.** If `seg.range.duration() > chunk_size`, split it into `⌈duration / chunk_size⌉` contiguous sub-ranges of length ≤ `chunk_size` and feed them through the normal path one at a time. The sub-ranges are recorded in `current_subs` exactly like any other segment, so provenance is preserved and downstream consumers see the original-segment boundaries via the join of consecutive sub-ranges.
  2. If `current_start` is None: set it to `seg.range.start()`.
  3. If `(seg.range.end() - current_start) > chunk_size` *and* `(current_end - current_start) > 0`:
     - Emit a `MergedChunk { start: current_start, end: current_end, subs: take(current_subs) }`.
     - Reset: `current_start = Some(seg.range.start())`, `current_subs.clear()`.
  4. Update `current_end = seg.range.end()`, `current_subs.push(seg.range)`.
- `flush()` (called on EOF):
  - If `current_start` is Some: emit the trailing chunk, reset.

The state machine guarantees:

1. **Monotonicity.** Output `MergedChunk`s are non-overlapping and strictly ordered by start time.
2. **Strict bound.** No emitted chunk spans more than `chunk_size`. The previous draft's bound (`chunk_size + max(seg_duration)`) was wrong: a single VAD segment longer than `chunk_size` would have been absorbed whole, because the `(current_end - current_start) > 0` guard fails on first entry to a new chunk. With the pre-split rule above, any segment whose duration exceeds `chunk_size` is broken into ≤ `chunk_size` pieces *before* the merge logic sees it, so the bound becomes a true `chunk_size`. This matters because Whisper's encoder is hard-capped at 30 s; a chunk over 30 s is silently truncated by whisper.cpp and alignment then runs against audio Whisper never transcribed.
3. **Provenance preserved.** `subs` lists every VAD segment (or hard-split sub-segment) whose union forms the chunk; downstream consumers know the precise speech-only intervals.

This logic is purely arithmetic on integer PTS; allocations are bounded by `current_subs`.

### 5.4 Sample buffer (`core/buffer.rs`)

```rust
pub(crate) struct SampleBuffer {
    base_pts: i64,                       // PTS of samples[0] (1/16000 timebase)
    next_pts: i64,                       // PTS of the next sample to be appended
    samples: Vec<f32>,
    cap: usize,
    gap_tolerance: u64,
}
```

Operations:

- `append(starts_at: Timestamp, packet: &[f32]) -> Result<(), TranscriberError>`:
  - On first call, set `base_pts = starts_at.pts()` and `next_pts = base_pts`.
  - Compute `delta = starts_at.pts() - next_pts`:
    - `delta < 0`: PTS regression. Return `PtsRegression`.
    - `delta == 0`: contiguous. Append `packet` directly.
    - `0 < delta <= gap_tolerance`: forward gap inside tolerance. Zero-fill `delta` samples, then append `packet`. Increment `next_pts` by `delta + packet.len()`.
    - `delta > gap_tolerance`: return `GapExceedsTolerance`. Caller is expected to handle a stream restart explicitly (e.g., flush, reset, re-init).
  - After append, if `samples.len() > cap`, return `Backpressure { buffered, cap }`.
- `extract(range: TimeRange) -> Arc<[f32]>`:
  - Slice `samples[(range.start_pts() - base_pts)..(range.end_pts() - base_pts)]`, copy into a fresh `Arc<[f32]>`.
  - Returns the Arc; the original buffer is not mutated. Trim happens separately.
- `trim_to(low_water: i64)`:
  - Drop `samples[0..(low_water - base_pts)]`, advance `base_pts`. Used by the dispatch state machine after a chunk is fully emitted; the new low-water is the lowest in-flight chunk's start (or the lowest pending-cut chunk's start, whichever is smaller).

Forward-gap tolerance addresses real-world ffmpeg behaviour: container PTS offsets, cross-file stitching, occasional packet drops, resample boundaries. The default `gap_tolerance_samples = 3 200` (200 ms) silently zero-fills typical micro-gaps; anything larger is surfaced for explicit caller handling. Zero-fill is correct because the VAD stream is independent of the audio stream — silence-filled samples will not produce VAD speech segments and will not be cut into a Whisper chunk.

The buffer is a flat `Vec<f32>` with periodic `drain(0..n)` on trim. For our packet rates (≪ 1 GB/s), the memmove cost is dominated by whisper inference time. A circular ring buffer is a future optimisation.

### 5.5 Dispatch state machine (`core/dispatch.rs`)

Tracks per-chunk lifecycle and enforces in-order event emission.

```rust
enum ChunkPhase {
    AwaitingAsr,               // RunAsr command issued
    AwaitingAlignment,         // ASR done; RunAlignment issued (if alignment enabled)
    Ready { transcript: Transcript },   // result built, awaiting in-order emission
    FailedReady { failure: WorkFailure }, // failure recorded, awaiting in-order emission
}

struct Dispatch {
    /// Lightweight chunk descriptors awaiting an in_flight slot.
    /// Holds (chunk_id, MergedChunk) tuples — NOT extracted samples.
    /// Samples remain in SampleBuffer until promotion; the buffer-cap
    /// mechanism is the single backpressure path.
    cut_pending: VecDeque<(ChunkId, MergedChunk)>,

    /// In-flight chunks ordered by chunk_id for in-order emission and
    /// low-water trim computation.
    in_flight: BTreeMap<ChunkId, ChunkRecord>,

    /// The next chunk_id whose Event has not yet been drained to
    /// `pending_events`. Events are emitted strictly in chunk_id
    /// order — chunk N+1's event waits in `in_flight[N+1].phase
    /// = Ready` until chunk N has emitted, even if N+1 finished
    /// inference first.
    next_emit_chunk_id: ChunkId,

    pending_commands: VecDeque<Command>,
    pending_events: VecDeque<Event>,
    word_alignment: bool,
    max_in_flight: usize,
}

struct ChunkRecord {
    chunk_id: ChunkId,
    range: TimeRange,
    samples: Arc<[f32]>,
    sub_segments: Vec<TimeRange>,
    phase: ChunkPhase,
    asr_result: Option<AsrResult>,
}
```

Transitions:

- **On `Cut::emit(merged_chunk)`** — called whenever the cut state machine produces a chunk descriptor:
  - Allocate a `chunk_id`.
  - If `in_flight.len() >= max_in_flight`: push `(chunk_id, merged_chunk)` to `cut_pending`. **Do not extract samples yet.** Samples remain in `SampleBuffer`; if upstream keeps pushing, `buffer_cap_samples` is the single back-pressure choke point and `push_samples` will return `Backpressure`. This bounds memory: pending chunks cost only the descriptor (a `TimeRange` + a small `Vec<TimeRange>` for sub_segments), not 30 s × 16 kHz × 4 bytes of audio per pending chunk.
  - Else: extract `samples` from `SampleBuffer`, build a `ChunkRecord` (phase `AwaitingAsr`), insert into `in_flight`, enqueue `Command::RunAsr`.
- **On `inject_asr_result(chunk_id, result)`**:
  - Look up the record (else `UnknownChunk`).
  - Save `record.asr_result = Some(result)`.
  - If `word_alignment` AND `result.text` is non-empty: enqueue `Command::RunAlignment` and set `phase = AwaitingAlignment`.
  - Else: build the `Transcript` (with empty `words`), set `phase = Ready { transcript }`. Then call `flush_in_order_events()` and `trim()`.
- **On `inject_alignment_result(chunk_id, result)`**:
  - Build the `Transcript` from `record.asr_result + result.words`, set `phase = Ready { transcript }`. Call `flush_in_order_events()` and `trim()`.
- **On `inject_failure(chunk_id, failure)`**:
  - Set `phase = FailedReady { failure }`. Call `flush_in_order_events()` and `trim()`.
- **`flush_in_order_events()`**:
  - While `in_flight.first()` exists with id `next_emit_chunk_id` and phase ∈ {Ready, FailedReady}:
    - Pop the entry; enqueue `Event::Transcript(transcript)` or `Event::Error { chunk_id, error: failure }`; advance `next_emit_chunk_id`.
  - This is the only place events are enqueued, and chunk_id strictly increases by 1 per emission. Out-of-order completion is therefore invisible to the caller.
- **`trim()`**:
  - Compute `low_water_buffer = min`(start_pts of every record still in `in_flight`, plus the start_pts of every chunk in `cut_pending`). If both are empty, `low_water_buffer` advances to `next_pts` (the buffer's high-water).
  - Call `SampleBuffer::trim_to(low_water_buffer)`.
  - If `in_flight.len() < max_in_flight` and `cut_pending` is non-empty: promote the front of `cut_pending` (extract samples from SampleBuffer, build ChunkRecord, enqueue `Command::RunAsr`).

Invariants:

1. **In-order emission.** `Event::Transcript` and `Event::Error` are produced in strict `chunk_id` order regardless of which inference worker finishes first. This is a contract; downstream BM25/FTS write order, future cross-file ranking, and any time-aligned join with diarization can rely on it.
2. **Bounded memory under back-pressure.** `cut_pending` entries hold only descriptors; they cost O(1) audio. The single audio back-pressure path is `buffer_cap_samples`, which trips `push_samples` and lets the caller pause ingest.
3. **No deadlock.** As long as workers are alive and inference completes, `flush_in_order_events()` always advances; promotions from `cut_pending` happen as soon as a slot frees.

### 5.6 Command and Event

The core's command and result types deliberately use ASR-prefixed names rather than Whisper-prefixed ones. This is the load-bearing piece of the §3.4 backend invariant: a future swap from whisper-rs to candle-whisper or a CTranslate2 binding only changes the runner's interpretation of these types, never the types themselves or the state machine that produces and consumes them.

```rust
pub enum Command {
    RunAsr {
        chunk_id: ChunkId,
        samples: Arc<[f32]>,
        sample_rate: u32,                // always SAMPLE_RATE_HZ in v1
        params: AsrParams,
    },
    #[cfg(feature = "alignment")]
    RunAlignment {
        chunk_id: ChunkId,
        samples: Arc<[f32]>,
        sub_segments: Vec<TimeRange>,    // for silence-aware alignment, §6.3
        text: smol_str::SmolStr,
        language: Lang,
        token_hints: Vec<AsrTokenHint>,  // optional CTC seed
    },
}

pub enum Event {
    Transcript(Transcript),
    Error { chunk_id: ChunkId, error: WorkFailure },
}

/// Universal ASR knobs. Backend-agnostic: contains no whisper-rs
/// types and no whisper.cpp-specific fields. Whisper-only tuning
/// lives in the runner's `WhisperPoolConfig`.
pub struct AsrParams {
    pub language_hint: Option<Lang>,
    pub beam_size: usize,
    pub temperature_schedule: SmallVec<[f32; 6]>,
    pub no_speech_threshold: f32,
    pub log_prob_threshold: f32,
    pub compression_ratio_threshold: f32,
    pub suppress_tokens: SmallVec<[i32; 16]>,
    pub condition_on_previous_text: bool,
    pub initial_prompt: Option<smol_str::SmolStr>,
}

/// Result of one chunk's ASR inference.
pub struct AsrResult {
    pub text: smol_str::SmolStr,
    pub language: Lang,           // the detected (or hint-confirmed) language
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,         // final temperature used after fallback retries
    pub tokens: Vec<AsrTokenHint>,
}

/// Optional per-token hint passed downstream to the aligner.
/// Carries the timestamp Whisper believes the token spans (in
/// 1/16000 timebase) plus the token's text. Used by the aligner
/// to seed CTC search; if empty, the aligner falls back to a
/// uniform prior.
pub struct AsrTokenHint {
    pub text: smol_str::SmolStr,
    pub range: TimeRange,
}

#[cfg(feature = "alignment")]
pub struct AlignmentResult {
    pub words: Vec<Word>,
}
```

The runner's job is to translate `AsrParams` into a `whisper_rs::FullParams` (or its candle/CT2 equivalent) and translate the backend's output back into `AsrResult`. This translation lives entirely in `runner/whisper_pool.rs`; the core never names whisper-rs.

`AsrParams` defaults are set by the runner's builder, not by the core. The core just ships whatever the runner constructs through to the worker via `Command::RunAsr`.

Per-chunk override of `AsrParams` is supported: `ManagedTranscriber::process_packet` accepts an optional `AsrParamsOverride` (a sparse struct of `Option<T>` fields layered onto the runner's defaults). This is how callers supply per-call language hints without re-building the runner.

---

## 6. Runner (`runner/`)

Default-on `runner` feature. Wires the core to whisper-rs and (with the `alignment` feature) to ort-based wav2vec2 forced alignment.

### 6.1 ManagedTranscriber

```rust
pub struct ManagedTranscriber {
    core: core::Transcriber,
    whisper_pool: WhisperPool,
    #[cfg(feature = "alignment")]
    alignment_pool: Option<AlignmentPool>,
    emit_rx: crossbeam_channel::Receiver<Event>,
    asr_params_default: AsrParams,
    drain_timeout: Duration,
}

impl ManagedTranscriber {
    pub fn builder(whisper_ctx: WhisperContext) -> ManagedTranscriberBuilder;

    /// Push one packet of audio + the VAD segments newly closed within
    /// or before that packet's range. Optionally override ASR params
    /// for any chunk produced from this packet — useful for per-call
    /// language hints when the caller has prior knowledge.
    pub fn process_packet(
        &mut self,
        starts_at: Timestamp,
        samples: &[f32],
        vad_segments: &[VadSegment],
        params_override: Option<AsrParamsOverride>,
    ) -> Result<(), RunnerError>;

    pub fn signal_eof(&mut self) -> Result<(), RunnerError>;

    pub fn poll_transcript(&mut self) -> Option<Transcript>;
    pub fn poll_error(&mut self) -> Option<(ChunkId, WorkFailure)>;

    /// Block until all in-flight work drains, bounded by the
    /// configured `drain_timeout`. Returns once `core.is_idle()` or
    /// `WorkerHangTimeout` if a worker exceeds its own per-job
    /// timeout. The default drain_timeout is 10× the longest expected
    /// per-chunk inference (set per builder).
    pub fn drain(&mut self) -> Result<(), RunnerError>;
}

pub struct ManagedTranscriberBuilder { /* core config, whisper pool config, alignment set */ }

impl ManagedTranscriberBuilder {
    pub fn chunk_size(self, d: Duration) -> Self;
    pub fn buffer_cap_samples(self, n: usize) -> Self;
    pub fn gap_tolerance_samples(self, n: u64) -> Self;
    pub fn language_policy(self, p: LanguagePolicy) -> Self;
    pub fn whisper_pool(self, cfg: WhisperPoolConfig) -> Self;
    pub fn asr_params(self, p: AsrParams) -> Self;
    /// Per-job worker timeout. Workers that exceed this on a single
    /// inference are interrupted and emit `WorkerHangTimeout`.
    /// Default 60 s for ASR, 30 s for alignment.
    pub fn worker_timeouts(self, asr: Duration, align: Duration) -> Self;
    /// Cap on `drain()`. Default 10× the longest worker timeout.
    pub fn drain_timeout(self, t: Duration) -> Self;

    /// Enables word-level forced alignment using the supplied registry.
    /// If never called, `Transcript.words` is always empty (alignment off).
    #[cfg(feature = "alignment")]
    pub fn with_alignment(self, set: AlignmentSet) -> Self;

    pub fn build(self) -> Result<ManagedTranscriber, RunnerError>;
}

/// Sparse override of AsrParams for per-packet customisation.
/// Each `Some(_)` field replaces the corresponding default for any
/// chunk produced from this packet.
pub struct AsrParamsOverride {
    pub language_hint: Option<Option<Lang>>,
    pub beam_size: Option<usize>,
    pub temperature_schedule: Option<SmallVec<[f32; 6]>>,
    pub initial_prompt: Option<Option<smol_str::SmolStr>>,
}
```

The builder's `build()` returns a `ManagedTranscriber` with worker threads spawned and channels wired. Internally, `with_alignment` flips the core's `word_alignment` flag and stashes the `AlignmentSet` for the alignment worker.

### 6.2 WhisperPool

```rust
pub struct WhisperPoolConfig {
    pub worker_count: usize,
    pub model_path: PathBuf,
    pub device: Device,
    pub max_queued_chunks: usize,    // queue cap before process_packet blocks
}

pub enum Device {
    Cpu,
    Cuda { device_id: i32 },
    Metal { device_id: Option<i32> },
    Vulkan { device_id: i32 },
}

struct WhisperPool {
    ctx: Arc<WhisperContext>,        // assumed shared if Send + Sync; see Open Risk §13.1
    workers: Vec<JoinHandle<()>>,
    work_tx: crossbeam_channel::Sender<AsrWorkItem>,
    result_tx: crossbeam_channel::Sender<(ChunkId, Result<AsrResult, WorkFailure>)>,
}

struct AsrWorkItem {
    chunk_id: ChunkId,
    samples: Arc<[f32]>,
    params: AsrParams,
}
```

**Worker count default.** `max(1, num_cpus::get_physical() / 2)` — leaves room for ffmpeg, silero, soundevents, lancedb, and the alignment worker.

**Worker structure (proposed; subject to the §13.1 spike).** Each worker owns a `WhisperState` borrowed from a shared `Arc<WhisperContext>`. Workers run a loop: `recv work` → `state.full(samples, &asr_to_full_params(params))` → `send result`. The translation `asr_to_full_params` lives here in `runner/whisper_pool.rs`; this is the only place in the crate that names `whisper_rs::FullParams`.

**Memory implication if shared-context turns out to be unsafe.** If `WhisperContext` is found to be `!Send + !Sync` for the build features we need (or if `WhisperState<'a>` cannot be moved into worker threads even with self-referential helpers), the fallback is per-worker contexts. Memory then scales as `worker_count × model_size` — for the tiny model that's 4 × 75 = 300 MiB, manageable; for large that's 4 × 3 GiB = 12 GiB, which forces `worker_count = 1` on machines without enough RAM. This is not a correctness risk but it changes the deployment story; the §13.1 spike resolves it before we commit code.

**Dispatch loop.** The `ManagedTranscriber` runs a small dispatch loop inline on the caller's thread inside `process_packet` and `poll_transcript`:

1. Drain `Command`s out of `core`.
2. For `RunAsr` commands, send to `whisper_pool.work_tx`.
3. For `RunAlignment` commands, send to the alignment pool (if enabled).
4. Drain `result_rx` (asr) and `align_rx` (alignment), call `core.inject_*_result` / `core.inject_failure`.
5. Drain `Event`s, push them to `emit_tx`.

The dispatch loop is single-threaded and inline; only the inference workers run in parallel. If `process_packet` discovers `whisper_pool.work_tx.is_full()` (i.e., `max_queued_chunks` reached), it blocks until a worker drains.

### 6.3 Aligner and AlignmentSet

#[cfg(feature = "alignment")]

```rust
pub struct Aligner {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    language: Lang,
    normalizer: Box<dyn TextNormalizer>,
    sample_rate: u32,           // wav2vec2's expected rate, typically 16_000
    hop_samples: u32,           // model frame stride, typically 320 (= 20ms @ 16kHz)
    blank_token_id: u32,
}

impl Aligner {
    pub fn from_paths(
        language: Lang,
        model_path: &Path,
        tokenizer_path: &Path,
        normalizer: Box<dyn TextNormalizer>,
    ) -> Result<Self, RunnerError>;

    pub(crate) fn align(
        &mut self,
        samples: &[f32],
        sub_segments: &[TimeRange],     // §6.3.2
        text: &str,
    ) -> Result<AlignmentResult, WorkFailure>;
}

/// Identifies an aligner in the registry. The `Any` variant is the
/// "match-anything-not-explicitly-registered" fallback aligner
/// (typically a multilingual XLSR / MMS model). Lifting the
/// fallback into the type system avoids a sentinel string in
/// `Lang` and prevents `Lang::ANY` from accidentally being passed
/// to whisper.cpp as a literal "*" language hint.
pub enum AlignerKey {
    Lang(Lang),
    Any,
}

pub struct AlignmentSet {
    aligners: HashMap<AlignerKey, Mutex<Aligner>>,
    fallback: AlignmentFallback,
}

pub enum AlignmentFallback {
    /// Unknown language: emit the chunk's Transcript with empty `words`.
    /// Default. Indexing pipeline never blocks on alignment unavailability.
    SkipChunk,
    /// Unknown language: emit Event::Error with LanguageUnsupportedForAlignment.
    Error,
}

pub trait TextNormalizer: Send {
    /// Returns (normalised_text_for_alignment, alignment_to_original_word_map).
    /// The map's i-th entry gives the byte range in the original `text` that
    /// the i-th word in the normalised text corresponds to. Used by the aligner
    /// to look up the original surface form (with punctuation/casing) for each
    /// emitted Word.
    fn normalize<'a>(&self, text: &'a str) -> Result<NormalizedText<'a>, NormalizationError>;
}

pub struct NormalizedText<'a> {
    pub normalized: String,                    // alignment input
    pub original_words: Vec<&'a str>,          // surface forms, in order
}

pub struct AlignmentSetBuilder { /* … */ }
```

#### 6.3.1 Lookup order

For a chunk with detected language `L`, the alignment worker looks up:

1. `AlignerKey::Lang(L)` — explicit registered aligner for the language.
2. `AlignerKey::Any` — multilingual fallback aligner.
3. Apply `fallback`: `SkipChunk` (emit empty `words`) or `Error` (emit `LanguageUnsupportedForAlignment`).

#### 6.3.2 Alignment algorithm (silence-aware, normalisation-aware)

WhisperX's alignment quality story has three load-bearing pieces beyond the textbook CTC algorithm: text normalisation, surface-form recovery, and silence-handling. v1 implements all three.

For each chunk with non-empty text:

0. **Mask non-speech regions.** Build `samples_for_aligner` as a copy of `samples` with sample positions outside the union of `sub_segments` zeroed. wav2vec2 distributes near-all probability to the blank token in long silence regions; CTC Viterbi paths are robust under this only if silence is *uniformly* silent. Zero-masking ensures non-speech regions don't contribute spurious phoneme probabilities and don't smear word boundaries onto silence.
1. **Normalise text.** Run the language's `TextNormalizer` to produce `(normalized, original_words)`. Normalisation lowercases, strips punctuation, expands contractions per the language's rules, and produces a list of original-surface-form word slices in order.
2. **Tokenise.** Tokenise `normalized` against the wav2vec2 vocab to produce `Y = [t_0, t_1, ..., t_{n-1}]` (vocab indices). Track word-boundary positions in `Y` (where one normalised word ends and the next begins).
3. **Encode.** Run `session` over `samples_for_aligner` (reshaped to wav2vec2's expected input shape). Output is logits `(T, V)`.
4. **Log-softmax** along V to get log-probabilities.
5. **CTC lattice.** Build the standard CTC alignment lattice over `(T, 2|Y|+1)` (interspersed with blanks).
6. **Viterbi.** Run highest-probability monotonic alignment of `Y` to `T`. If no valid path exists, return `AlignmentFailureKind::NoAlignmentPath`.
7. **Per-word ranges.** Walk the path; for each word boundary in `Y`, extract the start and end frame indices. Map frame index → sample index via `frame * hop_samples`, then to a `TimeRange` at the 1/16000 timebase.
8. **Score.** For each word, compute `score = exp(mean(log_p_t))` over the frames spanning the word.
9. **Surface form recovery.** The i-th word's `text` is `original_words[i]` (the original surface form with punctuation and casing), not the normalised form. This way `Transcript.text()` and `joined(Transcript.words().map(|w| w.text()))` differ only in punctuation glue, not in the words themselves.

The `text` recovered for a `Word` therefore preserves the punctuated/cased original; the alignment input was the normalised form. This is the v1 invariant for `Transcript.text` vs `Word.text`.

#### 6.3.3 Concurrency: v1 is sequential; parallelism is conditional on backend

v1 ships **one alignment worker** in the `AlignmentPool`. Alignment is therefore sequential across chunks, regardless of language. With Whisper running on N workers, alignment will be the throughput bottleneck only when alignment-time-per-chunk × throughput exceeds whisper-time-per-chunk × throughput / N — which is unusual on indexing workloads but possible.

The `Mutex<Aligner>` in `AlignmentSet` is forward-looking, not v1-functional: it allows a future multi-worker pool to operate on different languages in parallel. **It does not by itself enable parallel alignment of the same language**, because `ort::Session::run` is not guaranteed thread-safe across all execution providers (CUDA EP in particular). Two paths exist for v2 if alignment becomes the bottleneck:

- **Cross-language parallel only.** N alignment workers each grab the relevant `Mutex<Aligner>` per chunk; same-language chunks serialise behind one mutex. Easy.
- **Within-language parallel.** Replace `Mutex<Aligner>` with `Vec<Aligner>` (one Session per worker per language). Multiplies model memory by parallelism factor.

Neither is implemented in v1. The §11 throughput math accounts for sequential alignment.

### 6.4 Concurrency model summary

```
  caller thread                ASR workers (N)          alignment worker (1)
  -------------                ---------------          --------------------

  process_packet
        |
        v
  push_samples / push_vad_segment
        |
        v
  [dispatch loop, inline]
        |
        +---- RunAsr -------------> work_tx --> WhisperState::full
        |                                            |
        |  <-- result_rx <----------------------- return
        |
        +---- inject_asr_result
        |
        +---- RunAlignment -------> align_tx -> aligner.align (silence-aware)
        |                                            |
        |  <-- align_rx <------------------------ return
        |
        +---- inject_alignment_result
        |
        +---- flush_in_order_events
        |          |
        |          v
        +-----> emit_tx --> poll_transcript / poll_error
```

The dispatch loop runs *inline* on the caller's thread inside `process_packet` and `poll_transcript`. There is no background dispatcher thread; the only threads in the runner are the N ASR workers and the 1 alignment worker. This keeps the runner deterministic from the caller's perspective.

The flip side: very long-running `process_packet` calls can stall if all workers are busy and `max_queued_chunks` is reached. By default, `process_packet` blocks; if `WhisperPoolConfig::block_on_full_queue = false`, it returns `RunnerError::Backpressure` instead so the caller can apply its own pacing.

Worker hang protection: each worker tracks its current job's start time and is interrupted (the job is recorded as `WorkerHangTimeout`) if it exceeds the configured per-job timeout. This bounds `drain()` and prevents indefinite stalls on a misbehaving model.

---

## 7. Data flow (end-to-end)

A worked example. Assume `chunk_size = 30 s`, `worker_count = 2`, alignment enabled, `LanguagePolicy::AutoLockAfter(1)`.

1. Caller's pipeline emits a 100 ms audio packet (1 600 samples) at PTS 0.
2. Caller runs silero on the packet, gets zero or more new `SpeechSegment`s.
3. Caller calls `mt.process_packet(Timestamp::new(0, TIMEBASE), &samples, &vad_segs, None)`.
4. `ManagedTranscriber::process_packet`:
   - Calls `core.push_samples(...)`. SampleBuffer extends; small forward gaps are zero-filled silently.
   - For each VAD segment: `core.push_vad_segment(seg)`. Cut state machine accumulates; if any single segment exceeds 30 s it is hard-split first; possibly emits a `MergedChunk` if accumulated speech ≥ 30 s.
   - Drains commands from `core.poll_command()`. If a `RunAsr` is emitted, ships it to `whisper_pool`.
5. Caller continues for some seconds, accumulating ~3–10 merged chunks across ASR workers.
6. ASR worker A finishes chunk 0, sends `Ok(AsrResult)` to `result_rx`. Detected language is recorded by the dispatcher; with `AutoLockAfter(1)`, all subsequent chunks now use this as a hard hint.
7. Caller's next `process_packet` (or `poll_transcript`) drains `result_rx`, calls `core.inject_asr_result(0, result)`. Core enqueues `Command::RunAlignment` for chunk 0 (carrying both `samples` and `sub_segments` for silence-aware alignment).
8. Dispatch loop ships the alignment command to the alignment worker.
9. Alignment worker zero-masks non-VAD regions, normalises text, runs wav2vec2 + CTC, recovers per-word ranges and surface forms; sends `Ok(AlignmentResult)` to `align_rx`.
10. Next drain calls `core.inject_alignment_result(0, result)`. Core builds `Transcript` and stages it in `phase = Ready`. `flush_in_order_events()` runs: chunk 0 has `next_emit_chunk_id = 0`, so the Transcript is enqueued. Dispatch loop drains it to `emit_tx`. `next_emit_chunk_id` advances to 1.
11. **Out-of-order completion handled.** ASR worker B may now finish chunk 2 before chunk 1; alignment may complete chunk 2 before chunk 1. Chunk 2's `Transcript` sits in `phase = Ready` until chunk 1 emits, at which point `flush_in_order_events` cascades and emits both. Caller-visible order is strictly chunk-id order.
12. Caller calls `poll_transcript()` and gets the `Transcript` for chunk 0. Indexer writes it to lancedb. Repeats for chunks 1, 2, …
13. `core` periodically trims its `SampleBuffer` to `min(in_flight, cut_pending)` start_pts, freeing memory.
14. After all packets are pushed, caller calls `signal_eof()`, then `drain()` to flush remaining chunks (bounded by `drain_timeout`).

The net effect: transcripts arrive a few seconds after their audio's wall-clock arrival (ASR latency + alignment latency), **in strict chunk-id order regardless of worker completion order**. The pipeline never holds more than the configured ceilings of audio in memory.

---

## 8. Configuration and tunables

Defaults and rationale:

| Param                                         | Default                              | Notes |
|-----------------------------------------------|--------------------------------------|-------|
| `chunk_size`                                  | 30 s                                 | Whisper's encoder window; also the hard-split threshold for individual VAD segments. |
| `buffer_cap_samples`                          | 60 s × 16 kHz = 960 000              | Twice `chunk_size`; bounds buffered raw audio under transient backpressure. Sole choke point against runaway memory. |
| `gap_tolerance_samples`                       | 200 ms × 16 kHz = 3 200              | Forward gaps inside this are zero-filled silently; larger gaps are surfaced. Tuned for normal ffmpeg PTS jitter. |
| `max_in_flight`                               | `worker_count + 2`                   | Pipeline depth ceiling on extracted chunk audio (Arc-counted by workers). |
| `worker_count` (ASR)                          | `max(1, num_cpus::get_physical()/2)` | Half of physical cores leaves room for ffmpeg, silero, soundevents, lancedb. |
| `alignment_workers`                           | 1                                    | Sequential in v1; multi-worker is v2 and depends on §6.3.3. |
| `language_policy`                             | `AutoLockAfter(1)`                   | Detect on the first non-trivial chunk, lock for the rest. Matches WhisperX. |
| `AsrParams.beam_size`                         | 5                                    | WhisperX default. |
| `AsrParams.temperature_schedule`              | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`     | Standard fallback. |
| `AsrParams.no_speech_threshold`               | 0.6                                  | WhisperX default. |
| `AsrParams.log_prob_threshold`                | -1.0                                 | Triggers temperature retry. WhisperX default. |
| `AsrParams.compression_ratio_threshold`       | 2.4                                  | Triggers temperature retry. WhisperX default. |
| `AsrParams.condition_on_previous_text`        | false                                | Each `WhisperState::full` call is independent (no cross-chunk state reuse) regardless of this setting. The flag only controls whether Whisper's *intra*-chunk decoder uses prior segment text as a prompt for the next ~30-token segment. WhisperX defaults to `false` because intra-chunk prompt continuation enables degenerate hallucination loops on misrecognised segments; the indexing use case prioritises avoiding these over modest punctuation/casing continuity gains. Callers can flip to `true` if they observe intra-chunk fragmentation. |
| `AlignmentFallback`                           | `SkipChunk`                          | Unknown languages still emit a `Transcript`, just with empty `words`. |
| `with_alignment(...)`                         | not called (off)                     | Caller opts in by passing an `AlignmentSet`; otherwise `Transcript.words` is empty. |
| `Device`                                      | `Device::Cpu`                        | GPU selection is opt-in; defaulting to CPU avoids surprising GPU memory usage on first run. |
| `worker_timeouts.asr`                         | 60 s                                 | Per-job; protects against model stalls. |
| `worker_timeouts.alignment`                   | 30 s                                 | Per-job. |
| `drain_timeout`                               | 10 × max(worker_timeouts)            | Cap on `drain()`. Prevents deadlock on a hung worker. |
| `WhisperPoolConfig.block_on_full_queue`       | true                                 | `process_packet` blocks when worker queue is full. Set false for non-blocking back-pressure. |

All exposed on the builder; nothing is hard-coded.

---

## 9. Error handling

### 9.1 Per-chunk failures

Whisper or alignment failures for a single chunk become `Event::Error { chunk_id, error: WorkFailure }`. The chunk's audio buffer is dropped, its slot in `in_flight` is freed, and the next pending chunk is admitted. The pipeline does not stop.

The indexer can decide what to do: log + continue, retry the chunk by re-running whisper out-of-band, drop the time range, or surface the gap to the user. whispery does not retry internally.

### 9.2 Unsupported languages

Per `AlignmentFallback`:

- `SkipChunk` (default): the `Transcript` is emitted with `words: Vec::new()`. The indexer sees a normal segment with no word-level data.
- `Error`: emit `Event::Error { error: WorkFailure::LanguageUnsupported }` instead of `Event::Transcript`.

### 9.3 Whisper context load failure

`ManagedTranscriberBuilder::build()` returns `Err(RunnerError::WhisperContextLoad(_))`. No worker threads are spawned; no resources to clean up.

### 9.4 Aligner load failure

`AlignmentSetBuilder::register(key, ...)` returns `Err(RunnerError::AlignerLoad(_))` or `Err(RunnerError::TokenizerLoad(_))`. The caller chooses to drop that language, fall through to an `AlignerKey::Any` multilingual aligner if registered, or abort builder construction.

### 9.5 Push order

`push_samples` and `push_vad_segment` reject PTS regressions (`PtsRegression`); forward gaps inside `gap_tolerance_samples` are zero-filled silently; larger forward gaps return `GapExceedsTolerance` so the caller can handle a stream restart deliberately.

### 9.6 Backpressure

If `SampleBuffer` fills past its cap (e.g., all workers busy, no chunks completing, `cut_pending` holding many descriptors), the next `push_samples` returns `Backpressure`. The caller pauses ingestion until `poll_transcript` drains chunks and the buffer trims. The runner's `process_packet` translates this into a blocking wait by default; with `WhisperPoolConfig::block_on_full_queue = false`, it propagates `Backpressure` for caller-side pacing.

### 9.7 Worker hang

If an inference worker exceeds its per-job timeout, the dispatcher records it as `WorkFailure::WorkerHangTimeout` for the affected `chunk_id`, the chunk emits `Event::Error`, and the worker is recycled (a fresh `WhisperState` or `Aligner` is created from the shared model). Callers see continued operation rather than a deadlocked `drain()`.

---

## 10. Testing strategy

### 10.1 Core (no ML deps)

- **Unit tests for `cut.rs`.**
  - Push synthetic VAD segment sequences, assert emitted MergedChunks match expected boundaries.
  - **Single VAD segment > chunk_size** is hard-split into ≤ chunk_size sub-ranges; provenance preserved in `subs`.
  - **Zero-gap consecutive VAD segments** merge into one chunk; the boundary is captured in `subs`.
  - **Empty / single-segment** inputs flush correctly on EOF.
  - Property test (`quickcheck` feature): for any random sequence of non-overlapping VAD segments, no emitted chunk exceeds `chunk_size`.
- **Unit tests for `dispatch.rs`.**
  - Drive the state machine with mocked ASR/alignment results; assert command and event sequences.
  - **Out-of-order completion** (chunk 5 finishes before chunk 3) emits in chunk-id order.
  - **`cut_pending` does not extract samples;** verify `SampleBuffer` size with `max_in_flight` saturated.
  - **Failure cascade** (one chunk errors) does not block downstream chunks.
- **Unit tests for `buffer.rs`.**
  - Round-trip extract/trim correctness, especially around boundary conditions.
  - **Forward gap within tolerance** zero-fills silently.
  - **Forward gap above tolerance** returns `GapExceedsTolerance`.
  - **PTS regression** returns `PtsRegression`.
  - **Backpressure** trips precisely at `buffer_cap_samples`.
- **Integration test for `Transcriber`.** End-to-end: push synthetic packet stream + canned VAD segments + mocked ASR/alignment results, assert emitted `Transcript`s match expectations.
- **Fuzz harness** (under `arbitrary` feature). Random push/inject sequences must not panic and must preserve chunk-id-order and bounded-memory invariants.

### 10.2 Runner (with whisper-rs and, optionally, ort)

- **End-to-end test** using a tiny GGUF whisper model and a canned 30 s audio file with known transcript. Assert text matches within a Levenshtein distance threshold; assert at least one `Transcript` is emitted.
- **Multi-chunk test** with a 90 s file producing exactly 3 transcripts.
- **Backpressure test** with a tiny `buffer_cap_samples` to verify the runner blocks `process_packet` correctly.
- **Alignment test** (alignment feature on) with a tiny wav2vec2 model and a known phrase; assert each word's range overlaps the expected sample range.
- **Edge cases:**
  - **Whisper returns empty text** (chunk emits with `text=""`, `words=[]`, no error).
  - **Alignment produces fewer words than tokenised text** (recovers by filling missing words with empty-range Word entries OR returns `NoAlignmentPath`; spec which one).
  - **Single VAD segment > chunk_size** end-to-end produces N transcripts with consistent `vad_segments` provenance.
  - **Zero-gap consecutive VAD segments** end-to-end.
  - **Worker hang timeout** (mocked): a worker that never returns triggers `WorkerHangTimeout`.
  - **Per-call language hint** via `AsrParamsOverride` skips auto-detection.
  - **Language lock** with `AutoLockAfter(1)`: the second chunk's detection is bypassed.

### 10.3 Benchmarks

- `benches/cut.rs`: throughput of the cut state machine alone (millions of segments / sec target).
- `benches/dispatch.rs`: throughput of the dispatch state machine with mocked inference.
- A separate offline-only `examples/managed_runner.rs` provides a hand-runnable timing reference; not a CI bench.

### 10.4 CI matrix

CI builds the crate on Linux, macOS, and Windows (mirroring the existing template). Feature combinations covered:

- `--no-default-features` (core only, no_std-eligible)
- `--no-default-features --features std` (std core, no runner)
- `--no-default-features --features "std runner"` (default-equivalent, no alignment)
- `--features "runner alignment"` (full runner)

Whisper-rs on Windows requires CMake and a working C compiler; the CI matrix should fail loudly at PR time rather than silently at release time.

---

## 11. Performance considerations

- The cut and dispatch state machines do `O(1)` work per push and per inject; total CPU is dominated by ASR inference and alignment inference.
- `Arc<[f32]>` ownership transfer between state machine and workers avoids per-chunk reallocation; one extract from `SampleBuffer` materialises the chunk's samples.
- ASR worker count defaults to half of physical cores. With a tiny model and CPU inference this gives ~4–6× real-time on a typical 8-core machine; figure depends on the §13.1 spike outcome (shared-context vs per-worker context).
- **Memory ceiling, working memory only (excludes model weights):**
  - `SampleBuffer`: `buffer_cap_samples × 4 bytes` ≈ 3.84 MiB at default cap.
  - In-flight extracted chunks: `max_in_flight × chunk_samples × 4 bytes` ≈ `(workers + 2) × 1.92 MiB`. For `workers = 4`: 6 × 1.92 ≈ 11.5 MiB.
  - Per-worker ASR decoder workspace: ~10–30 MiB per `WhisperState` (model-dependent; mostly KV cache + intermediate tensors).
  - Per-alignment-job logits buffer: `T × V × 4 bytes`. For 30 s @ 50 Hz frame rate (typical wav2vec2 hop) and a 32-character vocab: 1500 × 32 × 4 ≈ 192 KiB per job. For phoneme vocab (≈80): ~480 KiB. Multiple jobs in flight only if alignment_workers > 1.
  - `cut_pending` queue: O(N descriptor entries × ~200 bytes), negligible.
  - **Working-memory total at default config (4 ASR workers + 1 alignment worker, alignment on):** roughly 4 + 12 + 4×20 + 1 × 0.5 ≈ **96 MiB**.
- **Model weights (loaded once):**
  - Whisper: 75 MiB (tiny) up to 3 GiB (large-v3).
  - Per-language wav2vec2: 50–500 MiB each. Multilingual fallback (XLSR / MMS large): up to 2 GiB.
  - If §13.1 forces per-worker `WhisperContext`, multiply Whisper weight by `worker_count`.
- Alignment is sequential in v1; if a profile shows alignment as the bottleneck, parallelising is a runner-only change conditional on the §6.3.3 ort thread-safety question.

---

## 12. Future work

- **Auto-download default wav2vec2 models** (mirroring WhisperX's `DEFAULT_ALIGN_MODELS_HF`). Includes SHA-256 verification at fetch time.
- **Bundled tiny Whisper model** as a `bundled-tiny` feature, mirroring `soundevents` and `textclap` ergonomics. Will use a build-time fetch with checksum verification rather than a checked-in binary, so cargo-clones stay cheap.
- **Multi-aligner-worker pool** if alignment becomes the throughput bottleneck. Choice between cross-language-only or within-language parallelism depends on §6.3.3 outcome.
- **Backend swap**: candle-whisper or whisper-ONNX runners, slot into the same `core` crate via a parallel runner module. Enabled by the §3.4 backend invariant.
- **Async runner** behind a feature flag, exposing `Stream<Item = Transcript>` for tokio integration.
- **Live captioning latency profile**: shorter `chunk_size` + flush-on-silence cut policy; benchmarked latency vs. quality trade. Out of scope for v1 indexing.
- **Diarization integration glue.** whispery itself stays speaker-agnostic (§1.6); a future `whispery-diarize` adjacent crate may provide the indexer's join helper.
- **Metrics / observability hooks.** Per-chunk inference latency, queue depths, alignment failure rate, temperature-fallback hit rate. Likely as a `metrics` feature exporting via the `metrics` crate facade.
- **Per-call ASR override** beyond the language hint: in v1 we ship `AsrParamsOverride { language_hint, beam_size, temperature_schedule, initial_prompt }`. Other fields can be added without breaking changes.
- **Model integrity verification** (`SHA-256` checking of loaded GGUF / wav2vec2 files at builder-time).
- **Per-language wav2vec2 default-model registry.** The list of recommended models per language (with licenses) can ship as a separate `whispery-models` crate or as a doc page; v1 leaves this to the caller.

---

## 13. Open risks

The following items must be resolved (or explicitly accepted) before implementation begins. Each carries a meaningful chance of forcing a re-architecture, so they get a named slot rather than buried in §12.

### 13.1 `WhisperContext` sharing across worker threads

`whisper_rs::WhisperState<'a>` borrows `&'a WhisperContext`. The shared-context concurrency model (one `Arc<WhisperContext>` plus N states) requires `WhisperContext: Send + Sync` and a way to move (or self-reference) the borrowed `WhisperState` into a worker thread. Whether this is supported depends on whisper-rs version, build features, and the underlying whisper.cpp build.

**Spike (≤ 1 day) before code starts:** prototype both:

1. `Arc<WhisperContext>` shared, per-worker `WhisperState` (likely via `whisper_rs::OwnedWhisperState` or self-referential `ouroboros`).
2. Per-worker `WhisperContext` (one model load per worker; memory scales `worker_count × model_size`).

Outcome decides §6.2's worker structure, §11's memory footprint, and the realistic ceiling on `worker_count` for large models. If only (2) is viable, default `worker_count` drops to 1 for any model whose load size exceeds available memory divided by physical cores.

### 13.2 ort `Session` thread-safety per execution provider

`ort::Session::run` is documented as thread-safe in the general case but has known quirks per execution provider (CUDA EP serialises internally; some Vulkan paths require single-threaded use). §6.3.3 commits v1 to a single alignment worker; the multi-worker future (v2) needs a clear answer per supported EP.

Not a v1 blocker, but document the known constraints alongside the §6.3 design so v2 work doesn't restart the analysis from scratch.

### 13.3 wav2vec2 model availability per language

The forced-alignment story requires a wav2vec2 phoneme/character model per language. Hugging Face has good coverage of major languages; long-tail languages may have only multilingual (XLSR / MMS) models. The spec defers model curation to a v2 `whispery-models` crate, but if the indexer's first deployment targets a language without a quality language-specific model, falling back to multilingual changes the alignment quality story. Confirm target-language coverage before committing to the forced-alignment path; if a target language has no usable aligner, that chunk emits with empty `words` per `AlignmentFallback::SkipChunk`.

### 13.4 P4 architectural questions — resolved

Both questions are resolved and recorded in §1.6 / §1.7:

- **§1.6 (diarization integration):** confirmed against `dia` v0.1.0. `dia::DiarizedSpan.range` uses `mediatime::TimeRange` at the 1/16000 timebase, identical to `Word.range`; the join is plain interval-overlap, no whispery-side API changes required.
- **§1.7 (deployment):** crate-only for v1. A wrapper service binary, if ever needed, is additive and does not change whispery's public surface.

These resolutions are load-bearing for the implementation plan; if either changes (e.g., `dia` ships a breaking API revision before whispery v1 lands), revisit §1.6 before merging.

---

## Appendix A — WhisperX `merge_chunks` reference

For comparison, the original Python (lightly cleaned):

```python
def merge_chunks(segments, chunk_size, onset, offset):
    curr_end = 0
    merged = []
    seg_idxs = []
    curr_start = segments[0].start
    for seg in segments:
        if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
            merged.append({
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,
            })
            curr_start = seg.start
            seg_idxs = []
        curr_end = seg.end
        seg_idxs.append((seg.start, seg.end))
    merged.append({
        "start": curr_start,
        "end": curr_end,
        "segments": seg_idxs,
    })
    return merged
```

`Cut::push_segment` plus `Cut::flush` is the streaming form of this loop, with one segment look-ahead replaced by per-segment incremental decisions.

## Appendix B — Decisions deferred

1. Whether `Lang` should be a typed enum over the Whisper-supported languages or a `SmolStr` newtype. v1 uses the newtype; revisit when we have a curated downstream consumer set.
2. Whether `Transcript` should derive `Clone`. v1 does not; the dispatcher moves it through a single channel. Revisit if the indexer needs to fan out the same chunk to multiple writers.
3. Whether the runner's dispatch loop should run on a dedicated background thread instead of inline on the caller's thread inside `process_packet`. v1 inline; revisit if profiling shows `process_packet` stalls dominating.
4. ~~Whether to maintain a per-Transcriber language cache.~~ **Resolved.** v1 implements `LanguagePolicy::AutoLockAfter(1)` as the default — language is detected on the first non-trivial chunk and locked for the rest of the session.
