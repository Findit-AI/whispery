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

- **Speaker diarization.** WhisperX has it; we don't, and there's no current downstream consumer.
- **Multi-channel audio.** Input is mono f32 16 kHz. Caller mixes down.
- **Resampling.** Caller delivers 16 kHz; we do not own ffmpeg or libsamplerate logic.
- **Async runner.** No `tokio` dependency in v1. The runner exposes a sync push + sync poll API.
- **Live captioning latency profile.** We optimise for indexing throughput; v1 cuts at ≤ ~30 s chunks for max Whisper quality. A live-captioning latency profile (short chunks, sub-second flush) is a configurable knob in v1 but its tuning is not validated.
- **DTW token timestamps.** Word-level timestamps come from forced alignment only; whisper.cpp's DTW path is not enabled.
- **Auto-downloading wav2vec2 models.** Callers register paths explicitly. Auto-fetch is a v2 add-on that doesn't change the API.

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
│   └── cut.rs
│
└── tests/
    ├── core_cut.rs
    ├── core_dispatch.rs
    └── runner_e2e.rs        cfg(feature = "runner")
```

### 3.2 Cargo features

| Feature        | Default | Pulls                          | Notes |
|----------------|:-:|--------------------------------|-------|
| `std`          | yes | `alloc` + `std`                | Core compiles to no_std + alloc with this off; runner is std-only. |
| `runner`       | yes | `whisper-rs`, `crossbeam-channel` | The bundled production runner. |
| `alignment`    | yes | `ort`, `tokenizers`, `ndarray` | Forced-alignment pieces in the runner. Disable to skip word-level. |
| `bundled-tiny` | no  | a checked-in tiny.gguf file   | Mirrors `soundevents` / `textclap` ergonomics. Mostly for examples and tests. |
| `serde`        | no  | `serde`                        | Derive Serialize/Deserialize on public types. |
| `arbitrary`    | no  | `arbitrary`                    | Fuzz harnesses. |
| `quickcheck`   | no  | `quickcheck`                   | Property tests. |

`alignment` requires `runner`. The Cargo manifest enforces this with a `required-features` constraint on the runner module.

### 3.3 Public surface

```rust
// Always public:
pub mod types;
pub mod core;
pub use types::{Transcript, Word, Lang, ChunkId, TranscriberError, WorkFailure};
pub use core::{Transcriber, TranscriberConfig, Command, Event, CommandKind, EventKind};

#[cfg(feature = "runner")]
pub mod runner;

#[cfg(feature = "runner")]
pub use runner::{ManagedTranscriber, ManagedTranscriberBuilder, WhisperPoolConfig};

#[cfg(feature = "alignment")]
pub use runner::{AlignmentSet, AlignmentSetBuilder, Aligner, AlignmentFallback};
```

### 3.4 Layering rule

The runner depends on the core; the core does not name anything in the runner. Enforced at module level: `core/` modules `use` only `crate::types`, `crate::time`, and standard alloc/core types. `runner/` modules may freely call into `core/`.

---

## 4. Public types

### 4.1 Time

All emitted time ranges use a 16 kHz timebase, so PTS values are sample indices.

```rust
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// 1 / 16_000 — PTS values are sample counts.
pub const TIMEBASE: mediatime::Timebase =
    mediatime::Timebase::new(1, NonZeroU32::new(SAMPLE_RATE_HZ).unwrap());
```

`Timebase::new` is `const fn`. Downstream consumers call `.rescale_to(other_tb)` to express the same instants in their preferred timebase.

### 4.2 Transcript

The per-chunk emission unit. One merged chunk produces exactly one `Transcript`.

```rust
pub struct Transcript {
    /// Bounds of the merged chunk in source-audio sample space.
    pub range: mediatime::TimeRange,

    /// Detected (or hint-supplied) language for this chunk.
    pub language: Lang,

    /// Joined text (whitespace-aware concatenation of `words`).
    /// Held verbatim from whisper before alignment, so it includes
    /// punctuation and casing that wav2vec2 does not produce.
    pub text: smol_str::SmolStr,

    /// Word-level results from forced alignment. Empty when alignment
    /// was disabled, the language was unsupported, or alignment failed.
    pub words: Vec<Word>,

    /// Whisper's segment-level confidence (mean log-probability).
    pub avg_logprob: f32,

    /// Whisper's per-segment no-speech probability. Useful for the
    /// indexer to filter out borderline-silent chunks.
    pub no_speech_prob: f32,

    /// The sub-VAD-segments that composed this merged chunk, in the
    /// same sample-space timebase. Useful for indexers that want
    /// precise speech-only intervals within a 30 s chunk.
    pub vad_segments: Vec<mediatime::TimeRange>,

    /// Monotonic chunk identity within a single Transcriber lifetime.
    pub chunk_id: ChunkId,
}
```

`Transcript` is not `Copy` and is not `Clone` by default — callers move it through the indexer. With the `serde` feature it derives `Serialize` and `Deserialize`.

### 4.3 Word

```rust
pub struct Word {
    pub text: smol_str::SmolStr,
    pub range: mediatime::TimeRange,

    /// Alignment confidence in [0, 1]; the CTC score mapped to a
    /// per-word average. NaN-free.
    pub score: f32,
}
```

### 4.4 Language

A newtype around `SmolStr`. Whisper.cpp returns ISO 639-1 strings; we accept and emit the same.

```rust
pub struct Lang(pub smol_str::SmolStr);

impl Lang {
    pub const EN: Self = Self(SmolStr::new_inline("en"));
    pub const ZH: Self = Self(SmolStr::new_inline("zh"));
    pub const JA: Self = Self(SmolStr::new_inline("ja"));
    // …a curated list of constants for common languages
    pub const ANY: Self = Self(SmolStr::new_inline("*"));

    pub fn as_str(&self) -> &str;
    pub fn from_str(s: &str) -> Result<Self, InvalidLang>;
}
```

`Lang::ANY` is the sentinel that the registry treats as a multilingual fallback.

### 4.5 Errors

```rust
pub enum TranscriberError {
    /// Push order violation: samples or VAD segments arrived with a
    /// PTS earlier than the high-water mark.
    OutOfOrder { kind: PushKind, advance: i64 },

    /// Sample buffer would exceed its configured cap. The caller is
    /// pushing faster than the runner can drain.
    Backpressure { buffered: usize, cap: usize },

    /// Caller asked for a chunk_id we no longer have state for.
    UnknownChunk(ChunkId),

    /// Caller called signal_eof and then pushed more samples.
    AfterEof,
}

pub enum WorkFailure {
    WhisperFailed { kind: WhisperFailureKind, message: String },
    AlignmentFailed { kind: AlignmentFailureKind, message: String, language: Lang },
    LanguageUnsupported { language: Lang },
}

#[cfg(feature = "runner")]
pub enum RunnerError {
    WhisperContextLoad(/* whisper-rs error */),
    WhisperPoolShutdown,
    AlignerLoad { language: Lang, source: /* ort error */ },
    Io(std::io::Error),
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
    pub fn inject_whisper_result(
        &mut self,
        chunk_id: ChunkId,
        out: WhisperResult,
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
    pub chunk_size: Duration,

    /// Max samples kept in the internal buffer before the buffer
    /// returns Backpressure. Default 60 s × 16 kHz = 960_000 samples.
    pub buffer_cap_samples: usize,

    /// Whether to enable word-level alignment. The runner's
    /// AlignmentSet is configured separately; this flag tells the
    /// state machine whether to emit Command::RunAlignment after
    /// each whisper completion.
    pub word_alignment: bool,

    /// Maximum chunks in flight (issued Command::RunWhisper but not
    /// yet event-emitted). Bounds memory and applies backpressure.
    pub max_in_flight: usize,
}

impl Default for TranscriberConfig { /* sensible defaults */ }
```

### 5.3 Cut state machine (`core/cut.rs`)

A direct port of WhisperX's `merge_chunks`, restated as an incremental state machine.

State:

```rust
struct Cut {
    chunk_size: Duration,                // mediatime::Duration

    // The chunk currently accumulating (None when between chunks).
    current_start: Option<mediatime::Timestamp>,
    current_end: mediatime::Timestamp,
    current_subs: Vec<mediatime::TimeRange>,
}
```

Transitions:

- `push_segment(seg)`:
  - If `current_start` is None: set it to `seg.range.start()`.
  - If `(seg.range.end() - current_start) > chunk_size` *and* `(current_end - current_start) > 0`:
    - Emit a `MergedChunk { start: current_start, end: current_end, subs: take(current_subs) }`.
    - Reset: `current_start = Some(seg.range.start())`, `current_subs.clear()`.
  - Update `current_end = seg.range.end()`, `current_subs.push(seg.range)`.
- `flush()` (called on EOF):
  - If `current_start` is Some: emit the trailing chunk, reset.

The state machine guarantees:

1. **Monotonicity.** Output `MergedChunk`s are non-overlapping and strictly ordered by start time.
2. **Boundedness.** No emitted chunk spans more than `chunk_size + max(seg_duration)` — i.e., one segment overshoot is allowed because we only flush *before* adding the segment that would push over.
3. **Provenance preserved.** `subs` lists every silero VAD segment whose union forms the chunk; downstream consumers know the precise speech-only intervals.

This logic is purely arithmetic on integer PTS, no allocations beyond `current_subs`.

### 5.4 Sample buffer (`core/buffer.rs`)

```rust
pub(crate) struct SampleBuffer {
    base_pts: i64,                       // PTS of samples[0] (1/16000 timebase)
    samples: Vec<f32>,
    cap: usize,
}
```

Operations:

- `append(starts_at: Timestamp, samples: &[f32]) -> Result<(), Backpressure>`:
  - On first call, set `base_pts = starts_at.pts()`.
  - On subsequent calls, require `starts_at.pts() == base_pts + samples.len() as i64`. Out-of-order or gappy push is `OutOfOrder` (the runner is expected to upstream-detect gaps and drive the state machine through silence regions deliberately).
  - Extend `samples`; if `samples.len() > cap`, return `Backpressure`.
- `extract(range: TimeRange) -> Arc<[f32]>`:
  - Slice `samples[(start - base_pts)..(end - base_pts)]`, copy into a fresh `Arc<[f32]>`.
  - Return the Arc.
- `trim_to(low_water: i64)`:
  - Drop `samples[0..(low_water - base_pts)]`, advance `base_pts`. Used by the dispatch state machine after a chunk is fully emitted; the new low-water is the lowest in-flight chunk's start.

The buffer is a flat `Vec<f32>` with periodic `drain(0..n)` on trim. For our packet rates (≪ 1 GB/s), the memmove cost is dominated by whisper inference time. A circular ring buffer is a future optimisation.

### 5.5 Dispatch state machine (`core/dispatch.rs`)

Tracks per-chunk lifecycle:

```rust
enum ChunkPhase {
    AwaitingWhisper,           // RunWhisper command issued
    AwaitingAlignment,         // whisper done; RunAlignment issued (if alignment enabled)
    Done,                      // event emitted
    Failed,                    // event emitted as Error
}

struct Dispatch {
    in_flight: BTreeMap<ChunkId, ChunkRecord>,  // ordered for low-water trim
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
    whisper_result: Option<WhisperResult>,
}
```

Transitions:

- On `Cut::emit(MergedChunk)`:
  - If `in_flight.len() >= max_in_flight`: stash the chunk on a `cut_pending` queue; don't issue command yet.
  - Else: extract `samples` from `SampleBuffer`, push `ChunkRecord` into `in_flight`, enqueue `Command::RunWhisper`.
- On `inject_whisper_result(chunk_id, result)`:
  - Mark `record.whisper_result = Some(result)`.
  - If `word_alignment` is true AND result has non-empty text: enqueue `Command::RunAlignment`. Set `phase = AwaitingAlignment`.
  - Else: build `Transcript` with empty `words`, enqueue `Event::Transcript`, set `phase = Done`, run trim.
- On `inject_alignment_result(chunk_id, result)`:
  - Build `Transcript` from `whisper_result + result.words`, enqueue `Event::Transcript`, set `phase = Done`, run trim.
- On `inject_failure(chunk_id, failure)`:
  - Enqueue `Event::Error`, set `phase = Failed`, run trim.
- On `trim`:
  - Compute `low_water = in_flight.values().filter(phase ∈ {AwaitingWhisper, AwaitingAlignment}).map(range.start_pts).min()`.
  - Call `SampleBuffer::trim_to(low_water)`.
  - Remove `Done`/`Failed` records that have been event-drained.
  - If a `cut_pending` chunk exists and `in_flight.len() < max_in_flight`: promote it.

### 5.6 Command and Event

```rust
pub enum Command {
    RunWhisper {
        chunk_id: ChunkId,
        samples: Arc<[f32]>,
        sample_rate: u32,                // always SAMPLE_RATE_HZ in v1
        params: WhisperParams,
    },
    RunAlignment {
        chunk_id: ChunkId,
        samples: Arc<[f32]>,
        text: smol_str::SmolStr,
        language: Lang,
        token_hints: Vec<WhisperTokenHint>, // optional; helps CTC seed
    },
}

pub enum Event {
    Transcript(Transcript),
    Error { chunk_id: ChunkId, error: WorkFailure },
}
```

`WhisperParams` is the runner's responsibility to interpret; the core just shuttles it. It includes language hint, beam size, temperature schedule, suppress_tokens, no_speech_threshold. Sensible defaults match WhisperX's choices except `condition_on_previous_text=false` (silence between VAD chunks breaks continuity).

`WhisperResult` (in injects):

```rust
pub struct WhisperResult {
    pub text: smol_str::SmolStr,
    pub language: Lang,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
    pub tokens: Vec<WhisperTokenHint>,
}
```

`AlignmentResult`:

```rust
pub struct AlignmentResult {
    pub words: Vec<Word>,
}
```

---

## 6. Runner (`runner/`)

Default-on `runner` feature. Wires the core to whisper-rs and (with the `alignment` feature) to ort-based wav2vec2 forced alignment.

### 6.1 ManagedTranscriber

```rust
pub struct ManagedTranscriber {
    core: core::Transcriber,
    whisper_pool: WhisperPool,
    alignment_pool: Option<AlignmentPool>,    // Some iff word-level alignment is enabled
    emit_rx: crossbeam_channel::Receiver<Event>,
}

impl ManagedTranscriber {
    pub fn builder(whisper_ctx: WhisperContext) -> ManagedTranscriberBuilder;

    pub fn process_packet(
        &mut self,
        starts_at: Timestamp,
        samples: &[f32],
        vad_segments: &[VadSegment],
    ) -> Result<(), RunnerError>;

    pub fn signal_eof(&mut self) -> Result<(), RunnerError>;

    pub fn poll_transcript(&mut self) -> Option<Transcript>;
    pub fn poll_error(&mut self) -> Option<(ChunkId, WorkFailure)>;

    /// Block until all in-flight work drains; returns once core.is_idle().
    pub fn drain(&mut self) -> Result<(), RunnerError>;
}

pub struct ManagedTranscriberBuilder {
    /* core config, whisper pool config, alignment set */
}

impl ManagedTranscriberBuilder {
    pub fn chunk_size(self, d: Duration) -> Self;
    pub fn buffer_cap_samples(self, n: usize) -> Self;
    pub fn whisper_pool(self, cfg: WhisperPoolConfig) -> Self;

    /// Enables word-level forced alignment using the supplied registry.
    /// If never called, `Transcript.words` is always empty (alignment off).
    #[cfg(feature = "alignment")]
    pub fn with_alignment(self, set: AlignmentSet) -> Self;

    pub fn build(self) -> Result<ManagedTranscriber, RunnerError>;
}
```

The builder's `build()` returns a `ManagedTranscriber` with worker threads spawned and channels wired. Internally, `with_alignment` flips the core's `word_alignment` flag and stashes the `AlignmentSet` for the alignment worker.

### 6.2 WhisperPool

```rust
pub struct WhisperPoolConfig {
    pub worker_count: usize,         // default: max(1, num_cpus::get() / 2)
    pub model_path: PathBuf,         // or use bundled-tiny via builder helper
    pub use_gpu: bool,
}

struct WhisperPool {
    ctx: Arc<WhisperContext>,        // shared model
    workers: Vec<JoinHandle<()>>,
    work_tx: crossbeam_channel::Sender<WhisperWorkItem>,
    result_tx: crossbeam_channel::Sender<(ChunkId, Result<WhisperResult, WorkFailure>)>,
}

struct WhisperWorkItem {
    chunk_id: ChunkId,
    samples: Arc<[f32]>,
    params: WhisperParams,
}
```

Each worker owns its own `WhisperState` borrowed from the shared `WhisperContext`. Workers run a loop: `recv work` → `state.full(samples, params)` → `send result`.

The `ManagedTranscriber` runs a small dispatch loop on a dedicated thread (or on the caller's thread via `process_packet`):

1. Drain `Command`s out of `core`.
2. For `RunWhisper` commands, send to `whisper_pool.work_tx`.
3. For `RunAlignment` commands, send to the aligner.
4. Drain `result_rx` (whisper) and `align_rx` (alignment), call `core.inject_*`.
5. Drain `Event`s, push them to `emit_tx`.

The dispatch loop is single-threaded; only the inference workers run in parallel.

### 6.3 Aligner and AlignmentSet

#[cfg(feature = "alignment")]

```rust
pub struct Aligner {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    language: Lang,
    /* phoneme/character vocab, sample-rate, n_mfcc, etc. */
}

impl Aligner {
    pub fn from_paths(
        language: Lang,
        model_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, RunnerError>;

    pub(crate) fn align(
        &mut self,
        samples: &[f32],
        text: &str,
    ) -> Result<AlignmentResult, WorkFailure>;
}

pub struct AlignmentSet {
    aligners: HashMap<Lang, Mutex<Aligner>>,
    fallback: AlignmentFallback,
}

pub enum AlignmentFallback {
    /// On unsupported language, emit the chunk's Transcript with empty `words`.
    SkipChunk,
    /// On unsupported language, error the chunk.
    Error,
}

pub struct AlignmentSetBuilder { /* … */ }
```

The builder accepts `(Lang, Aligner)` pairs and a fallback policy. `Lang::ANY` is consulted last. The `aligners` map is wrapped in `Mutex` per-language so multiple alignment workers can run different languages concurrently.

The `align()` algorithm is standard CTC forced alignment:

1. Run `session` over `samples` (reshape to wav2vec2's expected input shape), get logits `(T, V)` where V is the phoneme/character vocab size including a blank token.
2. Apply log-softmax along V.
3. Tokenize `text` with `tokenizer` to get a sequence `Y` of vocab indices.
4. Build the standard CTC alignment lattice over `(T, 2|Y|+1)` and run Viterbi to get the highest-probability monotonic alignment of `Y` to `T`.
5. Walk the path to extract per-token start/end frame indices.
6. Map frame indices back to sample indices via the model's hop size, group tokens into words by tokenizer-defined word boundaries, and produce `Vec<Word>`.

For v1 the `AlignmentPool` ships one alignment worker (sequential alignment). The `Mutex<Aligner>` per language in `AlignmentSet` already supports parallelism, so future versions can spin up multiple alignment workers without touching the public API. Whisper runs with N workers, so whisper is rarely the bottleneck for indexing throughput.

### 6.4 Concurrency model summary

```
caller thread                whisper workers (N)        align worker (1)
─────────────                ──────────────────         ────────────────
process_packet   ──┐
                  ▼
            [dispatch loop]
            drains Command
                ├── RunWhisper ──▶ whisper work_tx ──▶ WhisperState::full
                │                                      │
                │                                      ▼
                │   ◀── result via result_rx ──── return
                │
                ├── inject_whisper_result
                │
                ├── RunAlignment ──▶ align work_tx ──▶ aligner.align
                │                                      │
                │   ◀── result via align_rx ──────── return
                │
                ├── inject_alignment_result
                │
                └── drain Event ──▶ emit_tx ──▶ poll_transcript / poll_error
```

The dispatch loop runs *inline* on the caller's thread inside `process_packet` and `poll_transcript`. There is no background dispatcher thread. This keeps the runner deterministic from the caller's perspective; workers are the only background threads.

The flip side: very long-running `process_packet` calls can stall if all workers are busy and the buffer fills. The runner's `WhisperPoolConfig` allows tuning `max_queued_chunks` to bound this; when exceeded, `process_packet` either blocks (default) or returns `RunnerError::Backpressure`.

---

## 7. Data flow (end-to-end)

A worked example. Assume `chunk_size = 30 s`, `worker_count = 2`, alignment enabled.

1. Caller's pipeline emits a 100 ms audio packet (1 600 samples) at PTS 0.
2. Caller runs silero on the packet, gets zero or more new `SpeechSegment`s.
3. Caller calls `mt.process_packet(Timestamp::new(0, TIMEBASE), &samples, &vad_segs)`.
4. `ManagedTranscriber::process_packet`:
   - Calls `core.push_samples(...)`. SampleBuffer extends.
   - For each VAD segment: `core.push_vad_segment(seg)`. Cut state machine accumulates; possibly emits a `MergedChunk` if accumulated speech ≥ 30 s.
   - Drains commands from `core.poll_command()`. If a `RunWhisper` is emitted, ships it to `whisper_pool`.
5. Caller continues for some seconds, accumulating ~3–10 merged chunks across whisper workers.
6. Whisper worker A finishes chunk 0, sends `Ok(WhisperResult)` to `result_rx`.
7. Caller's next `process_packet` (or `poll_transcript`) drains `result_rx`, calls `core.inject_whisper_result(0, result)`. Core enqueues `Command::RunAlignment` for chunk 0.
8. Dispatch loop ships the alignment command to the alignment worker.
9. Alignment worker computes word-level timestamps, sends `Ok(AlignmentResult)` to `align_rx`.
10. Next drain calls `core.inject_alignment_result(0, result)`. Core builds `Transcript`, enqueues `Event::Transcript`. Dispatch loop drains it to `emit_tx`.
11. Caller calls `poll_transcript()` and gets the `Transcript` for chunk 0. Indexer writes it to lancedb.
12. Periodically `core` trims its `SampleBuffer` to the lowest in-flight chunk's start, freeing memory.
13. After all packets are pushed, caller calls `signal_eof()`, then `drain()` to flush remaining chunks.

The net effect: transcripts arrive a few seconds after their audio's wall-clock arrival (whisper's inference latency + alignment latency), in chunk-id order. The pipeline never holds more than `max_in_flight + buffer_cap_samples` worth of audio in memory.

---

## 8. Configuration and tunables

Defaults and rationale:

| Param                            | Default            | Notes |
|----------------------------------|--------------------|-------|
| `chunk_size`                     | 30 s               | Whisper's encoder window. |
| `buffer_cap_samples`             | 60 s × 16 kHz      | Twice chunk_size; bounds memory under transient backpressure. |
| `max_in_flight`                  | `worker_count + 2` | Allows pipeline depth without unbounded queueing. |
| `worker_count` (whisper)         | `max(1, num_cpus / 2)` | Half of cores leaves room for ffmpeg, silero, soundevents, lancedb. |
| `alignment_workers`              | 1                  | Sequential in v1. |
| `WhisperParams.beam_size`        | 5                  | WhisperX default. |
| `WhisperParams.temperature_schedule` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | Standard fallback. |
| `WhisperParams.no_speech_threshold` | 0.6             | WhisperX default. |
| `WhisperParams.condition_on_previous_text` | false    | VAD chunks have silence between them; cross-chunk continuity hurts more than it helps. |
| `AlignmentFallback`              | `SkipChunk`        | Unknown languages still emit a `Transcript`, just with empty `words`. |
| `with_alignment(...)`            | not called (off)    | Caller opts in by passing an `AlignmentSet`; otherwise `Transcript.words` is empty. |

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

`AlignmentSetBuilder::register(lang, paths)` returns `Err(RunnerError::AlignerLoad(_))`. The caller chooses to drop that language, fall through to `Lang::ANY`, or abort builder construction.

### 9.5 Out-of-order push

`push_samples` and `push_vad_segment` enforce monotonic time. Out-of-order push is `TranscriberError::OutOfOrder`. The caller is responsible for sequencing — an indexer driving from ffmpeg packets has natural monotonicity.

### 9.6 Backpressure

If the buffer fills past its cap (e.g., all workers busy, no chunks completing), the next `push_samples` returns `Backpressure`. The caller pauses ingestion until `poll_transcript` drains chunks and the buffer trims. The runner's `process_packet` translates this into a blocking wait by default.

---

## 10. Testing strategy

### 10.1 Core (no ML deps)

- **Unit tests for `cut.rs`.** Push synthetic VAD segment sequences, assert emitted MergedChunks match expected boundaries. Property test (via `quickcheck` feature): for any random sequence of non-overlapping VAD segments, no emitted chunk exceeds `chunk_size + max_segment_duration`.
- **Unit tests for `dispatch.rs`.** Drive the state machine with mocked whisper/alignment results; assert the command and event sequence is correct.
- **Unit tests for `buffer.rs`.** Round-trip extract/trim correctness, especially around boundary conditions.
- **Integration test for `Transcriber`.** End-to-end: push a synthetic packet stream + canned VAD segments, inject mock whisper/alignment results, assert emitted `Transcript`s.
- **Fuzz harness** (under `arbitrary` feature). Random push/inject sequences must not panic and must preserve chunk-order invariants.

### 10.2 Runner (with whisper-rs and ort)

- **End-to-end test** using a tiny GGUF whisper model and a canned 30 s audio file with known transcript. Assert text matches within a Levenshtein distance threshold; assert at least one `Transcript` is emitted.
- **Multi-chunk test** with a 90 s file producing exactly 3 transcripts.
- **Backpressure test** with a tiny `buffer_cap_samples` to verify the runner blocks `process_packet` correctly.
- **Alignment test** with a tiny wav2vec2 model and a known phrase; assert each word's range overlaps the expected sample range.

### 10.3 Benchmarks

- `benches/cut.rs`: throughput of the cut state machine alone (millions of segments / sec target).
- `benches/dispatch.rs`: throughput of the dispatch state machine with mocked inference.
- A separate offline-only `examples/managed_runner.rs` provides a hand-runnable timing reference; not a CI bench.

---

## 11. Performance considerations

- The cut and dispatch state machines do `O(1)` work per push and per inject; total CPU is dominated by whisper inference and alignment inference.
- `Arc<[f32]>` ownership transfer between state machine and workers avoids per-chunk reallocation; one extract from `SampleBuffer` materialises the chunk's samples.
- Whisper worker count defaults to half of physical cores. With a tiny model and CPU inference this gives 4–6× real-time on a typical 8-core machine.
- Memory ceiling: `buffer_cap_samples + max_in_flight × 30s × 16kHz × 4 bytes ≈ 4 MiB + 8 × 1.92 MiB ≈ 19 MiB` for default config. wav2vec2 models add another 50–500 MiB depending on model size. Whisper models are 75 MiB (tiny) – 3 GiB (large).
- Alignment is sequential in v1; if a profile shows alignment as the bottleneck, parallelising is a runner-only change.

---

## 12. Future work

- **Auto-download default wav2vec2 models** (mirroring WhisperX's `DEFAULT_ALIGN_MODELS_HF`).
- **Multi-aligner-worker pool** if alignment becomes the throughput bottleneck.
- **Backend swap**: candle-whisper or whisper-ONNX runners, slot into the same `core` crate via a parallel runner module.
- **Async runner** behind a feature flag, exposing `Stream<Item = Transcript>` for tokio integration.
- **Live captioning profile**: shorter chunk_size + flush-on-silence cut policy; benchmarked latency vs. quality trade.
- **Diarization** as a downstream module if the indexer ever needs speaker labels.

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
2. Whether `Transcript` should derive `Clone`. v1 does not; revisit if the indexer needs to fan out the same chunk to multiple writers.
3. Whether the runner's dispatch loop should run on a dedicated background thread instead of inline on the caller's thread inside `process_packet`. v1 inline; revisit if profiling shows process_packet stalls dominating.
4. Whether to maintain a per-Transcriber language cache (assume the audio is in one language and pass as a hint to subsequent chunks). v1 lets each chunk auto-detect; revisit if mis-detection rates are high in practice.
