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
- **Auto-downloading wav2vec2 models.** Callers register paths explicitly. Auto-fetch is a v2 add-on that doesn't change the API.
- **Bundled model files.** v1 ships with no checked-in GGUF or wav2vec2 weights. Callers point at on-disk model paths.

- **DTW token timestamps.** v1 does not enable DTW. The reasoning is concrete: whisper.cpp's DTW path is set at `WhisperContext` construction (`WhisperContextParameters::dtw_parameters`) and is mutually exclusive with `flash_attn`; enabling it has measurable inference-time cost on larger models for a result that is *less accurate* than wav2vec2 forced alignment. v1's `Transcript.words` come exclusively from wav2vec2 + CTC. v2 may add DTW as an alignment-fallback option for languages without a registered wav2vec2 aligner; that decision is deferred and does not change the public surface.

### 1.6 Integration with diarization (speaker-agnostic by design)

whispery is speaker-agnostic. Diarization runs as a sibling of whispery on the same source audio, not upstream or downstream of it; the indexer joins the two outputs by time-range overlap.

**Join contract (verified against `dia` v0.1.0).** The findit-studio diarization crate `dia` (in `/Users/user/Develop/findit-studio/dia`) emits `DiarizedSpan` values with a `range()` accessor returning `mediatime::TimeRange` at the **1/16000 analysis timebase** (dia operates on resampled 16 kHz audio just like whispery internally does). Whispery's *output* `TimeRange`s are in the caller-chosen output timebase (§4.1) — typically the original media's timebase. The two outputs use the same shape (`mediatime::TimeRange`) but different timebases; the indexer joins by canonicalising both sides to one timebase before computing interval overlap.

Practical paths the indexer can take:

- **Canonicalise to whispery's output timebase up front.** On every `DiarizedSpan` emission, call `dia_span.range().rescale_to(output_tb)` and store the rescaled range. Subsequent overlap queries compare same-timebase `TimeRange` values via plain endpoint comparison.
- **Canonicalise to dia's 1/16000 timebase up front.** Symmetric; rescale whispery's `Word.range()` to `ANALYSIS_TIMEBASE` on the way into the index.
- **Roll your own cross-timebase Ord newtype.** `mediatime::Timestamp` exposes `cmp_semantic(&other) -> Ordering` (128-bit cross-multiply, exact). `mediatime::TimeRange` itself does not implement `Ord` (it has only `Eq`/`Hash` derived structurally over `(start, end, timebase)` — two equal-instant ranges in different timebases compare *unequal*). An indexer that wants to keep ranges in their native timebases needs a thin wrapper that compares via `cmp_semantic` on `start()`/`end()`.

Either of the first two paths is straightforward and lossless when the output timebase is a clean multiple of 16 kHz (1/48000 = 3×, 1/8000 = ½×); see §4.1 for non-integer-ratio behaviour. mediatime does not ship an interval-tree data structure; the indexer chooses its own (BTreeMap, btree-range, or a third-party crate) keyed on the canonicalised endpoints.

dia exposes a sync push-callback streaming API (`Diarizer::process_samples<F>(.., emit: F)` where `F: FnMut(DiarizedSpan)`); the indexer collects emitted `DiarizedSpan`s into its own time-indexed structure and runs interval-overlap queries against it as `Transcript`s arrive. The lookup index, the overlap query, and the indexer-side row structure are entirely the indexer's glue — they are not part of any crate. Sketch (indexer code, *not* a whispery API):

```rust
// Indexer-side glue (NOT a whispery API). Pick whatever interval
// data structure suits — example uses a BTreeMap keyed on
// canonicalised end-of-span PTS.
let mut spans_by_end_in_output_tb: BTreeMap<i64, DiarizedSpan> = BTreeMap::new();
diarizer.process_samples(.., |span| {
    let canon = span.range().rescale_to(output_tb);
    spans_by_end_in_output_tb.insert(canon.end_pts(), span);
})?;

// As whispery emits transcripts, join each word against overlapping dia spans.
for word in transcript.words() {
    let w = word.range();
    // standard interval overlap: start_a < end_b AND start_b < end_a
    for (&_end, span) in spans_by_end_in_output_tb
        .range(w.start_pts() + 1 ..)
    {
        let canon = span.range().rescale_to(output_tb);
        if canon.start_pts() >= w.end_pts() { break; }
        index.insert(WordRow {
            chunk_id: transcript.chunk_id(),
            word_text: word.text().to_owned(),
            range: w,
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
// Access via accessors: span.range(), span.speaker_id(),
// span.is_new_speaker(), span.average_activation(),
// span.activity_count(), span.clean_mask_fraction().
```

`Transcript.vad_segments()` is the secondary anchor when overlap is ambiguous (e.g., a word that straddles silence inside the merged chunk): the indexer should weight overlap with VAD segments more heavily than overlap with the chunk's outer `range`.

**What whispery does not need to do.**

- whispery does not consume `DiarizedSpan`s. It does not depend on the `dia` crate.
- whispery does not assign or track speaker IDs. `dia`'s speaker IDs are session-local `u64`s; cross-file speaker stability is the indexer's concern (a global cluster table keyed by embeddings, owned outside both whispery and dia).
- whispery does not need to expose phonetic-level timestamps or per-VAD-segment confidence. The sample-accurate `Word.range` (in output timebase, derived from a 16 kHz analysis index — sub-sample-accurate when the output timebase is integer-ratio related to 16 kHz) is sufficient for dia's overlap-based join.

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

### 2.1 Scope of inherited conventions

This crate inherits *some* idioms from its sibling crates (silero, soundevents, textclap, dia) and introduces *others* unique to whispery. Being explicit so future maintainers don't pick precedents from a fictional version of the siblings:

**Genuinely inherited from one or more siblings:**

- Private fields with getter accessors (silero, soundevents, dia, textclap).
- `thiserror`-derived error enums (all siblings).
- Builder pattern with both consuming `with_*` and in-place `set_*` accessors (silero, soundevents).
- `cfg`-gating for optional model bundling and feature-conditional public surface (silero, soundevents).
- Crate-level `#![deny(missing_docs)]` and `#![forbid(unsafe_code)]` (silero).
- ort version pin matching `=2.0.0-rc.12` (silero, soundevents, textclap, dia).
- `mediatime::TimeRange` for emitted ranges (dia uses 1/16000; whispery uses the caller's output timebase per §4.1, but the type itself is the shared currency).

**Whispery-original (not inherited):**

- The Sans-I/O `core/` module split with feature-gated `runner/`. silero/soundevents are flat single-tier modules; dia uses Sans-I/O at the orchestrator level but does not split into a separate runner module. This split is whispery's choice driven by the cut/dispatch logic being meaningfully testable without ML deps.
- `mediatime::Timestamp` in `push_samples` to anchor the output timebase. silero's interfaces use raw sample indices and have no mediatime types. The two-timebase model (§4.1) is whispery-specific.
- `smol_str::SmolStr` for language codes and word text. textclap uses `smol_str` internally; siblings do not on their public surface.
- `smallvec::SmallVec` for inline-sized parameter collections. No sibling exposes smallvec on its public surface; this is a whispery internal optimisation surfaced only on `AsrParams`.

**Specifically not isomorphic despite naming similarity:** whispery's `VadSegment { start_sample: u64, end_sample: u64 }` carries the same 16 kHz sample indices silero produces, but in a two-field shape (no `sample_rate` because whispery requires 16 kHz throughout). silero's `SpeechSegment::new(start_sample, end_sample, sample_rate)` is three positional fields with `Copy`. Conversion at the boundary is one constructor call.

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
| `runner`       | yes | `whisper-rs ^0.13`, `crossbeam-channel ^0.5`, `num_cpus ^1` | The bundled production runner. Whisper-rs version pinned to a specific compatible major; revisited per release. |
| `alignment`    | no  | `ort = "=2.0.0-rc.12"`, `tokenizers ^0.23`, `ndarray ^0.16` | **Opt-in.** Forced-alignment pieces in the runner. Pulls the heaviest deps in the tree (~150 MB of build artefacts). Without this feature, `Transcript.words` is always empty. ort exact-pinned with `=` (Cargo's pre-release semantics would otherwise accept the 2.0.0 final and break siblings); same pin as silero/soundevents/textclap. |
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

// Re-exports of mediatime types that appear in whispery's public
// API (so consumers don't need to add a separate `mediatime`
// dependency just to name them; they may still do so to call
// methods like `rescale_to`).
//
// SemVer note: re-exporting mediatime types ties whispery's public
// API to mediatime's. A breaking change in mediatime (major-version
// bump) is automatically a breaking change for whispery, so the
// `mediatime` dependency is pinned to a single major in Cargo.toml
// and bumping it requires a coordinated whispery major release.
// Consumers can also depend on mediatime directly at the same major
// to avoid type aliasing issues.
pub use mediatime::{Timebase, Timestamp, TimeRange};

pub use types::{
    Transcript, Word, Lang, ChunkId, VadSegment,
    TranscriberError, WorkFailure, AsrFailureKind, AlignmentFailureKind,
    PushKind, WorkerKind,
};
pub use core::{
    Transcriber, TranscriberConfig, LanguagePolicy,
    Command, Event,
    AsrParams, AsrResult, SamplingStrategy,
};

#[cfg(feature = "runner")]
pub mod runner;

#[cfg(feature = "runner")]
pub use runner::{
    ManagedTranscriber, ManagedTranscriberBuilder,
    WhisperPoolConfig,
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

**Backend invariant.** The core's `AsrParams`, `AsrResult`, and `SamplingStrategy` types contain only universal ASR knobs (language hint, beam params, no_speech threshold, temperature ladder parameters). They must not name `whisper-rs` types directly, must not include whisper.cpp-specific config fields, and must not require the runner to extend them with whisper-only options. Any whisper-rs-specific tuning lives in the runner (`WhisperPoolConfig`) and is consumed by the runner's worker thread, not shipped through the state machine. This invariant is what makes a future swap to candle-whisper or a CTranslate2 binding a runner-only change. Each `AsrParams` field corresponds to either a whisper-rs `FullParams` setter or a parameter the runner consumes for its temperature retry loop; nothing aspirational lives there.

---

## 4. Public types

### 4.1 Time — internal 16 kHz indexing, external caller-chosen timebase

Whispery operates on **two timebases**:

- **Internal (analysis) timebase = 1/16 000.** All cut decisions, SampleBuffer indexing, and CTC alignment happen in 16 kHz sample-index space. This matches silero, soundevents, and dia internally.
- **External (output) timebase = caller-chosen.** Every public `TimeRange` whispery emits — `Transcript.range`, `Word.range`, `Transcript.vad_segments`, error chunk identifiers — is in the **same timebase as the caller's input `Timestamp`s**. The caller establishes this when they make their first `push_samples` call; subsequent calls must use the same timebase.

This matters because the indexer's downstream consumers (textclap at 48 kHz, lancedb rows that reference back to original media, the playback UI that seeks within the source file) want timestamps in the *original media's* time space, not in the resampled-for-analysis 16 kHz space. silero's `SpeechSegment` carries 16 kHz sample indices because silero is purely an analysis library; whispery sits one layer up and is responsible for translating analysis-time back to source-media time.

```rust
use core::num::NonZeroU32;

/// The internal analysis sample rate. All audio fed to whispery
/// must already be resampled to this rate (caller's responsibility).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Const helper for NonZeroU32 conversion (panic-on-zero is fine for
/// compile-time evaluation; the value is statically known nonzero).
const fn nz(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
        Some(n) => n,
        None => panic!("expected nonzero u32"),
    }
}

const SAMPLE_RATE_NZ: NonZeroU32 = nz(SAMPLE_RATE_HZ);

/// Internal analysis timebase. Used by SampleBuffer, the cut state
/// machine, and the alignment pipeline. Not visible on whispery's
/// public output surface — every emitted `TimeRange` is in the
/// caller's external timebase.
pub const ANALYSIS_TIMEBASE: mediatime::Timebase =
    mediatime::Timebase::new(1, SAMPLE_RATE_NZ);
```

`mediatime::Timebase::new` is `const fn` and accepts `(u32, NonZeroU32)`. The local `nz` helper sidesteps `Option::unwrap`'s const stabilisation timeline.

#### 4.1.1 The output timebase is fixed by the first `push_samples`

`Transcriber::push_samples(&mut self, starts_at: mediatime::Timestamp, samples: &[f32])` accepts a `Timestamp` whose timebase is *whatever the caller chose for source-media time*. Common choices:

- 1/48 000 — when the source media is sampled at 48 kHz (typical for textclap-aligned outputs).
- 1/90 000 — MPEG-TS standard.
- 1/1 000 — millisecond PTS.
- 30 000/1 001 — NTSC frame rate, when transcripts must align to video frames.

The first `push_samples` call records the caller's timebase as the **output timebase** for the entire `Transcriber` lifetime. Subsequent pushes that use a different timebase are rejected with `TranscriberError::InconsistentTimebase`. The output timebase is also exposed via `Transcriber::output_timebase()`.

#### 4.1.2 Sample-index ↔ output-timebase translation

Internally, whispery indexes samples by 16 kHz sample count from the start of the stream. To translate a 16 kHz sample index `s` to an output-timebase `Timestamp`, whispery uses `mediatime::Timebase::rescale_pts`:

```rust
// in Transcriber, after first push_samples
let base_pts: i64        = starts_at.pts();             // output tb
let output_tb: Timebase  = starts_at.timebase();
// to translate sample index s (16 kHz) to output-tb pts:
let output_pts = base_pts + Timebase::rescale_pts(s as i64, ANALYSIS_TIMEBASE, output_tb);
```

`rescale_pts` uses 128-bit cross-multiply; it's exact for integer-ratio rates (16k → 48k = 3×) and saturates for non-integer ratios (16k → 90k = 5.625×) at `i64::MAX/MIN` rather than erroring. The 16k → 48k path is exact and sample-aligned; non-integer ratios give the closest representable PTS, which for indexing pipelines is well below the user-visible threshold.

#### 4.1.3 silero ↔ whispery boundary conversion

The caller takes a `silero::SpeechSegment { start_sample, end_sample, sample_rate: SampleRate::Hz16000 }` and converts it to a whispery `VadSegment` carrying the same 16 kHz sample indices:

```rust
// indexer-side glue
let v = VadSegment::new(
    silero_segment.start_sample(),    // u64 at 16 kHz
    silero_segment.end_sample(),      // u64 at 16 kHz
);
transcriber.push_vad_segment(v)?;
```

Whispery does the sample-index → output-timebase conversion internally when building MergedChunks and downstream `Transcript`s; the caller never touches output-timebase arithmetic for VAD inputs.

**Working with mediatime instants.** `mediatime` does not implement `Add` / `Sub` operators on `Timestamp` or `Duration`; arithmetic uses methods:

- `Timestamp::duration_since(&earlier) -> Option<core::time::Duration>` — Some when `self >= earlier`.
- `Timestamp::saturating_sub_duration(d) -> Timestamp` — for backing off by a duration.
- `Timestamp::rescale_to(target_timebase) -> Timestamp` — saturates to `i64::MAX/MIN` on overflow rather than returning an error.
- `Timebase::rescale_pts(pts, from_tb, to_tb) -> i64` — same saturation semantics.

`mediatime::Duration` does **not** exist as a separate type; durations are `core::time::Duration`. Internal references in earlier drafts to `mediatime::Duration` are wrong; everything that reads "Duration" in this doc means `core::time::Duration`.

**TimeRange invariants.** `mediatime::TimeRange::new(start, end, tb)` panics if `end < start`; the half-open invariant is enforced at construction. Whispery requires that callers hand it `VadSegment`s with validated `start_sample <= end_sample` (silero already guarantees this). Internal code paths that build a `TimeRange` from arithmetic should use `try_new` and surface programmer errors via panic messages rather than `Result`.

**Round-trip constraint with downstream consumers.** Common output timebases round-trip losslessly to the 1/16000 analysis timebase if the rate is a clean multiple (1/48000 = 3×, 1/8000 = ½×). Non-integer ratios (1/90000 MPEG-TS, NTSC frame rate) are saturated to the nearest representable PTS, which is sub-millisecond and well below indexing-pipeline tolerances. A defensive assertion in tests guards against PTS values approaching `i64::MAX`. Downstream consumers can always call `.rescale_to(other_tb)` if they need yet another timebase.

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
    /// surface; downstream BM25/FTS indexes this directly. The word-level
    /// `words[].text()` values are the matching original surface forms with
    /// punctuation and casing preserved (§6.3.2 step 9), recovered after
    /// CTC alignment runs over a normalised form internally; joining
    /// `words[].text()` is therefore *almost* the same as `text()`, modulo
    /// whitespace glue and any words alignment dropped on a low-confidence path.
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
    /// Original surface form of the word, preserving casing and
    /// punctuation as Whisper emitted them. Recovered after CTC
    /// alignment via the normalisation map (§6.3.2 step 9); the
    /// word that wav2vec2 actually aligned was the lowercased,
    /// punctuation-stripped form. This is the value that should
    /// be displayed in click-to-play UIs and fed to BM25 if a
    /// per-word index is built. Note that words whose audio fell
    /// inside a silence-masked region (silence-aware alignment,
    /// §6.3.2 step 0) are absent from `Transcript.words` —
    /// `text` still represents the original surface form for the
    /// words that are present, but `words[].text` joined is not
    /// guaranteed to equal `Transcript.text` modulo whitespace.
    pub fn text(&self) -> &str;

    /// Sample-accurate range of the word in the caller's output
    /// timebase (the timebase of the first push_samples Timestamp;
    /// see §4.1). Half-open. When silence-aware alignment (§6.3.2
    /// step 0) zero-masks parts of the chunk's audio, the Viterbi
    /// path may produce alignment for only a subset of the original
    /// words; the dropped words are absent from `Transcript.words`
    /// entirely. Words that ARE present have a `range` covering only
    /// the frames the Viterbi path attributed to them — never frames
    /// inside masked regions, never adjacent words' frames. The
    /// `text` for a present word is still the full original surface
    /// form, even if alignment only saw a fraction of its audio.
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

A typed enum over Whisper.cpp's supported language set. Marked `#[non_exhaustive]` so new variants can be added when whisper.cpp adds languages, without forcing a semver-major; carries an `Other(SmolStr)` variant so unknown ISO codes flowing in from whisper's auto-detect don't fail an indexing run — they propagate through to the indexer's logs.

```rust
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Lang {
    En, Zh, De, Es, Ru, Ko, Fr, Ja, Pt, Tr,
    Pl, Ca, Nl, Ar, Sv, It, Id, Hi, Fi, Vi,
    // …99 known whisper.cpp variants total; full list in Appendix C.
    /// ISO 639-1 (or whisper-supplied) code that did not match any
    /// known variant. `from_iso639_1` and `as_str` round-trip through
    /// this for unknown codes; the indexer can log the SmolStr value
    /// and continue.
    Other(smol_str::SmolStr),
}

impl Lang {
    /// Total-function constructor: every `&str` produces a `Lang`.
    /// Known whisper.cpp codes canonicalise to their named variant;
    /// unknown codes go to `Lang::Other`. Never produces
    /// `Lang::Other("en")` for an enum-known code — see the
    /// canonicalisation invariant below.
    pub fn from_iso639_1(s: &str) -> Self;

    /// Stable round-trip with `from_iso639_1`. Named variants emit
    /// their canonical ISO 639-1 string; `Other(s)` emits `s`.
    pub fn as_str(&self) -> &str;
}

impl Display for Lang { /* writes the ISO code */ }
```

**Canonicalisation invariant.** For every named variant `V`, `Lang::from_iso639_1(V.as_str())` returns the named variant, **never** `Lang::Other(V.as_str().into())`. This is what makes structural `PartialEq`/`Hash` correct: `Lang::En != Lang::Other("en")` is fine because no path in the public API can produce `Lang::Other("en")`. The runner's worker does `Lang::from_iso639_1(detected.as_str())` to wrap whisper's output, so the canonicalisation runs on every detection.

**Why `Other(SmolStr)` rather than `Result<Lang, InvalidLang>`.** An indexing engine processing thousands of hours of audio shouldn't fail a chunk when whisper detects a language we haven't enumerated yet. `Other` lets the chunk's transcript flow through with whatever string whisper produced; the indexer logs the unusual code and decides whether to retry, drop, or accept. The alternative — erroring at the boundary — forces every caller to write recovery logic for an event that should be a soft warning.

**Variant addition is not a breaking change.** Because the enum is `#[non_exhaustive]`, adding `Lang::Yue` later forces external callers' matches to already have a `_` arm; their compiled code keeps working. The newly-named variant just stops appearing under `Lang::Other("yue")` and starts appearing under `Lang::Yue` — observable, but not a panic surface.

**`AlignerKey::Lang(Lang)` hashing.** With the enum, `AlignerKey::Lang(Lang::En)` hashes by discriminant (one byte) instead of the SmolStr's content. `AlignerKey::Lang(Lang::Other(s))` falls back to hashing `s`'s contents. Both paths are correct; the typed-variant path is just faster.

### 4.5 Errors

```rust
pub enum TranscriberError {
    /// PTS regression: caller pushed samples or a VAD segment with a
    /// timestamp earlier than the current high-water mark. Forward
    /// gaps are tolerated up to `gap_tolerance_samples` (§5.4).
    /// The check runs in output-PTS space (not 16 kHz space) to
    /// avoid spurious regressions on non-integer-ratio output
    /// timebases.
    PtsRegression { kind: PushKind, advance: i64 },

    /// Forward gap exceeds the configured tolerance. Caller likely
    /// has a stream restart or a packet drop larger than expected.
    /// State machine refuses to silently zero-fill an arbitrarily
    /// large hole. Recover via `restart_at(starts_at)`.
    GapExceedsTolerance { gap_samples: u64, tolerance_samples: u64 },

    /// Sample buffer would exceed its configured cap. The runner has
    /// not drained completed chunks fast enough; the caller should
    /// pause and call `poll_event` / `poll_transcript` until the
    /// buffer trims.
    Backpressure { buffered: usize, cap: usize },

    /// `push_vad_segment` was called before any `push_samples`.
    /// The output timebase is not yet established, so the cut state
    /// machine cannot produce a meaningful `TimeRange`. Push a
    /// sample packet first.
    OutputTimebaseUnset,

    /// `push_samples` was called with a `Timestamp` whose timebase
    /// does not match the timebase recorded from the first push.
    /// Whispery enforces a single output timebase per Transcriber
    /// lifetime; cross-timebase callers should use separate
    /// Transcriber instances or rescale on their side.
    InconsistentTimebase {
        expected: mediatime::Timebase,
        got: mediatime::Timebase,
    },

    /// Caller `inject_*`ed a chunk_id that does not match an
    /// in-flight chunk.
    UnknownChunk(ChunkId),

    /// Caller called `signal_eof` and then pushed more samples or
    /// VAD segments.
    AfterEof,
}

// Derives Clone + Debug. Clone is required because the dispatch
// state machine moves WorkFailure into Event::Error while runner
// code may also want to log it; the indexer can match on it
// without a heavy-clone cost (the contained String is the only
// non-trivial allocation).
#[derive(Clone, Debug)]
pub enum WorkFailure {
    AsrFailed { kind: AsrFailureKind, message: String },
    AlignmentFailed { kind: AlignmentFailureKind, message: String, language: Lang },
    LanguageUnsupportedForAlignment { language: Lang },
    WorkerHangTimeout { kind: WorkerKind, elapsed: Duration },
}

pub enum AsrFailureKind {
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

// Note: there is no `EmptyOutput` variant. A whisper-rs result with
// zero segments is normal output — usually a silent chunk — and is
// represented as a `Transcript` with empty `text` and an elevated
// `no_speech_prob`. Treating empty output as a failure would convert
// every silent chunk into Event::Error and contradict the
// `no_speech_prob` field's semantics.

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

// Note: there is no `InvalidLang` type. `Lang::from_iso639_1` is a
// total function — every input string produces a `Lang`, with
// unknown codes going to `Lang::Other(s)`. See §4.4 for the
// canonicalisation invariant that keeps Eq/Hash consistent.

#[cfg(feature = "runner")]
pub enum RunnerError {
    WhisperContextLoad { source: /* whisper-rs error */ },
    WhisperPoolShutdown,
    /// Worker queue is saturated and `WhisperPoolConfig::block_on_full_queue`
    /// is false. The caller must drain transcripts before pushing more
    /// audio. See §6.4.2 for the contract on what state has been
    /// committed when this is returned.
    Backpressure { buffered: usize, cap: usize },
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

**Two distinct error channels.** `RunnerError` is for *runner-level* failures returned synchronously from `process_packet`, `signal_eof`, `drain`, `build`: model load failure, channel disconnects, backpressure, push-order violations. `WorkFailure` is for *per-chunk inference* failures surfaced asynchronously via `Event::Error { chunk_id, error: WorkFailure }` (drained by `poll_error`). The dispatch loop guarantees these channels stay decoupled — a single chunk's ASR or alignment failure never causes `process_packet` to error; only structural runtime failures do.

---

## 5. Sans-I/O core

The core is a single struct, `Transcriber`, wrapping the cut state machine, the dispatch state machine, and a sample buffer. Its public API has six push/inject methods and two poll methods.

### 5.1 Transcriber surface

```rust
// Transcriber is `Send` (movable across threads) but `!Sync`
// (every public mutating method takes `&mut self`). A consumer
// that wants to drive it from multiple threads must wrap it in
// `Mutex<Transcriber>` themselves; whispery does not provide
// internal synchronisation.
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

    /// Append audio samples to the buffer.
    ///
    /// Empty packets (`samples.is_empty()`) are accepted as no-ops
    /// when `delta_pts_out == 0`. The first call records the output
    /// timebase from `starts_at.timebase()`.
    ///
    /// Errors:
    /// - `PtsRegression` — `starts_at.pts()` is earlier than the
    ///   buffer's expected next-PTS (in output-PTS space).
    /// - `GapExceedsTolerance` — forward gap larger than the
    ///   configured `gap_tolerance_samples`.
    /// - `Backpressure` — buffered samples would exceed the cap.
    /// - `InconsistentTimebase` — `starts_at.timebase()` differs
    ///   from the timebase recorded on the first push.
    /// - `AfterEof` — `signal_eof()` was previously called.
    pub fn push_samples(
        &mut self,
        starts_at: mediatime::Timestamp,
        samples: &[f32],
    ) -> Result<(), TranscriberError>;

    /// Push a VAD segment into the cut state machine. VAD segments
    /// must be strictly monotonically increasing in `start_sample`;
    /// out-of-order or duplicate segments are rejected as
    /// `PtsRegression { kind: VadSegment }`.
    ///
    /// Errors:
    /// - `OutputTimebaseUnset` — no `push_samples` has been called yet,
    ///   so the cut state machine has no output timebase to anchor
    ///   against. Push at least one sample packet first.
    /// - `PtsRegression { kind: VadSegment }` — `seg.start_sample` is
    ///   not strictly greater than the previous VAD segment's
    ///   `end_sample`.
    /// - `AfterEof` — `signal_eof()` was previously called.
    pub fn push_vad_segment(
        &mut self,
        seg: VadSegment,
    ) -> Result<(), TranscriberError>;

    /// Mark the input stream as ended. Flushes the cut state
    /// machine, allowing any partial accumulated chunk to emit.
    /// Idempotent (calling twice is `Ok(())` on the second call).
    /// Calling before any `push_samples` is a no-op (`Ok(())`)
    /// since there is nothing to flush.
    ///
    /// Errors: never returns Err in v1; signature carries
    /// `Result<(), TranscriberError>` for forward compatibility.
    pub fn signal_eof(&mut self) -> Result<(), TranscriberError>;

    /// Recovers from a `GapExceedsTolerance`. Flushes the cut state
    /// machine, drains `cut_pending` into `in_flight`, clears the
    /// live SampleBuffer, re-anchors PTS, and preserves chunk_id
    /// continuity so already-in-flight chunks from before the gap
    /// still emit normally. See §5.4.1.
    ///
    /// Errors:
    /// - `AfterEof` — calling restart_at after signal_eof is
    ///   rejected; once a stream has been ended it cannot be
    ///   re-anchored. Construct a fresh Transcriber instead.
    pub fn restart_at(
        &mut self,
        starts_at: mediatime::Timestamp,
    ) -> Result<(), TranscriberError>;

    /// Non-mutating predicate: would the next push of `samples_len`
    /// audio samples plus `vad_count` VAD segments fit under the
    /// configured caps without returning Backpressure? Provided so
    /// callers that need to decide before pushing can do so without
    /// the TOCTOU race that consulting `buffered_samples()` directly
    /// would have. See §6.4.2 for the contract on what state is
    /// committed when push returns Backpressure.
    pub fn would_accept(&self, samples_len: usize, vad_count: usize) -> bool;

    // ── Inject side ─────────────────────────────────────────────

    /// Errors:
    /// - `UnknownChunk(chunk_id)` — `chunk_id` is not the id of any
    ///   record currently in `in_flight` (either never issued, or
    ///   already resolved to a Transcript / Error).
    pub fn inject_asr_result(
        &mut self,
        chunk_id: ChunkId,
        out: AsrResult,
    ) -> Result<(), TranscriberError>;

    /// Errors:
    /// - `UnknownChunk(chunk_id)` — same semantics as
    ///   `inject_asr_result`. Calling `inject_alignment_result` on a
    ///   chunk whose phase is not `AwaitingAlignment` is also
    ///   rejected as `UnknownChunk` (the chunk is no longer waiting
    ///   for alignment).
    pub fn inject_alignment_result(
        &mut self,
        chunk_id: ChunkId,
        out: AlignmentResult,
    ) -> Result<(), TranscriberError>;

    /// Errors:
    /// - `UnknownChunk(chunk_id)` — `chunk_id` is not in `in_flight`.
    pub fn inject_failure(
        &mut self,
        chunk_id: ChunkId,
        failure: WorkFailure,
    ) -> Result<(), TranscriberError>;

    // ── Poll side ───────────────────────────────────────────────
    pub fn poll_command(&mut self) -> Option<Command>;
    pub fn poll_event(&mut self) -> Option<Event>;

    /// Re-park the front of the command queue. Used by the runner's
    /// dispatch loop when a try_send returns Full and the command
    /// must be retried on the next drive iteration. **Visibility:
    /// `pub(crate)`** — the runner module is the only legitimate
    /// caller, and exposing this on the public surface would invite
    /// abuse. Crate-internal because the runner module lives in the
    /// same crate; out-of-tree consumers driving the state machine
    /// themselves do not need to re-park (their own command queue
    /// is theirs to manage).
    ///
    /// Must be called at most once per `poll_command` call (no FIFO;
    /// only the most-recently-popped command can be unpolled).
    pub(crate) fn unpoll_command(&mut self, cmd: Command);

    /// True iff every queue is empty: no buffered samples, no
    /// pending command/event, no in_flight chunks, no cut_pending
    /// entries. Pre-restart in-flight chunks (those still working
    /// through whisper or alignment) keep `is_idle()` false until
    /// they emit; `restart_at` does not synthetically clear them.
    pub fn is_idle(&self) -> bool;
    pub fn buffered_samples(&self) -> usize;
    pub fn output_timebase(&self) -> Option<mediatime::Timebase>;  // None until first push_samples

    /// Authoritative output-timebase PTS the buffer expects for the
    /// next contiguous `push_samples` call. Returns `None` before the
    /// first push (no anchor established yet). Strictly-contiguous
    /// callers SHOULD use this value instead of computing their own
    /// running sum, because per-packet `mediatime::Timebase::rescale_pts`
    /// truncates and a sequence of N per-packet rescales does not
    /// always equal one rescale of the cumulative sample count when
    /// the analysis-to-output ratio is non-integer (1/30001, NTSC,
    /// MPEG-TS 1/90000). Whispery's regression check is anchored on
    /// the cumulative form, so a caller summing per-packet rescales
    /// will eventually drift by ±1 PTS and trip a spurious
    /// `PtsRegression`. Conformant callers compute:
    ///
    /// ```ignore
    /// let starts_at = transcriber.next_expected_starts_at()
    ///     .expect("call after the first push_samples");
    /// transcriber.push_samples(starts_at, &packet)?;
    /// ```
    pub fn next_expected_starts_at(&self) -> Option<mediatime::Timestamp>;
}
```

`VadSegment` carries silero-native 16 kHz sample indices. Whispery accepts these as input (private fields with getters) and does the analysis-time → output-timebase conversion internally, so the caller never needs to do PTS arithmetic for VAD inputs.

```rust
pub struct VadSegment {
    start_sample: u64,    // 16 kHz analysis-frame index
    end_sample: u64,      // 16 kHz analysis-frame index, exclusive
}

impl VadSegment {
    /// Panics if `end_sample <= start_sample`. The strict
    /// inequality matters: zero-duration VAD segments (`end ==
    /// start`) would emit zero-length MergedChunks downstream,
    /// which break alignment and confuse downstream consumers.
    /// silero never produces zero-duration segments; the
    /// constructor surfaces programmer error at the boundary.
    ///
    /// `panic!` in `const fn` is stable on Rust ≥ 1.57; the
    /// crate's MSRV (≥ 1.85, inherited from siblings) covers this.
    pub const fn new(start_sample: u64, end_sample: u64) -> Self;

    pub const fn start_sample(&self) -> u64;
    pub const fn end_sample(&self) -> u64;
    pub const fn sample_count(&self) -> u64;       // end - start
}
```

Conversion from `silero::SpeechSegment`:

```rust
let v = VadSegment::new(s.start_sample(), s.end_sample());
```

(Whispery does *not* depend on the silero crate; example only.)

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

A direct port of WhisperX's `merge_chunks`, restated as an incremental state machine, with one extension: hard-splitting any single VAD segment longer than `chunk_size`. All internal arithmetic is in 16 kHz sample-index space (`SampleRange`); conversion to the output timebase happens at emission time via `SampleBuffer::samples_to_output_range`.

State:

```rust
struct Cut {
    chunk_size_samples: u64,    // chunk_size duration × 16 kHz (e.g., 30 s × 16k = 480_000)

    // The chunk currently accumulating (None when between chunks).
    current_start: Option<u64>,        // 16 kHz sample index, inclusive
    current_end: u64,                  // 16 kHz sample index, exclusive
    current_subs: Vec<SubRange>,
}

pub(crate) struct SubRange {
    pub range: SampleRange,             // 16 kHz indices
    pub origin: SubOrigin,
}

pub(crate) enum SubOrigin {
    /// Came directly from a VadSegment as pushed.
    Vad { vad_seq: u32 },               // monotonic counter assigned by Cut on push
    /// Result of hard-splitting a VadSegment longer than chunk_size.
    /// The full original VAD segment can be reconstructed by joining
    /// all SubRanges with the same `vad_seq`.
    HardSplit { vad_seq: u32, part: u8, total_parts: u8 },
}

/// Output of the cut state machine. Crate-private; the public
/// surface only sees the resulting Transcript.
pub(crate) struct MergedChunk {
    pub range: SampleRange,                  // 16 kHz indices
    pub subs: Vec<SubRange>,                  // 16 kHz indices, with provenance
}
```

`SubRange` carries the origin tag so downstream code (notably the alignment pipeline) can distinguish "this is one logical VAD segment" from "this is one slice of a hard-split long VAD segment." `Transcript.vad_segments()` exposes only the *logical* (joined-by-vad_seq) VAD segments, in output timebase, since indexer-side consumers care about original speech intervals; the hard-split fragments are an internal cut-stage detail.

Transitions:

- `push_segment(seg: VadSegment)`:
  1. Allocate a fresh `vad_seq` (monotonic counter). Compute `len = seg.end_sample() - seg.start_sample()`.
  2. **Pre-split overlong segments.** If `len > chunk_size_samples`, split into `n = ceil(len / chunk_size_samples)` parts using the per-index formula:
     ```
     start_i = seg.start_sample + (i as u64 * len) / n     // integer division
     end_i   = seg.start_sample + ((i+1) as u64 * len) / n // for i < n-1
     end_{n-1} = seg.end_sample                             // last absorbs remainder
     ```
     This guarantees every part has length `len/n` (rounded down) or `len/n + 1`; the maximum part length is `ceil(len / n) ≤ chunk_size_samples`. (Naive `floor(len/n)` per part with the last absorbing the remainder is **not** the contract: with `len=29, chunk_size=10, n=3` it produces 9, 9, 11, violating the strict bound.) Push each sub-range through steps 3–5 in order, with `SubOrigin::HardSplit { vad_seq, part: i, total_parts: n }`. Otherwise the segment becomes a single `SubRange { origin: SubOrigin::Vad { vad_seq } }`.
  3. If `current_start` is None: set it to the sub-range's start sample index, **and set `current_end = sub.start` so the `current_end >= current_start` invariant holds before step 4 inspects it.**
  4. If `(sub.end - current_start.unwrap()) > chunk_size_samples` *and* `current_end > current_start.unwrap()` (the second clause guards against the degenerate "first segment is itself overflowing" case where step 3 just initialised current_end to current_start):
     - Emit `MergedChunk { range: SampleRange::new(current_start.unwrap(), current_end), subs: take(current_subs) }`.
     - Reset: `current_start = Some(sub.start)`, `current_subs.clear()`. Re-set `current_end = sub.start` to maintain the invariant for the next iteration.
  5. Update `current_end = sub.end`, `current_subs.push(sub)`.
- `flush()` (called on EOF):
  - If `current_start` is Some: emit the trailing chunk, reset.

The state machine guarantees:

1. **Monotonicity.** Output `MergedChunk`s are non-overlapping and strictly ordered by start sample.
2. **Strict bound.** No emitted `MergedChunk.range` spans more than `chunk_size_samples`. The previous draft's bound (`chunk_size + max(seg_duration)`) was wrong; the pre-split rule with **equal-length** fragments enforces a true ≤ `chunk_size` ceiling, never violated. This matters because Whisper's encoder is hard-capped at 30 s; a chunk over 30 s is silently truncated by whisper.cpp.
3. **Equal-length hard-split contract.** The split rule is `ceil(len / chunk_size_samples)` equal-length parts (last may be ≤ the others by up to n-1 samples). An implementer producing "three 30 s parts plus a 1 s tail" violates this contract and §10's tests will fail.
4. **Provenance reconstruction.** From `MergedChunk.subs`, downstream code can:
   - Recover the union of speech intervals (logical VAD segments) by joining `SubRange`s sharing the same `vad_seq`.
   - Distinguish hard-split fragments from real silero segments by inspecting `SubRange.origin`.

This logic is purely arithmetic on `u64` sample indices; allocations are bounded by `current_subs`.

### 5.4 Sample buffer (`core/buffer.rs`)

```rust
pub(crate) struct SampleBuffer {
    /// Output timebase recorded from the first push_samples call.
    /// All output TimeRanges are constructed in this timebase.
    output_tb: mediatime::Timebase,
    /// PTS (in output_tb) of the stream's sample-index 0.
    /// **Immutable after the first push.** All sample-index ↔ output
    /// PTS conversions are anchored here, so trim()-induced
    /// recomputation never accumulates truncation error on
    /// non-integer-ratio rates (NTSC, MPEG-TS).
    base_pts_out_anchor: i64,
    /// Total samples ever appended to the stream (since the last
    /// restart_at). Monotonic; never decremented. The Vec<f32>
    /// holds samples in [buffer_drop_offset, absolute_sample_offset).
    absolute_sample_offset: u64,
    /// Samples dropped by trim(). Monotonic; never decremented.
    /// The buffer's first live sample's stream-index is this value.
    buffer_drop_offset: u64,
    samples: Vec<f32>,
    cap: usize,
    gap_tolerance_samples: u64,
}
```

Operations:

- `append(starts_at: Timestamp, packet: &[f32]) -> Result<(), TranscriberError>`:
  - On first call, record `output_tb = starts_at.timebase()`, set `base_pts_out_anchor = starts_at.pts()`, set `absolute_sample_offset = 0` and `buffer_drop_offset = 0`.
  - On subsequent calls, require `starts_at.timebase() == output_tb` (or `InconsistentTimebase`).
  - **Compute expected output PTS for the next contiguous sample**:
    ```rust
    let expected_pts_out = base_pts_out_anchor
        + Timebase::rescale_pts(
              absolute_sample_offset as i64,
              ANALYSIS_TIMEBASE,
              output_tb,
          );
    let delta_pts_out = starts_at.pts() - expected_pts_out;
    ```
  - **Run the regression / gap check in output-PTS space**, not 16 k space — this avoids the truncation-induced false-positive PtsRegressions that occur on non-integer-ratio output timebases (e.g., NTSC) when caller pushes are strictly contiguous in their own timebase but rescale-and-back round-trip introduces ±1 PTS noise.
    - `delta_pts_out < 0`: `PtsRegression`.
    - `delta_pts_out == 0`: contiguous. Append packet, advance `absolute_sample_offset`.
    - `delta_pts_out > 0`: forward gap. Convert to samples *only for the zero-fill width*: `delta_samples = Timebase::rescale_pts(delta_pts_out, output_tb, ANALYSIS_TIMEBASE)`. If `delta_samples > gap_tolerance_samples` return `GapExceedsTolerance`; else zero-fill `delta_samples` samples, then append packet. Advance `absolute_sample_offset` by `delta_samples + packet.len() as u64`.
  - After append, if `samples.len() > cap`, return `Backpressure { buffered, cap }`.

- `extract(range: SampleRange) -> Arc<[f32]>`:
  - `range` is in stream-relative 16 kHz sample indices (i.e., absolute, not relative to the live buffer). Slice
    `samples[(range.start - buffer_drop_offset) as usize .. (range.end - buffer_drop_offset) as usize]`,
    copy into `Arc<[f32]>`. The original buffer is not mutated.

- `samples_to_output_range(range: SampleRange) -> TimeRange`:
  - **Always rescales from the immutable anchor**, so truncation error is at most ±1 PTS regardless of how many trims have occurred:
    ```rust
    let start_out = base_pts_out_anchor
        + Timebase::rescale_pts(range.start as i64, ANALYSIS_TIMEBASE, output_tb);
    let end_out = base_pts_out_anchor
        + Timebase::rescale_pts(range.end as i64, ANALYSIS_TIMEBASE, output_tb);
    TimeRange::new(start_out, end_out, output_tb)
    ```

- `trim_to(low_water_samples: u64)`:
  - Drop `samples[0..(low_water_samples - buffer_drop_offset) as usize]` and advance `buffer_drop_offset` to `low_water_samples`. **`base_pts_out_anchor` is not touched** (it's the immutable stream-zero anchor). This is what makes drift-free PTS arithmetic possible across many trims.

Forward-gap tolerance addresses real-world ffmpeg behaviour: container PTS offsets, cross-file stitching, occasional packet drops, resample boundaries. The default `gap_tolerance_samples = 3 200` (200 ms at 16 kHz) silently zero-fills typical micro-gaps; anything larger is surfaced for explicit caller handling. Zero-fill is correct because the VAD stream is independent of the audio stream — silence-filled samples will not produce VAD speech segments and will not be cut into a Whisper chunk.

#### 5.4.1 Recovery from `GapExceedsTolerance`

`Transcriber::restart_at(starts_at: Timestamp) -> Result<(), TranscriberError>` is the explicit recovery path. It:

1. **Drain `cut_pending` before clearing the buffer.** `cut_pending` holds `(chunk_id, MergedChunk)` descriptors whose `SampleRange` indices are still in the old anchor's frame; if those entries survived into the new frame, the next `trim()` would compute a stale low-water and the next promotion attempt would `extract` against indices that no longer exist in the (cleared) buffer — hard panic. So restart_at first promotes every queued entry: for each `(chunk_id, merged_chunk)` in `cut_pending`, extract `samples` from the buffer (still in old-frame indexing), call `samples_to_output_range(merged_chunk.range)` against the **old** anchor to get a correctly-anchored `TimeRange`, build a `ChunkRecord` (phase `AwaitingAsr`), insert into `in_flight`, and enqueue `Command::RunAsr`. The drain-on-restart path is allowed to temporarily exceed `max_in_flight` — restart_at is a one-time event and bounded by however many entries were already queued; the alternative (dropping the chunks) would lose transcription work the caller had already paid for.
2. Flushes the cut state machine (`Cut::flush()`), emitting any partial `MergedChunk` it was accumulating. The flush goes through the same path as step 1's drain (the partial chunk becomes one more queued entry that gets promoted to `in_flight`).
3. Clears the live SampleBuffer's `Vec<f32>` (it must be empty for the next push to start a fresh contiguous segment). Pre-restart in-flight chunks already hold their audio in their own `Arc<[f32]>`s — both the chunks that were already in_flight before the call AND the chunks just promoted from `cut_pending` in step 1.
4. **Re-anchors with a clean slate.** Sets:
   - `base_pts_out_anchor = starts_at.pts()`
   - `absolute_sample_offset = 0`
   - `buffer_drop_offset = 0`

   The next `push_samples(starts_at, packet)` call computes `expected_pts_out = starts_at.pts() + rescale(0, …) = starts_at.pts()`, matches the caller's PTS exactly, and proceeds as a fresh contiguous start. Resetting both offsets to 0 (rather than carrying the prior cumulative `absolute_sample_offset` forward) is what makes `delta_pts_out = 0` on the first post-restart push; carrying them forward as the v3 draft did would have triggered `PtsRegression` on every restart.

5. **chunk_id continues monotonically.** `next_chunk_id` is *not* reset. In-flight chunks 5, 6, 7 from before the restart will still emit normally via the in-order path; their stored `range` PTS values were computed against the old anchor at extraction time and are correct as-is. The first chunk produced after restart_at has a chunk_id one larger than the last pre-restart chunk (which itself may have been a step-1 drain or a step-2 flush).
6. **Trim's low-water is computed from `cut_pending` only** (not `in_flight`). Once a chunk is `in_flight`, its audio lives in its own `Arc<[f32]>` and is decoupled from the buffer; the buffer only needs to hold samples for chunks not yet extracted. After restart_at, `cut_pending` is empty (steps 1+2 drained it), so trim's low-water is the new buffer's high-water (zero) — the entire empty buffer is eligible for drop. This eliminates any risk of stale-anchor PTS values poisoning trim's computation across a restart.

`restart_at` is the *only* public API affordance for recovering from a gap; `signal_eof` is one-way and does not reset the buffer.

The buffer is a flat `Vec<f32>` with periodic `drain(0..n)` on trim. For our packet rates (≪ 1 GB/s), the memmove cost is dominated by whisper inference time. A circular ring buffer is a future optimisation.

Internal types:

```rust
pub(crate) struct SampleRange {
    start: u64,    // 16 kHz analysis-frame index, inclusive
    end: u64,      // 16 kHz analysis-frame index, exclusive
}
```

`SampleRange` never crosses the public surface; only `TimeRange` (in output timebase) does.

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
  - Compute `low_water_samples = min`(start sample-index of every chunk in `cut_pending`). **`in_flight` is excluded** because in-flight chunks have already had their samples extracted into `Arc<[f32]>` (decoupled from the buffer); the buffer only retains data for not-yet-extracted chunks.
  - If `cut_pending` is empty, `low_water_samples` advances to `absolute_sample_offset` (the buffer's high-water).
  - Call `SampleBuffer::trim_to(low_water_samples)`.
  - If `in_flight.len() < max_in_flight` and `cut_pending` is non-empty: promote the front of `cut_pending` (extract samples from SampleBuffer, build ChunkRecord, enqueue `Command::RunAsr`).

Invariants:

1. **In-order emission.** `Event::Transcript` and `Event::Error` are produced in strict `chunk_id` order regardless of which inference worker finishes first. This is a contract; downstream BM25/FTS write order, future cross-file ranking, and any time-aligned join with diarization can rely on it.
2. **chunk_id allocation is monotonic across success and failure.** Every `MergedChunk` emitted by Cut allocates exactly one `chunk_id`. Failures produce `Event::Error` carrying that same `chunk_id`; the next chunk's id is one larger. Consumers can rely on chunk_id sequences having no gaps (every id is either a Transcript or an Error).
3. **flush-before-trim contract.** Every `inject_*_result` and `inject_failure` path follows the same shape: build the per-chunk outcome, set `phase = Ready { transcript } | FailedReady { failure }`, call `flush_in_order_events()`, *then* call `trim()`. This ordering matters because `trim()` removes records from `in_flight` and recomputes the low-water mark; flushing first ensures that any newly-emit-eligible chunks are surfaced before their state is dropped. Tests exercise this by injecting results out of order and asserting that `flush_in_order_events` runs before any `in_flight.remove(...)` on every code path.
4. **Bounded memory under back-pressure.** `cut_pending` entries hold only descriptors; they cost O(1) audio. The single audio back-pressure path is `buffer_cap_samples`, which trips `push_samples` and lets the caller pause ingest.
   **Exception:** `restart_at` (§5.4.1 step 1) drains the entire `cut_pending` queue into `in_flight` synchronously, which may transiently push `in_flight.len()` above `max_in_flight`. The exceedance is bounded by the size of `cut_pending` at restart time (itself bounded by `max_queued_chunks`) and decays as the worker pool drains the queue normally. Trim's promotion guard (`if in_flight.len() < max_in_flight`) is suspended for the duration of the drain — it does not gate restart-time promotion. This is the only path that breaks the invariant; all steady-state code respects it.
5. **No deadlock.** As long as workers are alive and inference completes, `flush_in_order_events()` always advances; promotions from `cut_pending` happen as soon as a slot frees. The runner's dispatch loop (§6.4.1) defends against the inline-send saturation deadlock by always draining results before retrying sends.

### 5.6 Command and Event

The core's command and result types deliberately use ASR-prefixed names rather than Whisper-prefixed ones. This is the load-bearing piece of the §3.4 backend invariant: a future swap from whisper-rs to candle-whisper or a CTranslate2 binding only changes the runner's interpretation of these types, never the types themselves or the state machine that produces and consumes them. Each `AsrParams` field corresponds to a knob exposed by whisper-rs 0.13.x's `FullParams` *or* is consumed by the runner's own retry loop; nothing lives here that the runner can't faithfully drive.

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
    },
}

pub enum Event {
    Transcript(Transcript),
    Error { chunk_id: ChunkId, error: WorkFailure },
}

/// Universal ASR knobs. Backend-agnostic: contains no whisper-rs
/// types and no whisper.cpp-specific fields. Maps cleanly to
/// whisper-rs 0.13.x's FullParams setters and the runner's own
/// temperature retry loop.
pub struct AsrParams {
    /// Language hint passed to FullParams::set_language. None means
    /// auto-detect (FullParams::set_detect_language(true)).
    pub language_hint: Option<Lang>,

    /// Sampling strategy. The runner constructs a fresh FullParams
    /// per chunk via FullParams::new(strategy.into_whisper_rs()).
    pub strategy: SamplingStrategy,

    /// Initial decoding temperature; first attempt of the runner's
    /// retry ladder. Forwarded to whisper.cpp via the strategy's
    /// implicit temperature (Greedy uses 0.0 by default; the
    /// runner's ladder rebuilds FullParams with adjusted strategy
    /// for each retry).
    pub initial_temperature: f32,

    /// Increment applied to temperature on each retry attempt.
    /// Default 0.2 (matches WhisperX default ladder).
    pub temperature_increment: f32,

    /// Maximum total attempts (initial + retries). Default 6.
    /// The retry ladder is implemented in the runner because
    /// whisper-rs does not expose a temperature schedule; each
    /// retry is a fresh state.full() call.
    pub max_attempts: u8,

    /// log_prob threshold; on a result with avg_logprob below
    /// this, the runner moves to the next temperature. Default
    /// -1.0 (WhisperX default).
    /// **Layered-ladder suppression.** whisper.cpp internally
    /// implements its own temperature ladder via
    /// `temperature_inc` (loop step) and `max_decoding_failures`
    /// (secondary safeguard). The runner pins
    /// `temperature_inc = 0.0` in `full_params_from` (§5.6),
    /// which alone fully disables the internal ladder — the
    /// loop `for t = initial; t <= 1.0 + 1e-6; t += inc` iterates
    /// exactly once. `max_decoding_failures = 1` is set
    /// best-effort as a secondary safeguard. The runner's outer
    /// ladder is the sole authority: exactly one decoding attempt
    /// happens per `state.full()` call, at the runner-supplied
    /// temperature.
    ///
    /// If the active whisper-rs version is found (during the §13.1
    /// verification) to lack `set_temperature` or
    /// `set_temperature_inc`, the runner module must be built with
    /// an explicit alternative implementation that omits those
    /// calls — this is a compile-time situation, not a runtime
    /// fallback. In that alternative path the AsrParams temperature
    /// fields become advisory and whisper.cpp's internal ladder
    /// runs at its default settings.
    pub log_prob_threshold: f32,

    /// Compression-ratio threshold; on a result with compression
    /// ratio above this (degenerate / repetitive output), the
    /// runner moves to the next temperature. Default 2.4.
    pub compression_ratio_threshold: f32,

    /// no_speech threshold for marking a chunk as silence (the
    /// chunk emits with empty text and `no_speech_prob` above
    /// this is reported in `Transcript.no_speech_prob`). Default 0.6.
    pub no_speech_threshold: f32,

    /// Forwarded to FullParams::set_no_context. NOTE the polarity:
    /// this matches whisper-rs's setter semantics (true = do not use
    /// past transcription as initial prompt). Default true; v1 never
    /// reuses cross-chunk context (each WhisperState::full call is
    /// independent regardless of this setting). The flag controls
    /// only whisper.cpp's intra-chunk decoder behaviour where it
    /// uses prior segment text as a prompt for the next ~30-token
    /// segment within the same encoder call.
    pub no_context: bool,

    /// Forwarded to FullParams::set_suppress_blank. Default true.
    pub suppress_blank: bool,

    /// Forwarded to FullParams::set_suppress_nst (suppress
    /// non-speech tokens). Default false.
    pub suppress_non_speech_tokens: bool,

    /// Forwarded to FullParams::set_initial_prompt. Default None.
    pub initial_prompt: Option<smol_str::SmolStr>,

    /// Forwarded to FullParams::set_n_threads (whisper.cpp's
    /// in-call thread count, separate from the runner's worker
    /// pool size). Default 1; the runner's parallelism comes from
    /// multiple WhisperStates running concurrently on different
    /// chunks, so over-subscribing in-call threads is wasteful.
    /// Type matches whisper-rs's setter parameter
    /// (`std::os::raw::c_int`).
    /// **SemVer note:** `c_int` is `i32` on every platform Rust
    /// currently supports, so this is a no-op alias today. The
    /// alias exists to track whisper-rs's signature exactly; if
    /// whisper-rs ever changes its setter parameter type (or if
    /// `c_int` becomes platform-variant in some future target),
    /// that propagates as a breaking SemVer change here, and the
    /// fix is a coordinated whispery major release.
    pub n_threads: std::os::raw::c_int,
}

pub enum SamplingStrategy {
    Greedy { best_of: i32 },
    BeamSearch { beam_size: i32, patience: f32 },
}

/// Result of one chunk's ASR inference.
pub struct AsrResult {
    pub text: smol_str::SmolStr,
    pub language: Lang,           // the detected (or hint-confirmed) language
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,         // final temperature used after fallback retries
}

#[cfg(feature = "alignment")]
pub struct AlignmentResult {
    pub words: Vec<Word>,
}
```

**Notes on what's *not* in this list, and why.**

- No `suppress_tokens: Vec<TokenId>` field: whisper-rs 0.13.x exposes only `set_suppress_blank` and `set_suppress_nst` — no arbitrary token-id-array setter. If a future caller needs custom token suppression, this becomes a runner-only extension.
- No `temperature_schedule: SmallVec<[f32; 6]>` field: whisper-rs does not surface a per-temperature ladder API. The runner implements the ladder by re-constructing `FullParams` with adjusted temperature inputs and calling `state.full()` per attempt; the parameters expressed as `(initial, increment, max_attempts)` faithfully describe that loop.
- No `AsrTokenHint` / DTW token timestamps: v1 does not enable DTW (§1.5); wav2vec2 forced alignment does not need per-token seeds from Whisper. If the user later wants DTW as an alignment fallback, the addition is `AsrResult.dtw_token_timestamps: Vec<DtwToken>` plus a builder flag — additive.
- No GPU-backend enum: whisper-rs selects backends via Cargo features (cuda/metal/vulkan/hipblas/openblas/coreml). Runtime `gpu_device: i32` is the only knob whisper-rs exposes and lives on `WhisperPoolConfig`, not on `AsrParams`.

The runner's job is to translate `AsrParams` into `FullParams` (`runner/whisper_pool.rs`):

```rust
fn full_params_from(
    params: &AsrParams,
    attempt_temperature: f32,
    abort_flag: Arc<AtomicBool>,
) -> FullParams<'static, 'static> {
    let strategy = match params.strategy {
        SamplingStrategy::Greedy { best_of } =>
            whisper_rs::SamplingStrategy::Greedy { best_of },
        SamplingStrategy::BeamSearch { beam_size, patience } =>
            whisper_rs::SamplingStrategy::BeamSearch { beam_size, patience },
    };
    let mut p = FullParams::new(strategy);
    p.set_n_threads(params.n_threads);
    p.set_no_context(params.no_context);
    p.set_suppress_blank(params.suppress_blank);
    p.set_suppress_nst(params.suppress_non_speech_tokens);
    if let Some(lang) = &params.language_hint {
        p.set_language(Some(lang.as_str()));
    } else {
        p.set_detect_language(true);
    }
    if let Some(prompt) = &params.initial_prompt {
        p.set_initial_prompt(prompt.as_str());
    }
    p.set_print_special(false);
    p.set_print_progress(false);
    p.set_print_realtime(false);
    p.set_print_timestamps(false);

    // Disable whisper.cpp's internal temperature ladder so the
    // runner's outer ladder is the sole authority. With temperature_inc
    // pinned to 0.0, whisper.cpp's internal loop
    //   for t = initial; t <= 1.0 + 1e-6; t += temperature_inc
    // iterates exactly once at the runner-supplied temperature.
    //
    // set_temperature(...) and set_temperature_inc(...) are confirmed
    // present in whisper-rs 0.13.x. set_max_decoding_failures(...) is
    // a belt-and-braces secondary safeguard: it only matters when
    // temperature_inc > 0, so even if the active whisper-rs version
    // does not expose it, temperature_inc = 0.0 alone fully disables
    // the internal ladder. Implementations should treat the
    // max_decoding_failures call as best-effort — if the §13.1
    // verification finds the setter is absent, the runner is built
    // (cfg-gated or feature-gated) with that line omitted; behaviour
    // is unchanged. The AsrParams temperature fields become advisory
    // *only* if both temperature setters disappear from whisper-rs,
    // which is a compile-time issue requiring an explicit code-path
    // alternative, not a runtime fallback.
    p.set_temperature(attempt_temperature);
    p.set_temperature_inc(0.0);
    p.set_max_decoding_failures(1);  // best-effort; primary disable is temperature_inc=0

    // Wire worker hang protection. The abort flag is flipped by a
    // separate watchdog thread (or an on-worker check against the
    // job start time) when the per-job timeout expires.
    p.set_abort_callback_safe(move || abort_flag.load(Ordering::Relaxed));
    p
}
```

**Canonical context construction.** The runner builder accepts a pre-constructed `WhisperContext` (so callers control flash_attn, DTW, model path, GPU device explicitly):

```rust
let ctx = whisper_rs::WhisperContext::new_with_params(
    "path/to/ggml-base.en.bin",
    whisper_rs::WhisperContextParameters::default(),
)?;
let mt = ManagedTranscriber::builder(ctx)
    .with_alignment(alignment_set)
    .build()?;
```

This is the only place in the public API that names `whisper_rs` types directly; it stays in the runner.

This translation lives entirely in the runner; the core never names whisper-rs.

`AsrParams` defaults are set by the runner's builder. Per-chunk override of `AsrParams` is supported: `ManagedTranscriber::process_packet` accepts an optional `AsrParamsOverride` (sparse `Option<T>` fields layered onto the runner's defaults). This is how callers supply per-call language hints without re-building the runner.

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
    ///
    /// **Contract on `vad_segments`:** segments must be in strictly
    /// monotonically increasing `start_sample` order, and
    /// `vad_segments[i].end_sample() <= vad_segments[i+1].start_sample()`
    /// (no overlap, no duplicates). Violations are surfaced as
    /// `RunnerError::Transcriber(TranscriberError::PtsRegression {
    /// kind: PushKind::VadSegment, .. })` from the underlying
    /// state-machine push.
    ///
    /// **Empty packet (`samples.is_empty()`):** accepted as a no-op;
    /// the underlying `push_samples` returns `Ok(())` immediately
    /// when `delta_pts_out == 0` (i.e., `starts_at` matches
    /// `next_expected_starts_at`). VAD segments in the same call
    /// are still pushed.
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
/// chunk produced from this packet. Fields cover the per-call
/// adjustments callers most commonly need; bulk re-tuning is done
/// via the builder's `asr_params(...)`.
pub struct AsrParamsOverride {
    pub language_hint: Option<Option<Lang>>,
    pub strategy: Option<SamplingStrategy>,
    pub initial_temperature: Option<f32>,
    pub initial_prompt: Option<Option<smol_str::SmolStr>>,
}
```

The builder's `build()` returns a `ManagedTranscriber` with worker threads spawned and channels wired. Internally, `with_alignment` flips the core's `word_alignment` flag and stashes the `AlignmentSet` for the alignment worker.

### 6.2 WhisperPool

```rust
pub struct WhisperPoolConfig {
    pub worker_count: usize,
    pub model_path: PathBuf,
    /// Forwarded to WhisperContextParameters::use_gpu. Default false.
    /// When true, whisper-rs uses whichever backend was selected at
    /// crate compile time (cuda/metal/vulkan/hipblas/openblas/coreml
    /// Cargo features); there is no runtime backend enum.
    pub use_gpu: bool,
    /// Forwarded to WhisperContextParameters::gpu_device. Default 0.
    /// Single-GPU index; whisper-rs does not expose multi-GPU
    /// dispatch.
    pub gpu_device: i32,
    /// Forwarded to WhisperContextParameters::flash_attn. Default
    /// false. Mutually exclusive with DTW (which is not enabled in v1).
    pub flash_attn: bool,
    /// Queue cap before process_packet blocks (when
    /// `block_on_full_queue=true`) or returns Backpressure.
    pub max_queued_chunks: usize,
    /// Block process_packet when work_tx is full. Default true.
    /// Set false to surface RunnerError::Backpressure for caller-side
    /// pacing. See §6.4 for the contract on what state has been
    /// consumed when Backpressure is returned.
    pub block_on_full_queue: bool,

    /// Maximum time the saturation wait (§6.4.1) blocks on
    /// `Select::ready_timeout` before spinning. Acts as a safety
    /// timer for the case where a worker channel becomes ready
    /// without a successful readiness wake. Default 10 ms.
    pub dispatch_idle_poll: Duration,
}

struct WhisperPool {
    ctx: Arc<WhisperContext>,        // shared model; whisper-rs is officially thread-safe
    workers: Vec<JoinHandle<()>>,
    work_tx: crossbeam_channel::Sender<AsrWorkItem>,
    result_tx: crossbeam_channel::Sender<(ChunkId, Result<AsrResult, WorkFailure>)>,
}

struct AsrWorkItem {
    chunk_id: ChunkId,
    samples: Arc<[f32]>,
    params: AsrParams,
    /// Per-job timeout sourced from the runner builder's
    /// `worker_timeouts(asr, _)` call. Stamped on the AsrWorkItem
    /// at dispatch time so each in-flight chunk carries its own
    /// budget; the worker thread feeds this into the abort_flag
    /// watchdog wired in `full_params_from`.
    asr_timeout: Duration,
}
```

**Worker count default.**

- **CPU backend** (no `cuda`/`metal`/`vulkan`/`hipblas` features): `max(1, num_cpus::get_physical() / 2)` — leaves room for ffmpeg, silero, soundevents, lancedb, and the alignment worker. Worker concurrency translates directly to throughput on CPU.
- **GPU backend** (cuda/metal/vulkan/hipblas/coreml feature active): default `1`. whisper.cpp serialises on a single GPU regardless of how many `WhisperState`s are running concurrently — additional workers consume memory without adding throughput. Indexing pipelines that need parallel GPU inference need multi-GPU hardware *and* compile-time selection of a backend that supports per-stream isolation, which is out of scope for v1.

The runner detects the active backend at compile time via `cfg!(feature = "...")` and picks the appropriate default; the user can override via `WhisperPoolConfig::worker_count` if they have a specific reason.

**Worker structure.** whisper-rs 0.13.x's documentation states:

> "Because the library is thread-safe, contexts can be shared across threads, while states are used to manage individual transcription tasks."

Each worker owns its own `WhisperState` (via `WhisperContext::create_state`); the context is shared via `Arc<WhisperContext>` across all workers. `WhisperState` is owned (no lifetime parameter) so it moves into worker threads cleanly.

Each worker runs a loop:

```rust
loop {
    let job = recv work or break;
    let state = self.state.lock();    // per-worker state, no contention
    let result = run_with_temperature_ladder(state, &job, &self.ctx);
    result_tx.send((job.chunk_id, result));
}
```

`run_with_temperature_ladder` is the runner-level retry loop:

```rust
fn run_with_temperature_ladder(state, job, ctx) -> Result<AsrResult, WorkFailure> {
    let p = &job.params;
    let mut temperature = p.initial_temperature;
    let abort_flag = job.abort_flag.clone();   // shared with the watchdog
    for attempt in 0..p.max_attempts {
        // full_params_from disables whisper.cpp's internal ladder
        // (temperature_inc=0, max_decoding_failures=1) and pins this
        // attempt's temperature, so each state.full() call is one
        // decoding attempt at exactly `temperature`.
        let full = full_params_from(p, temperature, abort_flag.clone());
        let _outcome = state.full(full, job.samples.as_ref())?;
        let logprob = compute_avg_logprob(state);
        let cratio  = compute_compression_ratio(state);
        if logprob >= p.log_prob_threshold && cratio <= p.compression_ratio_threshold {
            return Ok(build_asr_result(state, temperature));
        }
        temperature += p.temperature_increment;
    }
    Err(WorkFailure::AsrFailed { kind: AsrFailureKind::AllTemperaturesFailed, message: ... })
}
```

The runner owns the temperature ladder because whisper-rs's `FullParams` doesn't surface a per-temperature schedule API; each retry rebuilds `FullParams` (with the next temperature, internal ladder disabled) and calls `state.full()` again. The runner's outer ladder is the sole authority — exactly one decoding attempt happens per `state.full()` call. This is how WhisperX's ladder works internally; we replicate it because we need control over which temperature each attempt actually uses.

**Memory implication.** With shared `Arc<WhisperContext>`, model weights load once; per-worker memory is dominated by the `WhisperState`'s decoder workspace (~10–30 MiB depending on model size). Default 4 CPU workers × 20 MiB ≈ 80 MiB working memory plus model weights (75 MiB tiny – 3 GiB large). On GPU defaults (`worker_count = 1`), only one state's working memory exists.

**Dispatch loop and back-pressure interactions.** The dispatch loop runs *inline* on the caller's thread inside `process_packet` and `poll_transcript`. See §6.4 for how it avoids the saturation deadlock that an unconditional inline send would cause.

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
    /// Vocab id of the CTC blank token. Read from the wav2vec2
    /// tokenizer's special-tokens map at `Aligner::from_paths` time
    /// (HF tokenizers expose this via `Tokenizer::token_to_id("<pad>")`
    /// or the model's `tokenizer_config.json::pad_token`). If the
    /// model uses a non-standard blank token name, the caller can
    /// override via a future `Aligner::with_blank_token_id` builder
    /// method; v1 reads the standard `<pad>` / `[PAD]` convention.
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

#### 6.3.1 Lookup order and `Any` semantics

For a chunk with detected language `L`, the alignment worker looks up:

1. `AlignerKey::Lang(L)` — explicit registered aligner for the language.
2. `AlignerKey::Any` — multilingual fallback aligner (typically a multilingual XLSR / MMS model). Used **only for registry misses**, never as a recovery path after a registered aligner failed.
3. Apply `fallback`: `SkipChunk` (emit empty `words`) or `Error` (emit `LanguageUnsupportedForAlignment`).

**Failure on a registered aligner does NOT silently fall through to `Any`.** If `Lang(L)` is registered but its `Aligner::align` returns `WorkFailure::AlignmentFailed`, the failure is surfaced via `Event::Error` for that chunk; the `Any` aligner is not consulted. Silent fallback would mask data-quality bugs (e.g., the EN aligner crashing on CJK characters because language detection misfired) and produce systematically wrong word ranges without any signal to the indexer.

If callers want a "try registered, then `Any` on any failure" behaviour, it has to be implemented at registry-construction time (e.g., wrap the language-specific aligner with a fallback shim that internally retries against the multilingual aligner before returning). Whispery's built-in semantics are strict: `Any` is the no-aligner fallback, not the failed-aligner fallback.

#### 6.3.2 Alignment algorithm (silence-aware, normalisation-aware)

WhisperX's alignment quality story has three load-bearing pieces beyond the textbook CTC algorithm: text normalisation, surface-form recovery, and silence-handling. v1 implements all three.

For each chunk with non-empty text:

0. **Mask non-speech regions.** Build `samples_for_aligner` as a copy of `samples` with sample positions outside the union of *logical* `sub_segments` (joined by `vad_seq` per §5.3) zeroed. wav2vec2 distributes near-all probability to the blank token in long silence regions; CTC Viterbi paths are robust under this only if silence is *uniformly* silent. Zero-masking ensures non-speech regions don't contribute spurious phoneme probabilities and don't smear word boundaries onto silence.
1. **Normalise text.** Run the language's `TextNormalizer` to produce `NormalizedText { normalized, original_words }`. Normalisation lowercases, strips punctuation, expands contractions per the language's rules, and produces `original_words: Vec<&str>` — original-surface-form word slices in normalised-word-index order. The number of normalised words is `n = original_words.len()`.
2. **Tokenise.** Tokenise `normalized` against the wav2vec2 vocab to produce `Y = [t_0, t_1, ..., t_{m-1}]` (vocab indices, with `m` typically larger than `n` because tokens are sub-word). Track which normalised-word index each token belongs to in `word_idx_per_token: Vec<Option<usize>>` of length `m`. wav2vec2 vocabularies include word-delimiter tokens (`|`), unknown tokens (`<unk>`), and special markers; these have no natural word index and get `None`. Step 7 skips frames mapped to `None` slots when accumulating per-word state.
3. **Encode.** Run `session` over `samples_for_aligner` (reshaped to wav2vec2's expected input shape). Output is logits `(T, V)`.
4. **Log-softmax** along V to get log-probabilities.
5. **CTC lattice.** Build the standard CTC alignment lattice over `(T, 2|Y|+1)` (interspersed with blanks).
6. **Viterbi.** Run highest-probability monotonic alignment of `Y` to `T`. If no valid path exists, return `AlignmentFailureKind::NoAlignmentPath`.
7. **Per-word frame ranges into a sparse vector.**
   - Allocate `per_word: Vec<Option<(u32 /* start_frame */, u32 /* end_frame */, f32 /* logprob_sum */, u32 /* frame_count */)>> = vec![None; n]`.
   - Walk the Viterbi path frame by frame. **Skip blank-emitting frames** (CTC blank token contributes no per-word state) and **skip frames whose mapped token index `tok` has `word_idx_per_token[tok] == None`** (delimiter / `<unk>` / special). For other emitting frames, the corresponding normalised-word index is `w = word_idx_per_token[tok].unwrap()`. Update `per_word[w]`: open the entry on first sight, extend `end_frame`, accumulate logprob.
   - **Words whose audio fell entirely inside the silence-mask region get no emitting frames and remain `None`.** This is the M4 indexing fix: the previous draft assumed alignment produces exactly `n` ordered words, which fails when zero-masking drops some. With the sparse vector, the per-word index `w` always references the right normalised word, and dropped/special tokens never index out of bounds.
8. **Compose Word entries.** For each `(i, slot)` in `per_word.iter().enumerate()`:
   - If `Some((sf, ef, lp_sum, lp_n))`: build `Word { text: original_words[i].into(), range: frames_to_output_range(sf, ef), score: exp(lp_sum / lp_n as f32) }`.
   - If `None`: skip the word (it had no audio support — most often because it landed in a silence-masked region). The dropped word is *not* added to `words`. The total chunk text on `Transcript.text` still contains the word, just its per-word range entry is absent.
9. **Surface form preserved.** Each emitted `Word.text` is the original surface form `original_words[i]` (with punctuation and casing as Whisper produced it), not the normalised form. This is the v1 invariant for `Transcript.text` vs `Word.text`.

`frames_to_output_range(start_frame, end_frame)` is `SampleBuffer::samples_to_output_range(SampleRange::new(chunk_first_sample + start_frame as u64 * hop_samples, chunk_first_sample + end_frame as u64 * hop_samples))`, where `chunk_first_sample` is the chunk's first 16 kHz sample index in stream coordinates. This produces a `TimeRange` in the caller's output timebase.

#### 6.3.3 Concurrency: v1 is sequential; the `Mutex<Aligner>` justification

v1 ships **one alignment worker** in the `AlignmentPool`. Alignment is therefore sequential across chunks, regardless of language. With Whisper running on N workers, alignment will be the throughput bottleneck only when alignment-time-per-chunk × throughput exceeds whisper-time-per-chunk × throughput / N — which is unusual on indexing workloads but possible.

**Why `Mutex<Aligner>` exists in v1 even with one worker.** `ort::Session::run` requires `&mut self` (the session's tensor-allocation arenas mutate during inference). The `AlignmentSet` is owned by `ManagedTranscriber` (which lives on the caller's thread) but its `Aligner`s are accessed from the alignment worker thread; sharing across threads requires `Arc<AlignmentSet>` and a per-language interior-mutability mechanism. `Mutex<Aligner>` is the simplest correct choice: cheap to acquire when uncontended (which it always is in the v1 single-alignment-worker case), and naturally extends to multi-worker without API changes.

**What `Mutex<Aligner>` does NOT enable.** It does not by itself enable parallel alignment of the same language: even with multiple workers, only one can hold a given `Mutex<Aligner>` at a time. Two paths exist for v2 if alignment becomes the bottleneck:

- **Cross-language parallel only.** N alignment workers each grab the relevant `Mutex<Aligner>` per chunk; same-language chunks serialise behind one mutex. Easy.
- **Within-language parallel.** Replace `Mutex<Aligner>` with `Vec<Aligner>` (one Session per worker per language). Multiplies model memory by parallelism factor. Also requires verifying `ort::Session::run` thread-safety for the chosen execution provider; CUDA EP serialises internally on a single GPU regardless.

Neither is implemented in v1. The §11 throughput math accounts for sequential alignment.

### 6.4 Concurrency model and saturation behaviour

```
  caller thread                ASR workers (N)          alignment worker (1)
  -------------                ---------------          --------------------

  process_packet
        |
        v
  push_samples / push_vad_segment
        |
        v
  [dispatch loop, inline; non-blocking try_send + always-drain pattern]
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

#### 6.4.1 Avoiding the saturation deadlock

A naive inline dispatch loop that calls `work_tx.send(item)` (blocking) deadlocks under saturation: when `work_tx` is full, the caller blocks; meanwhile workers continue producing results into `result_rx`, but no thread is draining `result_rx` (the dispatch loop is stuck on the send). If `result_rx` is bounded the workers block too — full deadlock; if unbounded, memory grows without bound.

The dispatch loop therefore uses a **non-blocking try_send + always-drain pattern**:

```rust
/// Returns Ok(true) if any forward progress was made (drained ≥ 1
/// result, sent ≥ 1 command, or emitted ≥ 1 event); Ok(false) if
/// nothing changed. Used by both `process_packet` (which calls
/// drive in a loop after pushing inputs) and the saturation wait
/// (below) to decide whether to keep spinning.
fn drive_one_step(&mut self) -> Result<bool, RunnerError> {
    let mut progress = false;

    // Phase 1: ALWAYS drain results first. This must complete before
    // we attempt to send any new work.
    while let Ok((chunk_id, asr_result)) = self.whisper_pool.result_rx.try_recv() {
        progress = true;
        match asr_result {
            Ok(r)  => self.core.inject_asr_result(chunk_id, r)?,
            Err(e) => self.core.inject_failure(chunk_id, e)?,
        }
    }
    #[cfg(feature = "alignment")]
    if let Some(ap) = self.alignment_pool.as_ref() {
        while let Ok((chunk_id, ar)) = ap.align_rx.try_recv() {
            progress = true;
            match ar {
                Ok(r)  => self.core.inject_alignment_result(chunk_id, r)?,
                Err(e) => self.core.inject_failure(chunk_id, e)?,
            }
        }
    }

    // Phase 2: drain core's pending Events to the consumer-facing
    // emit channel.
    while let Some(event) = self.core.poll_event() {
        progress = true;
        self.emit_tx.send(event).map_err(|_| RunnerError::WhisperPoolShutdown)?;
    }

    // Phase 3: drain core's pending Commands. try_send only.
    while let Some(cmd) = self.core.poll_command() {
        match self.try_dispatch(cmd) {
            DispatchOutcome::Sent       => progress = true,
            DispatchOutcome::Backpressure(parked) => {
                self.core.unpoll_command(parked);
                if !self.whisper_pool_config.block_on_full_queue {
                    return Err(RunnerError::Backpressure {
                        buffered: self.core.buffered_samples(),
                        cap: self.transcriber_config.buffer_cap_samples,
                    });
                }
                // block_on_full_queue=true: park, return, let caller's
                // outer loop spin via select! (below).
                return Ok(progress);
            }
            DispatchOutcome::Disconnected =>
                return Err(RunnerError::WhisperPoolShutdown),
        }
    }
    Ok(progress)
}

enum DispatchOutcome { Sent, Backpressure(Command), Disconnected }
```

`drive_one_step` is called from `process_packet` after `core.push_*` mutations and from `poll_transcript` before reading `emit_rx`. It returns:

- `Ok(true)` — at least one of: a result was injected, an event was emitted, a command was sent. The caller can keep looping.
- `Ok(false)` — nothing happened this pass. The caller is idle or saturated; consult `core.is_idle()` to distinguish.
- `Err(RunnerError::Backpressure)` — only when `block_on_full_queue = false` and a command's `try_send` returned Full. The command has been re-parked via `unpoll_command`; **the core's buffer state has already been advanced** (samples buffered, segments merged into possibly-pending chunks). Per §6.4.2, the caller must drain via `poll_*` before pushing again.
- `Err(RunnerError::WhisperPoolShutdown)` — fatal; a worker channel is disconnected.

When `block_on_full_queue=true` and `drive_one_step` returns `Ok(progress)` with `core.poll_command().is_some()` still pending (i.e., we hit Backpressure but parked), `process_packet` enters a saturation wait:

```rust
loop {
    if drive_one_step()? {
        // forward progress was made — try again, possibly clearing the parked command
        continue;
    }
    if self.core.poll_command().is_none() {
        break;  // genuinely idle, no parked command, exit
    }
    // No progress AND a parked command is waiting. Block until one of
    // the worker channels becomes receivable, OR a 10 ms safety timeout.
    //
    // Critically, we must NOT use crossbeam_channel::select! here:
    // the recv arms in select! perform the actual receive and bind
    // the message to the arm body, so an empty `=> {}` body would
    // silently drop the result and Phase 1 of the next drive_one_step
    // would find the channel empty. Instead we use `Select::ready_timeout`,
    // which signals readiness without consuming. The next drive_one_step
    // Phase 1 then drains the message normally via try_recv.
    let mut sel = crossbeam_channel::Select::new();
    sel.recv(&self.whisper_pool.result_rx);
    #[cfg(feature = "alignment")]
    if let Some(ap) = &self.alignment_pool {
        sel.recv(&ap.align_rx);
    }
    let _ = sel.ready_timeout(self.whisper_pool_config.dispatch_idle_poll);
    // Fall through to the next loop iteration; drive_one_step's
    // try_recv pulls the now-ready message. crossbeam's
    // ready_timeout is documented to occasionally return success
    // spuriously (no actual ready channel); that's harmless here
    // — the next try_recv just returns Empty and the outer loop
    // spins one more iteration before re-blocking.
    //
    // Disconnected channels: ready_timeout returns Ok when a
    // channel is closed (not just when a message is available),
    // so a panicked or shutdown worker thread will wake the
    // saturation wait via its closed `result_rx`. The next
    // drive_one_step's try_recv then returns
    // `Err(TryRecvError::Disconnected)`, which the dispatch loop
    // surfaces as `Err(RunnerError::WhisperPoolShutdown)`. No
    // silent stall.
}
```

Forward progress is guaranteed: as soon as any worker produces a result (freeing a slot via the in-order emit / trim path), the next `drive_one_step` lands the parked command. The only deadlock-equivalent is genuine model hang, bounded by per-job worker timeouts (§6.4.3).

`drain()` reuses the same select-with-10 ms-default wait loop, exiting when `core.is_idle() && core.buffered_samples() == 0 && core.poll_command().is_none() && all worker queues empty`. The total wait is capped at `drain_timeout` (§6.1).

`unpoll_command` is the core's affordance for re-parking a command: it accepts the most-recently-popped command back onto the front of the queue. The dispatch loop is the only caller; the operation is safe because the core's own state (`in_flight`, `cut_pending`) is unchanged when a command was emitted but not actually consumed by the runner.

#### 6.4.2 Backpressure contract (the side-effect rule)

When `process_packet` returns `RunnerError::Backpressure { .. }` (only in `block_on_full_queue=false` mode):

- **Inputs are accepted, not rejected.** The `samples` were buffered, the `vad_segments` were fed through the cut state machine; any `MergedChunk` they produced has been queued in `cut_pending`. The state machine has advanced.
- **The caller must drain via `poll_transcript` / `poll_error`** until `core.buffered_samples()` falls below the buffer cap and `cut_pending` empties. Then continue ingest.
- **The caller must NOT retry the same `process_packet` call** with the same arguments. Doing so would be a PTS regression (audio already buffered) or a duplicate VAD segment push. Both are state-machine errors.
- The contract is unchanged for `block_on_full_queue=true` — that mode just blocks until drain, then accepts the input. Callers writing portable code should treat both modes identically and always drain before pushing again, regardless of whether they got a `Backpressure` error or a transient block.

For callers that genuinely need a *non-mutating* check (decide before pushing whether the buffer can accept), `Transcriber::would_accept(&self, samples_len, vad_count) -> bool` is provided as a const-time predicate; `process_packet` does not consult it internally to avoid TOCTOU between the check and the actual mutation.

#### 6.4.3 Worker hang protection

Each worker tracks its current job's start time. The whisper-rs `set_abort_callback_safe(...)` callback is wired to compare elapsed time against the job's `asr_timeout` and signal abort if exceeded. On abort, the worker emits `WorkFailure::WorkerHangTimeout { kind: WorkerKind::Asr, elapsed }` and returns the chunk via `result_rx`.

**Recycle hysteresis (cost-bounded recycling).** On CPU backends, `WhisperState` recycle is cheap — a fresh `state = ctx.create_state()` allocates a small decoder workspace and is dropped/created in microseconds. On GPU backends, `create_state` allocates KV-cache buffers on the device (10s–100s of ms latency, GPU memory churn); aggressively recycling on every timeout would stall the entire pool when `worker_count = 1` (the GPU default). The runner therefore uses a **per-worker timeout streak counter**:

- On a successful `state.full()`, reset the streak to 0.
- On a `WorkerHangTimeout`, increment the streak. If `streak < timeout_streak_threshold` (default 3), keep the existing `WhisperState` — assume a transient bad chunk caused the hang and try the next work item with the same state.
- On `streak >= threshold`, drop and recreate the state. Reset the streak to 0.

For CPU workers (where recycle is cheap), the threshold is 1 (recycle every time). For GPU workers, the threshold is 3 by default; the runner exposes `WhisperPoolConfig::timeout_streak_threshold` for tuning. Documented p99 stall on GPU recycle: `gpu_state_create_latency * num_recycling_workers`, which on a single-worker GPU pool means the entire next-chunk processing pauses for a full state allocation.

**Streak-vs-chunk_id correspondence.** Each timeout corresponds to exactly one `chunk_id` (the chunk in flight when the watchdog fired). When the streak threshold is reached and the state is recycled, the recycled state begins clean for the *next* dequeued `chunk_id`; the recycle does not retroactively retry the timed-out chunks (they emit as `Event::Error { error: WorkFailure::WorkerHangTimeout }` per their original chunk_ids and decay out of `in_flight`). Streak counting is per-worker, not per-chunk_id; a worker that hits 3 timeouts on chunks N, N+2, N+5 (with successful chunks in between resetting the streak) keeps its state.

Alignment workers use a `std::time::Instant`-tracked equivalent; ort does not have a built-in cooperative abort, so timeout in alignment kills the work item without interrupting in-progress ONNX inference. The session is then dropped and reconstructed from disk if it was a transient model fault. The same streak-threshold hysteresis applies (default threshold 3 for GPU, 1 for CPU).

---

## 7. Data flow (end-to-end)

A worked example. Assume `chunk_size = 30 s`, `worker_count = 2`, alignment enabled, `LanguagePolicy::AutoLockAfter(1)`.

1. Caller's pipeline emits a 100 ms audio packet (1 600 samples) at PTS 0.
2. Caller runs silero on the packet, gets zero or more new `SpeechSegment`s.
3. Caller calls `mt.process_packet(Timestamp::new(0, output_tb), &samples, &vad_segs, None)` — where `output_tb` is the original media's timebase (e.g., 1/48000 for a 48 kHz source).
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
| `AsrParams.strategy`                          | `BeamSearch { beam_size: 5, patience: -1.0 }` | WhisperX default. Greedy is faster but less accurate. |
| `AsrParams.initial_temperature`               | 0.0                                  | First attempt of the runner's temperature ladder. |
| `AsrParams.temperature_increment`             | 0.2                                  | Step size on retry. WhisperX default. |
| `AsrParams.max_attempts`                      | 6                                    | Cap on the runner's temperature retries (covers 0.0–1.0 by 0.2 step). |
| `AsrParams.no_speech_threshold`               | 0.6                                  | WhisperX default. |
| `AsrParams.log_prob_threshold`                | -1.0                                 | Triggers temperature retry. WhisperX default. |
| `AsrParams.compression_ratio_threshold`       | 2.4                                  | Triggers temperature retry. WhisperX default. |
| `AsrParams.no_context`                        | true                                 | Forwarded to `FullParams::set_no_context`. **Polarity is opposite to the WhisperX `condition_on_previous_text` knob this replaced**: `no_context = true` corresponds to `condition_on_previous_text = false`, the WhisperX default. Each `WhisperState::full` call is independent regardless. The flag only controls whether whisper's *intra*-chunk decoder uses prior segment text as a prompt for the next ~30-token segment within one encoder call; the WhisperX-default behaviour avoids degenerate hallucination loops on misrecognised segments. |
| `AsrParams.suppress_blank`                    | true                                 | WhisperX / whisper.cpp default. |
| `AsrParams.suppress_non_speech_tokens`        | false                                | Forwarded to `FullParams::set_suppress_nst`. |
| `AsrParams.n_threads`                         | 1                                    | Per-chunk in-call thread count. The runner's parallelism comes from the worker pool; over-subscribing in-call threads is wasteful. |
| `AsrParams.initial_prompt`                    | None                                 | Forwarded to `FullParams::set_initial_prompt` when Some. |
| `AlignmentFallback`                           | `SkipChunk`                          | Unknown languages still emit a `Transcript`, just with empty `words`. |
| `with_alignment(...)`                         | not called (off)                     | Caller opts in by passing an `AlignmentSet`; otherwise `Transcript.words` is empty. |
| `WhisperPoolConfig.use_gpu`                   | false                                | GPU selection is opt-in; defaulting to CPU avoids surprising GPU memory usage on first run. Backend (CUDA / Metal / Vulkan / HIPBLAS / CoreML) is selected at crate compile time via Cargo features. |
| `WhisperPoolConfig.gpu_device`                | 0                                    | Single-GPU index; whisper-rs has no multi-GPU dispatch. |
| `WhisperPoolConfig.flash_attn`                | false                                | Mutually exclusive with DTW (which is not enabled in v1). |
| `WhisperPoolConfig.max_queued_chunks`         | `worker_count + 4`                   | Cap on the work_tx channel before saturation kicks in. Slightly larger than `max_in_flight` so a brief burst can be absorbed without parking commands. |
| `WhisperPoolConfig.timeout_streak_threshold`  | 1 on CPU, 3 on GPU                   | Recycle `WhisperState` on the Nth consecutive timeout. Higher on GPU because state recreation is expensive there. |
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

### 10.4 v3-v5 regression tests

These exercise specific defects caught during the design-review rounds; landing them as named tests prevents regressions on subsequent refactors.

- **PTS drift on non-integer-ratio output (NB1).** Drive a 1/30001 output timebase through 10 000 trim cycles; assert each emitted `Word.range` is within ±1 PTS of the analytical exact value (rescale_pts of the immutable anchor + absolute_sample_offset). The pre-fix mutable-anchor code drifts by ~982 PTS per 1 000 trims; the test would fail loudly there.
- **Saturation-wait does not lose results (NB-β).** Mock a worker pool with bounded `result_rx` capacity 1; saturate work_tx; verify every chunk_id sent eventually emits a `Transcript`. The pre-fix `select! { recv -> _ => {} }` silently dropped one result per saturation cycle; this test would fail by missing transcripts.
- **`restart_at` cut_pending drain (Round-4 latent).** Push enough audio + VAD to fill `cut_pending` to 4 entries; trigger a `GapExceedsTolerance`; call `restart_at(new_anchor)`; push a fresh contiguous segment; assert (a) no panic in the next `trim()`, (b) the 4 pre-restart pending chunks emit normal `Transcript`s in chunk_id order, (c) the first post-restart chunk has chunk_id one larger than the last pre-restart chunk.
- **`next_expected_starts_at()` correctness (W3).** With output_tb 1/30001 and packet length 1000, push 100 packets using `next_expected_starts_at()` for each subsequent `starts_at`; assert no `PtsRegression`. Then re-run with the caller's own per-packet running sum; assert it eventually trips `PtsRegression` (proving the accessor is necessary).
- **Layered-ladder suppression (M-κ).** Mock whisper-rs with a recording wrapper around `state.full()`; verify each call's `FullParams` has `temperature_inc == 0.0` and an explicit `set_temperature(t)` value matching the runner's expected ladder step. Two layered ladders would show as multiple internal-loop iterations within a single call.
- **`unpoll_command` round-trip (M12).** Drive a saturated work_tx; verify `core.poll_command` returns the same command on the second call after `unpoll_command(cmd)` was called (i.e., commands aren't lost or reordered through the park-resume cycle).
- **Park-and-resume across the wake/select cycle (M12 extended).** Saturate work_tx → drive_one_step parks the front command via `unpoll_command` and returns `Ok(false)` → fire a mocked worker result on `result_rx` → assert `Select::ready_timeout` wakes within `dispatch_idle_poll` → next `drive_one_step` drains the result *and* lands the parked command (i.e., neither phase starves the other).
- **Empty packet handling.** `push_samples(next_expected_starts_at(), &[])` returns `Ok(())` and does not advance state; `push_vad_segment` after still works.
- **Zero-duration `VadSegment::new`.** Constructor panics; this is enforced via `#[should_panic]` test.
- **PtsRegression in output-PTS space, not 16k space (M-δ).** Output_tb 30000/1001 (NTSC), strictly contiguous packet pushes for 100 packets via `next_expected_starts_at()` — assert no spurious `PtsRegression`.
- **`Lang` canonicalisation invariant.** For every named variant `V` in Appendix C, assert `Lang::from_iso639_1(V.as_str()) == V` and that the result does NOT match the `Lang::Other(_)` pattern. The first half tests round-trip; the second pins the contract that named codes never end up in `Other`. A pure-`Other` round-trip — `Lang::from_iso639_1("zzz") == Lang::Other("zzz".into())` — is also asserted.

### 10.5 CI matrix

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
  - **Working-memory total at default config (4 ASR workers + 1 alignment worker, alignment on):** roughly 4 + 12 + 4×20 + 1 × 0.5 ≈ **96 MiB**. Scales roughly linearly with `worker_count` via the per-worker decoder workspace term (≈20 MiB / worker on tiny); e.g., 8 ASR workers → ≈ 176 MiB working memory; 16 → ≈ 336 MiB. Model weights and the alignment-logits buffer are independent of `worker_count`.
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
- **Per-call ASR override** beyond the language hint: in v1 we ship `AsrParamsOverride { language_hint, strategy, initial_temperature, initial_prompt }`. The override surface deliberately exposes `initial_temperature` but not `temperature_increment` or `max_attempts` — the per-packet caller can shift the ladder's starting point but not reshape it. Reshaping the ladder requires re-building the runner with `asr_params(...)`. This minimality is intentional; full ladder reshape is rare and can become per-call later as additive fields without breaking changes.
- **Model integrity verification** (`SHA-256` checking of loaded GGUF / wav2vec2 files at builder-time).
- **Per-language wav2vec2 default-model registry.** The list of recommended models per language (with licenses) can ship as a separate `whispery-models` crate or as a doc page; v1 leaves this to the caller.

---

## 13. Open risks

The following items must be resolved (or explicitly accepted) before implementation begins. Each carries a meaningful chance of forcing a re-architecture, so they get a named slot rather than buried in §12.

### 13.1 `WhisperContext` sharing across worker threads — verification, not spike

whisper-rs 0.13.x's documentation explicitly states:

> "Because the library is thread-safe, contexts can be shared across threads, while states are used to manage individual transcription tasks." (whisper-rs README, §Architecture)

`WhisperState` in 0.13.x is owned (no `'a` lifetime parameter); `create_state(&self)` takes `&WhisperContext` and returns an owned state. The shared-context concurrency model is therefore officially supported: `Arc<WhisperContext>` + N owned `WhisperState`s, one per worker.

**Verification (≤ 2 hours) before code lands**, not a spike:

1. Compile-time `assert_send_sync::<WhisperContext>()` to confirm bounds for our build features (CPU, Metal, CUDA, etc.).
2. Single empirical benchmark: 4 worker threads concurrently calling `state.full()` on a shared context with the tiny model on CPU; assert no panics, output text correctness, ~3–4× throughput vs. single-worker.
3. Repeat on the active GPU backend at the deployment target; if throughput is flat (single-GPU serialisation) or worse, default `worker_count = 1` for that backend per §6.2.
4. Confirm `set_temperature(...)`, `set_temperature_inc(...)`, and (best-effort) `set_max_decoding_failures(...)` are present on `whisper_rs::FullParams` for the active version. The v4/v5 audits found the first two are present in 0.13.x; the third is the best-effort secondary safeguard documented in §5.6. If a future version renames or removes either of the first two, the alternative code path described in §5.6 (advisory temperature fields, no runner ladder) must be wired in via cfg.

If verification (1) fails, the fallback is per-worker contexts (model loaded N times); the §11 memory math already accounts for both scenarios. If (2) fails (e.g., a panic under contention), file a whisper-rs issue and pin to a known-good version.

The 2-hour verification gates the implementation plan; it is not a separate work item that delays it meaningfully.

### 13.2 ort `Session` thread-safety per execution provider

`ort::Session::run` is documented as thread-safe in the general case but has known quirks per execution provider (CUDA EP serialises internally; some Vulkan paths require single-threaded use). §6.3.3 commits v1 to a single alignment worker; the multi-worker future (v2) needs a clear answer per supported EP.

Not a v1 blocker, but document the known constraints alongside the §6.3 design so v2 work doesn't restart the analysis from scratch.

### 13.3 wav2vec2 model availability per language

The forced-alignment story requires a wav2vec2 phoneme/character model per language. Hugging Face has good coverage of major languages; long-tail languages may have only multilingual (XLSR / MMS) models. The spec defers model curation to a v2 `whispery-models` crate, but if the indexer's first deployment targets a language without a quality language-specific model, falling back to multilingual changes the alignment quality story. Confirm target-language coverage before committing to the forced-alignment path; if a target language has no usable aligner, that chunk emits with empty `words` per `AlignmentFallback::SkipChunk`.

### 13.4 P4 architectural questions — resolved

Both questions are resolved and recorded in §1.6 / §1.7:

- **§1.6 (diarization integration):** confirmed against `dia` v0.1.0. `dia::DiarizedSpan.range()` returns `mediatime::TimeRange` at the 1/16000 analysis timebase; whispery's `Word.range()` is in the caller-chosen output timebase. mediatime's 128-bit cross-multiply lives on `Timestamp::cmp_semantic` only — `TimeRange` itself has no `Ord` — so the indexer canonicalises ranges to one timebase up front and joins by interval overlap (see §1.6 for the three concrete paths). No whispery-side API changes required.
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

1. ~~Whether `Lang` should be a typed enum over the Whisper-supported languages or a `SmolStr` newtype.~~ **Resolved.** v1 uses a `#[non_exhaustive]` enum with an `Other(SmolStr)` escape hatch — see §4.4 and Appendix C.
2. Whether `Transcript` should derive `Clone`. v1 does not; the dispatcher moves it through a single channel. Revisit if the indexer needs to fan out the same chunk to multiple writers.
3. Whether the runner's dispatch loop should run on a dedicated background thread instead of inline on the caller's thread inside `process_packet`. v1 inline; revisit if profiling shows `process_packet` stalls dominating.
4. ~~Whether to maintain a per-Transcriber language cache.~~ **Resolved.** v1 implements `LanguagePolicy::AutoLockAfter(1)` as the default — language is detected on the first non-trivial chunk and locked for the rest of the session.

## Appendix C — `Lang` variant table

The full set of named variants on `Lang`, mirroring whisper.cpp's `g_lang` table. Variant names use Rust PascalCase; the wire form (`as_str` / `from_iso639_1`) uses the lowercase ISO 639-1 code (or the multi-letter form for `Haw` / `Yue`).

```rust
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Lang {
    En, Zh, De, Es, Ru, Ko, Fr, Ja, Pt, Tr,
    Pl, Ca, Nl, Ar, Sv, It, Id, Hi, Fi, Vi,
    He, Uk, El, Ms, Cs, Ro, Da, Hu, Ta, No,
    Th, Ur, Hr, Bg, Lt, La, Mi, Ml, Cy, Sk,
    Te, Fa, Lv, Bn, Sr, Az, Sl, Kn, Et, Mk,
    Br, Eu, Is, Hy, Ne, Mn, Bs, Kk, Sq, Sw,
    Gl, Mr, Pa, Si, Km, Sn, Yo, So, Af, Oc,
    Ka, Be, Tg, Sd, Gu, Am, Yi, Lo, Uz, Fo,
    Ht, Ps, Tk, Nn, Mt, Sa, Lb, My, Bo, Tl,
    Mg, As, Tt, Haw, Ln, Ha, Ba, Jw, Su, Yue,
    Other(smol_str::SmolStr),
}
```

100 named variants plus `Other`. The list is generated from whisper.cpp's `g_lang` table (v1.7.6); a `#[test]` round-trips every named variant through `from_iso639_1(v.as_str())` and asserts the result equals `v` (canonicalisation invariant). When whisper.cpp ships a new language, the v1.x patch is to insert the variant alphabetically here and add the `as_str` / `from_iso639_1` match arms — no other code paths need touching.

Naming notes:

- `Haw` is Hawaiian (3-letter code "haw"; ISO 639-2 since 639-1 doesn't list it).
- `Yue` is Cantonese (3-letter code "yue").
- `Jw` is Javanese (whisper.cpp uses the older "jw" code; modern ISO 639-1 is "jv"; we accept "jv" via `Other` for now and may rename in v2 if whisper.cpp aligns).
