# whispery

> **Plan C — forced alignment.** Word-level forced alignment via wav2vec2 + ort.

Sans-I/O cut/batch/whisper/align state machine for speech-to-text indexing pipelines. Inspired by [WhisperX](https://github.com/m-bain/whisperX).

After Plan C merges, you can drive a real whisper-rs inference + word-level alignment end-to-end:

```rust
use std::path::Path;
use std::time::Duration;
use whispery::{
    Aligner, AlignerKey, AlignmentFallback, AlignmentSetBuilder, EnglishNormalizer,
    Lang, LanguagePolicy, ManagedTranscriber, WhisperPoolOptions,
};

let aligner = Aligner::from_paths(
    Lang::En,
    Path::new("path/to/wav2vec2-base-960h.onnx"),
    Path::new("path/to/wav2vec2-base-960h-tokenizer.json"),
    Box::new(EnglishNormalizer::new()),
)?;

let set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::SkipChunk)
    .register(AlignerKey::Lang(Lang::En), aligner)
    .build();

let pool = WhisperPoolOptions::new("path/to/ggml-tiny.en.bin")
    .with_worker_count(2);
let mut runner = ManagedTranscriber::from_options(pool)?
    .chunk_size(Duration::from_secs(30))
    .language_policy(LanguagePolicy::Lock { hint: Lang::En })
    .with_alignment(set)
    .build()?;

// (push samples + VAD via process_packet, drain via poll_transcript;
// each Transcript.words() now carries word-level alignment)
# Ok::<(), whispery::RunnerError>(())
```

## Status

- Plan A — types + core. Public surface: `Transcript`, `Word`, `Lang`, `VadSegment`, errors, `Transcriber`, `Command`, `Event`. Mockable ASR / alignment via `inject_asr_result` / `inject_alignment_result`.
- Plan B — runner + whisper-rs. Adds `ManagedTranscriber`, `WhisperPoolOptions`, `RunnerError`, `AsrParamsOverride`. Saturation-deadlock-safe dispatch loop, per-job worker-hang timeout, temperature retry ladder.
- Plan C — alignment. Adds wav2vec2 forced alignment via `ort`. Lights up `Transcript.words`. Single alignment worker per spec §6.3.3.

## Try it

```bash
cargo run --example core_only        # Plan A: drive the core with mocked backends
# Real-model end-to-end. The fixture fetch is opt-in — set
# WHISPERY_FETCH_MODEL=1 to download ~75 MB of Whisper +
# JFK WAV on first build:
WHISPERY_FETCH_MODEL=1 \
  cargo test --features runner --test runner_e2e -- --test-threads=1
# Real wav2vec2 alignment. Add WHISPERY_FETCH_W2V=1 for the
# ~360 MB ONNX + tokenizer fetch:
WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 \
  cargo test --features alignment --test alignment_e2e -- --test-threads=1
```

Plain `cargo build` makes no network requests; fixture
downloads only run when both env vars are set.

## Bundled assets

The `wav2vec2-base-960h` tokenizer ships in the crate, but
**parsed at build time** — `build.rs` reads
`assets/wav2vec2_base_960h_tokenizer.json` (2 KB) and emits
Rust constants under `OUT_DIR`. At runtime you can reach them
via `whispery::bundled::wav2vec2_base_960h::{VOCAB,
PAD_TOKEN_ID, UNK_TOKEN_ID, DELIMITER_TOKEN_ID, token_to_id}`
under `feature = "alignment"` — no JSON parse, no
`serde_json`, no allocations on the alignment hot path. The
matching ONNX model (~378 MB) is too large for crates.io;
download it once from HuggingFace and pass the path to
`Aligner::from_paths`.

## WhisperX parity

For English forced alignment whispery uses the same upstream
weights as [WhisperX](https://github.com/m-bain/whisperX):

| Component | WhisperX (PyTorch) | whispery (ONNX) |
| --- | --- | --- |
| EN aligner | `torchaudio` `WAV2VEC2_ASR_BASE_960H` (= `facebook/wav2vec2-base-960h`) | [`onnx-community/wav2vec2-base-960h-ONNX`](https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX) (direct ONNX export of the same weights) |
| Tokenizer | bundled w/ torchaudio bundle | bundled in this crate (parsed at build, exposed as `bundled::wav2vec2_base_960h` constants) |

For other languages WhisperX picks language-specific
`jonatasgrosman/wav2vec2-large-xlsr-53-{lang}` checkpoints.
ONNX exports of those exist on the HuggingFace hub and slot
into `Aligner::from_paths` with the matching language
[`TextNormalizer`](crate::TextNormalizer) — supply your own ONNX +
tokenizer pair via `AlignmentSetBuilder::register`. Whispery
ships [`EnglishNormalizer`], [`ChineseNormalizer`], and
[`JapaneseNormalizer`]; mapping new languages amounts to
implementing the trait.

## Documentation

- [Design spec](docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md)
- [Plan A](docs/superpowers/plans/2026-04-29-whispery-plan-a-types-and-core.md)
- [Plan B](docs/superpowers/plans/2026-04-29-whispery-plan-b-runner-whisper-rs.md)
- [Plan C](docs/superpowers/plans/2026-04-29-whispery-plan-c-alignment.md)

## License

MIT or Apache-2.0, at your option.
