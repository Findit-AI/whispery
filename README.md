# whispery

> **Plan A — types + Sans-I/O core. The runner (Plan B) and forced-alignment pipeline (Plan C) ship in subsequent milestones.**

Sans-I/O cut/batch/whisper/align state machine for speech-to-text indexing pipelines. Inspired by [WhisperX](https://github.com/m-bain/whisperX).

The crate's design separates a pure state machine (this milestone) from the actual ML inference (Plan B with `whisper-rs`, Plan C with `ort`-based wav2vec2 forced alignment). After Plan A merges, you can drive the core end-to-end with mocked backends — see `examples/core_only.rs`.

## Status

- ✅ **Plan A — types + core.** Public surface: `Transcript`, `Word`, `Lang`, `VadSegment`, errors, `Transcriber`, `Command`, `Event`. Mockable ASR / alignment via `inject_asr_result` / `inject_alignment_result`.
- ⏳ **Plan B — runner + whisper-rs.** Adds `ManagedTranscriber` and a worker pool over `whisper-rs`.
- ⏳ **Plan C — alignment.** Adds wav2vec2 forced alignment via `ort`. Lights up `Transcript.words`.

## Try it

```bash
cargo run --example core_only
```

## Documentation

- [Design spec](docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md)
- [Plan A](docs/superpowers/plans/2026-04-29-whispery-plan-a-types-and-core.md)

## License

MIT or Apache-2.0, at your option.
