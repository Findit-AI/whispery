# whispery

> **Plan C — forced alignment.** Word-level forced alignment via wav2vec2 + ort.

Sans-I/O cut/batch/whisper/align state machine for speech-to-text indexing pipelines. Inspired by [WhisperX](https://github.com/m-bain/whisperX).

whispery is **Sans-I/O**: it doesn't own threads, channels, or
runtime. You drive it from a single thread (or wrap blocking
calls in your async runtime), feeding it samples + VAD and
pulling commands the runner answers with sync compute
primitives (`AsrSource`, `run_one_alignment`):

```rust,ignore
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;
use whispery::{
    AlignWorkItem, Aligner, AlignerKey, AlignmentFallback,
    AlignmentSetBuilder, AsrChunkContext, AsrSource, EnglishNormalizer,
    Lang, WhisperAsrSource, WhisperContext, WhisperContextParameters,
    run_one_alignment,
    core::{Command, Transcriber, TranscriberConfig},
    ort::session::RunOptions,
};

// 1. Wav2vec2 alignment registry.
let aligner = Aligner::from_paths(
    Lang::En,
    Path::new("path/to/wav2vec2-base-960h.onnx"),
    Path::new("path/to/wav2vec2-base-960h-tokenizer.json"),
    Box::new(EnglishNormalizer::new()),
)?
// Optional: tighten / loosen the composer's per-word post-pass.
// Defaults: 0.5 coverage + 80 ms intra-word silence tolerance —
// see `whispery::DEFAULT_MIN_SPEECH_COVERAGE` and
// `whispery::DEFAULT_MAX_INTRA_SILENT_RUN`.
.with_min_speech_coverage(0.5)
.with_max_intra_silent_run(Duration::from_millis(80));
let alignment_set = AlignmentSetBuilder::new()
    .with_fallback(AlignmentFallback::SkipChunk)
    .register(AlignerKey::Lang(Lang::En), aligner)
    .build();

// 2. Whisper ASR source (sync compute, no internal threads).
let whisper_ctx = Arc::new(WhisperContext::new_with_params(
    Path::new("path/to/ggml-large-v3-turbo.bin"),
    WhisperContextParameters::default(),
)?);
let mut asr_source = WhisperAsrSource::new(whisper_ctx)?;

// 3. Sans-I/O state machine. Each alignment chunk gets a FRESH
//    `RunOptions` (allocated inside the loop below) so an
//    external watchdog's `terminate()` for chunk N never
//    poisons chunk N+1 — ORT termination is sticky on the
//    handle. `abort_flag` is the cross-chunk cancellation
//    surface; `run_one_alignment` checks it on entry.
let mut transcriber = Transcriber::new(TranscriberConfig::default());
let abort_flag = Arc::new(AtomicBool::new(false));

// 4. Pump loop: caller owns threading + cancellation.
//    `transcriber.process_packet(samples, vad)` from your I/O thread,
//    then drive commands and drain transcripts:
while let Some(cmd) = transcriber.poll_command() {
    match cmd {
        Command::RunAsr { chunk_id, samples, params, .. } => {
            let result = asr_source.run_chunk(AsrChunkContext {
                samples: &samples,
                params: &params,
                abort_flag: &abort_flag,
                chunk_id,
            })?;
            transcriber.inject_asr_result(chunk_id, result)?;
        }
        Command::RunAlignment { chunk_id, samples, sub_segments: _, text, language, runs } => {
            let job = AlignWorkItem::from_run_alignment(
                &transcriber, chunk_id, samples, text, language,
                runs, abort_flag.clone(),
            ).expect("chunk in flight");
            // Fresh `RunOptions` per chunk — see Sans-I/O comment above.
            let run_options = RunOptions::new()?;
            let aligned = run_one_alignment(&alignment_set, &job, &run_options)?;
            transcriber.inject_alignment_result(chunk_id, aligned)?;
        }
    }
}
while let Some(_event) = transcriber.poll_event() { /* Transcript.words() carries alignment */ }
# Ok::<(), Box<dyn std::error::Error>>(())
```

Async users (tokio, smol) wrap `WhisperAsrSource::run_chunk`
and `run_one_alignment` in `spawn_blocking`, and wire shutdown
via their own cancellation tokens flipping the supplied
`abort_flag`. Calling the chunk's `run_options.terminate()`
from another thread cancels in-flight ORT inference mid-call;
the alignment pipeline polls `abort_flag` between coarse
stages too. Allocate one `RunOptions` per chunk so a
`terminate()` for chunk N is never observable on chunk N+1
(ORT termination is sticky on the handle).

## Status

- **Core** (`Transcriber`, `Command`, `Event`, `Transcript`, `Word`, `Lang`, `VadSegment`) — Sans-I/O cut/batch state machine. Mockable ASR / alignment via `inject_asr_result` / `inject_alignment_result`.
- **Runner sync primitives** (`AsrSource`, `WhisperAsrSource`, `run_one_alignment`, `AlignWorkItem`) — caller-owned threading and cancellation. Whispery itself spawns no threads.
- **Alignment** — wav2vec2 forced alignment via `ort` (load-dynamic). Per-language `Aligner`, registry via `AlignmentSet`, script-dispatched per-language runs for code-switched chunks.

## Try it

```bash
cargo test --lib --features alignment    # 425+ unit tests, no network
```

End-to-end runs against real Whisper + wav2vec2 weights are
the caller's responsibility — pass model paths into
`WhisperContext::new_with_params` and `Aligner::from_paths`,
and set `ORT_DYLIB_PATH` to your `libonnxruntime` (see "ONNX
Runtime" below). `cargo build` makes no network requests.

### Audio parity fixtures (~283 MB)

Whispery's parity tests reference 14 WAV clips + RTTM speaker
annotations stored out-of-tree at
[Findit-AI/audio-fixtures](https://github.com/Findit-AI/audio-fixtures).
Populate them locally with:

```bash
bash scripts/fetch_fixtures.sh
```

The script shallow-clones the sibling repo, lays files out
under `tests/parity/fixtures/<name>/{clip_16k.wav,reference.rttm}`
(the layout existing tests expect), and is idempotent. CI
runs the same script when `WHISPERY_FETCH_FIXTURES=1` is set
on the workflow run; default `cargo test` stays network-free.

## ONNX Runtime

The `alignment` feature uses `ort` in **load-dynamic** mode —
`cargo build --features alignment` succeeds on a clean
toolchain (no system `libonnxruntime` needed at build time).
To actually run an `Aligner`, point `ort` at a runtime
library:

```bash
# Pick one:
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib   # macOS
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so      # Linux
# Or place the dylib on the platform's default search path
# (e.g. brew's `onnxruntime` formula on macOS).
```

This is intentional: the build path stays portable and
network-free, while users keep full control over which
ONNX Runtime build (CPU / CoreML / CUDA / DirectML) they
ship with.

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

### Default deviates from WhisperX on pronounced OOV symbols

Whispery's **default** tokenizer policy fails closed (drops
the chunk's word alignment, surfaces a recoverable
`SemanticOutOfVocab` diagnostic, preserves the ASR transcript)
when a pronounced non-alphanumeric OOV character — e.g. `&` in
`AT&T`, `@`, `%`, or the `,` in `4,9` — appears in a
transcript. WhisperX would silently align such characters
against whichever vocab item wins each frame; whispery treats
that as honest-looking but wrong word ranges and refuses.

Parity claims here therefore scope to
**alphanumeric-only inputs**. Enable the
`whisperx-strict-tokenizer` Cargo feature to opt into the
WhisperX wildcard-everything policy if your downstream
consumers expect bit-equivalent output on pronounced symbols
and accept the silent-misalignment risk.

## Documentation

- [Design spec](docs/superpowers/specs/2026-04-28-whispery-cut-batch-whisper-design.md)
- [Plan A](docs/superpowers/plans/2026-04-29-whispery-plan-a-types-and-core.md)
- [Plan B](docs/superpowers/plans/2026-04-29-whispery-plan-b-runner-whisper-rs.md)
- [Plan C](docs/superpowers/plans/2026-04-29-whispery-plan-c-alignment.md)

## License

MIT or Apache-2.0, at your option.
