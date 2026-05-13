<div align="center">
<h1>asry</h1>
</div>
<div align="center">

Sans-I/O word-level forced alignment with WhisperX-equivalent accuracy.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/asry-8da0cb?style=for-the-badge&logo=GitHub" height="22">][GitHub-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fasry" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/asry/ci.yml?logo=GitHub-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/asry?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-asry-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/asry?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/asry?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## Overview

Sans-I/O cut/batch/whisper/align state machine for speech-to-text
indexing pipelines, inspired by [WhisperX][whisperx-url]. Asry
itself owns no threads, channels, or runtime — you drive it from a
single thread (or wrap blocking calls in your async runtime),
feeding samples + VAD and pulling commands the runner answers via
sync compute primitives ([`AsrSource`][rust-asrsource-url],
[`run_one_alignment`][rust-runonealignment-url]).

## Quick start

The wav2vec2-base-960h tokenizer ships inside the crate (parsed at
build time, no `serde_json` runtime dep) — only the encoder ONNX
and the Whisper ggml checkpoint are BYO. Both files live above
crates.io's 10 MB hard limit and cannot be bundled. Fetch them
once with the pinned commands below; each one verifies SHA-256
before installing so a republished or truncated upstream surfaces
as a hard failure rather than silently altering alignment output.

### Whisper ggml model (`ggml-large-v3-turbo.bin`, ~1.6 GB)

```sh
ASRY_WHISPER_MODEL_SHA256="1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
mkdir -p models
TMP="$(mktemp "${TMPDIR:-/tmp}/ggml-large-v3-turbo.XXXXXXXXXX")"
curl --fail --location \
  --output "$TMP" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
ACTUAL="$(shasum -a 256 "$TMP" | awk '{print $1}')"
if [ "$ACTUAL" != "$ASRY_WHISPER_MODEL_SHA256" ]; then
  echo "SHA-256 mismatch: expected $ASRY_WHISPER_MODEL_SHA256, got $ACTUAL" >&2
  rm -f "$TMP"; exit 1
fi
mv "$TMP" models/ggml-large-v3-turbo.bin
```

### Wav2vec2 alignment encoder (English, ~378 MB)

```sh
ASRY_W2V_EN_SHA256="00b7cc69516c1ab63c429e63a2b543e4d42bb77441ec5b98ee935de175b00de1"
TMP="$(mktemp "${TMPDIR:-/tmp}/wav2vec2-base-960h.XXXXXXXXXX")"
curl --fail --location \
  --output "$TMP" \
  "https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX/resolve/main/onnx/model.onnx"
ACTUAL="$(shasum -a 256 "$TMP" | awk '{print $1}')"
if [ "$ACTUAL" != "$ASRY_W2V_EN_SHA256" ]; then
  echo "SHA-256 mismatch: expected $ASRY_W2V_EN_SHA256, got $ACTUAL" >&2
  rm -f "$TMP"; exit 1
fi
mv "$TMP" models/wav2vec2-base-960h.onnx
```

(Asry's `build.rs` can fetch both fixtures for you when
`ASRY_FETCH_MODEL=1` / `ASRY_FETCH_W2V=1` are set on a
`cargo build`. The script enforces the same SHA-256 pins. Plain
`cargo build` makes no network requests.)

### Run an end-to-end alignment

```rust,ignore
use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};
use asry::{
  AlignWorkItem, Aligner, AlignerKey, AlignmentFallback,
  AlignmentSetBuilder, AsrChunkContext, AsrSource, EnglishNormalizer,
  Lang, WhisperAsrSource, WhisperContext, WhisperContextParameters,
  run_one_alignment,
  core::{Command, Transcriber, TranscriberConfig},
  ort::session::RunOptions,
};

let aligner = Aligner::from_paths(
  Lang::En,
  Path::new("models/wav2vec2-base-960h.onnx"),
  Path::new("models/wav2vec2-base-960h-tokenizer.json"),
  Box::new(EnglishNormalizer::new()),
)?;
let alignment_set = AlignmentSetBuilder::new()
  .with_fallback(AlignmentFallback::SkipChunk)
  .register(AlignerKey::Lang(Lang::En), aligner)
  .build();

let whisper_ctx = Arc::new(WhisperContext::new_with_params(
  Path::new("models/ggml-large-v3-turbo.bin"),
  WhisperContextParameters::default(),
)?);
let mut asr_source = WhisperAsrSource::new(whisper_ctx)?;

let mut transcriber = Transcriber::new(TranscriberConfig::default());
let abort_flag = Arc::new(AtomicBool::new(false));

while let Some(cmd) = transcriber.poll_command() {
  match cmd {
    Command::Asr { chunk_id, samples, params, .. } => {
      let result = asr_source.run_chunk(AsrChunkContext::new(
        &samples, &params, &abort_flag, chunk_id,
      ))?;
      transcriber.handle_asr(chunk_id, result)?;
    }
    Command::Alignment { chunk_id, samples, sub_segments: _, text, language, runs } => {
      // Sans-I/O OOV resolution: per-run detect + decide.
      // Each run gets its own decisions vec sized + ordered
      // by the events `detect_oov` produces for that run's
      // text + language. Whole-chunk fallback (when `runs`
      // is empty) gets one inner vec.
      // `default_oov_decisions` mirrors the historical
      // behaviour (alphanumeric → wildcard, pronounced
      // symbols → fail-closed); swap for
      // `wildcard_all_decisions` (WhisperX 1:1) or write
      // your own per-run / per-language policy.
      let oov_decisions: Vec<Vec<asry::core::ResolvedOov>> = if runs.is_empty() {
        let events = alignment_set.detect_oov(&text, &language)?;
        vec![asry::core::default_oov_decisions(&events)]
      } else {
        alignment_set.detect_oov_per_run(&runs)?
          .iter()
          .map(|events| asry::core::default_oov_decisions(events))
          .collect()
      };

      let job = AlignWorkItem::from_run_alignment(
        &transcriber, chunk_id, samples, text, language,
        runs, abort_flag.clone(),
        oov_decisions,
      ).expect("chunk in flight");
      // Fresh `RunOptions` per chunk so a watchdog's
      // `terminate()` for chunk N does not poison chunk N+1.
      let run_options = RunOptions::new()?;
      let aligned = run_one_alignment(&alignment_set, &job, &run_options)?;
      transcriber.handle_alignment(chunk_id, aligned)?;
    }
  }
}
while let Some(_event) = transcriber.poll_event() {
  /* Transcript.words() carries word-level alignment */
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`ORT_DYLIB_PATH` overrides the default `libonnxruntime` lookup if
you keep the dylib elsewhere. The `alignment` feature uses `ort`
in **load-dynamic** mode — `cargo build --features alignment`
succeeds on a clean toolchain (no system `libonnxruntime` needed
at build time), but you must supply one at run time.

Async users (tokio, smol) wrap `WhisperAsrSource::run_chunk` and
`run_one_alignment` in `spawn_blocking` and wire shutdown via their
own cancellation tokens flipping `abort_flag`. Calling the chunk's
`run_options.terminate()` from another thread cancels in-flight
ORT inference mid-call; the alignment pipeline polls `abort_flag`
between coarse stages too.

## Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `std` | yes | `std`-backed implementations of crate types. Chains `std` to `mediatime`, `smol_str`, and serde when present. |
| `runner` | yes | `WhisperAsrSource` + the in-house `whispercpp` 0.2.x bindings + the temperature retry ladder + the real-zlib compression-ratio gate (via `miniz_oxide`). Implies `std`. |
| `alignment` | no | wav2vec2 forced alignment via `ort` (load-dynamic) + tokenizers + ndarray. Lights up `Aligner`, `AlignmentSet`, `run_one_alignment`. Implies `runner`. |
| `serde` | no | Derive `serde::{Serialize, Deserialize}` on public state-machine types (`Transcript`, `Word`, `AsrParams`, …). Implies `runner`. |
| `metal` | no | Apple-only: enables `whispercpp/metal` so the encoder runs on the unified-memory Metal backend. Implies `runner`. |
| `coreml` | no | Apple-only: enables `whispercpp/coreml` so the encoder additionally dispatches to ANE if the caller has produced a CoreML companion `.mlmodelc`. Implies `runner`. |
| `bench-internals` | no | Re-exports `pub(crate)` alignment internals (scalar/SIMD normaliser variants, raw `ctc_viterbi`, `LogProbsTV`) under `asry::__bench` so the SIMD baseline bench can call them directly. Doc-hidden; never enable in shipping builds. Implies `alignment`. |
| `parity-dump-emission` | no | Diagnostic-only: writes `wy_seg<N>.{emission,trellis}.bin` + a `wy_seg<N>.tokens.json` companion to `ASRY_PARITY_DUMP_TRELLIS` whenever set. Implies `alignment`. Do NOT enable in shipping builds. |

The CTC parity tests run as part of the regular test suite —
the OOV policy is per-test runtime data (no Cargo feature):

```bash
cargo test --features alignment,bench-internals --test whisperx_unit_parity
# 8/8 — tests 1-6 + 8 use `default_oov_decisions` (asry
# default); test 7 (`4,9` digits-comma WhisperX issue #1372)
# uses `wildcard_all_decisions` to opt into WhisperX 1:1
# behaviour for pronounced symbols.
```

These tests port WhisperX's
`tests/test_word_timestamp_interpolation.py` 1:1 onto asry's
CTC pipeline. The 193 alignment-pipeline lib tests pin the same
algorithmic invariants stage-by-stage; median IoU 0.9955–0.9990
across 854 word pairs vs. WhisperX's recorded outputs (measured
during initial calibration; see `trellis_beam.rs:305-330`).

## Audio parity fixtures (~283 MB)

The 14 WAV clips + RTTM speaker annotations asry's
end-to-end parity tests reference live out-of-tree at
[`Findit-AI/audio-fixtures`][audio-fixtures-url] so they don't
bloat asry's git history. Populate them locally with:

```bash
bash scripts/fetch_fixtures.sh
```

The script shallow-clones the sibling repo and lays files out
under `tests/parity/fixtures/<name>/{clip_16k.wav,reference.rttm}`
(the layout existing tests expect). Idempotent + cleans the
clone unless `ASRY_FIXTURES_KEEP_CLONE=1`. CI runs the same
script when `ASRY_FETCH_FIXTURES=1` is set on the workflow
run; default `cargo test` stays network-free.

## License

`asry` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT studio authors.

### Bundled-asset attributions propagate to downstream binaries

`asry` parses one third-party asset into every compiled
binary via `include_str!` at build time:

| File | License | Source |
|---|---|---|
| `assets/wav2vec2_base_960h_tokenizer.json` (parsed at build into Rust constants under `OUT_DIR`; bundled when `alignment` is on) | **Apache-2.0** | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) (tokenizer.json) |

The full SPDX expression for an `alignment`-enabled build is
therefore `(MIT OR Apache-2.0) AND Apache-2.0`. When you
redistribute a binary that depends on `asry`, reproduce the
attribution somewhere a recipient can find — for instance, in
your application's "About" or third-party-licenses page.

Models you BYO at runtime (Whisper ggml checkpoints, wav2vec2
encoder ONNX, language-specific aligners) carry their own
licenses — see the source links above and on each HuggingFace
repo. Mirror copies under
[`huggingface.co/FinDIT-Studio`](https://huggingface.co/FinDIT-Studio)
re-export upstream weights without modification; the upstream
license applies.

[GitHub-url]: https://github.com/Findit-AI/asry
[CI-url]: https://github.com/Findit-AI/asry/actions/workflows/ci.yml
[codecov-url]: https://app.codecov.io/gh/Findit-AI/asry/
[doc-url]: https://docs.rs/asry
[crates-url]: https://crates.io/crates/asry
[whisperx-url]: https://github.com/m-bain/whisperX
[audio-fixtures-url]: https://github.com/Findit-AI/audio-fixtures
[rust-asrsource-url]: https://docs.rs/asry/latest/asry/trait.AsrSource.html
[rust-runonealignment-url]: https://docs.rs/asry/latest/asry/fn.run_one_alignment.html
