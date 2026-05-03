# whispery vs WhisperX alignment parity harness

A side-by-side runner that compares whispery's word-level forced
alignment output against [WhisperX][whisperx]'s on the same audio,
reporting per-word IoU statistics.

## Purpose

whispery's forced-alignment pipeline is intentionally a Rust
re-implementation of WhisperX's wav2vec2-based aligner: same model
family (`facebook/wav2vec2-base-960h` weights, ONNX-exported), same
CTC Viterbi algorithm, same word-grouping rules. The two should
agree to within a CTC hop (~20 ms / 0.02 s) on what is recognisably
the same word.

This is **not a bit-exactness check**. WhisperX runs faster-whisper
(CTranslate2) for ASR; whispery uses whisper.cpp via `whisper-rs`.
The two ASR backends emit slightly different surface forms
(punctuation, casing, occasional dropped/inserted words). Scoring is
therefore done by sequence-aligning normalised word lists and
computing IoU only on matched pairs — exactly the pattern dia uses
for its own pyannote.audio parity check.

## Layout

- `Cargo.toml` / `src/main.rs` — Rust binary `whispery-parity-runner`
  that pushes a WAV through `ManagedTranscriber` with English-locked
  language + wav2vec2 alignment, dumps words to JSON.
- `python/pyproject.toml` / `python/whisperx_runner.py` — uv-managed
  WhisperX runner that emits JSON in the same schema.
- `python/score.py` — Needleman-Wunsch sequence alignment over
  normalised word texts + per-pair IoU summary.
- `run.sh` — end-to-end driver.
- `out/` — per-fixture JSON outputs (gitignored).

This directory is **NOT** part of `cargo test`. It's a manual harness
for release-time validation, the same status as `dia/tests/parity/`.

## Prerequisites

1. **Model fixtures.** A one-time prep:
   ```bash
   WHISPERY_FETCH_MODEL=1 WHISPERY_FETCH_W2V=1 \
       cargo test --features alignment
   ```
   This populates `target/whispery-test-fixtures/` (or
   `$HOME/.cargo/target/whispery-test-fixtures/` when
   `CARGO_TARGET_DIR` isn't set) with `ggml-tiny.en.bin`,
   `wav2vec2-base-960h.onnx`, and `wav2vec2-base-960h-tokenizer.json`.

   Override paths via `--whisper-model` / `--w2v-model` /
   `--w2v-tokenizer` flags, or via `WHISPER_MODEL_PATH` /
   `WAV2VEC2_ONNX_PATH` / `WAV2VEC2_TOKENIZER_PATH` env vars.

2. **`ORT_DYLIB_PATH`.** whispery's `ort` is pinned to `load-dynamic`;
   the dylib is not bundled. Point this at a `libonnxruntime.{so,dylib,dll}`
   you've installed separately (e.g. via Homebrew, `pip install
   onnxruntime`, or the upstream releases).

3. **`uv`.** `brew install uv` or `pip install uv`. Used to manage the
   WhisperX venv at `tests/parity_whisperx/python/.venv`.

4. **System FFmpeg libraries.** The Rust runner links `ffmpeg-next`
   (its audio loader mirrors WhisperX's `load_audio` exactly, so the
   parity check sees byte-identical audio on both sides). On macOS:
   `brew install ffmpeg`. On Debian/Ubuntu:
   `apt-get install libavformat-dev libavcodec-dev libavutil-dev libswresample-dev pkg-config`.

   The `ffmpeg-next` major version is pinned to match the system
   FFmpeg ABI; see the comment block in
   `tests/parity_whisperx/Cargo.toml`. The harness currently targets
   FFmpeg 8 (the only major still maintained as of 2026); FFmpeg 7's
   header file `libavcodec/avfft.h` is referenced by `ffmpeg-next 7`'s
   bindgen pass and was dropped in FFmpeg 8, so mixing the two does
   not build.

GPU is **not required**. The Python runner pins device=`cpu` and
compute_type=`int8` for determinism — CTranslate2 GPU paths have
non-deterministic floating-point ordering that would add noise without
changing the alignment model.

## Audio fixtures

The harness accepts any 16 kHz mono WAV. Sister project `dia` ships a
set of curated diarization fixtures we can borrow as ASR/alignment
inputs:

```bash
./tests/parity_whisperx/run.sh \
    /path/to/dia/tests/parity/fixtures/02_pyannote_sample
```

Available fixtures (relative to `dia/tests/parity/fixtures/`):

| Fixture                | Duration |
|------------------------|----------|
| `01_dialogue/`         | 226.96 s |
| `02_pyannote_sample/`  | 30.00 s  |
| `03_dual_speaker/`     | 60.00 s  |
| `04_three_speaker/`    | 39.97 s  |
| `05_four_speaker/`     | 60.00 s  |
| `06_long_recording/`   | 977.73 s |

The diarization labels (`reference.rttm`) and pyannote intermediates
(`*.npy` / `*.npz`) in those directories are not used here.

## Run

```bash
cd whispery
./tests/parity_whisperx/run.sh \
    /Users/you/Develop/findit-studio/dia/tests/parity/fixtures/02_pyannote_sample
```

The script:
1. Brings up `tests/parity_whisperx/python/.venv` via `uv` (cached
   after first run).
2. Builds + runs `whispery-parity-runner` →
   `tests/parity_whisperx/out/whispery_<fixture>.json`.
3. Runs `whisperx_runner.py` →
   `tests/parity_whisperx/out/whisperx_<fixture>.json`.
4. Runs `score.py` over the two JSON files, prints a human-readable
   summary on stderr, writes `out/score_<fixture>.json`, and exits
   with the score's exit code (0 iff median IoU ≥ 0.7).

A direct WAV path also works:

```bash
./tests/parity_whisperx/run.sh /path/to/clip.wav
```

## Output schema

Both runners emit JSON in this shape:

```jsonc
{
  "runner": "whispery" | "whisperx",
  "clip_path": "/abs/path/to/clip.wav",
  // SHA-256 of the f32 byte buffer that the model actually
  // consumes — NOT the file bytes. Both runners load via
  // ffmpeg → s16le → f32/32768, so the two hashes match iff
  // both pipelines decoded the audio identically. A mismatch
  // is a fast diagnostic that the audio-loader paths have
  // diverged again.
  "clip_sha256": "...",
  "duration_s": 30.0,
  "transcript_count": 1,
  "words": [
    {
      "text": "hello",
      "start_s": 0.123,
      "end_s": 0.456,
      "score": 0.92
    },
    ...
  ]
}
```

`score.py` consumes two such files and emits:

```jsonc
{
  "whispery_word_count": 73,
  "whisperx_word_count": 75,
  "matched_pairs": 71,
  "dropped_by_whispery": 2,
  "dropped_by_whisperx": 4,
  "iou": {
    "count": 71,
    "mean": 0.86,
    "median": 0.92,
    "p10": 0.65,
    "p90": 0.97,
    "below_0.5": 3
  },
  "worst_5": [...],
  "threshold_median_iou": 0.7,
  "passed": true
}
```

## Caveats

- WhisperX uses **faster-whisper** (CTranslate2) for ASR; whispery
  uses **whisper.cpp** via `whisper-rs`. Word texts can differ on
  punctuation/casing decisions — `score.py` lowercases and strips
  ASCII boundary punctuation before sequence-aligning, so those
  cosmetic differences don't drop pairs.
- The two pipelines also segment differently: WhisperX merges
  faster-whisper segments into long stretches before alignment;
  whispery aligns one whisper.cpp chunk at a time. IoU on individual
  words is unaffected (alignment is per-word) but transcript-level
  counts will diverge.
- The 0.7 median-IoU threshold is a "functionally equivalent" bar,
  not a "bit-exact" one. Tighten it (e.g. `--threshold 0.85`) once
  whispery's alignment is stable on a wider corpus.

## Encoder export trade-off (advanced)

whispery defaults to the community ONNX export at
`onnx-community/wav2vec2-base-960h-ONNX`, which is what `build.rs`
fetches into `target/whispery-test-fixtures/wav2vec2-base-960h.onnx`.
A re-export from PyTorch eager is provided for parity-strict users via
`python/export_wav2vec2_onnx.py`.

**Punchline: the community ONNX is already bit-near-exact with
PyTorch eager.** When the audio loader is held constant (both consume
the same `np.float32` waveform from the WAV), max emission diff on
the 01_dialogue/seg-3 hallucination is **2.44e-3 nats** (mean 2.10e-4,
0 cells > 1e-2). Re-exporting yields **2.27e-3** — a wash. There is
no encoder gap to close.

The community export and the re-export produce **identical IoU
scores** on the 5-fixture suite. After the audio-loader switch to
`ffmpeg-next` (2026-05) the harness yields:

| Fixture                | Median IoU | below 0.5 |
|------------------------|------------|-----------|
| `01_dialogue/`         | 0.997      | 0         |
| `02_pyannote_sample/`  | 0.997      | 0         |
| `03_dual_speaker/`     | 0.995      | 0         |
| `04_three_speaker/`    | 0.999      | 0         |
| `05_four_speaker/`     | 0.996      | 0         |

The 19 below-0.5 outliers in `01_dialogue` previously corresponded
to a Whisper hallucination ("no, no, no..." × 112) where whispery's
per-word alignments lined up one-token offset against WhisperX's.
Replacing the harness's `hound`-based loader with one that mirrors
WhisperX's `load_audio` byte-for-byte (`ffmpeg → s16le → /32768.0`)
removed the loader-driven divergence and the off-by-one outliers
disappeared.

The previously-published "1.45 nat encoder divergence on seg 20" was
in fact an **audio-loader divergence**, not an encoder divergence.
WhisperX's `load_audio` runs ffmpeg → s16le → `np.float32 / 32768.0`,
which is a lossy round-trip for f32-encoded WAV files (this corpus is
all f32-encoded). whispery's old `hound`-based loader read f32 samples
directly. The ~30-PPM input mismatch propagated through 12 transformer
layers and produced ~1.45 nat divergence on numerically sensitive
frames. The harness now uses `ffmpeg-next` with the same
`s16 → /32768.0` quantization on whispery's side, so both sides see
byte-identical f32 buffers — the harness emits matching `clip_sha256`
fields on both runners' JSON whenever it does.

If you need bit-near parity with PyTorch eager on the encoder
(e.g. for a custom corpus where path-flip behaviour differs), use
`python/export_wav2vec2_onnx.py` to regenerate the ONNX:

```bash
# One-time: separate venv pinned at transformers 4.49 (the parity
# venv pins transformers 5.x for whisperX, which has a mask-shape
# bug that crashes torch.onnx.export tracing).
uv venv /tmp/wav2vec2-export-venv --python 3.12
uv pip install --python /tmp/wav2vec2-export-venv/bin/python \
    torch==2.6.0 transformers==4.49.0 onnx onnxscript numpy

# Run.
/tmp/wav2vec2-export-venv/bin/python \
    tests/parity_whisperx/python/export_wav2vec2_onnx.py
```

The script writes:

- `wav2vec2-base-960h.repro.onnx` — legacy `torch.onnx.export`
  (jit-trace, opset 17, `do_constant_folding=False`). Variable-length
  input axis. ~378 MB. Confirmed: max emission diff vs
  `Wav2Vec2ForCTC` PyTorch eager on this corpus is **2.27e-3 nats**
  when the input audio matches.

- `wav2vec2-base-960h.dynamo.onnx` — new FX-based exporter
  (`dynamo=True`, opset 18). On torch 2.6 + transformers 4.49 this
  path falls back to TorchScript graph capture (the wav2vec2 conv
  stack divides input length by 320, which torch.export's symbolic
  shape solver can't satisfy) and produces a **fixed-shape** graph
  baked at the export time's dummy length (80,000 samples). Not
  usable for variable-length inference. Kept as a diagnostic; see
  the docstring of `export_wav2vec2_onnx.py`.

Wire either into a parity run via:

```bash
WAV2VEC2_ONNX_PATH=/path/to/wav2vec2-base-960h.repro.onnx \
    ./tests/parity_whisperx/run.sh /path/to/fixture
```

Production whispery (the `cargo build` path) continues to use the
community export. Re-export is opt-in.

[whisperx]: https://github.com/m-bain/whisperX
