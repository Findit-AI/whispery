#!/usr/bin/env bash
# WhisperX vs whispery alignment parity harness.
#
# Requires:
# - `<crate>/models/{ggml-tiny.en.bin, wav2vec2-base-960h.onnx,
#   wav2vec2-base-960h-tokenizer.json}` (run `WHISPERY_FETCH_MODEL=1
#   WHISPERY_FETCH_W2V=1 cargo test --features alignment` once to populate)
# - `uv` on PATH (https://docs.astral.sh/uv/)
#
# `ORT_DYLIB_PATH` (load-dynamic mode) is auto-detected from the venv's
# `onnxruntime` package below — bring up the venv first if it's missing.
# An externally-set `ORT_DYLIB_PATH` overrides the auto-detect.
#
# Usage:
#   ./tests/parity_whisperx/run.sh <fixture-dir-or-wav>
#
# Accepts either a fixture directory (uses `clip_16k.wav` inside) or a
# direct WAV path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  echo "usage: $(basename "$0") <fixture-dir|wav-path>" >&2
  exit 64
}

if [ "$#" -ne 1 ]; then
  usage
fi

ARG="$1"
if [ -d "$ARG" ]; then
  CLIP="$ARG/clip_16k.wav"
  # Fixture-dir convention: parent dir name IS the fixture name
  # (matches the dia/.../NN_xxx layout).
  FIXTURE_NAME="$(basename "$ARG")"
elif [ -f "$ARG" ]; then
  CLIP="$ARG"
  # WAV-path mode: use the file's stem so multiple WAVs sharing
  # a parent directory don't collide on the output JSON paths
  # (the previous "basename of dirname" derivation gave every
  # WAV in the same dir a single shared FIXTURE_NAME).
  FIXTURE_NAME="$(basename "$ARG" .wav)"
else
  echo "[run.sh] $ARG is neither a directory nor a WAV file" >&2
  exit 65
fi

if [ ! -f "$CLIP" ]; then
  echo "[run.sh] no clip at $CLIP" >&2
  exit 66
fi

ABS_CLIP="$(cd "$(dirname "$CLIP")" && pwd)/$(basename "$CLIP")"

OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"
WHISPERY_OUT="$OUT_DIR/whispery_${FIXTURE_NAME}.json"
WHISPERX_OUT="$OUT_DIR/whisperx_${FIXTURE_NAME}.json"
SCORE_OUT="$OUT_DIR/score_${FIXTURE_NAME}.json"

echo "[run.sh] clip: $ABS_CLIP"
echo "[run.sh] outputs: $WHISPERY_OUT, $WHISPERX_OUT, $SCORE_OUT"

# 1) uv venv for the WhisperX side.
cd "$SCRIPT_DIR/python"
if [ ! -d .venv ]; then
  echo "[run.sh] creating uv venv at $(pwd)/.venv ..."
  uv venv
fi
echo "[run.sh] syncing whisperX dependencies (cached when unchanged) ..."
uv pip install -e . > /dev/null

# 2) Python runner FIRST. The order is intentional: the upstream
# `whisper-rs` `whisper_full_with_state: failed to encode` /
# `GenericError(-6)` bug currently gates whispery's whisper.cpp side
# (same root cause that `tests/runner_e2e.rs` and
# `tests/alignment_e2e.rs` are `#[ignore]`'d for). Until that's
# fixed, we drive whispery's aligner directly with WhisperX's
# transcript via `--inject-from`. We need WhisperX's JSON before
# whispery can run, hence the order swap.
#
# When the upstream bug is fixed, the natural follow-up is to flip
# the order back and drop `--inject-from` to exercise full pipeline
# parity. The non-inject path in main.rs is kept intact for that day.
cd "$SCRIPT_DIR/python"
echo "[run.sh] running whisperx_runner.py ..."
# Multilingual mode: pass `WHISPERY_PARITY_LANGUAGE=auto` (or any
# specific code: ja/zh/...) to enable language detection and
# load whisperx's matching wav2vec2 ONNX. Default keeps the
# legacy English-locked behaviour for the dia 5-fixture suite
# whose tiny.en model can only ASR English.
WHISPERX_LANG="${WHISPERY_PARITY_LANGUAGE:-en}"
WHISPERX_MODEL="${WHISPERY_PARITY_WHISPER_MODEL:-tiny.en}"
# tiny.en is English-only by design; the .en checkpoints don't
# carry the multilingual decoder. Auto/non-en runs need a
# multilingual base model — bump to `tiny` (or larger) when
# the user requests anything other than `en`.
if [ "$WHISPERX_LANG" != "en" ] && [ "$WHISPERX_MODEL" = "tiny.en" ]; then
  WHISPERX_MODEL="tiny"
  echo "[run.sh] non-English language ($WHISPERX_LANG); auto-bumped whisper model to '$WHISPERX_MODEL' (tiny.en is English-only)"
fi
uv run python whisperx_runner.py "$ABS_CLIP" \
  --whisper-model "$WHISPERX_MODEL" \
  --language "$WHISPERX_LANG" \
  --out "$WHISPERX_OUT"

# 3) Rust runner in inject mode. Reads WhisperX's transcript and
# feeds it into whispery's aligner directly — no whisper.cpp.
#
# Auto-detect `ORT_DYLIB_PATH` from the venv's bundled
# `libonnxruntime.<ver>.dylib`. ORT 2.0.0-rc.12 in load-dynamic mode
# silently hangs in `Session::commit_from_file` when the path is
# unset (instead of erroring), so an unset env var here is a sharp
# foot-gun. An explicitly-set `ORT_DYLIB_PATH` wins.
if [ -z "${ORT_DYLIB_PATH:-}" ]; then
  ORT_DYLIB_PATH=$(find "$SCRIPT_DIR/python/.venv" \
    -name 'libonnxruntime*.dylib' -o \
    -name 'libonnxruntime*.so' 2>/dev/null | head -1)
  if [ -z "$ORT_DYLIB_PATH" ]; then
    echo "[run.sh] could not locate libonnxruntime under $SCRIPT_DIR/python/.venv/" >&2
    echo "[run.sh] (the venv must be created by step 1 above before the Rust runner can dlopen ORT)" >&2
    exit 70
  fi
  export ORT_DYLIB_PATH
  echo "[run.sh] ORT_DYLIB_PATH=$ORT_DYLIB_PATH"
fi

cd "$ROOT"
echo "[run.sh] running whispery-parity-runner (--inject-from) ..."
cargo run \
  --release \
  --manifest-path tests/parity_whisperx/Cargo.toml \
  -p whispery-parity-runner \
  --bin whispery-parity-runner \
  -- "$ABS_CLIP" \
  --inject-from "$WHISPERX_OUT" \
  --out "$WHISPERY_OUT"

# 4) Score. Captures the score's exit code and propagates it; this
# is what the `run.sh` user actually cares about.
#
# `cd` back to the python dir so `uv run` finds the venv + project
# (the previous `cargo run` cd'd to $ROOT). `score.py` is in the
# python dir alongside `pyproject.toml`.
cd "$SCRIPT_DIR/python"
echo "[run.sh] scoring ..."
set +e
uv run python score.py "$WHISPERY_OUT" "$WHISPERX_OUT" --out "$SCORE_OUT"
SCORE_RC=$?
set -e

exit $SCORE_RC
