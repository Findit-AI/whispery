#!/bin/bash
# Populate `tests/parity/fixtures/` with WAV clips +
# RTTM speaker-reference annotations from the audio-fixtures
# repo. Idempotent — safe to re-run.
#
# Layout produced (matches the pre-extraction layout that the
# asry test code expected):
#
#   tests/parity/fixtures/<name>/
#     clip_16k.wav       <- from pcm_s16le/<name>.wav
#     reference.rttm     <- from references/<name>.rttm
#
# Source repo:
#   https://github.com/Findit-AI/audio-fixtures
#
# Override the source via ASRY_FIXTURES_REPO_URL /
# ASRY_FIXTURES_REPO_REF (defaults below). Set
# ASRY_FIXTURES_KEEP_CLONE=1 to leave the shallow clone
# under `target/audio-fixtures/` for inspection; default
# behaviour cleans it up after the copy.

set -euo pipefail

REPO_URL="${ASRY_FIXTURES_REPO_URL:-https://github.com/Findit-AI/audio-fixtures.git}"
REPO_REF="${ASRY_FIXTURES_REPO_REF:-main}"
DEST="tests/parity/fixtures"
CLONE_DIR="target/audio-fixtures"
CRATE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$CRATE_ROOT"

if [ -d "$DEST" ] && [ "${ASRY_FIXTURES_FORCE:-0}" != "1" ]; then
  if [ -f "$DEST/01_dialogue/clip_16k.wav" ] && [ -f "$DEST/01_dialogue/reference.rttm" ]; then
    echo "fetch_fixtures: $DEST already populated; set ASRY_FIXTURES_FORCE=1 to refetch."
    exit 0
  fi
fi

mkdir -p target
rm -rf "$CLONE_DIR"
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$CLONE_DIR"

mkdir -p "$DEST"
shopt -s nullglob
for wav in "$CLONE_DIR"/pcm_s16le/*.wav; do
  name="$(basename "$wav" .wav)"
  mkdir -p "$DEST/$name"
  cp "$wav" "$DEST/$name/clip_16k.wav"
  rttm="$CLONE_DIR/references/${name}.rttm"
  if [ -f "$rttm" ]; then
    cp "$rttm" "$DEST/$name/reference.rttm"
  else
    echo "fetch_fixtures: WARN no rttm for $name (skipping reference)"
  fi
done
shopt -u nullglob

if [ "${ASRY_FIXTURES_KEEP_CLONE:-0}" != "1" ]; then
  rm -rf "$CLONE_DIR"
fi

count=$(find "$DEST" -name 'clip_16k.wav' | wc -l | tr -d ' ')
echo "fetch_fixtures: populated $count fixture(s) under $DEST"
