"""Run WhisperX (faster-whisper ASR + wav2vec2-base-960h alignment) on
a 16 kHz mono WAV; emit JSON in the same schema whispery's runner uses,
with `runner = "whisperx"`. Pair with `score.py` for IoU comparison.

Pinned to CPU + int8 compute_type for determinism: GPU CTranslate2 has
floating-point non-determinism (cuBLAS algorithm selection) that would
add noise to the parity numbers without changing the actual alignment
model.

Usage:
    uv run python whisperx_runner.py <wav_path> --out <json_path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import soundfile as sf

import whisperx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_f32_buffer(audio) -> str:
    """SHA-256 of an `np.float32` array's little-endian byte
    representation. We use this (rather than `sha256_file`) so the
    hash matches whispery-parity-runner's `clip_sha256`, which is
    computed over the f32 buffer the model actually consumes. Both
    runners load via ffmpeg (WhisperX's `load_audio` shells out;
    whispery's runner uses `ffmpeg-next` bindings) and therefore
    produce byte-identical f32 buffers — the matching hash is what
    proves the audio-loader divergence the README documented at
    parity-runner v1 is closed.
    """
    # `audio.tobytes()` walks the array in C-order; for a 1-D
    # `np.float32` buffer that's the same byte layout whispery
    # writes when it casts its `Vec<f32>` to `&[u8]`.
    h = hashlib.sha256()
    h.update(audio.tobytes(order="C"))
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run WhisperX alignment on a WAV; emit whispery-parity JSON."
    )
    parser.add_argument("wav_path", type=Path, help="16 kHz mono WAV.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output JSON path (default: stdout)."
    )
    parser.add_argument(
        "--whisper-model",
        default="tiny.en",
        help="faster-whisper model name (default: tiny.en).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="`cpu` or `cuda`; default cpu for determinism.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="ctranslate2 compute_type (default: int8 for cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batched faster-whisper batch size (default: 8).",
    )
    args = parser.parse_args()

    wav_path = args.wav_path.resolve()
    if not wav_path.is_file():
        print(f"WAV not found: {wav_path}", file=sys.stderr)
        return 2

    # `soundfile.read` returns float32 in [-1, 1]; we don't actually
    # need the samples here (whisperx loads them itself) but we keep
    # the read so we can validate sample rate + channels and report
    # duration_s from the ground truth, not whatever whisperx
    # internally resampled to.
    audio_data, sample_rate = sf.read(str(wav_path), dtype="float32")
    if sample_rate != 16_000:
        print(
            f"{wav_path}: expected 16 kHz, got {sample_rate} Hz",
            file=sys.stderr,
        )
        return 2
    if audio_data.ndim != 1:
        print(
            f"{wav_path}: expected mono, got shape {audio_data.shape}",
            file=sys.stderr,
        )
        return 2
    duration_s = float(len(audio_data)) / sample_rate

    print(
        f"[whisperx-parity] wav={wav_path} dur={duration_s:.2f}s "
        f"model={args.whisper_model} device={args.device}",
        file=sys.stderr,
    )

    t0 = time.monotonic()

    # 1) faster-whisper ASR. WhisperX uses its own batched inference
    # wrapper; the call signature is stable across 3.4.x. The
    # returned `result` is a dict-of-lists with `segments`, each
    # carrying tokenwise text + coarse word ranges.
    asr_model = whisperx.load_model(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        # English-locked; matches the whispery runner's
        # `LanguagePolicy::Lock { hint: Lang::En }`.
        language="en",
        # Use silero VAD instead of the default pyannote VAD: the
        # pyannote VAD checkpoint on HuggingFace pickles
        # `omegaconf.ListConfig`, which torch >= 2.6 refuses to
        # unpickle under the new `weights_only=True` default.
        # Silero ships its weights via torch.hub in a flat tensor
        # format that loads cleanly under the same default.
        # Alignment is run over the whole audio anyway so VAD
        # choice has no effect on the per-word timing comparison.
        vad_method="silero",
    )
    audio = whisperx.load_audio(str(wav_path))

    # SHA-256 over the actual float32 buffer the model consumes.
    # See `sha256_f32_buffer` docstring for the rationale; this is
    # the byte-identity check that lets the harness verify both
    # runners decoded the audio the same way.
    clip_sha256 = sha256_f32_buffer(audio)

    result = asr_model.transcribe(audio, batch_size=args.batch_size)
    t_asr = time.monotonic()

    # 2) wav2vec2 forced alignment. Same model family that whispery
    # loads via ONNX (`facebook/wav2vec2-base-960h`); WhisperX picks
    # it automatically for `language_code="en"`.
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en",
        device=args.device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device=args.device,
        return_char_alignments=False,
    )
    t_align = time.monotonic()

    print(
        f"[whisperx-parity] asr={t_asr - t0:.2f}s align={t_align - t_asr:.2f}s",
        file=sys.stderr,
    )

    # WhisperX's aligned output: segments[].words[] each with
    # `{ "word", "start", "end", "score" }`. Some words may lack
    # `start`/`end` if alignment failed for that span; we drop those
    # so downstream IoU only sees fully-aligned words.
    #
    # We emit BOTH a per-segment `segments[]` array (so the Rust
    # runner can mirror WhisperX's per-segment alignment flow — each
    # segment is a separate `align_chunk` call with `t1 = seg.start_s`
    # as the audio anchor) AND a flat `words[]` (which `score.py`
    # consumes; kept for backwards compat).
    out_words = []
    out_segments = []
    for seg in aligned["segments"]:
        seg_words = []
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            entry = {
                "text": w["word"],
                "start_s": float(w["start"]),
                "end_s": float(w["end"]),
                # WhisperX's `score` is the wav2vec2 CTC
                # alignment score; same semantics as
                # whispery's `Word::score`.
                "score": float(w.get("score", 0.0)),
            }
            seg_words.append(entry)
            out_words.append(entry)

        # Segment text: prefer the `text` field WhisperX threads
        # through alignment (it carries the original Whisper
        # transcription verbatim). Fall back to joining word texts
        # when the field is absent — diagnostic-only, alignment is
        # driven by the per-segment word list.
        seg_text = seg.get("text", " ".join(w["text"] for w in seg_words))
        # WhisperX retains segment-level start/end after alignment
        # (alignment.py keeps `aligned_seg["start"] / ["end"]`).
        # Fall back to bracketing word ranges if missing.
        if "start" in seg and "end" in seg:
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
        elif seg_words:
            seg_start = min(w["start_s"] for w in seg_words)
            seg_end = max(w["end_s"] for w in seg_words)
        else:
            # Empty segment with no times either — skip; the Rust
            # runner is told to skip these too.
            continue

        out_segments.append(
            {
                "start_s": seg_start,
                "end_s": seg_end,
                "text": seg_text.strip() if isinstance(seg_text, str) else "",
                "words": seg_words,
            }
        )

    # Raw ASR segments — exactly what `result["segments"]` carries
    # before alignment. Whispery's parity runner consumes these
    # to match WhisperX's GLOBAL alignment behaviour: each raw
    # segment carries the full ASR text (potentially with
    # hallucinated repetitions on long silences) and the audio
    # span the ASR claimed for it. Aligning per-raw-segment is
    # what WhisperX itself does (`whisperx.align(result["segments"], ...)`).
    # The per-sentence breakdown WhisperX adds afterwards is a
    # consumer-side derivation, not the alignment unit.
    raw_asr_segments = []
    for seg in result["segments"]:
        if "start" not in seg or "end" not in seg or "text" not in seg:
            continue
        raw_asr_segments.append({
            "start_s": float(seg["start"]),
            "end_s": float(seg["end"]),
            "text": str(seg["text"]).strip(),
        })

    payload = {
        "runner": "whisperx",
        "clip_path": str(wav_path),
        "clip_sha256": clip_sha256,
        "duration_s": duration_s,
        "transcript_count": len(aligned["segments"]),
        "segments": out_segments,
        "raw_asr_segments": raw_asr_segments,
        "words": out_words,
    }

    serialized = json.dumps(payload, indent=2)
    if args.out is None:
        print(serialized)
    else:
        args.out.write_text(serialized + "\n")
        print(
            f"[whisperx-parity] wrote {len(out_words)} words "
            f"across {payload['transcript_count']} segments to {args.out}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
