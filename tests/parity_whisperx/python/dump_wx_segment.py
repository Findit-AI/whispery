"""Dump WhisperX alignment internals for a single segment of a clip.

This is a one-shot diagnostic tool — not part of the parity harness or
test suite. It mirrors what `whisperx.alignment.align()` does for ONE
chosen segment but writes intermediate state (trellis cells, path,
char_segments, word_segments, logit statistics) to a JSON for diff
against an analogous whispery dump.

Usage:
    uv run python dump_wx_segment.py <wav_path> <segment_index> --out <json>

The segment list is read from a sibling whisperX JSON
(`out/whisperx_<fixture>.json`) created by `whisperx_runner.py`; we
DON'T re-run ASR here. Defaults to `out/whisperx_<basename>.json`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import whisperx
from whisperx.alignment import (
    backtrack_beam,
    get_trellis,
    merge_repeats,
    merge_words,
)

SAMPLE_RATE = 16_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("seg_index", type=int)
    parser.add_argument("--whisperx-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--align-model",
        default=None,
        help="Override the alignment model name (e.g. "
        "`facebook/wav2vec2-base-960h` to mirror the whispery ONNX). "
        "Defaults to whisperx's default torchaudio bundle.",
    )
    args = parser.parse_args()

    wav_path = args.wav_path.resolve()
    if args.whisperx_json is None:
        out_dir = Path(__file__).parent.parent / "out"
        wxj = out_dir / f"whisperx_{wav_path.parent.name}.json"
    else:
        wxj = args.whisperx_json
    payload = json.loads(wxj.read_text())
    raw = payload["raw_asr_segments"]
    if args.seg_index < 0 or args.seg_index >= len(raw):
        print(f"seg index out of range: {args.seg_index} vs len {len(raw)}")
        return 2
    seg = raw[args.seg_index]
    t1 = float(seg["start_s"])
    t2 = float(seg["end_s"])
    text = seg["text"]
    print(f"segment {args.seg_index}: [{t1:.3f}, {t2:.3f}]s text={text[:80]!r}...")

    # Load full audio.
    audio = whisperx.load_audio(str(wav_path))
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    f1 = int(t1 * SAMPLE_RATE)
    f2 = int(t2 * SAMPLE_RATE)
    waveform_segment = audio_t[:, f1:f2]
    if waveform_segment.shape[-1] < 400:
        lengths = torch.tensor([waveform_segment.shape[-1]])
        waveform_segment = torch.nn.functional.pad(
            waveform_segment, (0, 400 - waveform_segment.shape[-1])
        )
    else:
        lengths = None
    print(f"waveform_segment shape = {tuple(waveform_segment.shape)}")

    # Load align model.
    model, meta = whisperx.load_align_model(
        language_code="en", device="cpu", model_name=args.align_model
    )
    model_dictionary = meta["dictionary"]
    blank_id = 0
    for char, code in model_dictionary.items():
        if char == "[pad]" or char == "<pad>":
            blank_id = code
    print(f"blank_id = {blank_id}, vocab size = {len(model_dictionary)}")

    # Build clean_char/tokens just like alignment.py does.
    num_leading = len(text) - len(text.lstrip())
    num_trailing = len(text) - len(text.rstrip())
    clean_char, clean_cdx = [], []
    for cdx, char in enumerate(text):
        char_ = char.lower().replace(" ", "|")
        if cdx < num_leading:
            pass
        elif cdx > len(text) - num_trailing - 1:
            pass
        elif char_ in model_dictionary:
            clean_char.append(char_)
            clean_cdx.append(cdx)
        else:
            clean_char.append("*")
            clean_cdx.append(cdx)
    text_clean = "".join(clean_char)
    tokens = [model_dictionary.get(c, -1) for c in text_clean]
    print(f"text_clean ({len(text_clean)} chars) = {text_clean[:80]!r}...")
    print(f"tokens (first 30) = {tokens[:30]}")

    with torch.inference_mode():
        emissions = model(waveform_segment, lengths=lengths)[0]
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()
    T = emission.shape[0]
    V = emission.shape[1]
    print(f"emission T={T} V={V}")

    # Logit stats.
    em = emission.numpy()
    em_stats = {
        "T": int(T),
        "V": int(V),
        "mean": float(em.mean()),
        "std": float(em.std()),
        "min": float(em.min()),
        "max": float(em.max()),
        # Per-frame blank logprob mean / std across T frames
        "blank_mean": float(em[:, blank_id].mean()),
        "blank_std": float(em[:, blank_id].std()),
        # Frame-by-frame argmax chain (first 80 frames)
        "argmax_first_80": [int(x) for x in np.argmax(em[:80], axis=1)],
    }

    trellis = get_trellis(emission, tokens, blank_id)
    tr = trellis.numpy()
    print(f"trellis shape = {tr.shape}, finite count = {np.isfinite(tr).sum()}")

    # Trellis corner / diagonal samples.
    NTOK = len(tokens)
    corners = {
        "[0,0]": float(tr[0, 0]),
        "[0,1]": float(tr[0, 1]) if NTOK > 1 else None,
        "[1,0]": float(tr[1, 0]),
        "[T-1,0]": float(tr[T - 1, 0]),
        "[T-1,N-1]": float(tr[T - 1, NTOK - 1]),
        "[T-2,N-1]": float(tr[T - 2, NTOK - 1]) if T > 1 else None,
        "[T-1,N-2]": float(tr[T - 1, NTOK - 2]) if NTOK > 1 else None,
    }

    path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)
    if path is None:
        print("backtrack_beam returned None")
        return 1
    path_serial = [
        {"token_index": p.token_index, "time_index": p.time_index, "score": float(p.score)}
        for p in path
    ]
    print(f"path len = {len(path_serial)}")

    char_segments = merge_repeats(path, text_clean)
    char_serial = [
        {
            "label": s.label,
            "start": int(s.start),
            "end": int(s.end),
            "score": float(s.score),
        }
        for s in char_segments
    ]
    print(f"char_segments len = {len(char_serial)}")

    word_segments = merge_words(char_segments, separator="|")
    word_serial = [
        {
            "label": s.label,
            "start": int(s.start),
            "end": int(s.end),
            "score": float(s.score),
        }
        for s in word_segments
    ]
    print(f"word_segments len = {len(word_serial)}")

    # WhisperX's frame -> seconds conversion uses
    # `ratio = duration / (T - 1)` where `duration = t2 - t1`.
    ratio = (t2 - t1) / (T - 1) if T > 1 else 0.0
    word_times = []
    for w in word_segments:
        start_s = round(w.start * ratio + t1, 3)
        end_s = round(w.end * ratio + t1, 3)
        word_times.append(
            {"label": w.label, "start_s": start_s, "end_s": end_s, "score": float(w.score)}
        )

    out = {
        "segment_index": args.seg_index,
        "t1": t1,
        "t2": t2,
        "text": text,
        "text_clean": text_clean,
        "n_samples": int(f2 - f1),
        "blank_id": int(blank_id),
        "vocab_size": int(V),
        "tokens": tokens,
        "emission_stats": em_stats,
        "trellis_corners": corners,
        # Per-frame blank emission for the first 100 frames (tiny;
        # useful to compare absolute values).
        "blank_emission_first_100": [float(em[i, blank_id]) for i in range(min(100, T))],
        # Trellis column 0 first 100 (the cumsum init).
        "trellis_col0_first_100": [float(tr[i, 0]) for i in range(min(100, T))],
        # Path (full).
        "path": path_serial,
        "char_segments": char_serial,
        "word_segments_frames": word_serial,
        "word_segments_times": word_times,
        "ratio": ratio,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
