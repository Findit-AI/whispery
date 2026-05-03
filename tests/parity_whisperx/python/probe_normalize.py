"""Compare HF wav2vec2 emissions: raw waveform vs processor-normalized.

WhisperX's `alignment.py` skips `Wav2Vec2Processor` entirely on the
HuggingFace path (`facebook/wav2vec2-base-960h`); it feeds the raw
audio buffer to `model.forward`. The processor would mean/var
normalize. This probe runs both forms and reports the gap so we can
assess how much of the whispery↔WhisperX divergence is rooted in
"whispery normalizes (correct) vs WhisperX doesn't (the de facto
reference)".
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import whisperx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("seg_index", type=int)
    parser.add_argument("--whisperx-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wav_path = args.wav_path.resolve()
    if args.whisperx_json is None:
        out_dir = Path(__file__).parent.parent / "out"
        wxj = out_dir / f"whisperx_{wav_path.parent.name}.json"
    else:
        wxj = args.whisperx_json
    payload = json.loads(wxj.read_text())
    seg = payload["raw_asr_segments"][args.seg_index]
    t1 = float(seg["start_s"])
    t2 = float(seg["end_s"])

    audio = whisperx.load_audio(str(wav_path))
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    f1 = int(t1 * 16000)
    f2 = int(t2 * 16000)
    waveform = audio_t[:, f1:f2]
    if waveform.shape[-1] < 400:
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    with torch.inference_mode():
        emis_raw = torch.log_softmax(model(waveform).logits, dim=-1)[0]

    # Processor normalize.
    samples_np = waveform[0].numpy()
    inputs = processor(
        samples_np, sampling_rate=16_000, return_tensors="pt"
    ).input_values
    print(f"raw mean/std: {samples_np.mean():.6f} / {samples_np.std():.6f}")
    print(
        f"processed mean/std: {inputs[0].numpy().mean():.6f} / "
        f"{inputs[0].numpy().std():.6f}"
    )
    with torch.inference_mode():
        emis_proc = torch.log_softmax(model(inputs).logits, dim=-1)[0]

    em_raw = emis_raw.numpy()
    em_proc = emis_proc.numpy()
    delta = em_proc - em_raw

    blank_id = 0
    out = {
        "T_raw": int(em_raw.shape[0]),
        "T_proc": int(em_proc.shape[0]),
        "stats_raw": {
            "mean": float(em_raw.mean()),
            "std": float(em_raw.std()),
            "blank_mean": float(em_raw[:, blank_id].mean()),
            "blank_std": float(em_raw[:, blank_id].std()),
        },
        "stats_proc": {
            "mean": float(em_proc.mean()),
            "std": float(em_proc.std()),
            "blank_mean": float(em_proc[:, blank_id].mean()),
            "blank_std": float(em_proc[:, blank_id].std()),
        },
        "delta_stats": {
            "max_abs": float(np.abs(delta).max()),
            "mean_abs": float(np.abs(delta).mean()),
            "rms": float(np.sqrt((delta * delta).mean())),
            "blank_max_abs": float(np.abs(delta[:, blank_id]).max()),
            "blank_mean_abs": float(np.abs(delta[:, blank_id]).mean()),
        },
        # Per-frame argmax agreement first 100
        "argmax_agree_first_100": int(
            (np.argmax(em_raw[:100], axis=1) == np.argmax(em_proc[:100], axis=1)).sum()
        ),
        "argmax_total": int(em_raw.shape[0]),
        "argmax_agree_total": int(
            (np.argmax(em_raw, axis=1) == np.argmax(em_proc, axis=1)).sum()
        ),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
