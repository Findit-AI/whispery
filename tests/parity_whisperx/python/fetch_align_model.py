"""Download a HuggingFace wav2vec2-CTC alignment model + tokenizer
and export to ONNX, ready to feed into `Aligner::from_paths`.

Mirrors WhisperX's `DEFAULT_ALIGN_MODELS_HF` (and the torchaudio
"VOXPOPULI_*" / "WAV2VEC2_*" pipelines via their underlying HF
checkpoints) so whispery can drop into 1:1 language coverage.

Usage:
    # By language code (canonical):
    /tmp/wav2vec2-export-venv/bin/python fetch_align_model.py ja
    /tmp/wav2vec2-export-venv/bin/python fetch_align_model.py zh

    # By explicit HF model name:
    /tmp/wav2vec2-export-venv/bin/python fetch_align_model.py \\
        --model jonatasgrosman/wav2vec2-large-xlsr-53-japanese

    # Custom output dir:
    /tmp/wav2vec2-export-venv/bin/python fetch_align_model.py ja \\
        --out-dir /path/to/dest

Output layout (per model):
    <out-dir>/<safe-name>.onnx
    <out-dir>/<safe-name>-tokenizer.json
where <safe-name> = <hf-org>--<hf-name> with `/` and `_` replaced
by `-` for filesystem safety. The whispery `Aligner::from_paths`
accepts both files directly.

Environment requirements: this script uses `torch.onnx.export` with
the legacy tracing exporter. The whisperX parity venv pins
`transformers == 5.7` which BREAKS `torch.onnx.export` (the new
`sdpa_mask` path crashes mid-trace). Run from a SEPARATE venv with
`transformers == 4.49` (or any 4.4x line):

    uv venv /tmp/wav2vec2-export-venv --python 3.12
    /tmp/wav2vec2-export-venv/bin/pip install \\
        torch==2.6.0 transformers==4.49.0 onnx onnxscript numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

# 1:1 mirror of WhisperX's DEFAULT_ALIGN_MODELS_TORCH +
# DEFAULT_ALIGN_MODELS_HF, but resolved to HF model names so we can
# uniformly download and ONNX-export them. The torchaudio
# "VOXPOPULI_ASR_BASE_10K_*" pipelines wrap the same upstream HF
# checkpoints by `facebook/wav2vec2-base-10k-voxpopuli-ft-{lang}`,
# and "WAV2VEC2_ASR_BASE_960H" is `facebook/wav2vec2-base-960h`.
DEFAULT_MODELS: dict[str, str] = {
    # Torchaudio-bundled (in WhisperX): English + voxpopuli set.
    "en": "facebook/wav2vec2-base-960h",
    "fr": "facebook/wav2vec2-base-10k-voxpopuli-ft-fr",
    "de": "facebook/wav2vec2-base-10k-voxpopuli-ft-de",
    "es": "facebook/wav2vec2-base-10k-voxpopuli-ft-es",
    "it": "facebook/wav2vec2-base-10k-voxpopuli-ft-it",
    # HuggingFace community models (DEFAULT_ALIGN_MODELS_HF).
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
}

SAMPLE_RATE = 16_000
DUMMY_LEN = 5 * SAMPLE_RATE  # 80,000 samples — covers every shape via dynamic axis


class CTCWrapper(torch.nn.Module):
    """Expose only `.logits` so the exported ONNX graph has a single output."""

    def __init__(self, m: Wav2Vec2ForCTC):
        super().__init__()
        self.m = m

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.m(input_values).logits


def safe_name(hf_path: str) -> str:
    """Convert an HF org/name path into a filesystem-safe stem.

    `org/name` -> `org--name`. Underscores stay intact (HF model
    names commonly include them). Dashes in the path itself are
    preserved.
    """
    return hf_path.replace("/", "--")


def export_onnx(model_name: str, out_dir: Path, opset: int) -> Path:
    """Download model + processor from HF, ONNX-export, save tokenizer.

    Returns the path to the .onnx file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(model_name)
    onnx_path = out_dir / f"{stem}.onnx"
    tokenizer_path = out_dir / f"{stem}-tokenizer.json"

    if onnx_path.exists() and tokenizer_path.exists():
        print(f"[skip] {model_name} already at {onnx_path} (+tokenizer)")
        return onnx_path

    print(f"[download] {model_name}")
    t0 = time.time()
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"[download] done in {time.time() - t0:.1f}s")

    # Tokenizer.json — whispery's `Aligner::from_paths` reads this
    # via the `tokenizers` crate (HF tokenizers binding).
    # `processor.tokenizer.save_pretrained` writes a directory; we
    # extract just the tokenizer.json that the tokenizers crate
    # actually consumes.
    t0 = time.time()
    tmp_tok_dir = out_dir / f"{stem}-tokenizer-staging"
    processor.tokenizer.save_pretrained(str(tmp_tok_dir))
    src_tok = tmp_tok_dir / "tokenizer.json"
    if not src_tok.exists():
        # Some older HF tokenizers don't auto-emit `tokenizer.json` from
        # `save_pretrained`; rebuild it from `vocab.json`.
        vocab_path = tmp_tok_dir / "vocab.json"
        if vocab_path.exists():
            print(f"[tokenizer] rebuilding tokenizer.json from vocab.json")
            vocab = json.loads(vocab_path.read_text())
            # Minimal HF tokenizers WordLevel format. The `unk_token`
            # is read from the saved tokenizer config when present;
            # default to "<unk>" otherwise.
            tok_config_path = tmp_tok_dir / "tokenizer_config.json"
            unk_token = "<unk>"
            if tok_config_path.exists():
                tok_config = json.loads(tok_config_path.read_text())
                unk_token = tok_config.get("unk_token", "<unk>")
            tokenizer_json = {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": ""},
                    "behavior": "Isolated",
                    "invert": False,
                },
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": vocab,
                    "unk_token": unk_token,
                },
            }
            src_tok.parent.mkdir(parents=True, exist_ok=True)
            src_tok.write_text(json.dumps(tokenizer_json, indent=2))
        else:
            raise SystemExit(
                f"[tokenizer] save_pretrained did not emit tokenizer.json or "
                f"vocab.json under {tmp_tok_dir}; cannot proceed"
            )
    src_tok.replace(tokenizer_path)
    # Clean up staging dir.
    for p in tmp_tok_dir.glob("*"):
        p.unlink()
    tmp_tok_dir.rmdir()
    print(f"[tokenizer] wrote {tokenizer_path} in {time.time() - t0:.1f}s")

    # ONNX export — legacy tracing path (`dynamo=False`). The newer
    # `dynamo=True` path is faster + closer to eager but is gated on
    # opset>=18 and requires a recent torch; legacy is the safer
    # default for the canonical fetcher. The diagnostic-only
    # `export_wav2vec2_onnx.py` script next door covers both paths
    # for parity-strict re-export.
    print(f"[onnx] exporting opset={opset} -> {onnx_path}")
    t0 = time.time()
    dummy = torch.zeros(1, DUMMY_LEN, dtype=torch.float32)
    wrapped = CTCWrapper(model).eval()
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {1: "T_samples"},
            "logits": {1: "T_frames"},
        },
        opset_version=opset,
        # Disable constant folding — it reorders float-add chains
        # and is a known source of ORT-vs-eager numeric drift.
        do_constant_folding=False,
        dynamo=False,
        export_params=True,
    )
    sz = onnx_path.stat().st_size
    print(f"[onnx] done in {time.time() - t0:.1f}s, {sz / 1e6:.1f} MB")
    return onnx_path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download + ONNX-export a wav2vec2 alignment model.",
    )
    p.add_argument(
        "lang",
        nargs="?",
        type=str,
        help="ISO 639 language code (e.g. 'ja', 'zh'). Use --list to see all.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit HuggingFace model path (overrides --lang resolution).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "WHISPERY_TEST_FIXTURES",
                "/Users/user/.cargo/target/whispery-test-fixtures",
            )
        ),
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (default 17; legacy exporter supports 11..17).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the language→model registry and exit.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download every model in the registry. Network/disk heavy.",
    )
    args = p.parse_args()

    if args.list:
        for code, model in DEFAULT_MODELS.items():
            print(f"  {code:5} {model}")
        return 0

    if args.all:
        if args.lang or args.model:
            print(
                "[fatal] --all is mutually exclusive with --lang / --model",
                file=sys.stderr,
            )
            return 64
        for code, model in DEFAULT_MODELS.items():
            print(f"=== {code}: {model} ===")
            export_onnx(model, args.out_dir, args.opset)
        return 0

    if args.model:
        model_name = args.model
    elif args.lang:
        if args.lang not in DEFAULT_MODELS:
            print(
                f"[fatal] no default model for language {args.lang!r}; "
                f"pass --model <hf-path> or pick one of: "
                f"{', '.join(sorted(DEFAULT_MODELS))}",
                file=sys.stderr,
            )
            return 64
        model_name = DEFAULT_MODELS[args.lang]
    else:
        p.print_help(sys.stderr)
        return 64

    export_onnx(model_name, args.out_dir, args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
