"""Compare WhisperX-vs-whispery trellis + emission dumps.

Inputs (paths supplied via flags; defaults match what
`dump_wx_segment.py` and the `parity-dump-emission`-feature build of
the parity runner write to `tests/parity_whisperx/out/`):

  --wx-base out/wx_seg20_dump        # WhisperX dump basename
  --wy-base out/wy_seg20_dump        # whispery dump basename

Each basename has companions `<base>.emission.bin`, `<base>.trellis.bin`,
and `<base>.tokens.json` (whispery) or `<base>.json` (WhisperX, which
also contains tokens). All `*.bin` files share the same header layout:
two LE-u32 dimensions then row-major LE-f32 data.

Reports:
1. Emission diff: how much do ORT (whispery) and PyTorch (WhisperX)
   disagree on the wav2vec2 outputs for this same segment of audio?
2. Trellis diff: how much do the resulting forward DP cells diverge?
3. (Optional) path diff: where do the resulting beam paths first
   disagree?

This is a *diagnostic* — it doesn't fix anything, just quantifies the
gap so you can decide whether to ship as-is or invest more in
ORT/PyTorch parity.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def _read_bin(path: Path):
    with path.open("rb") as f:
        d0 = struct.unpack("<I", f.read(4))[0]
        d1 = struct.unpack("<I", f.read(4))[0]
        n = d0 * d1
        arr = np.frombuffer(f.read(4 * n), dtype="<f4").copy()
    return d0, d1, arr.reshape(d0, d1)


def _stats(name: str, diffs: np.ndarray) -> None:
    mask = np.isfinite(diffs)
    finite = diffs[mask]
    print(f"{name}: finite_pairs={len(finite)} max={finite.max():.4e} "
          f"mean={finite.mean():.4e} median={np.median(finite):.4e}")
    for thresh in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        cnt = int((finite > thresh).sum())
        print(f"  count > {thresh:>4}: {cnt}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wx-base", type=Path, required=True,
                   help="WhisperX dump basename (no extension).")
    p.add_argument("--wy-base", type=Path, required=True,
                   help="whispery dump basename (no extension).")
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    wx_em_path = args.wx_base.with_suffix(".emission.bin")
    wy_em_path = args.wy_base.with_suffix(".emission.bin")
    wx_tr_path = args.wx_base.with_suffix(".trellis.bin")
    wy_tr_path = args.wy_base.with_suffix(".trellis.bin")

    wx_t, wx_v, wx_em = _read_bin(wx_em_path)
    wy_t, wy_v, wy_em = _read_bin(wy_em_path)
    print(f"WhisperX emission: T={wx_t} V={wx_v}")
    print(f"whispery emission: T={wy_t} V={wy_v}")
    if (wx_t, wx_v) != (wy_t, wy_v):
        print("EMISSION SHAPE MISMATCH; cannot diff cell-by-cell")
        return 2

    abs_diff_em = np.abs(wx_em - wy_em)
    print()
    print("=== EMISSION DIFF (ORT vs PyTorch wav2vec2) ===")
    _stats("emission", abs_diff_em)

    # Top diffs
    flat = abs_diff_em.flatten()
    top = np.argsort(flat)[::-1][: args.top]
    print(f"top-{args.top} cells:")
    for i in top:
        ti, vi = i // wy_v, i % wy_v
        print(
            f"  (t={ti:>5}, v={vi:>3}): wy={wy_em[ti, vi]:.4f} "
            f"wx={wx_em[ti, vi]:.4f} |diff|={flat[i]:.4e}"
        )

    wx_tr_t, wx_tr_n, wx_tr = _read_bin(wx_tr_path)
    wy_tr_t, wy_tr_n, wy_tr = _read_bin(wy_tr_path)
    if (wx_tr_t, wx_tr_n) != (wy_tr_t, wy_tr_n):
        print("TRELLIS SHAPE MISMATCH; cannot diff cell-by-cell")
        return 2

    print()
    print("=== TRELLIS DIFF (whispery's ORT-driven forward DP vs WhisperX's PyTorch-driven) ===")
    # Mask out -inf vs -inf etc — only care about finite-vs-finite.
    fmask = np.isfinite(wx_tr) & np.isfinite(wy_tr)
    abs_diff_tr = np.where(fmask, np.abs(wx_tr - wy_tr), np.nan)
    _stats("trellis", abs_diff_tr)

    # Largest trellis diffs — these tell us if there are particular
    # frames where the path comparison is going to flip.
    finite_idx = np.flatnonzero(fmask)
    finite_diffs = abs_diff_tr.flatten()[finite_idx]
    if finite_diffs.size:
        order = np.argsort(finite_diffs)[::-1][: args.top]
        print(f"top-{args.top} trellis cells:")
        for k in order:
            i = finite_idx[k]
            ti, ji = i // wy_tr_n, i % wy_tr_n
            print(
                f"  (t={ti:>5}, j={ji:>5}): wy={wy_tr[ti, ji]:.4f} "
                f"wx={wx_tr[ti, ji]:.4f} |diff|={finite_diffs[k]:.4e}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
