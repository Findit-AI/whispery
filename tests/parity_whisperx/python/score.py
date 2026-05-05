"""Compare two parity-runner JSON outputs (one whispery, one whisperX)
and report word-alignment IoU statistics.

Approach:
1. Normalise word texts: lowercase, strip ASCII boundary punctuation
   (`,.;:!?"'()[]{}`).
2. Sequence-align the normalised lists with Needleman-Wunsch, scoring
   matches +1, mismatches/gaps -1.
3. For each matched pair, compute the time-range IoU.
4. Emit a JSON summary on stdout and a human-readable summary on stderr.

Exit code 0 iff ALL of:
- median IoU >= 0.95
- mean IoU >= 0.95
- below_0.5 == 0

The bar reflects the achievable baseline when both runners align the
same audio + text + segment anchors (median 0.995-0.999, mean
0.983-0.998, 0 below-0.5 outliers across the 5-fixture set, 854 word
pairs). Loosening it would mask the kind of silent regression that
once put 03_dual_speaker at median 0.196 / 70 below-0.5 because the
parity runner was feeding whispery WhisperX's POST-alignment
sub-segments instead of the raw ASR segments WhisperX itself aligns
against. See `tests/parity_whisperx/README.md` for context.

Override via `--threshold` (median floor) and `--allow-below-0-5`
(maximum allowed outlier count) for diagnostic / experimental runs.
The defaults are the production parity bar.

Usage:
    uv run python score.py <whispery.json> <whisperx.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# ASCII-only boundary punctuation. Both runners can emit unicode
# punctuation (smart quotes, em-dashes), but the wav2vec2-base-960h
# tokenizer's vocab is ASCII-only so neither runner will *align*
# non-ASCII tokens — they only show up in surface forms. Stripping
# only ASCII matches what `EnglishNormalizer` does internally on the
# Rust side.
_BOUNDARY_PUNCT = ",.;:!?\"'()[]{}-_<>/\\"


def _normalize(text: str) -> str:
    return text.lower().strip(_BOUNDARY_PUNCT).strip()


@dataclass
class WordRow:
    text: str
    norm: str
    start_s: float
    end_s: float
    score: float


def _load(path: Path) -> tuple[str, list[WordRow]]:
    payload = json.loads(path.read_text())
    rows: list[WordRow] = []
    for w in payload["words"]:
        norm = _normalize(w["text"])
        if not norm:
            # Empty post-normalisation: drop. These come from
            # punctuation-only "words" that some pipelines emit.
            continue
        rows.append(
            WordRow(
                text=w["text"],
                norm=norm,
                start_s=float(w["start_s"]),
                end_s=float(w["end_s"]),
                score=float(w.get("score", 0.0)),
            )
        )
    return payload.get("runner", path.stem), rows


def _iou(a: WordRow, b: WordRow) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    if union <= 0.0:
        return 0.0
    return inter / union


def _align(
    a: list[WordRow], b: list[WordRow]
) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch over normalised text. Returns a list of pairs
    `(i, j)` where `i` indexes `a` and `j` indexes `b`; either may be
    `None` to indicate a gap."""
    n, m = len(a), len(b)
    # +1 match, -1 mismatch / gap. Hand-rolled because we don't want
    # a python-Levenshtein dep just for one DP.
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = -i
    for j in range(1, m + 1):
        score[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_cost = 1 if a[i - 1].norm == b[j - 1].norm else -1
            score[i][j] = max(
                score[i - 1][j - 1] + diag_cost,
                score[i - 1][j] - 1,
                score[i][j - 1] - 1,
            )
    # Traceback. Tie-break: prefer diagonal (match-or-substitute) so
    # we maximise matched pairs rather than walking the gap edges.
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        diag_cost = 1 if a[i - 1].norm == b[j - 1].norm else -1
        if score[i][j] == score[i - 1][j - 1] + diag_cost:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] - 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    while i > 0:
        pairs.append((i - 1, None))
        i -= 1
    while j > 0:
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def _refine_repeated_runs(
    rows_a: list[WordRow],
    rows_b: list[WordRow],
    pairs: list[tuple[int | None, int | None]],
) -> list[tuple[int | None, int | None]]:
    """Post-process NW pairings: within a maximal run of consecutive
    pairs that ALL share the same normalised text (matched or
    gapped on either side), re-pair by greedy max-IoU matching.
    Items left without a non-zero-IoU partner become drops on
    their respective sides instead of staying as phantom
    matched-but-zero-IoU pairs.

    Why: NW's text-equality scoring is +1 regardless of timing;
    when one runner emits one extra (or one fewer) token in a
    repetition like "Watch! Watch! Watch! Watch!", NW pairs them
    sequentially from the left, leaving every same-text pair in
    the run with IoU≈0 due to a uniform time shift. The actual
    alignments differ by a single repetition; greedy by IoU
    recovers the correct N-1 pairings + 2 drops.
    """
    refined: list[tuple[int | None, int | None]] = []
    n = len(pairs)
    i = 0
    while i < n:
        # Determine the "anchor" norm for the run starting at i.
        # The run is non-empty iff pair[i] has a normalised word
        # on at least one side (matched or gapped).
        anchor_a, anchor_b = pairs[i]
        if anchor_a is not None and anchor_b is not None:
            if rows_a[anchor_a].norm != rows_b[anchor_b].norm:
                # NW substitution (different words). Not a same-
                # text region; pass through.
                refined.append(pairs[i])
                i += 1
                continue
            run_norm = rows_a[anchor_a].norm
        elif anchor_a is not None:
            run_norm = rows_a[anchor_a].norm
        else:
            run_norm = rows_b[anchor_b].norm

        # Extend the run forward over consecutive pairs whose
        # non-None side(s) all share `run_norm`.
        run_end = i
        while run_end + 1 < n:
            na, nb = pairs[run_end + 1]
            ok = True
            if na is not None and rows_a[na].norm != run_norm:
                ok = False
            if nb is not None and rows_b[nb].norm != run_norm:
                ok = False
            if not ok:
                break
            run_end += 1

        # Single-pair run: nothing to refine, pass through.
        if run_end == i:
            refined.append(pairs[i])
            i += 1
            continue

        # Collect the a- and b-indices in this run.
        a_indices = [pairs[k][0] for k in range(i, run_end + 1) if pairs[k][0] is not None]
        b_indices = [pairs[k][1] for k in range(i, run_end + 1) if pairs[k][1] is not None]

        # Greedy max-IoU matching. Threshold at IoU > 0: a pair
        # with zero overlap is no better than a drop, and keeping
        # it as a "matched" pair drags the stats down with phantom
        # zero-IoU entries.
        candidates: list[tuple[float, int, int]] = []
        for ii_, ai in enumerate(a_indices):
            for jj_, bi in enumerate(b_indices):
                iou = _iou(rows_a[ai], rows_b[bi])
                if iou > 0.0:
                    candidates.append((iou, ii_, jj_))
        candidates.sort(reverse=True)
        used_a: set[int] = set()
        used_b: set[int] = set()
        new_pairs: list[tuple[int | None, int | None]] = []
        for _iou_score, ii_, jj_ in candidates:
            if ii_ in used_a or jj_ in used_b:
                continue
            new_pairs.append((a_indices[ii_], b_indices[jj_]))
            used_a.add(ii_)
            used_b.add(jj_)
        # Items not paired become drops on their side.
        for ii_ in range(len(a_indices)):
            if ii_ not in used_a:
                new_pairs.append((a_indices[ii_], None))
        for jj_ in range(len(b_indices)):
            if jj_ not in used_b:
                new_pairs.append((None, b_indices[jj_]))

        refined.extend(new_pairs)
        i = run_end + 1

    return refined


def _drop_distant_phantoms(
    rows_a: list[WordRow],
    rows_b: list[WordRow],
    pairs: list[tuple[int | None, int | None]],
    max_distance_s: float = 2.0,
) -> list[tuple[int | None, int | None]]:
    """Demote same-text matched pairs whose time centres are more
    than `max_distance_s` apart (and IoU == 0) into drops on each
    side. Such pairs are different OCCURRENCES of the same word
    that NW happened to align by sequence — not algorithmic
    alignment drift. Treating them as IoU=0 matches drags the
    stats down with phantom failures; treating them as drops is
    truthful (different word events, no comparison possible).

    Threshold of 2s is conservative: actual CTC drift between two
    aligners on the same chunk is typically ≤ 100 ms; centre-to-
    centre distance > 2s strongly indicates different occurrences.
    """
    refined: list[tuple[int | None, int | None]] = []
    for ai, bi in pairs:
        if ai is None or bi is None:
            refined.append((ai, bi))
            continue
        wa = rows_a[ai]
        wb = rows_b[bi]
        if _iou(wa, wb) > 0.0:
            refined.append((ai, bi))
            continue
        center_a = (wa.start_s + wa.end_s) / 2.0
        center_b = (wb.start_s + wb.end_s) / 2.0
        if abs(center_a - center_b) > max_distance_s:
            # Different occurrences — split into two drops.
            refined.append((ai, None))
            refined.append((None, bi))
        else:
            refined.append((ai, bi))
    return refined


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "count": n,
        "mean": float(statistics.fmean(sv)),
        "median": float(statistics.median(sv)),
        "p10": float(sv[max(0, int(0.10 * (n - 1)))]),
        "p90": float(sv[min(n - 1, int(0.90 * (n - 1)))]),
        "min": float(sv[0]),
        "max": float(sv[-1]),
        "below_0.5": int(sum(1 for v in sv if v < 0.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score two parity-runner JSON outputs against each other."
    )
    parser.add_argument("whispery_json", type=Path)
    parser.add_argument("whisperx_json", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary here (default: stdout).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Median IoU floor for exit-code 0 (default: 0.95).",
    )
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=0.95,
        help="Mean IoU floor for exit-code 0 (default: 0.95).",
    )
    parser.add_argument(
        "--allow-below-0-5",
        type=int,
        default=0,
        help="Maximum allowed pairs with IoU < 0.5 (default: 0).",
    )
    args = parser.parse_args()

    name_a, rows_a = _load(args.whispery_json)
    name_b, rows_b = _load(args.whisperx_json)

    pairs = _align(rows_a, rows_b)
    # Refine same-text runs: fix the repeated-token off-by-N
    # phantom-mismatch pattern (e.g., "Watch! Watch! Watch! ...").
    pairs = _refine_repeated_runs(rows_a, rows_b, pairs)
    # Drop cross-region phantom matches: same-text pairs whose
    # time centres are seconds apart aren't algorithmic drift,
    # they're different occurrences NW happened to pair.
    pairs = _drop_distant_phantoms(rows_a, rows_b, pairs)
    matched: list[tuple[WordRow, WordRow, float]] = []
    dropped_a = 0
    dropped_b = 0
    for i, j in pairs:
        if i is None:
            dropped_b += 1
        elif j is None:
            dropped_a += 1
        else:
            wa = rows_a[i]
            wb = rows_b[j]
            if wa.norm == wb.norm:
                matched.append((wa, wb, _iou(wa, wb)))
            else:
                # Substitution at the alignment level — we don't
                # score IoU on different words, but they're still
                # not a "drop" on either side. Track separately.
                dropped_a += 1
                dropped_b += 1

    iou_values = [iou for _, _, iou in matched]
    iou_stats = _stats(iou_values)

    matched_sorted = sorted(matched, key=lambda t: t[2])
    worst = [
        {
            "iou": round(iou, 4),
            "whispery": {
                "text": wa.text,
                "start_s": round(wa.start_s, 3),
                "end_s": round(wa.end_s, 3),
            },
            "whisperx": {
                "text": wb.text,
                "start_s": round(wb.start_s, 3),
                "end_s": round(wb.end_s, 3),
            },
        }
        for wa, wb, iou in matched_sorted[:5]
    ]

    median = iou_stats.get("median", 0.0)
    mean = iou_stats.get("mean", 0.0)
    below_0_5 = iou_stats.get("below_0.5", 0)
    passed = bool(
        len(matched) > 0
        and median >= args.threshold
        and mean >= args.mean_threshold
        and below_0_5 <= args.allow_below_0_5
    )
    summary = {
        "whispery_word_count": len(rows_a),
        "whisperx_word_count": len(rows_b),
        "matched_pairs": len(matched),
        "dropped_by_whispery": dropped_a,
        "dropped_by_whisperx": dropped_b,
        "iou": iou_stats,
        "worst_5": worst,
        "threshold_median_iou": args.threshold,
        "threshold_mean_iou": args.mean_threshold,
        "threshold_max_below_0_5": args.allow_below_0_5,
        "passed": passed,
    }

    serialized = json.dumps(summary, indent=2)
    if args.out is None:
        print(serialized)
    else:
        args.out.write_text(serialized + "\n")

    print(
        f"\n[parity score] {name_a} ({len(rows_a)} words) vs "
        f"{name_b} ({len(rows_b)} words)",
        file=sys.stderr,
    )
    print(
        f"  matched={len(matched)} dropped_by_a={dropped_a} dropped_by_b={dropped_b}",
        file=sys.stderr,
    )
    if iou_stats["count"] == 0:
        print(
            "  no matched pairs — alignment outputs disagree on every word",
            file=sys.stderr,
        )
        return 1
    print(
        f"  IoU mean={iou_stats['mean']:.3f} median={iou_stats['median']:.3f} "
        f"p10={iou_stats['p10']:.3f} p90={iou_stats['p90']:.3f} "
        f"below_0.5={iou_stats['below_0.5']}",
        file=sys.stderr,
    )
    if worst:
        print("  worst 5:", file=sys.stderr)
        for w in worst:
            print(
                f"    iou={w['iou']:.3f} whispery="
                f"{w['whispery']['text']!r}@[{w['whispery']['start_s']:.2f},"
                f"{w['whispery']['end_s']:.2f}] whisperx="
                f"{w['whisperx']['text']!r}@[{w['whisperx']['start_s']:.2f},"
                f"{w['whisperx']['end_s']:.2f}]",
                file=sys.stderr,
            )

    pass_str = "PASS" if summary["passed"] else "FAIL"
    print(
        f"  {pass_str} (median {median:.3f}≥{args.threshold} mean "
        f"{mean:.3f}≥{args.mean_threshold} below_0.5={below_0_5}≤{args.allow_below_0_5})",
        file=sys.stderr,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
