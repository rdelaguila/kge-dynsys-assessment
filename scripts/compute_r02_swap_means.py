"""R02: per-detector mean(swap F1 - canonical F1), regenerated from
progress/parallel_operator_sweep.json rather than hand-typed.

Canonical operator per detector: ODE-A2 uses knee, TransE/DistMult/MuRE
use Otsu (per Section 3.4 of the manuscript). The swap is the opposite
operator for that same detector. Delta = F1(swap) - F1(canonical),
averaged over the 9 stratified cells.

Also reports win/loss/tie counts per detector and in aggregate (36
detector-cell comparisons total), as a regenerated cross-check of the
18/16/2 the manuscript already prints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_OPERATOR = {
    "ODE-A2": "knee",
    "TransE": "otsu",
    "DistMult": "otsu",
    "MuRE": "otsu",
}
TIE_TOLERANCE = 0.0  # exact-equality tie definition, as used in the manuscript's 18/16/2


def main() -> Dict:
    data = json.load(open(REPO_ROOT / "progress" / "parallel_operator_sweep.json"))
    per_detector: Dict[str, List[float]] = {d: [] for d in CANONICAL_OPERATOR}
    wins_canonical = wins_swap = ties = 0

    for cell in data["cells"]:
        for detector, canonical_op in CANONICAL_OPERATOR.items():
            swap_op = "otsu" if canonical_op == "knee" else "knee"
            det_block = cell["detectors"].get(detector)
            if det_block is None or canonical_op not in det_block or swap_op not in det_block:
                continue
            f1_canonical = det_block[canonical_op]["f1"]
            f1_swap = det_block[swap_op]["f1"]
            delta = f1_swap - f1_canonical
            per_detector[detector].append(delta)
            if delta > TIE_TOLERANCE:
                wins_swap += 1
            elif delta < -TIE_TOLERANCE:
                wins_canonical += 1
            else:
                ties += 1

    means = {d: (sum(vals) / len(vals) if vals else None) for d, vals in per_detector.items()}
    return {
        "tie_definition": "exact equality (delta == 0.0)",
        "n_comparisons": sum(len(v) for v in per_detector.values()),
        "canonical_wins": wins_canonical,
        "swap_wins": wins_swap,
        "ties": ties,
        "per_detector_deltas": per_detector,
        "per_detector_mean_delta": means,
    }


if __name__ == "__main__":
    result = main()
    print(f"n_comparisons={result['n_comparisons']}  "
          f"canonical_wins={result['canonical_wins']}  "
          f"swap_wins={result['swap_wins']}  ties={result['ties']}")
    for d, m in result["per_detector_mean_delta"].items():
        n_pos = sum(1 for x in result["per_detector_deltas"][d] if x > 0)
        n_neg = sum(1 for x in result["per_detector_deltas"][d] if x < 0)
        n_tie = sum(1 for x in result["per_detector_deltas"][d] if x == 0)
        print(f"  {d:10s} mean_delta={m:+.9f}  wins/losses/ties={n_pos}/{n_neg}/{n_tie}")

    out_path = REPO_ROOT / "progress" / "r02_swap_means_recomputed.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
