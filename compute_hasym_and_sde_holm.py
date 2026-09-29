"""H_asym determinism (Milestone C) and SDE: the HASym_DET_3 and
HASym_SDE_REVIEW_3 families, recomputed from persisted per-seed pairs.

Closeout review, 2026-09-26 (statistics integration pass) @rdelaguila

Two independent defects in the historical driver
(src/experiments/run_c_h_asym_deterministic.py) motivated
this script, per the review's own worked-out numbers:

  1. Its ``rank_biserial`` field is actually a sign-count index
     ((n_pos-n_neg)/(n_pos+n_neg)), not the rank-sum rank-biserial
     r=(W+-W-)/(W++W-) the manuscript's own methods section defines.
  2. Its ``p_holm`` field is a raw copy of the two-sided p-value -- no
     Holm correction was ever applied within the driver itself.

This script does NOT patch that driver (it is a historical experiment
record, left untouched). As of this integration pass, the actual
computation lives in ``src/statistical_tests.py``
(``validate_seeds_complete``, ``recompute_gradient_family``,
``compute_hasym_det_3``, ``compute_hasym_sde_review_3``) so that this
standalone script and ``run_all.py``'s production pipeline (via
``compute_final_statistical_report``) share one calculation, not two
divergent copies. This module re-exports those names for backward
compatibility and adds the S04 diagnostic/CLI behaviour specific to
this script.

  - HASym_DET_3: f1_ref vs f1_zero (gradient active vs inactive),
    deterministic observable, 3 datasets, two-sided, Holm-3.
  - HASym_SDE_REVIEW_3: same contrast, SDE observable, 3 datasets,
    two-sided, Holm-3 -- a mandatory third family, not an optional
    exploratory addendum. Kept in an entirely separate family from
    HASym_DET_3 (see statistical_tests.py tests: family independence).

It also runs the review's own diagnostic: does a genuine rank-based
rank-biserial, computed from these exact persisted pairs, match the
manuscript's printed effect sizes? The answer is reported as data, not
forced either way.

Inputs (read-only):
  progress/milestone_c_h_asym_{FB15k-237,codex-m,WN18RR}.json  (94aa267)
  progress/sde_b6_wilcoxon_{FB15k-237,codex-m,WN18RR}.json

Output (new file):
  progress/hasym_and_sde_holm_recomputed.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from statistical_tests import (  # noqa: E402
    paired_wilcoxon,
    holm_correction_named,
    FINAL_DATASETS as DATASETS,
    EXPECTED_SEEDS_PER_GRADIENT_DATASET as EXPECTED_SEEDS_PER_DATASET,
    HASYM_DET_FAMILY_ID,
    HASYM_SDE_FAMILY_ID,
    validate_seeds_complete,
    recompute_gradient_family as _recompute_gradient_family,
)

MANUSCRIPT_PRINTED_RANK_BISERIAL = {
    "FB15k-237": -0.60,
    "codex-m": -0.05,
    "WN18RR": -1.000,
}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def recompute_family(prefix: str, family_id: str) -> List[Dict]:
    """Thin wrapper over statistical_tests.recompute_gradient_family,
    forwarding this module's own REPO_ROOT (kept as a real wrapper, not a
    raw alias, so tests can monkeypatch REPO_ROOT here)."""
    return _recompute_gradient_family(prefix, family_id, repo_root=REPO_ROOT)


def diagnose_manuscript_rank_biserial(det_rows: List[Dict]) -> List[Dict]:
    """S04: is the manuscript's printed rank-biserial the genuine
    rank-based rank-biserial computed from these exact persisted pairs?
    Report, don't force."""
    out = []
    for r in det_rows:
        printed = MANUSCRIPT_PRINTED_RANK_BISERIAL[r["dataset"]]
        out.append({
            "dataset": r["dataset"],
            "manuscript_printed": printed,
            "genuine_rank_biserial_from_persisted_pairs": r["rank_biserial_recomputed"],
            "matches_manuscript": abs(printed - r["rank_biserial_recomputed"]) < 1e-6,
            "sign_index_from_persisted_pairs": r["sign_index_recomputed"],
            "matches_manuscript_via_sign_index": abs(printed - r["sign_index_recomputed"]) < 1e-6,
        })
    return out


def main() -> Dict:
    det_rows = recompute_family("milestone_c_h_asym", HASYM_DET_FAMILY_ID)
    sde_rows = recompute_family("sde_b6_wilcoxon", HASYM_SDE_FAMILY_ID)
    diagnosis = diagnose_manuscript_rank_biserial(det_rows)
    return {
        "HASym_DET_3": det_rows,
        "HASym_SDE_REVIEW_3": sde_rows,
        "manuscript_rank_biserial_diagnosis": diagnosis,
    }


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "progress" / "hasym_and_sde_holm_recomputed.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}\n")

    print("=== HASym_DET_3 (deterministic, two-sided, Holm-3) ===")
    for r in result["HASym_DET_3"]:
        print(f"  {r['dataset']:14s} p_raw={r['p_value_recomputed']:.9f} "
              f"p_holm={r['p_holm_recomputed']:.9f} "
              f"rank_biserial={r['rank_biserial_recomputed']:+.4f} "
              f"sign_index={r['sign_index_recomputed']:+.4f} "
              f"(p_raw matches historical: {r['p_value_match']})")

    print("\n=== HASym_SDE_REVIEW_3 (SDE, two-sided, Holm-3, mandatory third family) ===")
    for r in result["HASym_SDE_REVIEW_3"]:
        print(f"  {r['dataset']:14s} p_raw={r['p_value_recomputed']:.9f} "
              f"p_holm={r['p_holm_recomputed']:.9f} "
              f"rank_biserial={r['rank_biserial_recomputed']:+.4f} "
              f"sign_index={r['sign_index_recomputed']:+.4f} "
              f"(p_raw matches historical: {r['p_value_match']})")

    print("\n=== S04 diagnosis: origin of manuscript's printed rank-biserial ===")
    for d in result["manuscript_rank_biserial_diagnosis"]:
        print(f"  {d['dataset']:14s} printed={d['manuscript_printed']:+.3f} "
              f"genuine_rank_biserial={d['genuine_rank_biserial_from_persisted_pairs']:+.4f} "
              f"(matches: {d['matches_manuscript']}) "
              f"sign_index={d['sign_index_from_persisted_pairs']:+.4f} "
              f"(matches via sign_index: {d['matches_manuscript_via_sign_index']})")
