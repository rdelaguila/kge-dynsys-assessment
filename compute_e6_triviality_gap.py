"""E6 / triviality-gap: the E6_RANDOM_9 family, from raw per-seed data only.

Closeout review, 2026-09-26 (statistics integration pass). The historical
producer of ``progress/postfix_trivialitygap_bootstrap.json`` (commit
35a1f7b) was searched for and not located. This script is a from-scratch
reconstruction of the same documented analysis from its declared raw
inputs -- it is not a recovery of the historical script, and this file's
own docstring says so rather than presenting itself as the original.

As of this integration pass, the actual computation lives in
``src/statistical_tests.py`` (``load_e6_ode_f1_by_seed``,
``load_e6_kge_f1_by_seed``, ``compute_e6_cell``, ``compute_e6_random_9``)
so that this standalone script and ``run_all.py``'s production pipeline
(via ``compute_final_statistical_report``) share one calculation, not two
divergent copies. This module only re-exports those names (for backward
compatibility with existing tests and callers) and adds the
historical-comparison/CLI behaviour specific to this script.

Design (identifier E6_RANDOM_9, per the review's naming table):
  - 3 datasets (FB15k-237, codex-m, WN18RR) x 3 KGE baselines
    (TransE, DistMult, MuRE; ComplEx excluded, per stage specification) =
    9 cells.
  - Per cell: 10 paired seeds. Pairing is by ``protocol.perturbation_seed``
    read from the ODE file (not by filesystem iteration order) and by the
    ``seed_<n>`` directory component for the KGE file (that file carries
    no internal seed field of its own).
  - Contrast: F1(KGE) - F1(ODE), tested one-sided (alternative='greater'),
    Holm-Bonferroni within exactly this 9-cell family (family_id
    'E6_RANDOM_9'), using the already-reviewed and tested
    ``paired_wilcoxon``/``holm_correction_named`` from statistical_tests.py.
    No new Holm implementation, no recomputation of ranks by hand.

Inputs (read-only, never written to):
  paper_results/<ds>/bootstrap_postfix/seed_<n>/random/metrics/metrics_random_entity_swap_5.0.json   (ODE, post-fix)
  paper_results/<ds>/bootstrap_r0_canonical/seed_<n>/random/metrics/metrics_random_entity_swap_5.0.json  (KGE, canonical)

Output (new file, does not overwrite the historical artefact):
  progress/e6_triviality_gap_reconstructed.json

Historical regression reference (progress/postfix_trivialitygap_bootstrap.json,
is loaded and diffed against this reconstruction's own
p_raw/p_holm/rank_biserial per cell; first divergence is reported, not
silently resolved in either direction.
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
    FINAL_KGE_METHODS as KGE_METHODS,
    E6_FAMILY_ID as FAMILY_ID,
    EXPECTED_SEEDS_PER_E6_CELL as EXPECTED_SEEDS_PER_CELL,
    load_e6_ode_f1_by_seed,
    load_e6_kge_f1_by_seed,
    compute_e6_cell as compute_cell,
    compute_e6_random_9,
)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_ode_f1_by_seed(dataset: str) -> Dict[int, float]:
    """Thin wrapper over statistical_tests.load_e6_ode_f1_by_seed, forwarding
    this module's own REPO_ROOT (kept as a real wrapper, not a raw alias,
    so tests can monkeypatch REPO_ROOT here)."""
    return load_e6_ode_f1_by_seed(dataset, repo_root=REPO_ROOT)


def load_kge_f1_by_seed(dataset: str, method: str) -> Dict[int, float]:
    """Thin wrapper over statistical_tests.load_e6_kge_f1_by_seed, same
    REPO_ROOT-forwarding rationale as load_ode_f1_by_seed above."""
    return load_e6_kge_f1_by_seed(dataset, method, repo_root=REPO_ROOT)


def main() -> Dict:
    return compute_e6_random_9(repo_root=REPO_ROOT)


def compare_to_historical(reconstructed: Dict) -> List[Dict]:
    hist_path = REPO_ROOT / "progress" / "postfix_trivialitygap_bootstrap.json"
    hist = _load(hist_path)
    hist_by_key = {f"{c['dataset']}/{c['kge']}": c for c in hist["cells"]}
    diffs = []
    for c in reconstructed["cells"]:
        key = f"{c['dataset']}/{c['kge']}"
        h = hist_by_key.get(key)
        if h is None:
            diffs.append({"cell": key, "status": "missing_in_historical"})
            continue
        diffs.append({
            "cell": key,
            "p_raw_reconstructed": c["p_raw"], "p_raw_historical": h["p_raw"],
            "p_raw_match": abs(c["p_raw"] - h["p_raw"]) < 1e-9,
            "p_holm_reconstructed": c["p_holm"], "p_holm_historical": h["p_holm"],
            "p_holm_match": abs(c["p_holm"] - h["p_holm"]) < 1e-9,
            "rank_biserial_reconstructed": c["rank_biserial"],
            "rank_biserial_historical": h["rank_biserial"],
            "rank_biserial_match": abs(c["rank_biserial"] - h["rank_biserial"]) < 1e-9,
        })
    return diffs


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "progress" / "e6_triviality_gap_reconstructed.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")

    comparison = compare_to_historical(result)
    n_mismatch = sum(1 for d in comparison if not d.get("p_holm_match", False))
    print(f"Comparison vs historical (progress/postfix_trivialitygap_bootstrap.json): "
          f"{len(comparison) - n_mismatch}/{len(comparison)} cells match p_holm exactly.")
    for d in comparison:
        if not d.get("p_holm_match", False):
            print("  MISMATCH:", json.dumps(d, indent=2))
