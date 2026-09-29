"""Minimal table-generation verification for the reproducibility release.

Exercises generate_tables.py's VERIFIED fast statistical steps (the ones
that only read already-persisted per-seed data, no retraining) against
real data, and checks: the producers run, they generate the expected
number of tables/rows, no essential column is missing, and no
placeholder or missing value is silently accepted as a valid result.

The expensive steps (milestone_postfix_section4_rerun.py,
run_c_h_asym_deterministic.py, run_sde_b6_wilcoxon.py -- all of which
retrain/re-integrate the ODE) are skipped here by default and are
exercised separately by the mandatory pre-packaging smoke test instead of
by this unit test.

generate_tables.py's NOT_VERIFIED_STEPS (E1-E5 evidence, tripartite
consensus, ranking panel) are deliberately NOT exercised here: running
them during packaging validation overwrote their own committed reference
output with a different, incomplete result (see the packaging validation
report), so this test does not assert they work until that is
root-caused.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EXPENSIVE_STEPS = [
    "native_and_E6_ode_side",
    "HASym_deterministic_per_seed",
    "HASym_SDE_per_seed",
]


def test_generate_tables_fast_steps_succeed():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "generate_tables.py"), "--skip", *EXPENSIVE_STEPS],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    report_path = REPO_ROOT / "GENERATE_TABLES_REPORT.json"
    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)

    ran_labels = {s["label"] for s in report["steps"]}
    assert ran_labels == {
        "E6_statistics", "HASym_and_SDE_statistics",
        "R02_operator_sweep_raw", "R02_swap_means",
    }
    for step in report["steps"]:
        assert step["ok"], f"{step['label']} failed: {step['stderr_tail']}"


def test_e6_table_has_nine_cells_no_placeholders():
    from compute_e6_triviality_gap import main as compute_e6
    result = compute_e6()
    assert len(result["cells"]) == 9
    for c in result["cells"]:
        for key in ("p_raw", "p_holm", "rank_biserial", "mean_diff"):
            assert c[key] is not None
            assert c[key] == c[key]  # NaN check: NaN != NaN


def test_hasym_tables_have_three_cells_each_no_placeholders():
    from compute_hasym_and_sde_holm import main as compute_hasym
    result = compute_hasym()
    for family in ("HASym_DET_3", "HASym_SDE_REVIEW_3"):
        rows = result[family]
        assert len(rows) == 3
        for r in rows:
            assert r["p_value_recomputed"] is not None
            assert r["p_holm_recomputed"] is not None
            assert r["rank_biserial_recomputed"] == r["rank_biserial_recomputed"]


def test_r02_table_has_four_detectors_nine_cells_each():
    from compute_r02_swap_means import main as compute_r02
    result = compute_r02()
    assert result["n_comparisons"] == 36  # 4 detectors x 9 conditions
    assert result["canonical_wins"] + result["swap_wins"] + result["ties"] == 36


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
