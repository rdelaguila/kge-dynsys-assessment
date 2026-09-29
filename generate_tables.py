"""Thin orchestrator: regenerate every manuscript table and statistical
result from the outputs of ``src/run_all.py`` (the main entry point).

This script implements no metrics, no statistics and no scientific logic
of its own. It only calls the existing producer scripts in ``scripts/``
and ``src/experiments/`` in a working order, and reports which ones
succeeded. Each producer writes its own output files, exactly as it does
when invoked directly -- this wrapper does not centralise or reformat
their outputs.

Prerequisite: `python3 src/run_all.py --datasets <ds> --stages all
--no-interactive` must have been run for the datasets you want tables
for -- this script only aggregates outputs that already exist on disk.

Usage:
    python3 generate_tables.py [--datasets fb15k-237 codex-m WN18RR]

Writes a summary of what ran and what it produced to
``GENERATE_TABLES_REPORT.json`` in the repository root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Each entry: (label, command, output_paths_to_check). Order matters --
# later producers read the output of earlier ones.
#
# VERIFIED steps: confirmed, by actually running them against real data
# during packaging, to produce the expected structure (9/3/3 cells, no
# missing fields). NOT_VERIFIED steps are included as source code for
# completeness but disabled by default here, because running them fresh
# was observed to overwrite their own committed reference output with a
# different, incomplete or structurally different result (see the
# packaging validation report) -- re-enable only after that is
# root-caused and fixed.
VERIFIED_STEPS = [
    ("native_and_E6_ode_side",
     [PY, "scripts/milestone_postfix_section4_rerun.py"],
     ["paper_results/WN18RR/post_fix_rerun", "paper_results/WN18RR/bootstrap_postfix"]),
    ("E6_statistics",
     [PY, "scripts/compute_e6_triviality_gap.py"],
     ["progress/e6_triviality_gap_reconstructed.json"]),
    ("HASym_deterministic_per_seed",
     [PY, "src/experiments/run_c_h_asym_deterministic.py"],
     ["progress/milestone_c_h_asym_WN18RR.json"]),
    ("HASym_SDE_per_seed",
     [PY, "src/experiments/run_sde_b6_wilcoxon.py"],
     ["progress/sde_b6_wilcoxon_WN18RR.json"]),
    ("HASym_and_SDE_statistics",
     [PY, "scripts/compute_hasym_and_sde_holm.py"],
     ["progress/hasym_and_sde_holm_recomputed.json"]),
    ("R02_operator_sweep_raw",
     [PY, "scripts/compute_parallel_operator_sweep.py"],
     ["progress/parallel_operator_sweep.json"]),
    ("R02_swap_means",
     [PY, "scripts/compute_r02_swap_means.py"],
     ["progress/r02_swap_means_recomputed.json"]),
]

NOT_VERIFIED_STEPS = [
    ("E1_E5_upstream_audit",
     [PY, "scripts/audit_phase2_E_rerun.py"],
     ["progress/audit_phase2_E_rerun.json"]),
    ("E1_E5_upstream_audit_E2E3",
     [PY, "scripts/audit_phase2_E2E3_postfix.py"],
     ["progress/audit_phase2_E2E3_postfix.json"]),
    ("E1_E5_evidence",
     [PY, "scripts/extract_e1_e5_evidence.py"],
     ["progress/e1_e5_evidence_extracted.json"]),
    ("tripartite_consensus",
     [PY, "scripts/compute_tripartite_postfix.py"],
     ["progress/tripartite_postfix.json"]),
    ("ranking_full_panel",
     [PY, "scripts/compute_full_panel.py"],
     ["progress/20260623/full_panel.json"]),
]

STEPS = VERIFIED_STEPS


def run_step(label: str, cmd: list, expected_outputs: list) -> dict:
    print(f"\n{'=' * 70}\n[generate_tables] {label}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    outputs_present = {p: (REPO_ROOT / p).exists() for p in expected_outputs}
    if not ok:
        print(f"  FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
    else:
        print(f"  ok ({elapsed:.1f}s) -- outputs present: {outputs_present}")
    return {
        "label": label, "command": " ".join(cmd), "returncode": result.returncode,
        "ok": ok, "elapsed_s": elapsed, "outputs_present": outputs_present,
        "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["fb15k-237", "codex-m", "WN18RR"],
                        help="Informational only -- most producers already iterate the three "
                             "paper datasets internally with their own defaults.")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Step labels to skip (see STEPS in this file).")
    parser.add_argument("--include-not-verified", action="store_true",
                        help="Also run NOT_VERIFIED_STEPS (E1-E5, tripartite consensus, "
                             "ranking panel). WARNING: these were observed, during packaging, "
                             "to overwrite their own committed reference output with a "
                             "different or incomplete result when actually run -- do not "
                             "trust their output without independently checking it first.")
    args = parser.parse_args()

    steps = STEPS + (NOT_VERIFIED_STEPS if args.include_not_verified else [])
    if args.include_not_verified:
        print("WARNING: running NOT_VERIFIED_STEPS. These are known, from packaging-time "
              "testing, to sometimes produce incomplete or unexpected output. Check results "
              "against the expected row/cell counts before trusting them.")

    report = {"steps": []}
    for label, cmd, expected_outputs in steps:
        if label in args.skip:
            print(f"[generate_tables] skipping {label} (requested)")
            continue
        report["steps"].append(run_step(label, cmd, expected_outputs))

    n_ok = sum(1 for s in report["steps"] if s["ok"])
    n_total = len(report["steps"])
    report["summary"] = f"{n_ok}/{n_total} producers completed successfully"
    print(f"\n{'=' * 70}\n{report['summary']}\n{'=' * 70}")

    out_path = REPO_ROOT / "GENERATE_TABLES_REPORT.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")

    sys.exit(0 if n_ok == n_total else 1)


if __name__ == "__main__":
    main()
