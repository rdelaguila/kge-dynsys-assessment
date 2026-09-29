"""Compute threshold-free metrics (PR-AUC + AUROC) for KG-Debug-ODE.

Tier-0: reads the 9 captured ablation_audit cells and emits per-cell
PR-AUC + AUROC for the ODE energy (pre-fix) and the three KGE plausibility
scores (TransE/DistMult/MuRE).

Tier-1.5 extension: if `--ode-postfix` is passed, replaces the ODE energy
with the post-fix arrays dumped by the instrumented
`milestone_postfix_section4_rerun.py` (one npz per cell).

Read-only over persisted artefacts. No model is invoked. Coste compute
nominal (sklearn calls only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("WN18RR",    "transitivity",  4650),
    ("codex-m",   "cardinality",  30930),
    ("WN18RR",    "asymmetry",   13950),
    ("FB15k-237", "transitivity", 46517),
    ("WN18RR",    "cardinality", 13950),
    ("codex-m",   "asymmetry",   20620),
    ("codex-m",   "transitivity", 10310),
    ("codex-m",   "cardinality",  10310),
    ("FB15k-237", "cardinality",  15505),
]
KGE_METHODS = ("TransE", "DistMult", "MuRE")


def load_cell(ds: str, vtype: str, n: int) -> dict:
    p = REPO_ROOT / "paper_results" / ds / "ablation_audit" / f"{vtype}_{n}.json"
    return json.loads(p.read_text())


def load_postfix_ode(ds: str, vtype: str, n: int) -> np.ndarray | None:
    p = REPO_ROOT / "progress" / "20260623" / f"ode_postfix_{ds}_{vtype}_{n}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["eval_energies"], z["gt"]


def per_method_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    finite = np.isfinite(y_score)
    yt = y_true[finite]
    ys = y_score[finite]
    return {
        "n_finite": int(finite.sum()),
        "n_pos": int(yt.sum()),
        "prevalence": float(yt.mean()),
        "prauc": float(average_precision_score(yt, ys)),
        "auroc": float(roc_auc_score(yt, ys)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ode-postfix", action="store_true",
                    help="use post-fix ODE arrays from progress/20260623/ode_postfix_*.npz")
    args = ap.parse_args()

    out_rows = []
    for ds, vtype, n in CELLS:
        try:
            cell = load_cell(ds, vtype, n)
        except FileNotFoundError:
            print(f"  [skip] {ds}/{vtype}/{n}: ablation_audit missing")
            continue

        triples = cell["triples"]
        y_true = np.array([t["is_violation"] for t in triples], dtype=bool)
        prevalence = y_true.mean()

        row = {
            "dataset": ds, "violation": vtype, "level_n": n,
            "n_eval": len(triples), "n_violations": int(y_true.sum()),
            "prevalence": float(prevalence), "methods": {},
        }

        # ODE (pre-fix from ablation_audit OR post-fix from .npz)
        if args.ode_postfix:
            postfix = load_postfix_ode(ds, vtype, n)
            if postfix is None:
                print(f"  [warn] {ds}/{vtype}/{n}: post-fix ODE npz missing, skipping ODE")
                ode_score = None
            else:
                ode_energies, ode_gt = postfix
                # Sanity: gt mask must match (perturbator+sampler deterministic with seed=42)
                if len(ode_gt) != len(y_true):
                    raise RuntimeError(f"Length mismatch {ds}/{vtype}/{n}: postfix={len(ode_gt)} vs audit={len(y_true)}")
                if not np.array_equal(ode_gt.astype(bool), y_true):
                    diff = (ode_gt.astype(bool) != y_true).sum()
                    raise RuntimeError(f"GT mask mismatch {ds}/{vtype}/{n}: {diff} triples differ")
                ode_score = ode_energies
        else:
            ode_score = np.array([t.get("ode_energy", np.nan) for t in triples], dtype=float)

        if ode_score is not None:
            row["methods"]["ODE"] = per_method_metrics(y_true, ode_score)

        # KGE methods: higher = more plausible, so anomaly score = -kge_raw
        for kge in KGE_METHODS:
            kge_score = np.array([-t["kge_raw"][kge] for t in triples], dtype=float)
            row["methods"][kge] = per_method_metrics(y_true, kge_score)

        out_rows.append(row)

        # Print one-line summary
        methods_str = " | ".join(
            f"{m}: PR-AUC {row['methods'][m]['prauc']:.3f} AUROC {row['methods'][m]['auroc']:.3f}"
            for m in ("ODE",) + KGE_METHODS if m in row["methods"]
        )
        print(f"  {ds:12s} / {vtype:14s} / n={n:>5d}  prev={prevalence:.3f}  ||  {methods_str}")

    out_path = REPO_ROOT / "progress" / "20260623" / (
        "threshold_free_postfix.json" if args.ode_postfix else "threshold_free_prefix.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "regime": "post-fix ODE" if args.ode_postfix else "pre-fix ODE",
        "cells": out_rows,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
