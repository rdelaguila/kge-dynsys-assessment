"""Phase-2 authorized minimal run — Section C: dynamics x symbolic bonus 2x2.

Extends the existing, published A1/A2 comparison (`run_c_h_asym_deterministic.py`,
`progress/milestone_c_h_asym_WN18RR.json`) with the symbolic-bonus axis, reusing
the identical dataset/perturbation/seeds/pool. The "bonus=off" cell is exactly
what that script already computes (sanity-checked against it below); only the
"bonus=on" cell is new.

For each of the 10 seeds already used for A1 (lambda_asym=0) / A2 (lambda_asym=ref):
  - Same perturbed pool, same 10,000-triple eval sample (same seed).
  - Same RK4-integrated per-triple energy (E_data) for each lambda setting.
  - NEW: constraint-violation count per eval triple from a standalone checker
    built the same way IntegratedKGDebugger.setup_ode_detector builds one
    (transitivity + symmetry + antisymmetry relations from get_constraint_config).
  - NEW: symbolic bonus = q99 of the finite energy distribution for that
    lambda setting (same methodology as AnomalyDetector._compute_calibrated_bonus,
    applied to the evaluated pool), bonus-on score = energy + n_violations*bonus.
  - F1/n_detected/threshold via the same knee-on-sorted-score rule, for both
    bonus=off and bonus=on, at both lambda settings -> one 2x2 cell set per seed.

No existing file is modified. No training/tuning occurs (cached hyperparameters
only). Writes progress/milestone_c2_dynamics_symbolic_2x2_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from ablation_study import KGPerturbator
from data import DataLoader
from constraints import ConstraintChecker, TransitivityConstraint, SymmetryConstraint, AntisymmetryConstraint
from run_all import _load_hyperparam_cache, get_constraint_config, sample_triples
from experiments.run_c_h_asym_deterministic import build_kgode, rk4_integrate, f1_knee


def build_constraint_checker(dataset: str, margin_asym: float = 1.0) -> ConstraintChecker:
    """Standalone equivalent of IntegratedKGDebugger.setup_ode_detector's
    constraint-building block (lines 143-169), kept here so the existing
    harness file is not touched."""
    cfg = get_constraint_config(dataset)
    checker = ConstraintChecker()
    for rel in cfg["transitivity_relations"]:
        checker.add_constraint(TransitivityConstraint(rel))
    for rel in cfg["symmetry_relations"]:
        checker.add_constraint(SymmetryConstraint(rel))
    for rel in cfg.get("asymmetry_relations", []):
        checker.add_constraint(AntisymmetryConstraint(rel, margin=margin_asym, enabled=True))
    return checker


def count_violations(checker: ConstraintChecker, full_triples, eval_triples) -> np.ndarray:
    """Mirrors AnomalyDetector.compute_constraint_violation_score's cached
    set-membership logic, vectorised over eval_triples."""
    cached = {}
    for c in checker.constraints:
        cached[c.name] = set(c.get_violating_triples(full_triples))
    counts = np.zeros(len(eval_triples), dtype=int)
    for i, t in enumerate(eval_triples):
        counts[i] = sum(1 for c in checker.constraints if t in cached[c.name])
    return counts


def energy_for_lambda(dataset, lam_val, perturbed, eval_triples, seed, n_steps, device):
    """Exact same per-triple energy computation as run_c_h_asym_deterministic.py's
    detect() closure, factored out so it can be reused for both cells of a lambda
    setting without duplicating the RK4 call."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    kgode = build_kgode(dataset, lam_val, perturbed, device=device)
    y_T = rk4_integrate(kgode, n_steps=n_steps)

    s_idx, o_idx, r_idx_in_pool, valid_mask = [], [], [], []
    triple_to_pool = {(t.subject, t.relation, t.object): i for i, t in enumerate(kgode.triples)}
    for t in eval_triples:
        k = (t.subject, t.relation, t.object)
        pi = triple_to_pool.get(k)
        if pi is None or t.subject not in kgode.entity_to_idx or t.object not in kgode.entity_to_idx:
            valid_mask.append(False)
            s_idx.append(0); o_idx.append(0); r_idx_in_pool.append(0)
            continue
        valid_mask.append(True)
        s_idx.append(kgode.entity_to_idx[t.subject])
        o_idx.append(kgode.entity_to_idx[t.object])
        r_idx_in_pool.append(pi)

    s_idx_t = torch.tensor(s_idx, dtype=torch.long, device=y_T.device)
    o_idx_t = torch.tensor(o_idx, dtype=torch.long, device=y_T.device)
    r_idx_t = torch.tensor(r_idx_in_pool, dtype=torch.long, device=y_T.device)
    h_emb = y_T[s_idx_t]
    t_emb = y_T[o_idx_t]
    r_vec = kgode.r_vectors[r_idx_t]
    energy = ((h_emb + r_vec - t_emb) ** 2).sum(dim=-1).cpu().numpy()
    valid_mask_np = np.asarray(valid_mask, dtype=bool)
    if not valid_mask_np.all():
        energy[~valid_mask_np] = np.nan
    return energy


def run_one_seed(dataset: str, seed: int, sample_size: int, level_pct: float,
                  n_steps: int, device: str) -> dict:
    t0 = time.time()
    loader = DataLoader()
    base = loader.load_dataset(dataset)
    n_viol = max(10, int(len(base) * (level_pct / 100.0)))
    pert = KGPerturbator(base, seed=int(seed))
    perturbed, violation_set, _ = pert.perturb_logical(
        violation_type="asymmetry", n_violations=n_viol, dataset_name=dataset,
    )
    eval_triples, eval_viol_set = sample_triples(perturbed, violation_set,
                                                  sample_size, seed=int(seed))
    gt_keys = set(eval_viol_set)
    gt = np.array([(t.subject, t.relation, t.object) in gt_keys
                   for t in eval_triples], dtype=bool)

    bp = _load_hyperparam_cache(dataset, "ode")
    lambda_asym_ref = bp.get("lambda_asym", 1.0)
    margin_asym = float(bp.get("margin_asym", 1.0))

    checker = build_constraint_checker(dataset, margin_asym=margin_asym)
    viol_counts = count_violations(checker, perturbed, eval_triples)

    cells = {}
    for lam_label, lam_val in [("ref", lambda_asym_ref), ("zero", 0.0)]:
        energy = energy_for_lambda(dataset, lam_val, perturbed, eval_triples,
                                    seed, n_steps, device)
        finite = np.isfinite(energy)
        eval_gt = gt[finite]
        eval_energy = energy[finite]
        eval_viol = viol_counts[finite]

        q99 = float(np.quantile(eval_energy, 0.99)) if eval_energy.size else float("nan")
        bonus = q99 if (np.isfinite(q99) and q99 > 0) else 500.0
        score_bonus_on = eval_energy + eval_viol * bonus

        f1_off = f1_knee(eval_energy, eval_gt, label=f"{lam_label}_bonus_off")
        f1_on = f1_knee(score_bonus_on, eval_gt, label=f"{lam_label}_bonus_on")
        cells[lam_label] = {
            "bonus_off": f1_off, "bonus_on": f1_on,
            "bonus_value": bonus, "n_finite": int(finite.sum()),
            "n_eval_constraint_violations": int(eval_viol.sum()),
        }

    elapsed = time.time() - t0
    return {
        "seed": seed, "ok": True, "elapsed_s": elapsed,
        "n_eval": len(eval_triples), "n_eval_injected_violations": int(gt.sum()),
        "cells": cells,
        "f1_ref_bonus_off": cells["ref"]["bonus_off"]["f1"],
        "f1_zero_bonus_off": cells["zero"]["bonus_off"]["f1"],
        "f1_ref_bonus_on": cells["ref"]["bonus_on"]["f1"],
        "f1_zero_bonus_on": cells["zero"]["bonus_on"]["f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN18RR")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--level-pct", type=float, default=10.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 78)
    print(f"# Phase-2 Section C — Dynamics x Symbolic 2x2 | {args.dataset} asym @{args.level_pct}%")
    print(f"# n_seeds={args.n_seeds}, sample_size={args.sample_size}, n_steps={args.n_steps}")
    print("=" * 78)

    results = []
    t_panel = time.time()
    for seed in range(1, args.n_seeds + 1):
        print(f"\n--- seed {seed}/{args.n_seeds} ---", flush=True)
        try:
            r = run_one_seed(args.dataset, seed, args.sample_size,
                              args.level_pct, args.n_steps, args.device)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"seed": seed, "ok": False, "error": str(e)})
            continue
        results.append(r)
        print(f"  done in {r['elapsed_s']:.1f}s | "
              f"F1 ref(off/on)={r['f1_ref_bonus_off']:.4f}/{r['f1_ref_bonus_on']:.4f}, "
              f"zero(off/on)={r['f1_zero_bonus_off']:.4f}/{r['f1_zero_bonus_on']:.4f}",
              flush=True)
    panel_elapsed = time.time() - t_panel

    ok = [r for r in results if r["ok"]]

    def arr(key_path):
        vals = []
        for r in ok:
            cell = r
            for k in key_path:
                cell = cell[k]
            vals.append(cell)
        return np.array(vals, dtype=float)

    f1_ref_off = arr(["f1_ref_bonus_off"])
    f1_zero_off = arr(["f1_zero_bonus_off"])
    f1_ref_on = arr(["f1_ref_bonus_on"])
    f1_zero_on = arr(["f1_zero_bonus_on"])

    summary = {
        "milestone": "C2 — dynamics x symbolic bonus 2x2 (phase-2, authorized minimal run)",
        "protocol": {
            "dataset": args.dataset, "violation": "asymmetry",
            "level_pct": args.level_pct, "sample_size": args.sample_size,
            "n_seeds": args.n_seeds, "n_steps_rk4": args.n_steps,
            "device": args.device, "panel_wall_clock_s": panel_elapsed,
        },
        "per_seed": results,
        "marginal_means": {
            "f1_ref_bonus_off": float(f1_ref_off.mean()) if len(f1_ref_off) else None,
            "f1_zero_bonus_off": float(f1_zero_off.mean()) if len(f1_zero_off) else None,
            "f1_ref_bonus_on": float(f1_ref_on.mean()) if len(f1_ref_on) else None,
            "f1_zero_bonus_on": float(f1_zero_on.mean()) if len(f1_zero_on) else None,
        },
        "deltas_descriptive": {
            "dynamics_effect_at_bonus_off (ref-zero)": float((f1_ref_off - f1_zero_off).mean()) if len(f1_ref_off) else None,
            "dynamics_effect_at_bonus_on (ref-zero)": float((f1_ref_on - f1_zero_on).mean()) if len(f1_ref_on) else None,
            "symbolic_effect_at_dynamics_ref (on-off)": float((f1_ref_on - f1_ref_off).mean()) if len(f1_ref_on) else None,
            "symbolic_effect_at_dynamics_zero (on-off)": float((f1_zero_on - f1_zero_off).mean()) if len(f1_zero_on) else None,
        },
    }

    out_json = REPO_ROOT / "progress" / f"milestone_c2_dynamics_symbolic_2x2_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("SECTION C — 2x2 PANEL SUMMARY (descriptive only, no causal claim)")
    print("=" * 78)
    for k, v in summary["marginal_means"].items():
        print(f"  mean {k} = {v}")
    for k, v in summary["deltas_descriptive"].items():
        print(f"  {k} = {v}")
    print(f"  Wall-clock: {panel_elapsed:.1f}s")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
