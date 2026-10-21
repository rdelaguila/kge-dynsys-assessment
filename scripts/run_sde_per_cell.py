"""SDE per-cell variance dumper (Phase 2 / A3 generator).

Re-runs the SDE (configuration (3) of §3.7) on a given (dataset, violation_type,
n_violations, seed) cell, post-§3.8.6-fix, and dumps the per-triple variance
array + ground-truth mask to a format identical to the milestone A1/A2 dumps:

    progress/20260623/ode_postfix_A3_<ds>_<vtype>_<n>.npz
      .eval_energies = variances (np float)
      .gt            = is_violation bool mask
      .arch          = "A3"

This is read-only over the model: re-integrates SDE on cached embeddings,
does not re-train any KGE / ODE model. Wall-clock ~5 min/cell on CPU.

Mirrors `src/experiments/run_sde_pivot_v2.py` but parametrised on vtype and
n_violations instead of hardcoded asymmetry@10%.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from data import DataLoader
from ablation_study import KGPerturbator
from run_all import sample_triples
from sde_system import KGSDESystem, compute_per_triple_variance
sys.path.insert(0, str(REPO_ROOT / "src" / "experiments"))
from run_sde_pivot_v2 import build_kgode_with_lambda_asym

DEFAULT_SIGMA = 0.05
DEFAULT_K = 10
DEFAULT_N_STEPS = 50


def run_cell(dataset: str, vtype: str, n_violations: int, seed: int,
             sigma: float, k: int, n_steps: int, sample_size: int, device: str):
    print(f"\n[A3 SDE] {dataset}/{vtype}/n={n_violations} seed={seed} σ={sigma} k={k}", flush=True)
    t0 = time.time()

    loader = DataLoader()
    base = loader.load_dataset(dataset)
    print(f"  loaded {len(base)} triples", flush=True)

    perturbator = KGPerturbator(base, seed=int(seed))
    perturbed, violation_set, _ = perturbator.perturb_logical(
        violation_type=vtype, n_violations=int(n_violations), dataset_name=dataset,
    )
    if not violation_set:
        return {"error": "no_violations_generated"}
    print(f"  perturbed: {len(perturbed)}; injected violations: {len(violation_set)}", flush=True)

    eval_triples, eval_viol = sample_triples(
        perturbed, violation_set, sample_size, seed=int(seed)
    )
    print(f"  eval: {len(eval_triples)}; in_eval_violations: {len(eval_viol)}", flush=True)

    # Build KGODESystem with lambda_asym=ref (configuration (3) uses the same
    # deterministic drift; the difference is the stochastic wrapper).
    from run_all import _load_hyperparam_cache
    bp = _load_hyperparam_cache(dataset, "ode")
    lam_asym = bp.get("lambda_asym", 1.0)
    np.random.seed(int(seed)); torch.manual_seed(int(seed))
    kgode = build_kgode_with_lambda_asym(dataset, lam_asym, perturbed)
    print(f"  KGODE ready; n_entities={kgode.n_entities}", flush=True)

    sde = KGSDESystem(kgode, sigma=sigma)
    variances = compute_per_triple_variance(
        sde, eval_triples,
        k_trajectories=k, t_span=kgode.config.t_span,
        n_steps=n_steps, seed_base=int(seed), verbose=False,
    )
    v_np = variances.cpu().numpy()

    gt = np.array([(t.subject, t.relation, t.object) in eval_viol
                   for t in eval_triples], dtype=bool)
    finite = np.isfinite(v_np)

    out_dir = REPO_ROOT / "progress" / "20260623"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ode_postfix_A3_{dataset}_{vtype}_{n_violations}.npz"
    np.savez(out_path, eval_energies=v_np, gt=gt,
             valid=np.ones_like(v_np, dtype=bool), finite=finite, arch="A3")
    print(f"  wrote {out_path.name}  ({time.time()-t0:.1f}s)", flush=True)
    return {"ok": True, "wall_s": time.time()-t0,
            "variance_mean": float(np.nanmean(v_np)),
            "n_finite": int(finite.sum()), "n_gt_pos": int(gt.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--vtype", required=True, choices=["asymmetry", "transitivity", "cardinality"])
    ap.add_argument("--n-violations", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    return run_cell(args.dataset, args.vtype, args.n_violations, args.seed,
                    args.sigma, args.k, args.n_steps, args.sample_size, args.device)


if __name__ == "__main__":
    main()
