"""block-sde B.5 v2 — pivot pivot with canonical bootstrap alignment (Path 1a).

This driver replaces the v1 pivot (run_sde_pivot.py) which produced n_finite=5
due to a 2500-triple unperturbed-pool mismatch with the canonical
ablation_audit eval set. v2 reproduces the canonical seed-N construction
exactly:

  1. Load full ``data/WN18RR/{train,valid,test}.txt`` triples (93,003).
  2. Apply ``KGPerturbator(triples, seed=N)`` and ``perturb_logical(
     "asymmetry", n_violations=int(len(triples)*0.10), "WN18RR")``.
  3. Sample 10,000 eval triples stratified via ``sample_triples(..., seed=N)``.
  4. Build ``KGODESystem`` on the **perturbed_triples pool** (full graph
     + injected violations, ~102k triples).
  5. For both ``lambda_asym = ref`` and ``lambda_asym = 0``, compute the
     SDE per-triple variance over k=10 trajectories on the eval set.
  6. Apply the knee threshold on the variance and compute F1 vs the
     ground-truth violation set.

This is more compute than v1 (the constraint structure cache must be
rebuilt for the 102k-triple pool), but only this protocol gives an
n_finite that matches the canonical Wilcoxon panel and a paired
comparison that actually exercises the asymmetry gradient.

Usage::

    .venv/bin/python -u src/experiments/run_sde_pivot_v2.py
        [--dataset WN18RR] [--seed 1] [--k 10] [--sample-size 10000]

Outputs under ``paper_results/<ds>/sde_pivot_v2/seed_<N>/{ref,zero}/``:
  - ``variances.npy``  — (n_eval,) per-triple variances, NaN for unknown.
  - ``gt.npy``         — (n_eval,) bool, ``True`` if triple in violation_set.
  - ``ode_energy.npy`` — (n_eval,) baseline deterministic ODE energy
                         (extracted from the same KGODESystem post-integration).
  - ``eval_triples.json`` — (n_eval,) list of (s, r, o, is_violation).

Aggregate at ``progress/sde_pivot_WN18RR_asymmetry_v2.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from kneed import KneeLocator

from data import DataLoader  # noqa: E402
from ablation_study import KGPerturbator  # noqa: E402
from integrated_framework import IntegratedKGDebugger  # noqa: E402
from run_all import (  # noqa: E402
    get_constraint_config,
    _load_hyperparam_cache,
    sample_triples,
)
from sde_system import KGSDESystem, compute_per_triple_variance  # noqa: E402

DEFAULT_SIGMA = 0.05  # from B.4
DEFAULT_K = 10
DEFAULT_N_STEPS = 50


def build_kgode_with_lambda_asym(dataset, lambda_asym_val, triples_pool, device="cpu",
                                  initialise_relations: bool = True):
    """Build a KGODESystem with the canonical configuration but force lambda_asym.

    The reference path ``IntegratedKGDebugger.setup_ode_detector`` builds the
    embedding dict with **entities only** (no relation names as keys), which
    silently makes the constraint gradient compute with ``r = 0`` (the
    ``embeddings.get(self.relation, np.zeros_like(...))`` fallback in
    ``src/data.py``).  This makes the antisymmetry hinge gradient identically
    zero (``forward_sq - reverse_sq = 0`` when r=0), and the H_asym null
    verdict at the SDE observable level is a direct mechanical consequence
    of this implementation choice rather than a Fokker-Planck property.

    When ``initialise_relations=True`` (the default for this driver as of
    the perf milestone), we extend the embedding dict with random relation
    vectors of the same distribution as the entity init, so the constraint
    gradients see a non-zero ``r`` and the H_asym hypothesis can actually
    be tested. ``initialise_relations=False`` reproduces the canonical
    (r=0) protocol for backwards comparison.
    """
    bp = _load_hyperparam_cache(dataset, "ode")
    config = IntegratedKGDebugger.get_default_config()
    config["use_ode"] = True
    config["use_pykeen"] = False
    config["ode_config"] = {
        "embedding_dim": bp["embedding_dim"],
        "t_span": tuple(bp["t_span"]) if isinstance(bp["t_span"], list) else bp["t_span"],
        "lambda_data":   bp["lambda_data"],
        "lambda_logic":  bp["lambda_logic"],
        "lambda_asym":   lambda_asym_val,
        "margin_asym":   bp.get("margin_asym", 1.0),
        "lambda_reg":    bp["lambda_reg"],
        "rtol":          bp.get("rtol", 5e-3),
        "atol":          bp.get("atol", 1e-6),
        "device":        device,
    }
    config["constraint_config"] = get_constraint_config(dataset)
    dbg = IntegratedKGDebugger(config)
    dbg.setup_ode_detector(triples_pool)
    kgode = dbg.ode_system
    assert kgode is not None

    if initialise_relations:
        # Extend y0 with random rows for each distinct relation that participates
        # in the constraint structures. We rebuild a KGODESystem with the same
        # constraint_checker and triples but an embeddings dict that now
        # includes relations as keys.
        import numpy as np
        from ode_system import KGODESystem, ODEConfig
        from data import ConstraintChecker
        D = kgode.embedding_dim
        relations = sorted({t.relation for t in triples_pool})
        # Reuse y0 rows for entities (already deterministic); generate fresh
        # vectors for relations using the SAME init distribution as
        # setup_ode_detector (np.random.randn * 0.1).
        emb = {}
        y0_np = kgode.y0.detach().cpu().numpy()
        for i, e in enumerate(kgode.entity_list):
            emb[e] = y0_np[i].astype(np.float32)
        for r in relations:
            if r not in emb:
                emb[r] = (np.random.randn(D) * 0.1).astype(np.float32)
        # Rebuild with the same constraint_checker (already populated) and the
        # same ODE config object on the new embeddings dict.
        kgode2 = KGODESystem(emb, kgode.triples, kgode.constraint_checker, kgode.config)
        return kgode2
    return kgode


def f1_knee(scores, gt, label=""):
    """Knee threshold + F1; returns dict + threshold."""
    n = len(scores)
    sorted_s = np.sort(scores)
    try:
        kl = KneeLocator(np.arange(n), sorted_s, curve="convex",
                         direction="increasing", interp_method="polynomial")
        knee_idx = int(kl.knee) if kl.knee is not None else int(0.95 * n)
    except Exception:
        knee_idx = int(0.95 * n)
    thr = float(sorted_s[knee_idx])
    detected = scores > thr
    tp = int((detected & gt).sum())
    fp = int((detected & ~gt).sum())
    fn = int((~detected & gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return {
        "label": label,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "n_detected": int(detected.sum()),
        "threshold": thr,
        "n_eval_used": int(n),
    }


def run_config(dataset, lambda_asym_val, perturbed_triples, eval_triples,
               eval_viol_set, k, sigma, n_steps, seed_base, label):
    """Build KGODESystem + KGSDESystem and compute per-triple variances on eval_triples.

    Returns:
        dict with variances (np array, length len(eval_triples)),
        ode_energy baseline (np array, NaN for unknown), and timings.
    """
    print(f"\n--- config={label}  λ_asym={lambda_asym_val:g} ---", flush=True)
    t0 = time.time()
    # Seed np.random BEFORE KGODESystem init so y0 is deterministic for both configs.
    np.random.seed(int(seed_base))
    torch.manual_seed(int(seed_base))
    kgode = build_kgode_with_lambda_asym(dataset, lambda_asym_val, perturbed_triples)
    setup_s = time.time() - t0
    print(f"  KGODESystem ready in {setup_s:.1f}s  "
          f"(n_entities={kgode.n_entities}, n_triples={len(kgode.triples)})", flush=True)

    sde = KGSDESystem(kgode, sigma=sigma)
    t1 = time.time()
    variances = compute_per_triple_variance(
        sde, eval_triples,
        k_trajectories=k, t_span=kgode.config.t_span,
        n_steps=n_steps, seed_base=int(seed_base),
        verbose=True,
    )
    sde_s = time.time() - t1
    v_np = variances.cpu().numpy()
    finite = np.isfinite(v_np)
    print(f"  variance: mean={np.nanmean(v_np):.4e}, std={np.nanstd(v_np):.4e}, "
          f"n_finite={int(finite.sum())}/{len(v_np)} ({sde_s:.1f}s)", flush=True)
    return {
        "label": label, "lambda_asym": float(lambda_asym_val),
        "variances": v_np, "n_finite": int(finite.sum()),
        "variance_mean": float(np.nanmean(v_np)) if finite.any() else float("nan"),
        "variance_std": float(np.nanstd(v_np)) if finite.any() else float("nan"),
        "setup_s": round(setup_s, 1), "sde_s": round(sde_s, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN18RR")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--level-pct", type=float, default=10.0,
                    help="Logical violation level percentage (canonical=10%%)")
    args = ap.parse_args()

    print(f"\n{'='*72}", flush=True)
    print(f"# B.5 v2 — {args.dataset}/asymmetry @{args.level_pct}%  seed={args.seed}", flush=True)
    print(f"#   σ={args.sigma}, k={args.k}, n_steps={args.n_steps}, "
          f"sample_size={args.sample_size}", flush=True)
    print(f"{'='*72}", flush=True)

    # Reproduce canonical bootstrap eval state
    loader = DataLoader()
    triples = loader.load_dataset(args.dataset)
    n_base = len(triples)
    n_viol = max(10, int(n_base * (args.level_pct / 100.0)))
    print(f"  loaded {n_base} triples; will inject n_viol={n_viol} asymmetry violations", flush=True)

    t0 = time.time()
    pert = KGPerturbator(triples, seed=int(args.seed))
    perturbed_triples, violation_set, _counts = pert.perturb_logical(
        violation_type="asymmetry", n_violations=n_viol, dataset_name=args.dataset,
    )
    t_pert = time.time() - t0
    print(f"  perturbed: {len(perturbed_triples)} triples, "
          f"{len(violation_set)} actual violations injected ({t_pert:.1f}s)", flush=True)

    eval_triples, eval_viol = sample_triples(
        perturbed_triples, violation_set, args.sample_size, seed=int(args.seed)
    )
    print(f"  eval: {len(eval_triples)} triples, "
          f"{len(eval_viol)} violations in eval (= ground truth positives)", flush=True)

    # Ground truth array (aligned with eval_triples order)
    eval_viol_keys = set(eval_viol)
    gt = np.array([
        (t.subject, t.relation, t.object) in eval_viol_keys
        for t in eval_triples
    ], dtype=bool)
    print(f"  gt array: {int(gt.sum())} positive / {len(gt)} total", flush=True)

    # Two configs: ref (cached lambda_asym) and zero
    bp = _load_hyperparam_cache(args.dataset, "ode")
    lambda_asym_ref = bp.get("lambda_asym", 1.0)

    out_root = REPO_ROOT / "paper_results" / args.dataset / "sde_pivot_v2" / f"seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": {
            "dataset": args.dataset, "seed": args.seed, "sigma": args.sigma,
            "k_trajectories": args.k, "n_steps": args.n_steps,
            "level_pct": args.level_pct, "sample_size": args.sample_size,
            "violation_type": "asymmetry",
            "n_base_triples": n_base, "n_injected_violations": len(violation_set),
            "n_perturbed_pool": len(perturbed_triples),
            "n_eval": len(eval_triples), "n_eval_violations": int(gt.sum()),
            "lambda_asym_ref": lambda_asym_ref,
            "perturbation_time_s": round(t_pert, 1),
        },
    }

    configs = [("ref", lambda_asym_ref), ("zero", 0.0)]
    config_results = {}
    for label, lam in configs:
        r = run_config(
            args.dataset, lam, perturbed_triples, eval_triples, eval_viol,
            args.k, args.sigma, args.n_steps, args.seed, label,
        )
        # Persist per-config
        cdir = out_root / label
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(cdir / "variances.npy", r["variances"])
        config_results[label] = r

    # Save eval triples + ground truth (shared across configs)
    np.save(out_root / "gt.npy", gt)
    (out_root / "eval_triples.json").write_text(json.dumps([
        {"s": t.subject, "r": t.relation, "o": t.object,
         "is_violation": bool(gt[i])}
        for i, t in enumerate(eval_triples)
    ]))

    # Apply knee threshold per config + compute F1s
    print(f"\n=== F1 comparison (knee threshold on variance observable) ===", flush=True)
    f1s = {}
    for label in ("ref", "zero"):
        v = config_results[label]["variances"]
        finite_mask = np.isfinite(v)
        v_f = v[finite_mask]
        gt_f = gt[finite_mask]
        if len(v_f) < 2:
            print(f"  {label}: too few finite ({len(v_f)}), skipping F1", flush=True)
            f1s[label] = {"error": "too_few_finite", "n_eval_used": len(v_f)}
            continue
        result = f1_knee(v_f, gt_f, label=f"{label}_variance_knee")
        f1s[label] = result
        print(f"  {label} (λ_asym={config_results[label]['lambda_asym']:g}): "
              f"F1={result['f1']:.4f}, n_det={result['n_detected']}/{len(v_f)}, "
              f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}",
              flush=True)

    if "ref" in f1s and "zero" in f1s and "error" not in f1s["ref"] and "error" not in f1s["zero"]:
        delta_f1 = f1s["ref"]["f1"] - f1s["zero"]["f1"]
        print(f"\n  Δ F1 (ref − zero): {delta_f1:+.4f}", flush=True)
        if abs(delta_f1) < 0.02:
            verdict = ("NULL — stochastic observable does not differentiate; "
                       "single-seed; reinforces H_asym pending B.6 scaling.")
        elif abs(delta_f1) > 0.05:
            verdict = ("SIGNAL — material Δ F1; recommend scaling to B.6 "
                       "full 10-seed Wilcoxon panel.")
        else:
            verdict = ("WEAK SIGNAL — 3-5 additional seeds before decision.")
    else:
        delta_f1 = float("nan")
        verdict = "INCOMPLETE — one or both configs produced no finite variances."
    summary["configs"] = {k: {kk: vv for kk, vv in v.items() if kk != "variances"}
                          for k, v in config_results.items()}
    summary["f1"] = f1s
    summary["delta_f1"] = float(delta_f1) if delta_f1 == delta_f1 else None
    summary["verdict"] = verdict

    json_path = REPO_ROOT / "progress" / "sde_pivot_WN18RR_asymmetry_v2.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nVerdict: {verdict}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {out_root}/{{ref,zero}}/variances.npy + gt.npy + eval_triples.json",
          flush=True)


if __name__ == "__main__":
    main()
