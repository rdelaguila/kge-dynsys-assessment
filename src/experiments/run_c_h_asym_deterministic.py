"""Milestone C — Re-evaluate the §4.7 H_asym Wilcoxon row with the
relation-init fix applied to IntegratedKGDebugger.setup_ode_detector.

This is the deterministic counterpart of the SDE B.6 panel
(`run_sde_b6_wilcoxon.py`). For each seed in 1..N:

  - Build the canonical perturbed pool: KGPerturbator(triples, seed).perturb_logical(
        violation_type='asymmetry', n_violations=10%, dataset)
  - Sample 10,000 eval triples (sample_triples, stratified, seed).
  - Build a KGODESystem with lambda_asym=ref (cached). RK4-integrate the
    drift from y0 to terminal. Per-triple energy ||h(T) + r - t(T)||².
    Knee threshold on sorted energies → detection set → F1.
  - Repeat with lambda_asym=0.
  - Δ F1 (ref − zero) for that seed.

Aggregate over N seeds with Wilcoxon paired signed-rank + rank-biserial.

Skips the KGE/random ablations entirely (those don't depend on
lambda_asym; the H_asym row is ODE-only).

The relation-init fix is now baked into IntegratedKGDebugger; we go
through that same path for parity with the canonical §4.7 protocol.
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
from scipy.stats import wilcoxon

from ablation_study import KGPerturbator
from data import DataLoader
from integrated_framework import IntegratedKGDebugger
from run_all import (
    _load_hyperparam_cache,
    get_constraint_config,
    sample_triples,
)


def build_kgode(dataset: str, lambda_asym_val: float, perturbed_triples, device="cpu"):
    """Build a KGODESystem via the canonical IntegratedKGDebugger path
    (now with the relation-init fix). Returns the ode_system attribute."""
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
    dbg.setup_ode_detector(perturbed_triples)
    return dbg.ode_system


def rk4_integrate(kgode, n_steps: int) -> torch.Tensor:
    """Plain RK4 integration of kgode.ode_dynamics over kgode.config.t_span.
    Returns terminal y (shape (n_entities, D))."""
    t0, t1 = kgode.config.t_span
    dt = (t1 - t0) / n_steps
    y = kgode.y0.clone()
    t = float(t0)
    with torch.no_grad():
        for _ in range(n_steps):
            k1 = kgode.ode_dynamics(t, y)
            k2 = kgode.ode_dynamics(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = kgode.ode_dynamics(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = kgode.ode_dynamics(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
    return y


def f1_knee(scores: np.ndarray, gt: np.ndarray, label: str = "") -> dict:
    """Knee threshold + F1 on a sorted-energy detection."""
    from kneed import KneeLocator
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
    return dict(label=label, tp=tp, fp=fp, fn=fn,
                precision=float(p), recall=float(r), f1=float(f1),
                n_detected=int(detected.sum()), threshold=thr)


def run_one_seed(dataset: str, seed: int, sample_size: int, level_pct: float,
                 n_steps: int, device: str) -> dict:
    """Run one paired comparison for one seed."""
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

    def detect(lam_val: float) -> dict:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        kgode = build_kgode(dataset, lam_val, perturbed, device=device)
        y_T = rk4_integrate(kgode, n_steps=n_steps)
        # Per-triple energy on eval_triples
        s_idx = []
        o_idx = []
        r_idx_in_pool = []
        triple_to_pool = {(t.subject, t.relation, t.object): i for i, t in enumerate(kgode.triples)}
        valid_mask = []
        for t in eval_triples:
            k = (t.subject, t.relation, t.object)
            pi = triple_to_pool.get(k)
            if (pi is None
                or t.subject not in kgode.entity_to_idx
                or t.object not in kgode.entity_to_idx):
                valid_mask.append(False)
                s_idx.append(0)
                o_idx.append(0)
                r_idx_in_pool.append(0)
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
        # Knee on finite values
        finite = np.isfinite(energy)
        eval_gt = gt[finite]
        eval_scores = energy[finite]
        return f1_knee(eval_scores, eval_gt,
                       label=f"lambda_asym={lam_val:g}"), int(finite.sum())

    f1_ref, n_ref = detect(lambda_asym_ref)
    f1_zero, n_zero = detect(0.0)

    elapsed = time.time() - t0
    return {
        "seed": seed, "ok": True, "elapsed_s": elapsed,
        "f1_ref": f1_ref["f1"], "f1_zero": f1_zero["f1"],
        "delta_f1": f1_ref["f1"] - f1_zero["f1"],
        "tp_ref": f1_ref["tp"], "fp_ref": f1_ref["fp"], "fn_ref": f1_ref["fn"],
        "tp_zero": f1_zero["tp"], "fp_zero": f1_zero["fp"], "fn_zero": f1_zero["fn"],
        "n_detected_ref": f1_ref["n_detected"],
        "n_detected_zero": f1_zero["n_detected"],
        "threshold_ref": f1_ref["threshold"],
        "threshold_zero": f1_zero["threshold"],
        "n_finite_ref": n_ref,
        "n_finite_zero": n_zero,
        "n_eval": len(eval_triples),
        "n_eval_violations": int(gt.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN18RR")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--level-pct", type=float, default=10.0)
    ap.add_argument("--n-steps", type=int, default=100,
                    help="RK4 integration steps")
    ap.add_argument("--device", default="cpu",
                    help="cpu or mps; cpu uses float64 internals for parity with scalar reference")
    args = ap.parse_args()

    print("=" * 78)
    print(f"# Milestone C — H_asym deterministic Wilcoxon | {args.dataset} asym @{args.level_pct}%")
    print(f"# n_seeds={args.n_seeds}, sample_size={args.sample_size}, n_steps={args.n_steps}, "
          f"device={args.device}, RELATION-INIT fix in IntegratedKGDebugger.")
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
              f"F1 ref={r['f1_ref']:.4f}, zero={r['f1_zero']:.4f}, "
              f"Δ={r['delta_f1']:+.4f}, n_eval_viol={r['n_eval_violations']}",
              flush=True)
    panel_elapsed = time.time() - t_panel

    ok = [r for r in results if r["ok"]]
    if len(ok) < 2:
        print(f"\nNot enough successful seeds ({len(ok)}); aborting Wilcoxon.")
        return

    deltas = np.array([r["delta_f1"] for r in ok])
    f1_refs = np.array([r["f1_ref"] for r in ok])
    f1_zeros = np.array([r["f1_zero"] for r in ok])

    try:
        wstat, pval = wilcoxon(f1_refs, f1_zeros, alternative="two-sided", zero_method="pratt")
    except ValueError as e:
        wstat, pval = float("nan"), float("nan")
        print(f"Wilcoxon failed: {e}")

    n_pos = int((deltas > 0).sum())
    n_neg = int((deltas < 0).sum())
    n_zero = int((deltas == 0).sum())
    rb = (n_pos - n_neg) / max(n_pos + n_neg, 1)
    supported = bool((not np.isnan(pval)) and (pval < 0.05))

    summary = {
        "milestone": "C — H_asym deterministic with relation-init fix",
        "protocol": {
            "dataset": args.dataset,
            "violation": "asymmetry",
            "level_pct": args.level_pct,
            "sample_size": args.sample_size,
            "n_seeds": args.n_seeds,
            "n_steps_rk4": args.n_steps,
            "device": args.device,
            "panel_wall_clock_s": panel_elapsed,
        },
        "per_seed": ok,
        "stats": {
            "mean_delta_f1": float(deltas.mean()),
            "median_delta_f1": float(np.median(deltas)),
            "stddev_delta_f1": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "min_delta_f1": float(deltas.min()),
            "max_delta_f1": float(deltas.max()),
            "n_positive": n_pos, "n_negative": n_neg, "n_zero_diffs": n_zero,
            "rank_biserial": float(rb),
            "wilcoxon_W": float(wstat) if not np.isnan(wstat) else None,
            "wilcoxon_p_two_sided": float(pval) if not np.isnan(pval) else None,
            "p_holm": float(pval) if not np.isnan(pval) else None,
            "supported_at_alpha_005": supported,
        },
    }

    out_json = REPO_ROOT / "progress" / f"milestone_c_h_asym_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("MILESTONE C — H_ASYM DETERMINISTIC PANEL SUMMARY")
    print("=" * 78)
    print(f"  Successful seeds: {len(ok)}/{args.n_seeds}")
    print(f"  Δ F1 per seed: {[f'{d:+.4f}' for d in deltas]}")
    print(f"  Mean Δ F1:        {summary['stats']['mean_delta_f1']:+.5f}")
    print(f"  Median Δ F1:      {summary['stats']['median_delta_f1']:+.5f}")
    print(f"  Stddev:           {summary['stats']['stddev_delta_f1']:.5f}")
    print(f"  Range:            [{summary['stats']['min_delta_f1']:+.5f}, {summary['stats']['max_delta_f1']:+.5f}]")
    print(f"  Signs:            +{n_pos} / -{n_neg} / 0:{n_zero}")
    print(f"  Rank-biserial:    {rb:+.3f}")
    print(f"  Wilcoxon W:       {wstat}")
    print(f"  Wilcoxon p:       {pval}")
    print(f"  α=0.05 supported: {supported}")
    print(f"  Wall-clock:       {panel_elapsed:.1f} s")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
