"""Milestone: focused ODE-only re-run post-fix de las tablas §4.3/§4.5/§4.6/§4.7.

Alcance:
  - 9 cells de random ablation: 3 datasets × 3 levels (5/10/15%).
  - 27 cells de logical ablation: 3 datasets × 3 violation_types × 3 levels.
  - Bootstrap E6: 3 datasets × 10 seeds × random@5%.

Reutiliza:
  - KGE detections existentes (no se re-entrena KGE; el bug solo afecta ODE).
  - El perf module `constraint_gradient_vectorized` está activo via HEAD.

Outputs:
  - paper_results/<ds>/post_fix_rerun/metrics/metrics_random_entity_swap_<level>.json
  - paper_results/<ds>/post_fix_rerun/metrics/metrics_logical_<vtype>_<n>.json
  - paper_results/<ds>/bootstrap_postfix/seed_<N>/random/metrics/metrics_random_entity_swap_5.0.json

Determinismo: random_seed=42 a nivel native (matches publicado); perturbation_seed=N for bootstrap seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from data import (
    DataLoader, Triple, ConstraintChecker,
    TransitivityConstraint, SymmetryConstraint, AntisymmetryConstraint,
)
from ode_system import KGODESystem, ODEConfig
from run_all import (
    get_constraint_config, _load_hyperparam_cache, sample_triples,
)
from ablation_study import KGPerturbator


LEVEL_BY_DATASET = {
    "WN18RR":    {5.0: 4650, 10.0: 9300, 15.0: 13950},
    "FB15k-237": {5.0: 15505, 10.0: 31011, 15.0: 46517},
    "codex-m":   {5.0: 10310, 10.0: 20620, 15.0: 30930},
}

VIOLATION_TYPES = ("asymmetry", "transitivity", "cardinality")


def build_cc(dataset: str, relations_present: set) -> ConstraintChecker:
    cfg = get_constraint_config(dataset)
    cc = ConstraintChecker()
    for r in cfg.get("transitivity_relations", []):
        if r in relations_present: cc.add_constraint(TransitivityConstraint(r, weight=1.0))
    for r in cfg.get("symmetry_relations", []):
        if r in relations_present: cc.add_constraint(SymmetryConstraint(r, weight=1.0))
    for r in cfg.get("asymmetry_relations", []):
        if r in relations_present:
            cc.add_constraint(AntisymmetryConstraint(r, weight=1.0, margin=1.0,
                                                      treats_as_gradient=True))
    return cc


def build_kgode_postfix(dataset: str, triples_pool, seed: int = 42, device: str = "cpu",
                         lambda_asym_override: float | None = None):
    bp = _load_hyperparam_cache(dataset, "ode")
    D = bp["embedding_dim"]
    entities = sorted({t.subject for t in triples_pool} | {t.object for t in triples_pool})
    relations = sorted({t.relation for t in triples_pool})
    np.random.seed(seed); torch.manual_seed(seed)
    emb = {e: (np.random.randn(D) * 0.1).astype(np.float32) for e in entities}
    for r in relations:
        emb[r] = (np.random.randn(D) * 0.1).astype(np.float32)
    cc = build_cc(dataset, set(relations))
    lam_asym = bp.get("lambda_asym", 1.0) if lambda_asym_override is None else float(lambda_asym_override)
    odecfg = ODEConfig(
        t_span=tuple(bp["t_span"]) if isinstance(bp["t_span"], list) else bp["t_span"],
        lambda_data=bp["lambda_data"], lambda_logic=bp["lambda_logic"],
        lambda_asym=lam_asym,
        margin_asym=bp.get("margin_asym", 1.0),
        lambda_reg=bp["lambda_reg"],
        rtol=bp.get("rtol", 5e-3), atol=bp.get("atol", 1e-6),
        device=device,
    )
    return KGODESystem(emb, triples_pool, cc, odecfg)


def rk4_terminal(kgode, n_steps=50):
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


def per_triple_energy(kgode, y_T):
    H = y_T[kgode.s_indices]
    T = y_T[kgode.o_indices]
    R = kgode.r_vectors
    return ((H + R - T) ** 2).sum(dim=-1).cpu().numpy()


def knee_threshold(scores: np.ndarray) -> float:
    from kneed import KneeLocator
    n = len(scores)
    sorted_s = np.sort(scores)
    try:
        kl = KneeLocator(np.arange(n), sorted_s, curve="convex",
                         direction="increasing", interp_method="polynomial")
        knee_idx = int(kl.knee) if kl.knee is not None else int(0.95 * n)
    except Exception:
        knee_idx = int(0.95 * n)
    return float(sorted_s[knee_idx])


def f1_for(detection_mask: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((detection_mask & gt).sum())
    fp = int((detection_mask & ~gt).sum())
    fn = int((~detection_mask & gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return dict(true_positives=tp, false_positives=fp, false_negatives=fn,
                precision=p, recall=r, f1=f1,
                n_detected=int(detection_mask.sum()),
                n_violations=int(gt.sum()),
                n_total=int(len(gt)))


def detect_one_cell(dataset: str, stage_type: str, vtype: str | None, level: float,
                    n_violations: int, sample_size: int = 10000,
                    perturbation_seed: int = 42, n_steps: int = 50,
                    device: str = "cpu",
                    lambda_asym_override: float | None = None,
                    arch_label: str = "A2") -> dict:
    """Detect anomalies in one cell with ODE post-fix.

    For random ablation, vtype is None and level is the percentage.
    For logical ablation, vtype is one of {asymmetry, transitivity, cardinality}
    and n_violations is the absolute injection count.
    """
    loader = DataLoader()
    base = loader.load_dataset(dataset)

    perturbator = KGPerturbator(base, seed=int(perturbation_seed))
    if stage_type == "random":
        perturbed, violation_set = perturbator.perturb(
            percentage=level, perturbation_types=["entity_swap"],
        )
    else:
        perturbed, violation_set, _ = perturbator.perturb_logical(
            violation_type=vtype, n_violations=n_violations, dataset_name=dataset,
        )

    if not violation_set:
        return {"error": "no_violations_generated"}

    eval_triples, eval_viol = perturbed, violation_set
    if sample_size and sample_size < len(perturbed):
        eval_triples, eval_viol = sample_triples(
            perturbed, violation_set, sample_size, seed=int(perturbation_seed),
        )

    n_eval_viol = len(eval_viol)

    # ODE detect
    kgode = build_kgode_postfix(dataset, perturbed, seed=42, device=device,
                                 lambda_asym_override=lambda_asym_override)
    y_T = rk4_terminal(kgode, n_steps=n_steps)
    energies_pool = per_triple_energy(kgode, y_T)

    triple_to_idx = {(t.subject, t.relation, t.object): i
                     for i, t in enumerate(kgode.triples)}
    eval_energies = []
    valid = []
    for t in eval_triples:
        k = (t.subject, t.relation, t.object)
        i = triple_to_idx.get(k)
        if i is None:
            valid.append(False); eval_energies.append(np.nan)
        else:
            valid.append(True); eval_energies.append(float(energies_pool[i]))
    eval_energies = np.array(eval_energies)
    valid = np.array(valid)
    gt = np.array([(t.subject, t.relation, t.object) in eval_viol
                   for t in eval_triples], dtype=bool)
    finite = np.isfinite(eval_energies)
    if finite.sum() < 10:
        return {"error": "too_few_valid_eval_triples", "n_finite": int(finite.sum())}

    # [2026-06-23] Dump per-triple arrays for threshold-free metrics (PR-AUC/AUROC).
    _dump_dir = Path("progress/20260623")
    _dump_dir.mkdir(parents=True, exist_ok=True)
    np.savez(_dump_dir / f"ode_postfix_{arch_label}_{dataset}_{vtype or stage_type}_{n_violations}.npz",
             eval_energies=eval_energies, gt=gt, valid=valid, finite=finite,
             arch=arch_label)

    thr = knee_threshold(eval_energies[finite])
    detection = (eval_energies > thr) & finite
    ode_metrics = f1_for(detection, gt)
    ode_metrics["threshold_knee"] = thr

    return {
        "ode_postfix": ode_metrics,
        "protocol": {
            "dataset": dataset, "stage_type": stage_type, "vtype": vtype,
            "level": level, "n_violations": n_violations,
            "sample_size": sample_size, "perturbation_seed": perturbation_seed,
            "n_steps_rk4": n_steps,
        },
    }


def run_native_tables(datasets, n_steps: int, sample_size: int = 10000,
                       lambda_asym_override: float | None = None,
                       arch_label: str = "A2"):
    """Re-generate Tables 3 + 4a/4b/4c — native perturbation_seed=42."""
    results = {}
    for ds in datasets:
        ds_out = REPO_ROOT / "paper_results" / ds / "post_fix_rerun" / "metrics"
        ds_out.mkdir(exist_ok=True, parents=True)
        results[ds] = {}

        # Random ablation @5/10/15%
        for level in (5.0, 10.0, 15.0):
            print(f"\n[{ds}] random @ {level}%")
            t0 = time.time()
            r = detect_one_cell(ds, "random", None, level, n_violations=0,
                                sample_size=sample_size, perturbation_seed=42,
                                n_steps=n_steps,
                                lambda_asym_override=lambda_asym_override,
                                arch_label=arch_label)
            r["wall_clock_s"] = time.time() - t0
            results[ds].setdefault("random", {})[level] = r
            with open(ds_out / f"metrics_random_entity_swap_{level}.json", "w") as f:
                json.dump(r, f, indent=2)
            if "error" not in r:
                m = r["ode_postfix"]
                print(f"  ODE post-fix F1 = {m['f1']:.4f}  (TP={m['true_positives']}, "
                      f"FP={m['false_positives']}, FN={m['false_negatives']}, "
                      f"n_det={m['n_detected']}, n_viol={m['n_violations']})  [{r['wall_clock_s']:.1f}s]")
            else:
                print(f"  ERROR: {r['error']}")

        # Logical ablation × 3 violation_types × 3 levels
        for vtype in VIOLATION_TYPES:
            for level in (5.0, 10.0, 15.0):
                n_inject = LEVEL_BY_DATASET[ds][level]
                # Adjust per-level for logical
                n_inject_for_level = LEVEL_BY_DATASET[ds][level]
                print(f"\n[{ds}] logical {vtype} @ n={n_inject_for_level}")
                t0 = time.time()
                r = detect_one_cell(ds, "logical", vtype, level, n_violations=n_inject_for_level,
                                    sample_size=sample_size, perturbation_seed=42,
                                    n_steps=n_steps,
                                    lambda_asym_override=lambda_asym_override,
                                    arch_label=arch_label)
                r["wall_clock_s"] = time.time() - t0
                results[ds].setdefault(f"logical_{vtype}", {})[level] = r
                with open(ds_out / f"metrics_logical_{vtype}_{n_inject_for_level}.json", "w") as f:
                    json.dump(r, f, indent=2)
                if "error" not in r:
                    m = r["ode_postfix"]
                    print(f"  ODE post-fix F1 = {m['f1']:.4f}  (TP={m['true_positives']}, "
                          f"FP={m['false_positives']}, FN={m['false_negatives']}, "
                          f"n_det={m['n_detected']}, n_viol={m['n_violations']})  [{r['wall_clock_s']:.1f}s]")
                else:
                    print(f"  ERROR: {r['error']}")
    return results


def run_bootstrap_e6(datasets, n_seeds: int, n_steps: int, sample_size: int = 10000):
    """Re-generate Bootstrap E6 — random@5% × 10 seeds, ODE post-fix only."""
    results = {}
    for ds in datasets:
        results[ds] = {}
        for seed in range(1, n_seeds + 1):
            seed_out = REPO_ROOT / "paper_results" / ds / "bootstrap_postfix" / f"seed_{seed}" / "random" / "metrics"
            seed_out.mkdir(exist_ok=True, parents=True)
            print(f"\n[{ds}] bootstrap seed {seed}/{n_seeds} random@5%")
            t0 = time.time()
            r = detect_one_cell(ds, "random", None, 5.0, n_violations=0,
                                sample_size=sample_size, perturbation_seed=seed,
                                n_steps=n_steps)
            r["wall_clock_s"] = time.time() - t0
            results[ds][seed] = r
            with open(seed_out / "metrics_random_entity_swap_5.0.json", "w") as f:
                json.dump(r, f, indent=2)
            if "error" not in r:
                m = r["ode_postfix"]
                print(f"  ODE post-fix F1 = {m['f1']:.4f} [{r['wall_clock_s']:.1f}s]")
            else:
                print(f"  ERROR: {r['error']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["WN18RR", "FB15k-237", "codex-m"])
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--n-bootstrap-seeds", type=int, default=10)
    ap.add_argument("--skip-native", action="store_true")
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--lambda-asym-override", type=float, default=None,
                    help="Override cached lambda_asym (e.g. 0.0 for A1, --simplest)")
    ap.add_argument("--arch-label", default=None,
                    help="Label for arch in the dumped npz filename (A1/A2/A3); auto-set from override.")
    args = ap.parse_args()
    # Auto-label
    if args.arch_label is None:
        args.arch_label = "A1" if args.lambda_asym_override == 0.0 else "A2"

    print("=" * 80)
    print("# MILESTONE — Section 4 post-fix re-run (ODE only)")
    print("=" * 80)
    print(f"  datasets: {args.datasets}")
    print(f"  n_steps_rk4: {args.n_steps}")
    print(f"  sample_size: {args.sample_size}")
    print(f"  bootstrap seeds: {args.n_bootstrap_seeds}")

    t_total = time.time()
    summary = {}

    if not args.skip_native:
        summary["native"] = run_native_tables(args.datasets, args.n_steps, args.sample_size,
                                                lambda_asym_override=args.lambda_asym_override,
                                                arch_label=args.arch_label)
    if not args.skip_bootstrap:
        summary["bootstrap_e6"] = run_bootstrap_e6(args.datasets, args.n_bootstrap_seeds,
                                                    args.n_steps, args.sample_size)

    elapsed = time.time() - t_total
    summary["wall_clock_total_s"] = elapsed
    summary["protocol"] = {
        "datasets": args.datasets, "n_steps": args.n_steps,
        "sample_size": args.sample_size,
        "n_bootstrap_seeds": args.n_bootstrap_seeds,
        "fix_active": True,
    }
    out = REPO_ROOT / "progress" / "milestone_postfix_section4_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTotal wall-clock: {elapsed:.1f}s")
    print(f"Persisted: {out}")


if __name__ == "__main__":
    main()
