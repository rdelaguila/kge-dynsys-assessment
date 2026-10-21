"""Progressive S0-S4 ablation (R2.1 closure), explicitly authorized.

Isolates, within ONE internally-paired family per (dataset, seed), the
marginal contribution of: zero-integration, data-only evolution,
symmetry/transitivity forces, antisymmetry force, and the post-hoc
symbolic checker/bonus. Reuses exactly the proven building blocks already
used elsewhere in this review (KGPerturbator, sample_triples, the
build_kgode/rk4_integrate/f1_knee pattern from
run_c_h_asym_deterministic.py, and the standalone constraint-checker +
post-hoc bonus pattern from run_c2_dynamics_symbolic_2x2.py) rather than
inventing new mechanics, so that every stage's comparability rests on
code already exercised and reported in this review.

For a given (dataset, seed): the perturbed pool, the 10,000-triple sample,
and the ground-truth labels are built ONCE and reused identically across
all five stages -- guaranteed by construction (same in-memory objects),
not by after-the-fact comparison of separate runs. This is why old 2x2
data (WN18RR only, 2 of 5 stages, built by a separate script invocation)
is NOT reused here: proving its exact equivalence would cost as much as
recomputing it, and splicing reused + fresh stages for one dataset but not
the other two would itself violate internal pairing.

Stages (all five share the same per-seed perturbed pool / sample / labels):
  S0 zero-integration : kgode.y0, kgode.r_vectors directly, no RK4 at all.
  S1 data-only         : lambda_logic=0, lambda_asym=0.
  S2 data+sym/trans    : lambda_logic=ref, lambda_asym=0.    (= A1)
  S3 +antisymmetry     : lambda_logic=ref, lambda_asym=ref.  (= A2)
  S4 +post-hoc checker : S3's embeddings, symbolic bonus added post-hoc
                         (q99-calibrated on S3's finite residuals for that
                         seed, exactly as run_c2_dynamics_symbolic_2x2.py).

Outputs an isolated directory (not paper_results/, not progress/'s
existing milestone files): progress/r2_1_progressive_ablation/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from kneed import KneeLocator

from ablation_study import KGPerturbator
from data import DataLoader
from constraints import ConstraintChecker, TransitivityConstraint, SymmetryConstraint, AntisymmetryConstraint
from run_all import _load_hyperparam_cache, get_constraint_config, sample_triples
from integrated_framework import IntegratedKGDebugger
from ode_system import KGODESystem, ODEConfig

OUT_DIR = REPO_ROOT / "progress" / "r2_1_progressive_ablation"
SAMPLE_SIZE = 10000
LEVEL_PCT = 10.0
N_STEPS = 100  # matches the H_asym deterministic panel exactly


def build_kgode(dataset, lambda_logic_val, lambda_asym_val, perturbed_triples, device="cpu"):
    bp = _load_hyperparam_cache(dataset, "ode")
    cfg = IntegratedKGDebugger.get_default_config()
    cfg["use_ode"] = True
    cfg["use_pykeen"] = False
    cfg["ode_config"] = {
        "embedding_dim": bp["embedding_dim"],
        "t_span": tuple(bp["t_span"]) if isinstance(bp["t_span"], list) else bp["t_span"],
        "lambda_data": bp["lambda_data"],
        "lambda_logic": lambda_logic_val,
        "lambda_asym": lambda_asym_val,
        "margin_asym": bp.get("margin_asym", 1.0),
        "lambda_reg": bp["lambda_reg"],
        "rtol": bp.get("rtol", 5e-3),
        "atol": bp.get("atol", 1e-6),
        "device": device,
    }
    cfg["constraint_config"] = get_constraint_config(dataset)
    dbg = IntegratedKGDebugger(cfg)
    dbg.setup_ode_detector(perturbed_triples)
    return dbg.ode_system


def rk4_integrate(kgode, n_steps: int) -> torch.Tensor:
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


def residual_for_terminal_state(kgode, y_terminal, eval_triples):
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
    s_idx_t = torch.tensor(s_idx, dtype=torch.long, device=y_terminal.device)
    o_idx_t = torch.tensor(o_idx, dtype=torch.long, device=y_terminal.device)
    r_idx_t = torch.tensor(r_idx_in_pool, dtype=torch.long, device=y_terminal.device)
    h_emb = y_terminal[s_idx_t]
    t_emb = y_terminal[o_idx_t]
    r_vec = kgode.r_vectors[r_idx_t]
    energy = ((h_emb + r_vec - t_emb) ** 2).sum(dim=-1).cpu().numpy()
    valid_mask_np = np.asarray(valid_mask, dtype=bool)
    if not valid_mask_np.all():
        energy[~valid_mask_np] = np.nan
    return energy


def knee_threshold(x: np.ndarray):
    """Explicit, non-improvised handling of degenerate distributions:
    constant/near-constant x -> KneeLocator typically returns None -> fall
    back to the 95th percentile (same fallback already used in production
    and in run_c_h_asym_deterministic.py's f1_knee). Never fabricate a
    threshold; report knee_found=False when this fallback fires."""
    n = len(x)
    if n == 0 or np.allclose(x, x[0]):
        return float(np.percentile(x, 95)) if n else float("nan"), False
    sorted_e = np.sort(x)[::-1]
    xs = np.arange(n)
    try:
        knl = KneeLocator(xs, sorted_e, curve="convex", direction="decreasing", interp_method="polynomial")
    except Exception:
        return float(np.percentile(x, 95)), False
    if knl.knee is None:
        return float(np.percentile(x, 95)), False
    return float(sorted_e[int(knl.knee)]), True


def f1_from_scores(scores: np.ndarray, gt: np.ndarray, k_positive: int):
    finite = np.isfinite(scores)
    n_nonfinite = int((~finite).sum())
    s, g = scores[finite], gt[finite]
    thr, knee_found = knee_threshold(s)
    detected = s >= thr
    tp = int((detected & g).sum()); fp = int((detected & ~g).sum()); fn = int((~detected & g).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    # NDCG@K, K = effective positives in this stage's finite eval set
    K = int(g.sum())
    ndcg = None
    if K > 0:
        order = np.argsort(-s)
        rel = g[order][:K].astype(float)
        discounts = 1.0 / np.log2(np.arange(2, K + 2))
        dcg = float((rel * discounts).sum())
        ideal = np.ones(min(K, int(g.sum())))
        idcg = float((ideal * discounts[: len(ideal)]).sum())
        ndcg = dcg / idcg if idcg > 0 else None
    return {
        "f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn,
        "n_detected": int(detected.sum()), "n_eval_finite": int(finite.sum()),
        "n_nonfinite": n_nonfinite, "threshold": thr, "knee_found": knee_found,
        "ndcg_at_k": ndcg, "k_positive_effective": K,
    }


def build_constraint_checker(dataset, margin_asym):
    cfg = get_constraint_config(dataset)
    checker = ConstraintChecker()
    for rel in cfg["transitivity_relations"]:
        checker.add_constraint(TransitivityConstraint(rel))
    for rel in cfg["symmetry_relations"]:
        checker.add_constraint(SymmetryConstraint(rel))
    for rel in cfg.get("asymmetry_relations", []):
        checker.add_constraint(AntisymmetryConstraint(rel, margin=margin_asym, enabled=True))
    return checker


def count_violations(checker, full_triples, eval_triples):
    cached = {c.name: set(c.get_violating_triples(full_triples)) for c in checker.constraints}
    counts = np.zeros(len(eval_triples), dtype=int)
    for i, t in enumerate(eval_triples):
        counts[i] = sum(1 for c in checker.constraints if t in cached[c.name])
    return counts


def run_one_seed(dataset: str, seed: int) -> dict:
    t0 = time.time()
    loader = DataLoader()
    base = loader.load_dataset(dataset)
    n_viol = max(10, int(len(base) * (LEVEL_PCT / 100.0)))
    pert = KGPerturbator(base, seed=seed)
    perturbed, violation_set, _ = pert.perturb_logical(
        violation_type="asymmetry", n_violations=n_viol, dataset_name=dataset,
    )
    eval_triples, eval_gt_set = sample_triples(perturbed, violation_set, SAMPLE_SIZE, seed=seed)
    gt_keys = set(eval_gt_set)
    gt = np.array([(t.subject, t.relation, t.object) in gt_keys for t in eval_triples], dtype=bool)
    sample_hash = hashlib.sha256(
        "|".join(f"{t.subject}:{t.relation}:{t.object}" for t in eval_triples).encode()
    ).hexdigest()[:16]

    bp = _load_hyperparam_cache(dataset, "ode")
    lambda_logic_ref = bp["lambda_logic"]
    lambda_asym_ref = float(bp.get("lambda_asym", 1.0))
    margin_asym = float(bp.get("margin_asym", 1.0))

    checker = build_constraint_checker(dataset, margin_asym)
    viol_counts_full = count_violations(checker, perturbed, eval_triples)

    stages = {}

    # S0: zero-integration -- build kgode (embedding init only), read y0 directly, no RK4.
    np.random.seed(seed); torch.manual_seed(seed)
    kgode0 = build_kgode(dataset, lambda_logic_ref, lambda_asym_ref, perturbed)
    e_s0 = residual_for_terminal_state(kgode0, kgode0.y0, eval_triples)
    stages["S0_zero_integration"] = f1_from_scores(e_s0, gt, int(gt.sum()))

    # S1: data-only (logic=0, asym=0)
    np.random.seed(seed); torch.manual_seed(seed)
    kgode1 = build_kgode(dataset, 0.0, 0.0, perturbed)
    y1 = rk4_integrate(kgode1, N_STEPS)
    e_s1 = residual_for_terminal_state(kgode1, y1, eval_triples)
    stages["S1_data_only"] = f1_from_scores(e_s1, gt, int(gt.sum()))

    # S2: data + sym/trans (logic=ref, asym=0) -- equivalent to A1
    np.random.seed(seed); torch.manual_seed(seed)
    kgode2 = build_kgode(dataset, lambda_logic_ref, 0.0, perturbed)
    y2 = rk4_integrate(kgode2, N_STEPS)
    e_s2 = residual_for_terminal_state(kgode2, y2, eval_triples)
    stages["S2_data_sym_trans"] = f1_from_scores(e_s2, gt, int(gt.sum()))

    # S3: + antisymmetry (logic=ref, asym=ref) -- equivalent to A2
    np.random.seed(seed); torch.manual_seed(seed)
    kgode3 = build_kgode(dataset, lambda_logic_ref, lambda_asym_ref, perturbed)
    y3 = rk4_integrate(kgode3, N_STEPS)
    e_s3 = residual_for_terminal_state(kgode3, y3, eval_triples)
    stages["S3_plus_antisymmetry"] = f1_from_scores(e_s3, gt, int(gt.sum()))

    # S4: S3's own embeddings, post-hoc bonus added to the SAME residual (no re-integration)
    finite3 = np.isfinite(e_s3)
    q99 = float(np.quantile(e_s3[finite3], 0.99)) if finite3.any() else float("nan")
    bonus = q99 if (np.isfinite(q99) and q99 > 0) else 500.0
    e_s4 = e_s3 + viol_counts_full * bonus
    e_s4[~finite3] = np.nan
    s4_metrics = f1_from_scores(e_s4, gt, int(gt.sum()))
    s4_metrics["bonus_value"] = bonus
    s4_metrics["n_eval_with_violation"] = int((viol_counts_full > 0).sum())
    stages["S4_plus_checker_bonus"] = s4_metrics

    elapsed = time.time() - t0
    return {
        "dataset": dataset, "seed": seed, "elapsed_s": elapsed,
        "n_eval": len(eval_triples), "n_eval_positive": int(gt.sum()),
        "sample_sha256_16": sample_hash,
        "lambda_logic_ref": lambda_logic_ref, "lambda_asym_ref": lambda_asym_ref,
        "stages": stages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["FB15k-237", "codex-m", "WN18RR"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__,
    }

    for dataset in args.datasets:
        results = []
        t_ds = time.time()
        for seed in args.seeds:
            print(f"\n--- {dataset} seed {seed}/{args.seeds[-1]} ---", flush=True)
            try:
                r = run_one_seed(dataset, seed)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                results.append({"dataset": dataset, "seed": seed, "ok": False, "error": f"{type(e).__name__}: {e}"})
                continue
            r["ok"] = True
            results.append(r)
            s = r["stages"]
            print(f"  done in {r['elapsed_s']:.1f}s | "
                  f"S0={s['S0_zero_integration']['f1']:.4f} S1={s['S1_data_only']['f1']:.4f} "
                  f"S2={s['S2_data_sym_trans']['f1']:.4f} S3={s['S3_plus_antisymmetry']['f1']:.4f} "
                  f"S4={s['S4_plus_checker_bonus']['f1']:.4f}", flush=True)
            out_seed = OUT_DIR / f"{dataset}_seed_{seed}.json"
            with open(out_seed, "w") as f:
                json.dump(r, f, indent=2)
        ds_elapsed = time.time() - t_ds
        out_summary = OUT_DIR / f"{dataset}_summary.json"
        with open(out_summary, "w") as f:
            json.dump({"dataset": dataset, "seeds": args.seeds, "elapsed_s": ds_elapsed,
                       "env": env, "per_seed": results}, f, indent=2)
        print(f"\n=== {dataset} done in {ds_elapsed:.1f}s, wrote {out_summary} ===")


if __name__ == "__main__":
    main()
