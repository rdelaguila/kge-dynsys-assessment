"""Phase-2 authorized minimal run — Section B: ODE score-weight sensitivity.

s_ODE(h,r,t) = w_E * E_data + w_sigma * max(Deltasigma_h, Deltasigma_t) + |C| * b

Native ODE detection-set membership is decided by a knee threshold on the
PRE-BONUS composite (w_E*E_data + w_sigma*max_instability) -- verified by
reading src/integrated_framework.py::run_ode_detection directly: the
KneeLocator call operates on `raw_energies = anomaly_score - n_viol*bonus`,
not on the full bonus-inclusive score. So the bonus term never moves
detection-set membership in the current pipeline; only w_E and w_sigma do.
This script varies exactly those two weights, holding the bonus mechanism
and thresholding rule (knee on the sorted pre-bonus composite, the rule used
for all three datasets) identical to production.

One re-integration per condition (no training -- cached hyperparameters
only), then the 3x3 (w_E, w_sigma) grid is pure post-hoc arithmetic on the
captured per-triple components -- no further integration per grid cell.

Conditions: WN18RR, seed 42 (canonical).
  - random_5pct:      KGPerturbator.perturb(5.0, ['entity_swap'])
  - logical_asym_10pct: KGPerturbator.perturb_logical('asymmetry', 10% nominal)

Writes progress/milestone_b_weight_sensitivity_WN18RR.json. No LaTeX, no
figures, no other dataset/condition touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from kneed import KneeLocator

from ablation_study import KGPerturbator
from data import DataLoader
from integrated_framework import IntegratedKGDebugger
from run_all import _load_hyperparam_cache, get_constraint_config, sample_triples

SAMPLE_SIZE = 10000

DATASET = "WN18RR"
SEED = 42
W_E_GRID = [2.5, 5.0, 10.0]
W_SIGMA_GRID = [50.0, 100.0, 200.0]
NOMINAL = (5.0, 100.0)


def build_condition_pool(condition: str):
    loader = DataLoader()
    base = loader.load_dataset(DATASET)
    pert = KGPerturbator(base, seed=SEED)
    if condition == "random_5pct":
        perturbed, changed_set = pert.perturb(percentage=5.0, perturbation_types=["entity_swap"])
        gt_set = changed_set
    elif condition == "logical_asym_10pct":
        n_viol = max(10, int(len(base) * 0.10))
        perturbed, violation_set, _ = pert.perturb_logical(
            violation_type="asymmetry", n_violations=n_viol, dataset_name=DATASET,
        )
        gt_set = violation_set
    else:
        raise ValueError(condition)
    return perturbed, gt_set


def knee_threshold(x: np.ndarray) -> tuple[float, int]:
    """Verbatim copy of integrated_framework.py's unimodal-branch knee logic
    (curve='convex', direction='decreasing', interp_method='polynomial'),
    kept local so that file is not touched."""
    n = len(x)
    sorted_e = np.sort(x)[::-1]
    xs = np.arange(n)
    knl = KneeLocator(xs, sorted_e, curve="convex", direction="decreasing", interp_method="polynomial")
    knee_idx = knl.knee
    if knee_idx is not None:
        return float(sorted_e[knee_idx]), int(knee_idx)
    return float(np.percentile(x, 95)), -1


def f1_from_mask(detected: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((detected & gt).sum())
    fp = int((detected & ~gt).sum())
    fn = int((~detected & gt).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def run_condition(condition: str) -> dict:
    t0 = time.time()
    perturbed, gt_set = build_condition_pool(condition)
    # CORRECTED (per the explicit protocol-confirmation step below): the
    # canonical pipeline (src/run_all.py::run_ablation_stage) never runs ODE
    # detection on the full perturbed pool -- it draws the same 10,000-triple
    # stratified sample used for KGE, then calls
    # run_complete_analysis(eval_triples, full_triples=perturbed_triples):
    # the ODE system (and hence the per-triple components below) is built
    # only from eval_triples; full_triples is passed through only for the
    # constraint checker's get_violating_triples() calls, which need the
    # whole graph to decide e.g. whether an inverse triple exists anywhere.
    eval_triples, eval_gt_set = sample_triples(perturbed, gt_set, SAMPLE_SIZE, seed=SEED)
    gt_keys = set(eval_gt_set)

    bp = _load_hyperparam_cache(DATASET, "ode")
    config = IntegratedKGDebugger.get_default_config()
    config["use_ode"] = True
    config["use_pykeen"] = False
    config["ode_config"] = {
        "embedding_dim": bp["embedding_dim"],
        "t_span": tuple(bp["t_span"]) if isinstance(bp["t_span"], list) else bp["t_span"],
        "lambda_data": bp["lambda_data"],
        "lambda_logic": bp["lambda_logic"],
        "lambda_asym": float(bp.get("lambda_asym", 1.0)),
        "margin_asym": float(bp.get("margin_asym", 1.0)),
        "lambda_reg": bp["lambda_reg"],
        "rtol": bp.get("rtol", 0.01),
        "atol": bp.get("atol", 1e-6),
        "device": bp.get("device", "cpu"),
        "top_k_ratio": 0.05,
    }
    config["constraint_config"] = get_constraint_config(DATASET)

    dbg = IntegratedKGDebugger(config)
    dbg.setup_ode_detector(eval_triples)

    from anomaly_detector import AnomalyDetector
    solution = dbg.ode_system.solve()
    dbg.ode_system.solution = solution
    detector = AnomalyDetector(
        dbg.ode_system, solution, dbg.constraint_checker,
        full_triples=perturbed, symbolic_bonus=config.get("symbolic_bonus", None),
    )

    # min_score=0.0 -> every triple in the pool gets a report (score is a sum
    # of non-negative terms, so >=0.0 is a no-op filter), with the raw
    # components now exposed by the anomaly_detector.py patch.
    reports = detector.detect_all_anomalies(top_k=None, min_score=0.0)
    integration_elapsed = time.time() - t0

    n = len(reports)
    gt = np.zeros(n, dtype=bool)
    energy = np.empty(n, dtype=float)
    max_instab = np.empty(n, dtype=float)
    n_viol = np.empty(n, dtype=int)
    for i, r in enumerate(reports):
        k = (r.triple.subject, r.triple.relation, r.triple.object)
        gt[i] = k in gt_keys
        energy[i] = r.energy
        max_instab[i] = r.max_instability
        n_viol[i] = r.n_violations

    bonus = float(detector.symbolic_bonus)

    grid_results = []
    nominal_detected = None
    for w_e in W_E_GRID:
        for w_s in W_SIGMA_GRID:
            x = w_e * energy + w_s * max_instab  # pre-bonus composite, matches production's thresholding input
            thr, knee_idx = knee_threshold(x)
            detected = x >= thr
            f1 = f1_from_mask(detected, gt)
            cell = {
                "w_E": w_e, "w_sigma": w_s,
                "n_detected": int(detected.sum()), "threshold": thr, "knee_idx": knee_idx,
                "f1": f1["f1"], "precision": f1["precision"], "recall": f1["recall"],
                "tp": f1["tp"], "fp": f1["fp"], "fn": f1["fn"],
                "is_nominal": (w_e, w_s) == NOMINAL,
            }
            grid_results.append(cell)
            if (w_e, w_s) == NOMINAL:
                nominal_detected = detected

    for cell in grid_results:
        w_e, w_s = cell["w_E"], cell["w_sigma"]
        x = w_e * energy + w_s * max_instab
        thr = cell["threshold"]
        detected = x >= thr
        inter = int((detected & nominal_detected).sum())
        union = int((detected | nominal_detected).sum())
        cell["jaccard_vs_nominal"] = inter / union if union else 1.0

    elapsed = time.time() - t0
    return {
        "condition": condition, "dataset": DATASET, "seed": SEED,
        "n_full_pool_triples": len(perturbed),
        "sample_size": n, "sample_positive_count": int(gt.sum()),
        "symbolic_bonus_calibrated": bonus,
        "integration_elapsed_s": integration_elapsed,
        "total_elapsed_s": elapsed,
        "grid": grid_results,
    }


def main():
    print("=" * 78)
    print(f"# Phase-2 Section B — Weight sensitivity | {DATASET} seed={SEED}")
    print(f"# w_E grid={W_E_GRID}  w_sigma grid={W_SIGMA_GRID}  nominal={NOMINAL}")
    print("=" * 78)

    out = {"conditions": {}}
    for condition in ["random_5pct", "logical_asym_10pct"]:
        print(f"\n--- condition: {condition} ---", flush=True)
        t0 = time.time()
        res = run_condition(condition)
        print(f"  done in {res['total_elapsed_s']:.1f}s "
              f"(integration {res['integration_elapsed_s']:.1f}s) | "
              f"sample_size={res['sample_size']} sample_pos={res['sample_positive_count']} "
              f"bonus={res['symbolic_bonus_calibrated']:.3f}", flush=True)
        for cell in res["grid"]:
            tag = " <- NOMINAL" if cell["is_nominal"] else ""
            print(f"    w_E={cell['w_E']:.1f} w_sigma={cell['w_sigma']:.0f} "
                  f"n_detected={cell['n_detected']:6d} F1={cell['f1']:.4f} "
                  f"Jaccard_vs_nominal={cell['jaccard_vs_nominal']:.4f} "
                  f"thr={cell['threshold']:.4f}{tag}")
        out["conditions"][condition] = res

    out_json = REPO_ROOT / "progress" / "milestone_b_weight_sensitivity_WN18RR_10k_corrected.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
