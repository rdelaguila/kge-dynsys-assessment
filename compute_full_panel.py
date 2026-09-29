"""Full multimetric panel — A1, A2, A3 ODE configurations + 3 KGE.

Combines:
- A1 = ODE-det λ_asym = 0  (the "simplest" config; matches §4.6.1 point 2 headline)
- A2 = ODE-det λ_asym = ref  (canonical config; matches §5.9, §4.7)
- A3 = SDE per-triple variance  (configuration (3); matches §3.7)
- TransE, DistMult, MuRE plausibility (KGE plausibility, sign-flipped)

For each (dataset, vtype, n) cell, computes for each detector:
  - F1 at knee/Otsu (loaded from milestone metrics JSONs where available)
  - Hit@K, NDCG@K, BEDROC@α=20, EF@1%, AUROC, pAUC@5%, PR-AUC

Output: progress/20260623/full_panel.json + CSV-friendly tabular dump.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, ndcg_score, roc_curve, auc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DUMP_DIR = REPO_ROOT / "progress" / "20260623"

VTYPES = ("asymmetry", "transitivity", "cardinality")
DATASETS = ("FB15k-237", "codex-m", "WN18RR")
KGE_METHODS = ("TransE", "DistMult", "MuRE")
ODE_ARCHS = ("A1", "A2", "A3")
LEVELS_BY_DS = {
    "FB15k-237": {5: 15505, 10: 31011, 15: 46517},
    "WN18RR":    {5: 4650,  10: 9300,  15: 13950},
    "codex-m":   {5: 10310, 10: 20620, 15: 30930},
}
PARTIAL_FPR_ALPHA = 0.05


def bedroc(y_true, y_score, alpha=20.0):
    order = np.argsort(-y_score, kind="stable")
    y_sorted = np.asarray(y_true)[order].astype(int)
    N = len(y_sorted)
    n = int(y_sorted.sum())
    if n == 0 or n == N:
        return float("nan")
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    R_a = n / N
    s = float(np.sum(np.exp(-alpha * pos_ranks / N)))
    denom_cosh = np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * R_a)
    if abs(denom_cosh) < 1e-30:
        return float("nan")
    factor = (R_a * np.sinh(alpha / 2)) / denom_cosh
    return float((s / n) * factor + 1.0 / (1.0 - np.exp(alpha * (1 - R_a))))


def enrichment_factor(y_true, y_score, top_k):
    order = np.argsort(-y_score, kind="stable")
    top = order[:top_k]
    tp = int(np.asarray(y_true)[top].sum())
    precision_at_k = tp / max(top_k, 1)
    prev = float(np.asarray(y_true).sum()) / max(len(y_true), 1)
    if prev == 0:
        return float("nan")
    return precision_at_k / prev


def metrics_for(y_true, y_score, K):
    finite = np.isfinite(y_score)
    yt = y_true[finite].astype(bool)
    ys = y_score[finite]
    if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
        return {k: float("nan") for k in
                ("hit_at_K", "ndcg_at_K", "bedroc", "ef_at_1pct", "auroc", "pauc_norm", "prauc")}
    K_eff = min(K, len(yt))
    order = np.argsort(-ys, kind="stable")
    top_K = order[:K_eff]
    in_top = np.zeros(len(yt), dtype=bool)
    in_top[top_K] = True
    tp_K = int((in_top & yt).sum())
    hit_at_K = tp_K / max(int(yt.sum()), 1)
    rel = yt.astype(int)[None, :]; sc = ys[None, :]
    ndcg = float(ndcg_score(rel, sc, k=K_eff))
    try:
        fpr, tpr, _ = roc_curve(yt, ys)
        mask = fpr <= PARTIAL_FPR_ALPHA
        pauc_norm = float(auc(fpr[mask], tpr[mask])) / PARTIAL_FPR_ALPHA if mask.sum() >= 2 else float("nan")
    except Exception:
        pauc_norm = float("nan")
    bedroc20 = bedroc(yt, ys, alpha=20.0)
    K_one_pct = max(1, round(0.01 * len(yt)))
    ef_1pct = enrichment_factor(yt, ys, top_k=K_one_pct)
    try:
        auroc = float(roc_auc_score(yt, ys))
    except ValueError:
        auroc = float("nan")
    prauc = float(average_precision_score(yt, ys))
    return {"hit_at_K": float(hit_at_K), "ndcg_at_K": ndcg,
            "bedroc": bedroc20, "ef_at_1pct": float(ef_1pct),
            "auroc": auroc, "pauc_norm": pauc_norm, "prauc": prauc}


def load_ode_npz(arch, ds, vtype, n):
    p = DUMP_DIR / f"ode_postfix_{arch}_{ds}_{vtype}_{n}.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["eval_energies"]), np.asarray(z["gt"]).astype(bool)


def load_kge_from_audit(ds, vtype, n):
    """Load TransE/DistMult/MuRE per-triple scores from the persistent ablation_audit.
    Returns dict {KGE: (scores, gt)} or None if cell not captured."""
    p = REPO_ROOT / "paper_results" / ds / "ablation_audit" / f"{vtype}_{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    triples = d["triples"]
    y_true = np.array([t["is_violation"] for t in triples], dtype=bool)
    out = {}
    for kge in KGE_METHODS:
        out[kge] = (np.array([-t["kge_raw"][kge] for t in triples], dtype=float), y_true)
    return out


def load_f1_from_milestone(arch, ds, vtype, n):
    """Best-effort load of the post-fix F1 from the JSON the milestone wrote."""
    # The milestone with A2 writes to post_fix_rerun; A1 overwrites with same path.
    # We can't differentiate by arch in the metrics JSONs, so we only return the
    # currently persisted F1 for the last-run arch. Mark as N/A if absent.
    p = REPO_ROOT / "paper_results" / ds / "post_fix_rerun" / "metrics" / f"metrics_logical_{vtype}_{n}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "ode_postfix" in d:
        m = d["ode_postfix"]
        return {"f1": m.get("f1"), "precision": m.get("precision"),
                "recall": m.get("recall"), "tp": m.get("true_positives"),
                "n_detected": m.get("n_detected"),
                "threshold_knee": m.get("threshold_knee")}
    return None


def main():
    rows = []
    cells_covered = []
    for ds in DATASETS:
        for vt in VTYPES:
            for lvl, n in LEVELS_BY_DS[ds].items():
                cell_row = {"dataset": ds, "vtype": vt, "level_pct": lvl, "level_n": n,
                            "detectors": {}}
                # KGE per-triple from audit (only available for the 9 ablation_audit cells)
                kge_dict = load_kge_from_audit(ds, vt, n)
                # K = n_violations from y_true if available
                K = None
                gt_ref = None
                # ODE configs
                for arch in ODE_ARCHS:
                    energies, gt = load_ode_npz(arch, ds, vt, n)
                    if energies is None:
                        cell_row["detectors"][f"ODE-{arch}"] = {"status": "missing"}
                        continue
                    if gt_ref is None:
                        gt_ref = gt
                        K = int(gt.sum())
                    m = metrics_for(gt, energies, K)
                    cell_row["detectors"][f"ODE-{arch}"] = m
                if kge_dict is not None:
                    for kge, (scores, gt) in kge_dict.items():
                        if K is None:
                            K = int(gt.sum())
                        m = metrics_for(gt, scores, K)
                        cell_row["detectors"][kge] = m
                else:
                    for kge in KGE_METHODS:
                        cell_row["detectors"][kge] = {"status": "no_kge_audit_for_cell"}
                # n_pos / prevalence
                if gt_ref is not None:
                    cell_row["n_eval"] = int(len(gt_ref))
                    cell_row["n_pos"] = int(gt_ref.sum())
                    cell_row["prevalence"] = float(gt_ref.sum() / len(gt_ref))
                    cell_row["K"] = K
                else:
                    cell_row["n_eval"] = None
                    cell_row["prevalence"] = None
                rows.append(cell_row)
                cells_covered.append((ds, vt, lvl, n,
                                      any(arch in cell_row["detectors"]
                                          and cell_row["detectors"][arch].get("hit_at_K") is not None
                                          for arch in ("ODE-A1", "ODE-A2", "ODE-A3"))))

    out = {"protocol": {
              "ode_archs": list(ODE_ARCHS),
              "kge_methods": list(KGE_METHODS),
              "K_definition": "K = n_violations in eval set",
              "pauc_alpha": PARTIAL_FPR_ALPHA,
              "bedroc_alpha": 20.0,
              "ef_top_pct": 0.01,
          },
          "cells": rows}
    out_path = DUMP_DIR / "full_panel.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    # Markdown summary
    print(f"\n{'cell':<40} {'ODE-A1 NDCG':>10} {'ODE-A2 NDCG':>10} {'ODE-A3 NDCG':>10} {'MuRE NDCG':>10} | "
          f"{'A1 EF1%':>8} {'A2 EF1%':>8} {'A3 EF1%':>8} {'MuRE EF1%':>10}")
    for row in rows:
        d = row["detectors"]
        def gv(arch, key):
            v = d.get(arch, {}).get(key, float("nan"))
            return f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "  —"
        cell_label = f"{row['dataset']:>10s}/{row['vtype']:<13s}/n={row['level_n']:>5d}"
        line = (f"{cell_label:<40} "
                f"{gv('ODE-A1','ndcg_at_K'):>10} {gv('ODE-A2','ndcg_at_K'):>10} {gv('ODE-A3','ndcg_at_K'):>10} {gv('MuRE','ndcg_at_K'):>10} | "
                f"{gv('ODE-A1','ef_at_1pct'):>8} {gv('ODE-A2','ef_at_1pct'):>8} {gv('ODE-A3','ef_at_1pct'):>8} {gv('MuRE','ef_at_1pct'):>10}")
        print(line)


if __name__ == "__main__":
    main()
