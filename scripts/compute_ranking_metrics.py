"""Compute ranking-aware metrics that respect the ODE's unimodal heavy-tail
distribution: Hit@K, NDCG@K, partial AUC (FPR <= alpha), BEDROC@alpha=20,
Enrichment Factor @ K=1% pool.

Read-only over the persisted artefacts of Tier-1.5. No model invoked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score, roc_curve, auc, roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC (Truchon & Bayly 2007), canonical formula.

    Concentrates ~80% of weight on the first 8% of the ranking when alpha=20.
    Returns a value in [0, 1].
    """
    order = np.argsort(-y_score, kind="stable")
    y_sorted = np.asarray(y_true)[order].astype(int)
    N = len(y_sorted)
    n = int(y_sorted.sum())
    if n == 0 or n == N:
        return float("nan")
    pos_ranks = np.where(y_sorted == 1)[0] + 1  # 1-indexed
    R_a = n / N
    s = float(np.sum(np.exp(-alpha * pos_ranks / N)))
    # Truchon-Bayly normalisation:
    denom_cosh = np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * R_a)
    if denom_cosh == 0:
        return float("nan")
    factor = (R_a * np.sinh(alpha / 2)) / denom_cosh
    bedroc_val = (s / n) * factor + 1.0 / (1.0 - np.exp(alpha * (1 - R_a)))
    return float(bedroc_val)


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, top_k: int) -> float:
    """EF@K = Precision@K / prevalence. Random detector gives EF=1."""
    order = np.argsort(-y_score, kind="stable")
    top = order[:top_k]
    tp = int(np.asarray(y_true)[top].sum())
    precision_at_k = tp / max(top_k, 1)
    prevalence = float(np.asarray(y_true).sum()) / max(len(y_true), 1)
    if prevalence == 0:
        return float("nan")
    return precision_at_k / prevalence

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
PARTIAL_FPR_ALPHA = 0.05  # operating region: FPR <= 5%


def load_cell(ds, vtype, n):
    return json.loads(
        (REPO_ROOT / "paper_results" / ds / "ablation_audit" / f"{vtype}_{n}.json").read_text()
    )


def load_postfix_ode(ds, vtype, n):
    p = REPO_ROOT / "progress" / "20260623" / f"ode_postfix_{ds}_{vtype}_{n}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["eval_energies"], z["gt"]


def metrics_for_score(y_true: np.ndarray, y_score: np.ndarray, K: int) -> dict:
    finite = np.isfinite(y_score)
    yt = y_true[finite]
    ys = y_score[finite]
    n = len(yt)
    n_pos = int(yt.sum())
    # Top-K by score (descending)
    order = np.argsort(-ys, kind="stable")
    top_K = order[:K]
    in_top = np.zeros(n, dtype=bool)
    in_top[top_K] = True
    tp_K = int((in_top & yt).sum())
    hit_at_K = tp_K / max(n_pos, 1)         # = recall@K = precision@K when K=n_pos
    # NDCG@K — sklearn expects relevance scores in batch form
    rel = yt.astype(int)[None, :]
    sc = ys[None, :]
    ndcg = float(ndcg_score(rel, sc, k=K))
    # Partial AUC (FPR <= alpha) via trapezoidal area on the ROC curve restricted
    fpr, tpr, _ = roc_curve(yt, ys)
    mask = fpr <= PARTIAL_FPR_ALPHA
    if mask.sum() >= 2:
        pauc_raw = float(auc(fpr[mask], tpr[mask]))
        pauc_norm = pauc_raw / PARTIAL_FPR_ALPHA
    else:
        pauc_norm = float("nan")
    # BEDROC@α=20 — early-retrieval enriched (Truchon & Bayly)
    bedroc20 = bedroc(yt, ys, alpha=20.0)
    # EF@1% pool — Enrichment Factor at top-1%
    K_one_pct = max(1, round(0.01 * n))
    ef_1pct = enrichment_factor(yt, ys, top_k=K_one_pct)
    # AUROC and PR-AUC (for completeness; PR-AUC kept for appendix)
    try:
        auroc = float(roc_auc_score(yt, ys))
    except ValueError:
        auroc = float("nan")
    prauc = float(average_precision_score(yt, ys))
    return {
        "K": K, "n_pos": n_pos,
        "hit_at_K": hit_at_K,
        "ndcg_at_K": ndcg,
        "pauc_norm": pauc_norm,
        "bedroc_alpha20": bedroc20,
        "ef_at_1pct": ef_1pct,
        "auroc": auroc,
        "prauc": prauc,
    }


def main():
    rows = []
    for ds, vtype, n in CELLS:
        try:
            cell = load_cell(ds, vtype, n)
        except FileNotFoundError:
            print(f"  [skip] {ds}/{vtype}/{n}")
            continue

        triples = cell["triples"]
        y_true = np.array([t["is_violation"] for t in triples], dtype=bool)
        n_pos = int(y_true.sum())
        K = n_pos  # K = n_violations: "if you flag as many as there really are, how many do you hit"

        ode_postfix = load_postfix_ode(ds, vtype, n)
        if ode_postfix is None:
            ode_score = None
        else:
            ode_energies, ode_gt = ode_postfix
            assert np.array_equal(ode_gt.astype(bool), y_true)
            ode_score = ode_energies

        methods_results = {}
        if ode_score is not None:
            methods_results["ODE"] = metrics_for_score(y_true, ode_score, K)
        for kge in KGE_METHODS:
            kge_score = np.array([-t["kge_raw"][kge] for t in triples], dtype=float)
            methods_results[kge] = metrics_for_score(y_true, kge_score, K)

        row = {
            "dataset": ds, "violation": vtype, "level_n": n,
            "n_eval": len(triples), "n_pos": n_pos, "K": K,
            "methods": methods_results,
        }
        rows.append(row)

        # one-line summary: ODE vs MuRE on the 6 headline metrics
        def fmt(m, key):
            v = methods_results.get(m, {}).get(key, float("nan"))
            return f"{v:.3f}"
        print(f"  {ds:12s}/{vtype:14s}/n={n:>5d} prev={n_pos/len(triples):.3f}")
        print(f"    ODE   F1=(see milestone)  Hit@K={fmt('ODE','hit_at_K')}  NDCG@K={fmt('ODE','ndcg_at_K')}  BEDROC={fmt('ODE','bedroc_alpha20')}  EF@1%={fmt('ODE','ef_at_1pct')}  AUROC={fmt('ODE','auroc')}  pAUC@5%={fmt('ODE','pauc_norm')}  PR-AUC={fmt('ODE','prauc')}")
        print(f"    MuRE                       Hit@K={fmt('MuRE','hit_at_K')}  NDCG@K={fmt('MuRE','ndcg_at_K')}  BEDROC={fmt('MuRE','bedroc_alpha20')}  EF@1%={fmt('MuRE','ef_at_1pct')}  AUROC={fmt('MuRE','auroc')}  pAUC@5%={fmt('MuRE','pauc_norm')}  PR-AUC={fmt('MuRE','prauc')}")

    out = REPO_ROOT / "progress" / "20260623" / "ranking_metrics_postfix.json"
    out.write_text(json.dumps({"cells": rows, "alpha_pauc": PARTIAL_FPR_ALPHA}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
