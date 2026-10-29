"""R2 shared evaluation helpers (round-2 protocol; see ../README.md). Import with the repository root as cwd.

Reused unchanged: compute_full_panel.metrics_for, statistical_tests.paired_wilcoxon / bootstrap_ci / holm_bonferroni.
Verbatim copies (the source modules execute their analysis at import time): pct / lexi from stage2_block_a.py and
region from r2_2_exclusive_recovery.py; `exclusive` applies region with the same top-K rule and partition invariants.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "reviewer_controls/scripts")

import numpy as np
from scipy import stats
from scipy.stats import hypergeom

from scipy.stats import rankdata

from compute_full_panel import metrics_for
from statistical_tests import bootstrap_ci, holm_bonferroni, paired_wilcoxon

N_BOOT_SEEDS, BOOT_SEED = 50_000, 12345


# pct and lexi: verbatim from reviewer_controls/scripts/stage2_block_a.py (that module parses sys.argv on import)
def pct(x):
    return rankdata(x, method="average") / (len(x) + 1.0)


def lexi(C, x):
    """Checker first; within equal C, the base score's percentile rank (scale-free)."""
    return C + pct(x)


def fusion(a, b):
    return (pct(a) + pct(b)) / 2


def metrics(gt, s):
    m = metrics_for(gt, s, int(gt.sum()))
    return {"ndcg_at_K": m["ndcg_at_K"], "auroc": m["auroc"], "prauc": m["prauc"]}


def region(name, idx, gt, N, P):  # verbatim from scripts/postprocess_review/r2_2_exclusive_recovery.py
    n = len(idx)
    tp = int(gt[list(idx)].sum()) if n else 0
    prev = P / N
    if n == 0:
        return {f"{name}_n": 0, f"{name}_tp": 0, f"{name}_negatives": 0, f"{name}_precision": "NA", f"{name}_enrichment": "NA",
                f"{name}_delta_precision": "NA", f"{name}_expected_tp_random": 0.0, f"{name}_p_enrichment": "NA", f"{name}_p_depletion": "NA"}
    prec = tp / n
    return {f"{name}_n": n, f"{name}_tp": tp, f"{name}_negatives": n - tp, f"{name}_precision": prec, f"{name}_enrichment": prec / prev,
            f"{name}_delta_precision": prec - prev, f"{name}_expected_tp_random": n * P / N,
            f"{name}_p_enrichment": float(hypergeom.sf(tp - 1, N, P, n)), f"{name}_p_depletion": float(hypergeom.cdf(tp, N, P, n))}


def exclusive(gt, a, b, a_name="a"):
    """Top-K (K = P) regions of score a versus score b, as in R2.2 (stable argsort on -score)."""
    N, P = len(gt), int(gt.sum())
    K = P
    A = set(np.argsort(-a, kind="stable")[:K].tolist())
    B = set(np.argsort(-b, kind="stable")[:K].tolist())
    a_only, shared, b_only = A - B, A & B, B - A
    assert len(a_only) + len(shared) == K == len(b_only) + len(shared)
    row = {"N": N, "P": P, "K": K, "prevalence": P / N}
    row.update(region(f"{a_name}_only", a_only, gt, N, P))
    row.update(region("shared", shared, gt, N, P))
    row.update(region("kge_only", b_only, gt, N, P))
    a_tp = row[f"{a_name}_only_tp"] + row["shared_tp"]
    k_tp = row["kge_only_tp"] + row["shared_tp"]
    u_tp = a_tp + row["kge_only_tp"]
    row.update({f"{a_name}_tp_total": a_tp, "kge_tp_total": k_tp,
                f"{a_name}_recall_at_K": a_tp / P, "kge_recall_at_K": k_tp / P,
                f"fraction_{a_name}_tp_exclusive": row[f"{a_name}_only_tp"] / a_tp if a_tp else "NA",
                "union_n": len(A | B), "union_tp": u_tp, "union_recall": u_tp / P})
    return row


def wilcoxon_method(d):
    """'exact' when scipy's default call returns the exact-distribution p for these differences, else 'approx'."""
    nz = d[d != 0]
    if len(nz) < 1:
        return "undefined"
    p_auto = stats.wilcoxon(d, zero_method="wilcox", correction=False).pvalue
    try:
        p_exact = stats.wilcoxon(nz, method="exact").pvalue
    except Exception:
        return "approx"
    return "exact" if np.isclose(p_auto, p_exact, rtol=0, atol=1e-15) else "approx"


def contrast(d):
    d = np.asarray(d, float)
    n = len(d)
    w = paired_wilcoxon(list(d), [0.0] * n)
    ci = bootstrap_ci(list(d), n_resamples=N_BOOT_SEEDS, rng_seed=BOOT_SEED)
    ok = w.get("status") == "ok"
    return {"n_available": n, "n_nonzero": int((d != 0).sum()), "mean": float(d.mean()), "median": float(np.median(d)),
            "sd": float(d.std(ddof=1)) if n > 1 else float("nan"), "min": float(d.min()), "max": float(d.max()),
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum()),
            "ci95_lo": ci["ci_lo"] if ci else float("nan"), "ci95_hi": ci["ci_hi"] if ci else float("nan"),
            "bootstrap": f"percentile, mean, {N_BOOT_SEEDS} resamples over realisations, seed {BOOT_SEED}",
            "wilcoxon_status": w.get("status"), "w_plus": w.get("w_plus", ""), "w_minus": w.get("w_minus", ""),
            "p_raw": w["p_value"] if ok else 1.0,
            "rank_biserial": w["rank_biserial"] if ok else "undefined (all differences zero)",
            "wilcoxon_method": wilcoxon_method(d) if ok else "n/a"}


def family(name, items):
    """items: list of (label, differences). Holm over the complete family."""
    rows = [{"family": name, "contrast": lab, **contrast(d)} for lab, d in items]
    for r, p in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
        r["p_holm"] = p
        r["family_size"] = len(rows)
    return rows
