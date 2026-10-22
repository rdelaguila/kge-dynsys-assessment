"""Block B analysis — paired same-checker comparisons on the asymmetry-10% samples (seeds 1-10, three datasets).

Run from the repository root:  python reviewer_controls/scripts/stage6_block_b_analysis.py reviewer_controls/frozen_inputs reviewer_controls/work

Reproduction check: S1/S2/S3 F1 recomputed from the stored residuals with the S0-S4 knee convention must equal
the persisted progress/r2_1_progressive_ablation/<ds>_seed_<s>.json values (reported, not assumed).
Configurations per seed (larger = more anomalous; KGE = -raw): C, degree, S1, S2, S3, TransE, TransE_d50, each
base also with the checker (lexicographic: C first, then the base percentile rank; additive q99 as secondary).
Pre-specified primary family (NDCG@K, two-sided exact Wilcoxon over seeds, Holm over 9 tests):
    B1 (S3+C) - (TransE+C)     B2 (S1+C) - (TransE+C)     B3 [(S3+C) - S3] - [(TransE+C) - TransE]
Secondary families (each Holm over 9): the same three contrasts on AUROC, on F1 (knee, additive rule),
on F1 (Otsu, additive rule), and with TransE_d50 in place of TransE on NDCG.
Descriptive: Spearman between S1 at 10/25/50/100 RK4 steps and TransE (10 epochs).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/experiments")
sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from compute_full_panel import metrics_for
from compute_parallel_operator_sweep import otsu_threshold
from run_s0_s4_progressive_ablation import f1_from_scores
from stage2_block_a import additive, lexi
from stage4_paired import contrast
from statistical_tests import holm_bonferroni

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DATASETS = ("FB15k-237", "codex-m", "WN18RR")


def f1_otsu(gt, s):
    det = s > otsu_threshold(s)
    tp = int((det & gt).sum()); fp = int((det & ~gt).sum()); fn = int((~det & gt).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-10)


def all_metrics(gt, s, tied=False, rng=None):
    P = int(gt.sum())
    if tied:
        acc = [metrics_for(gt, s + rng.random(len(s)) * 1e-9, P) for _ in range(200)]
        m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        det = s > 0
        tp = int((det & gt).sum()); fp = int((det & ~gt).sum()); fn = int((~det & gt).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        m["f1_knee"] = m["f1_otsu"] = 2 * p * r / max(p + r, 1e-10)
        return m
    m = metrics_for(gt, s, P)
    m["f1_knee"] = f1_from_scores(s, gt, P)["f1"]
    m["f1_otsu"] = f1_otsu(gt, s)
    return m


def main():
    rng = np.random.default_rng(0)
    rows, repro, sim = [], [], []
    per = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        for seed in range(1, 11):
            inp = np.load(OUT / "inputs" / f"{ds}__asym10_seed{seed}.npz")
            z = np.load(OUT / "block_b_raw" / f"{ds}__seed{seed}.npz")
            assert str(z["sha"]) == str(inp["sha"])
            gt, C, deg = inp["gt"].astype(bool), inp["C"].astype(float), inp["degree"].astype(float)
            s04 = json.loads((FROZEN / "progress/r2_1_progressive_ablation" / f"{ds}_seed_{seed}.json").read_text())
            key = {"S1": "S1_data_only", "S2": "S2_data_sym_trans", "S3": "S3_plus_antisymmetry"}
            for st, k in key.items():
                f1 = f1_from_scores(np.asarray(z[st], float), gt, int(gt.sum()))
                repro.append({"dataset": ds, "seed": seed, "stage": st, "f1_recomputed": f1["f1"],
                              "f1_persisted": s04["stages"][k]["f1"],
                              "ndcg_recomputed": f1["ndcg_at_k"], "ndcg_persisted": s04["stages"][k]["ndcg_at_k"],
                              "abs_diff_f1": abs(f1["f1"] - s04["stages"][k]["f1"])})
            base = {"S1": z["S1"], "S2": z["S2"], "S3": z["S3"], "TransE": -z["TransE_raw"],
                    "TransE_d50": -z["TransE_d50_raw"]}
            base = {k: np.asarray(v, float) for k, v in base.items()}
            assert all(np.isfinite(v).all() for v in base.values()), (ds, seed, "non-finite")
            cfg = {"C": all_metrics(gt, C, True, rng), "degree": all_metrics(gt, deg)}
            for k, s in base.items():
                cfg[k] = all_metrics(gt, s)
                cfg[f"{k}+C"] = all_metrics(gt, lexi(C, s))
                cfg[f"{k}+C_add_q99"] = all_metrics(gt, additive(C, s, 0.99)[0])
            for k, m in cfg.items():
                rows.append({"dataset": ds, "seed": seed, "config": k, "P": int(gt.sum()),
                             "C_coverage": float((C > 0).mean()),
                             **{x: m[x] for x in ("ndcg_at_K", "auroc", "prauc", "f1_knee", "f1_otsu")}})
            for step in (10, 25, 50):
                sim.append({"dataset": ds, "seed": seed, "S1_steps": step,
                            "spearman_vs_TransE": float(spearmanr(z[f"S1_step{step}"], base["TransE"]).statistic)})
            sim.append({"dataset": ds, "seed": seed, "S1_steps": 100,
                        "spearman_vs_TransE": float(spearmanr(base["S1"], base["TransE"]).statistic)})
            per[ds].append(cfg)
    fams = {}

    def add(fam, ds, label, d):
        fams.setdefault(fam, []).append({"family": fam, "dataset": ds, "contrast": label, **contrast(np.asarray(d))})

    for ds in DATASETS:
        P = per[ds]
        for metric, fam, kb, suffix in (("ndcg_at_K", "B-NDCG (primary)", "TransE", "+C"),
                                        ("auroc", "B-AUROC", "TransE", "+C"),
                                        ("f1_knee", "B-F1-knee (additive)", "TransE", "+C_add_q99"),
                                        ("f1_otsu", "B-F1-otsu (additive)", "TransE", "+C_add_q99"),
                                        ("ndcg_at_K", "B-NDCG d50", "TransE_d50", "+C")):
            v = lambda c, k: c[k][metric]
            add(fam, ds, f"(S3{suffix})-({kb}{suffix})", [v(c, "S3" + suffix) - v(c, kb + suffix) for c in P])
            add(fam, ds, f"(S1{suffix})-({kb}{suffix})", [v(c, "S1" + suffix) - v(c, kb + suffix) for c in P])
            add(fam, ds, f"checker gain S3 minus {kb}",
                [(v(c, "S3" + suffix) - v(c, "S3")) - (v(c, kb + suffix) - v(c, kb)) for c in P])
    out_rows = []
    for fam, lst in fams.items():
        for r, p in zip(lst, holm_bonferroni([r["p_raw"] for r in lst])):
            out_rows.append({**r, "p_holm": p})
    res = OUT / "block_b"
    res.mkdir(exist_ok=True)
    for name, data_ in (("per_seed_metrics", rows), ("paired_contrasts", out_rows),
                        ("reproduction_check_S1_S3", repro), ("S1_vs_TransE_similarity", sim)):
        with open(res / f"{name}.csv", "w", newline="") as f:
            keys = list(dict.fromkeys(k for r in data_ for k in r))
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(data_)
    print("wrote", res)


if __name__ == "__main__":
    main()
