"""Block A-bis — checker-only and A2+checker on the paired perturbation seeds (no training, no integration).

Run from the repository root:  python reviewer_controls/scripts/stage4_paired.py reviewer_controls/frozen_inputs reviewer_controls/work

random-5%, seeds 1-10 (A2 residuals persisted for all 10,000 positions):
    configurations C, degree, A2, A2+C (lexicographic), A2+C (additive, q99)
asymmetry-10%, seeds 1-10 (only per-seed S3/S4 summaries persisted):
    configurations C, degree, compared with the persisted S3 and S4 F1 / NDCG@K of the same samples
V2: the number of positions with C>0 must equal the persisted S4 n_eval_with_violation for every seed.
Pre-specified families (two-sided exact Wilcoxon, Holm within family, bootstrap CI seed 12345):
    R-NDCG: per dataset (A2+C)-A2 and C-A2 on NDCG@K           (6 tests)
    R-F1  : per dataset (A2+C)-A2 (knee) and C(>0)-A2 (knee)    (6 tests)
    S-NDCG: per dataset C-S3 and C-S4 on NDCG@K                 (6 tests)
    S-F1  : per dataset C(>0)-S3 and C(>0)-S4 on F1             (6 tests)
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/experiments")
sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from compute_full_panel import metrics_for
from run_s0_s4_progressive_ablation import f1_from_scores
from statistical_tests import bootstrap_ci, holm_bonferroni, paired_wilcoxon
from stage2_block_a import additive, lexi

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DATASETS = ("FB15k-237", "codex-m", "WN18RR")
BOOT_SEED, N_TIE = 12345, 200


def tied_metrics(gt, s, rng):
    P = int(gt.sum())
    acc = [metrics_for(gt, s + rng.random(len(s)) * 1e-9, P) for _ in range(N_TIE)]
    m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
    det = s > 0
    tp = int((det & gt).sum()); fp = int((det & ~gt).sum()); fn = int((~det & gt).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    m["f1"] = 2 * p * r / max(p + r, 1e-10)
    m["n_det"] = int(det.sum())
    return m


def scored_metrics(gt, s):
    m = metrics_for(gt, s, int(gt.sum()))
    k = f1_from_scores(s, gt, int(gt.sum()))
    m["f1"], m["n_det"] = k["f1"], k["n_detected"]
    return m


def contrast(diff):
    w = paired_wilcoxon(list(diff), [0.0] * len(diff))
    ci = bootstrap_ci(list(diff), rng_seed=BOOT_SEED)
    return {"mean": float(np.mean(diff)), "sd": float(np.std(diff, ddof=1)), "ci95_lo": ci["ci_lo"],
            "ci95_hi": ci["ci_hi"], "rank_biserial": w.get("rank_biserial", 0.0), "p_raw": w.get("p_value", 1.0),
            "n_pos": w.get("n_positive"), "n_neg": w.get("n_negative"), "n_zero": w.get("n_zero")}


def main():
    rng = np.random.default_rng(0)
    rows, families, v2 = [], {"R-NDCG": [], "R-F1": [], "S-NDCG": [], "S-F1": []}, []
    for ds in DATASETS:
        per = {}
        for s in range(1, 11):
            # random-5%
            inp = np.load(OUT / "inputs" / f"{ds}__random5_seed{s}.npz")
            T = ["\t".join(t) for t in inp["triples"]]
            gt, C, deg = inp["gt"].astype(bool), inp["C"].astype(float), inp["degree"].astype(float)
            rec = json.load(gzip.open(FROZEN / "canonical_persist/detections" / ds / "postfix" /
                                      f"random_entity_swap_5.0_seed{s}__A2.json.gz", "rt"))
            uniq_pos = len({t for t, g in zip(T, gt) if g})
            assert rec["sample_triples_sha256"] == str(inp["sha"]) and rec["n_pos"] == uniq_pos, (ds, s)
            a2 = np.array([rec["scores"][t] for t in T], float)
            cfg = {"C": tied_metrics(gt, C, rng), "degree": scored_metrics(gt, deg), "A2": scored_metrics(gt, a2),
                   "A2+C": scored_metrics(gt, lexi(C, a2)), "A2+C_add_q99": scored_metrics(gt, additive(C, a2, .99)[0])}
            for k, m in cfg.items():
                rows.append({"dataset": ds, "condition": "random5", "seed": s, "config": k, "P": int(gt.sum()),
                             "C_coverage": float((C > 0).mean()), **{x: m[x] for x in ("ndcg_at_K", "auroc", "prauc", "f1", "n_det")}})
            per.setdefault("random5", []).append(cfg)
            # asymmetry-10%
            inp = np.load(OUT / "inputs" / f"{ds}__asym10_seed{s}.npz")
            gt, C, deg = inp["gt"].astype(bool), inp["C"].astype(float), inp["degree"].astype(float)
            s04 = json.loads((FROZEN / "progress/r2_1_progressive_ablation" / f"{ds}_seed_{s}.json").read_text())
            sha16 = hashlib.sha256("|".join(f"{a}:{b}:{c}" for a, b, c in inp["triples"]).encode()).hexdigest()[:16]
            assert sha16 == s04["sample_sha256_16"] and s04["n_eval_positive"] == int(gt.sum()), (ds, s, "S0-S4 sample")
            v2.append({"dataset": ds, "seed": s, "C_gt0_recomputed": int((C > 0).sum()),
                       "S4_n_eval_with_violation": s04["stages"]["S4_plus_checker_bonus"]["n_eval_with_violation"]})
            cfg = {"C": tied_metrics(gt, C, rng), "degree": scored_metrics(gt, deg)}
            for st, key in (("S3", "S3_plus_antisymmetry"), ("S4", "S4_plus_checker_bonus")):
                cfg[st] = {"ndcg_at_K": s04["stages"][key]["ndcg_at_k"], "f1": s04["stages"][key]["f1"],
                           "n_det": s04["stages"][key]["n_detected"], "auroc": float("nan"), "prauc": float("nan")}
            for k, m in cfg.items():
                rows.append({"dataset": ds, "condition": "asym10", "seed": s, "config": k, "P": int(gt.sum()),
                             "C_coverage": float((C > 0).mean()), **{x: m[x] for x in ("ndcg_at_K", "auroc", "prauc", "f1", "n_det")}})
            per.setdefault("asym10", []).append(cfg)
        R, S_ = per["random5"], per["asym10"]
        for metric, fam in (("ndcg_at_K", "R-NDCG"), ("f1", "R-F1")):
            for a, b in (("A2+C", "A2"), ("C", "A2")):
                d = np.array([c[a][metric] - c[b][metric] for c in R])
                families[fam].append({"dataset": ds, "contrast": f"{a} - {b}", "metric": metric, **contrast(d)})
        for metric, fam in (("ndcg_at_K", "S-NDCG"), ("f1", "S-F1")):
            for a, b in (("C", "S3"), ("C", "S4")):
                d = np.array([c[a][metric] - c[b][metric] for c in S_])
                families[fam].append({"dataset": ds, "contrast": f"{a} - {b}", "metric": metric, **contrast(d)})
    out_rows = []
    for fam, lst in families.items():
        adj = holm_bonferroni([r["p_raw"] for r in lst])
        for r, p in zip(lst, adj):
            out_rows.append({"family": fam, "p_holm": p, **r})
    res = OUT / "block_a_bis"
    res.mkdir(exist_ok=True)
    for name, data_ in (("per_seed_metrics", rows), ("paired_contrasts", out_rows), ("V2_checker_reproduction", v2)):
        with open(res / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data_[0].keys()))
            w.writeheader(); w.writerows(data_)
    print("wrote", res)


if __name__ == "__main__":
    main()
