"""Additional tests 1-3 and optional tests 4-5 (analysis only; inputs from stages 1, 5 and 7).

Run from the repository root:  python reviewer_controls/scripts/stage8_extra_analysis.py reviewer_controls/frozen_inputs reviewer_controls/work

Pre-specified families (two-sided exact Wilcoxon over the 10 perturbation seeds, Holm over the 3 datasets of
each family unless stated; bootstrap CI seed 12345). Larger score = more anomalous; KGE = -raw.
T1  replication of the complementarity result on asymmetry-10% (Block B samples; S3 = A2 configuration):
    T1a NDCG  Fusion(S3,TransE) - TransE
    T1b NDCG  Fusion(S3,TransE) - Fusion(permuted S3,TransE)   [mean of 200 within-sample permutations]
    T1c       S3-only enrichment - 1 at budget K = P            [K = P is an evaluation budget only]
    T1d AUROC S1 - TransE
    descriptive: the same with S1 in place of S3; KGE-only and shared enrichment; exclusive TP fractions
T2  random-5% (seeds 1-10): TransE versus the persisted A2 residuals under one common criterion
    T2a NDCG TransE - A2   T2b AUROC TransE - A2   T2c F1-Otsu TransE - A2   T2d NDCG (TransE+C) - (A2+C)
T3  training-seed variability: Block B primary contrasts recomputed with TransE trained with PyKEEN seeds 42, 43, 44
    (one Holm family of 9 per training seed); SD of TransE NDCG across training seeds vs across perturbation seeds
T4  degree adjustment (label-free OLS of each score on log1p(degree) within each sample):
    T4a NDCG S3adj - TransEadj   T4b NDCG S1adj - TransEadj   T4c AUROC S3adj - TransEadj   T4d AUROC S1adj - TransEadj
T5  checker scope: checker evaluated inside the 10,000-triple sample (as in the controlled KGE pipeline) versus on
    the complete perturbed graph (as in S4): coverage and checker-only metrics, and (S3+C)-(TransE+C) NDCG with each.
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

import data as _data
_data._cache_load = lambda path: None

from compute_full_panel import metrics_for
from data import Triple
from run_all import _load_hyperparam_cache
from run_s0_s4_progressive_ablation import build_constraint_checker, count_violations
from stage2_block_a import fast_auroc, fast_ndcg, lexi, pct
from stage4_paired import contrast
from stage6_block_b_analysis import f1_otsu
from statistical_tests import holm_bonferroni

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DATASETS = ("FB15k-237", "codex-m", "WN18RR")
N_PERM = 200
rng = np.random.default_rng(0)


def topk(s, K):
    return set(np.argsort(-s, kind="stable")[:K].tolist())


def exclusive(gt, a, b):
    N, P = len(gt), int(gt.sum())
    A, B = topk(a, P), topk(b, P)
    out = {}
    for name, R in (("a_only", A - B), ("shared", A & B), ("b_only", B - A)):
        tp = sum(gt[i] for i in R)
        out[f"{name}_n"] = len(R)
        out[f"{name}_tp"] = int(tp)
        out[f"{name}_enrichment"] = (tp / len(R)) / (P / N) if R else float("nan")
    ta = sum(gt[i] for i in A); tb = sum(gt[i] for i in B)
    out["a_tp_exclusive_frac"] = out["a_only_tp"] / ta if ta else float("nan")
    out["b_tp_exclusive_frac"] = out["b_only_tp"] / tb if tb else float("nan")
    return out


def adjust_degree(s, deg):
    x = np.log1p(deg)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    return s - X @ beta


def family(rows, name, items):
    """items: list of (dataset, label, diff array)."""
    lst = [{"family": name, "dataset": ds, "contrast": lab, **contrast(np.asarray(d))} for ds, lab, d in items]
    for r, p in zip(lst, holm_bonferroni([r["p_raw"] for r in lst])):
        r["p_holm"] = p
        rows.append(r)


def main():
    fam_rows, desc_rows, t3_rows, t5_rows, wrapper = [], [], [], [], []
    T1, T2, T3, T4 = {k: [] for k in "abcd"}, {k: [] for k in "abcd"}, {}, {k: [] for k in "abcd"}
    T5 = []
    for ds in DATASETS:
        margin_asym = float(_load_hyperparam_cache(ds, "ode").get("margin_asym", 1.0))
        acc = {k: [] for k in ("t1a", "t1b", "t1c", "t1d", "t2a", "t2b", "t2c", "t2d", "t4a", "t4b", "t4c", "t4d")}
        t3 = {42: [], 43: [], 44: []}
        tr_ndcg = []
        t5d = {"full": [], "in": []}
        for s in range(1, 11):
            # ---------- asymmetry-10% (Block B samples) ----------
            inp = np.load(OUT / "inputs" / f"{ds}__asym10_seed{s}.npz")
            z = np.load(OUT / "block_b_raw" / f"{ds}__seed{s}.npz")
            ex = np.load(OUT / "extra_raw" / f"trainseed__{ds}__seed{s}.npz")
            assert str(z["sha"]) == str(inp["sha"]) == str(ex["sha"])
            gt, C, deg = inp["gt"].astype(bool), inp["C"].astype(float), inp["degree"].astype(float)
            S1, S3 = np.asarray(z["S1"], float), np.asarray(z["S3"], float)
            T = {42: -np.asarray(z["TransE_raw"], float), 43: -np.asarray(ex["TransE_seed43_raw"], float),
                 44: -np.asarray(ex["TransE_seed44_raw"], float)}
            TE = T[42]
            fus = (pct(S3) + pct(TE)) / 2
            noise = [fast_ndcg(gt, (pct(rng.permutation(S3)) + pct(TE)) / 2) for _ in range(N_PERM)]
            exS3, exS1 = exclusive(gt, S3, TE), exclusive(gt, S1, TE)
            acc["t1a"].append(fast_ndcg(gt, fus) - fast_ndcg(gt, TE))
            acc["t1b"].append(fast_ndcg(gt, fus) - float(np.mean(noise)))
            acc["t1c"].append(exS3["a_only_enrichment"] - 1 if exS3["a_only_n"] else 0.0)
            acc["t1d"].append(fast_auroc(gt, S1) - fast_auroc(gt, TE))
            fus1 = (pct(S1) + pct(TE)) / 2
            desc_rows.append({"test": "T1", "dataset": ds, "seed": s, "ndcg_TransE": fast_ndcg(gt, TE),
                              "ndcg_S3": fast_ndcg(gt, S3), "ndcg_S1": fast_ndcg(gt, S1),
                              "ndcg_fusion_S3": fast_ndcg(gt, fus), "ndcg_fusion_S1": fast_ndcg(gt, fus1),
                              "ndcg_fusion_noise_mean": float(np.mean(noise)),
                              "ndcg_fusion_noise_q975": float(np.quantile(noise, .975)),
                              "auroc_S1": fast_auroc(gt, S1), "auroc_S3": fast_auroc(gt, S3), "auroc_TransE": fast_auroc(gt, TE),
                              **{f"S3vsTransE_{k}": v for k, v in exS3.items()},
                              **{f"S1vsTransE_{k}": v for k, v in exS1.items()}})
            # ---------- T3 training seeds ----------
            for seed, te in T.items():
                t3[seed].append({"b1": fast_ndcg(gt, lexi(C, S3)) - fast_ndcg(gt, lexi(C, te)),
                                 "b2": fast_ndcg(gt, lexi(C, S1)) - fast_ndcg(gt, lexi(C, te)),
                                 "b3": (fast_ndcg(gt, lexi(C, S3)) - fast_ndcg(gt, S3)) - (fast_ndcg(gt, lexi(C, te)) - fast_ndcg(gt, te)),
                                 "ndcg_te": fast_ndcg(gt, te)})
            tr_ndcg.append([fast_ndcg(gt, T[k]) for k in (42, 43, 44)])
            t3_rows.append({"dataset": ds, "seed": s, **{f"spearman_TransE42_vs_{k}": float(spearmanr(T[42], T[k]).statistic) for k in (43, 44)},
                            **{f"ndcg_TransE_seed{k}": fast_ndcg(gt, T[k]) for k in (42, 43, 44)}})
            # ---------- T4 degree adjustment ----------
            S3a, S1a, TEa = adjust_degree(S3, deg), adjust_degree(S1, deg), adjust_degree(TE, deg)
            acc["t4a"].append(fast_ndcg(gt, S3a) - fast_ndcg(gt, TEa))
            acc["t4b"].append(fast_ndcg(gt, S1a) - fast_ndcg(gt, TEa))
            acc["t4c"].append(fast_auroc(gt, S3a) - fast_auroc(gt, TEa))
            acc["t4d"].append(fast_auroc(gt, S1a) - fast_auroc(gt, TEa))
            # ---------- T5 checker scope ----------
            sample = [Triple(*t) for t in inp["triples"]]
            Cin = count_violations(build_constraint_checker(ds, margin_asym), sample, sample).astype(float)
            for scope, CC in (("full", C), ("in", Cin)):
                m = metrics_for(gt, CC + rng.random(len(CC)) * 1e-9, int(gt.sum()))
                det = CC > 0
                tp = int((det & gt).sum()); fp = int((det & ~gt).sum()); fn = int((~det & gt).sum())
                pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                t5d[scope].append(fast_ndcg(gt, lexi(CC, S3)) - fast_ndcg(gt, lexi(CC, TE)))
                t5_rows.append({"dataset": ds, "seed": s, "scope": scope, "coverage": float(det.mean()),
                                "checker_f1": 2 * pr * rc / max(pr + rc, 1e-10), "checker_ndcg": m["ndcg_at_K"],
                                "checker_auroc": m["auroc"], "S3C_minus_TransEC_ndcg": t5d[scope][-1]})
            # ---------- T2 random-5% ----------
            inp = np.load(OUT / "inputs" / f"{ds}__random5_seed{s}.npz")
            rx = np.load(OUT / "extra_raw" / f"random5__{ds}__seed{s}.npz")
            assert str(rx["sha"]) == str(inp["sha"])
            gt, C = inp["gt"].astype(bool), inp["C"].astype(float)
            import gzip
            rec = json.load(gzip.open(FROZEN / "canonical_persist/detections" / ds / "postfix" /
                                      f"random_entity_swap_5.0_seed{s}__A2.json.gz", "rt"))
            assert rec["sample_triples_sha256"] == str(inp["sha"])
            A2 = np.array([rec["scores"]["\t".join(t)] for t in inp["triples"]], float)
            TE = -np.asarray(rx["TransE_seed42_raw"], float)
            acc["t2a"].append(fast_ndcg(gt, TE) - fast_ndcg(gt, A2))
            acc["t2b"].append(fast_auroc(gt, TE) - fast_auroc(gt, A2))
            acc["t2c"].append(f1_otsu(gt, TE) - f1_otsu(gt, A2))
            acc["t2d"].append(fast_ndcg(gt, lexi(C, TE)) - fast_ndcg(gt, lexi(C, A2)))
            desc_rows.append({"test": "T2", "dataset": ds, "seed": s, "ndcg_TransE": fast_ndcg(gt, TE),
                              "ndcg_A2": fast_ndcg(gt, A2), "auroc_TransE": fast_auroc(gt, TE), "auroc_A2": fast_auroc(gt, A2),
                              "f1otsu_TransE": f1_otsu(gt, TE), "f1otsu_A2": f1_otsu(gt, A2),
                              "ndcg_random": float(gt.mean())})
        for k in "abcd":
            T1[k].append((ds, {"a": "NDCG Fusion(S3,TransE)-TransE", "b": "NDCG Fusion(S3,TransE)-Fusion(noise,TransE)",
                               "c": "S3-only enrichment - 1 (K=P)", "d": "AUROC S1-TransE"}[k], acc[f"t1{k}"]))
            T2[k].append((ds, {"a": "NDCG TransE-A2", "b": "AUROC TransE-A2", "c": "F1-Otsu TransE-A2",
                               "d": "NDCG (TransE+C)-(A2+C)"}[k], acc[f"t2{k}"]))
            T4[k].append((ds, {"a": "NDCG S3adj-TransEadj", "b": "NDCG S1adj-TransEadj",
                               "c": "AUROC S3adj-TransEadj", "d": "AUROC S1adj-TransEadj"}[k], acc[f"t4{k}"]))
        for seed in (42, 43, 44):
            T3.setdefault(seed, []).extend([(ds, f"(S3+C)-(TransE+C) NDCG", [r["b1"] for r in t3[seed]]),
                                            (ds, f"(S1+C)-(TransE+C) NDCG", [r["b2"] for r in t3[seed]]),
                                            (ds, f"checker gain S3 minus TransE NDCG", [r["b3"] for r in t3[seed]])])
        tr = np.array(tr_ndcg)
        desc_rows.append({"test": "T3-variance", "dataset": ds,
                          "sd_across_training_seeds_mean": float(tr.std(axis=1, ddof=1).mean()),
                          "sd_across_perturbation_seeds": float(tr[:, 0].std(ddof=1))})
        T5.append((ds, "NDCG (S3+C)-(TransE+C), in-sample checker", t5d["in"]))
    for k in "abcd":
        family(fam_rows, f"T1{k}", T1[k]); family(fam_rows, f"T2{k}", T2[k]); family(fam_rows, f"T4{k}", T4[k])
    for seed, items in T3.items():
        family(fam_rows, f"T3 seed{seed}", items)
    family(fam_rows, "T5", T5)
    res = OUT / "extra_tests"
    res.mkdir(exist_ok=True)
    for name, data_ in (("family_contrasts", fam_rows), ("descriptive", desc_rows), ("T3_training_seeds", t3_rows),
                        ("T5_checker_scope", t5_rows)):
        with open(res / f"{name}.csv", "w", newline="") as f:
            keys = list(dict.fromkeys(k for r in data_ for k in r))
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(data_)
    print("wrote", res)


if __name__ == "__main__":
    main()
