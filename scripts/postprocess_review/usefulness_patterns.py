"""Descriptive pattern analyses on the moderator matrices (read-only; no model, no regression).

Writes CROSS_DATASET_PATTERN.csv, PERTURBATION_LEVEL_PATTERN.csv, SCOPE_B_CONTINGENCY.csv, HYPOTHESIS_COUNTS.json.
Usage: python usefulness_patterns.py <usefulness_moderators_dir>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

O = Path(sys.argv[1])
A = pd.read_csv(O / "USEFULNESS_MODERATOR_MATRIX.csv")
B = pd.read_csv(O / "USEFULNESS_MODERATOR_PAIRWISE.csv")
for c in ("a2_only_enrichment", "kge_only_enrichment", "shared_enrichment", "a2_only_p_enrichment"):
    B[c] = pd.to_numeric(B[c], errors="coerce")
out = {}

# 5.1 / 5.2 force alignment (Scope A)
def summ(df):
    return {k: {"median": float(df[k].median()), "min": float(df[k].min()), "max": float(df[k].max())} for k in ("a2_f1_lift", "a2_ndcg_lift", "a2_auroc")} | {"n": int(len(df))}
al, nal = A[A.force_aligned], A[~A.force_aligned]
out["5.1_aligned"] = summ(al) | {"aligned_f1_lift_le_1": int((al.a2_f1_lift <= 1).sum()), "aligned_ndcg_lift_le_1": int((al.a2_ndcg_lift <= 1).sum()),
                                 "aligned_auroc_le_0.5": int((al.a2_auroc <= 0.5).sum()),
                                 "aligned_counterexamples": al[(al.a2_f1_lift <= 1) | (al.a2_ndcg_lift <= 1) | (al.a2_auroc <= 0.5)][["dataset", "generator", "level_pct"]].values.tolist()}
out["5.1_not_aligned"] = summ(nal)
nn = nal[(nal.a2_f1_lift > 1) | (nal.a2_ndcg_lift > 1) | (nal.a2_auroc > 0.5)]
out["5.2_not_aligned_but_above"] = nn[["dataset", "generator", "level_pct", "a2_f1_lift", "a2_ndcg_lift", "a2_auroc"]].values.tolist()
out["5.2_not_aligned_all_three_above"] = nal[(nal.a2_f1_lift > 1) & (nal.a2_ndcg_lift > 1) & (nal.a2_auroc > 0.5)][["dataset", "generator", "level_pct"]].values.tolist()
# 5.3 coverage among aligned
cov = al.force_coverage_positives
out["5.3_coverage_values"] = sorted(set(round(float(v), 4) for v in cov))
for k in ("a2_f1_lift", "a2_ndcg_lift", "a2_auroc"):
    rho, p = spearmanr(cov, al[k])
    out[f"5.3_spearman_coverage_vs_{k}"] = {"rho": float(rho) if rho == rho else None, "n": int(len(al)), "label": "EXPLORATORY, n small, 17 of 18 ties"}
# 5.4 relation concentration (Scope A)
out["5.4_concentration"] = A.groupby(["dataset", "generator"]).agg(n_pos_rel=("n_positive_relations", "median"), max_share=("max_positive_relation_share", "median"),
                                                                    H_norm=("positive_relation_entropy_normalized", "median"), f1_lift=("a2_f1_lift", "median")).round(3).reset_index().values.tolist()

# 6 Scope B contingency
B["kge_weak_ndcg"] = B.kge_ndcg_lift <= 1
B["kge_weak_auroc"] = B.kge_auroc <= 0.5
B["a2_only_above_prev"] = B.a2_only_enrichment > 1
B["a2_only_strong"] = (B.a2_only_enrichment > 1) & (B.a2_only_p_enrichment < 0.025)
rows = []
for weak in ("kge_weak_ndcg", "kge_weak_auroc"):
    for outcome in ("a2_only_above_prev", "a2_only_strong"):
        for fa in (True, False):
            for w in (True, False):
                sub = B[(B.force_aligned == fa) & (B[weak] == w)]
                rows.append({"kge_weakness_indicator": weak, "a2_outcome": outcome, "force_aligned": fa, "kge_weak": w, "n_comparisons": len(sub),
                             "n_outcome_true": int(sub[outcome].sum()),
                             "cells": "; ".join(f"{r.dataset}/{r.generator}{r.level_pct}/{r.kge}" for r in sub[sub[outcome]].itertuples())})
pd.DataFrame(rows).to_csv(O / "SCOPE_B_CONTINGENCY.csv", index=False)

# 8 hypotheses counts (Scope A for H1/H4, Scope B for H2/H3/H5)
useful_A = (A.a2_f1_lift > 1) & (A.a2_ndcg_lift > 1) & (A.a2_auroc > 0.5)
out["H1_aligned_useful_all_three"] = A[A.force_aligned & useful_A][["dataset", "generator", "level_pct"]].values.tolist()
out["H1_aligned_not_useful"] = int((A.force_aligned & ~useful_A).sum())
out["H2_kge_weak_ndcg"] = {"n": int(B.kge_weak_ndcg.sum()), "a2_only_strong": int((B.kge_weak_ndcg & B.a2_only_strong).sum()),
                           "a2_only_above": int((B.kge_weak_ndcg & B.a2_only_above_prev).sum()),
                           "cells_weak": [f"{r.dataset}/{r.generator}{r.level_pct}/{r.kge}" for r in B[B.kge_weak_ndcg].itertuples()]}
conj = B.force_aligned & B.kge_weak_ndcg
out["H3_conjunction"] = {"n": int(conj.sum()), "a2_only_strong": int((conj & B.a2_only_strong).sum()), "a2_only_above": int((conj & B.a2_only_above_prev).sum()),
                         "cells": [f"{r.dataset}/{r.generator}{r.level_pct}/{r.kge} enr={r.a2_only_enrichment:.3f}" for r in B[conj].itertuples()]}
out["H3_strong_cases"] = [f"{r.dataset}/{r.generator}{r.level_pct}/{r.kge} kge_ndcg_lift={r.kge_ndcg_lift:.3f} kge_auroc={r.kge_auroc:.3f} enr={r.a2_only_enrichment:.3f}" for r in B[B.a2_only_strong].itertuples()]
sw = B.drop_duplicates(["dataset", "generator", "level_pct"])[["dataset", "generator", "level_pct", "a2_operator_swap_delta_f1", "a2_f1_lift", "a2_ndcg_lift", "a2_auroc"]]
out["H5_operator_swap"] = sw.round(4).values.tolist()

# 9 cross-dataset pattern
cd = []
for g in ("ASYMMETRY", "CLOSURE_TRANSITIVITY", "CARDINALITY"):
    for k, ref in (("a2_f1_lift", 1.0), ("a2_ndcg_lift", 1.0), ("a2_auroc", 0.5)):
        dirs = {}
        for ds in ("FB15k-237", "codex-m", "WN18RR"):
            v = A[(A.generator == g) & (A.dataset == ds)][k]
            above = int((v > ref).sum())
            dirs[ds] = "above" if above == len(v) else "below_or_equal" if above == 0 else "mixed"
        vals = list(dirs.values())
        top = max(set(vals), key=vals.count)
        cons = "consistent_3_of_3" if vals.count(top) == 3 else "consistent_2_of_3" if vals.count(top) == 2 else "dataset_specific"
        cd.append({"generator": g, "outcome": k, "reference": ref, **{f"{ds}_direction": d for ds, d in dirs.items()}, "pattern": cons, "majority_direction": top})
pd.DataFrame(cd).to_csv(O / "CROSS_DATASET_PATTERN.csv", index=False)

# 10 level pattern
lv = []
for (ds, g), df in A.groupby(["dataset", "generator"]):
    df = df.sort_values("level_pct")
    for k in ("a2_f1_lift", "a2_ndcg_lift", "a2_auroc"):
        v = df[k].values
        d = np.diff(v)
        span = float(v.max() - v.min())
        pat = ("effectively_flat" if span < 0.01 else "monotonic_increasing" if (d >= 0).all() and (d > 0).any() else
               "monotonic_decreasing" if (d <= 0).all() and (d < 0).any() else "non_monotonic")
        lv.append({"dataset": ds, "generator": g, "outcome": k, "level_5": v[0], "level_10": v[1], "level_15": v[2], "range": span, "pattern": pat,
                   "note": "flat threshold 0.01 absolute (descriptive)"})
pd.DataFrame(lv).to_csv(O / "PERTURBATION_LEVEL_PATTERN.csv", index=False)
out["10_level_pattern_counts"] = pd.DataFrame(lv).groupby(["outcome", "pattern"]).size().reset_index().values.tolist()
(O / "HYPOTHESIS_COUNTS.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str)[:7000])
