"""Objective A: node-degree confounding audit of the A2 ranking score (read-only, descriptive).

Scope: the 27 A2 constraint-ranking cells (3 datasets x 3 generators x 3 levels, seed 42), using exactly the
canonical ranking-panel inputs:
  A2 score : <root>/progress/20260623/ode_postfix_A2_<ds>_<vt>_<n>.npz  eval_energies (higher = more anomalous)
  sample   : <root>/canonical_persist/samples/<sha24>.json.gz  (triples in panel order, labels)
  record   : <root>/canonical_persist/detections/<ds>/postfix/logical_<vt>_<n>_seed42__A2.json.gz
Degree is computed on the FROZEN native graph (src/data.py::DataLoader.load_dataset = train+valid+test):
  degree_total(e) = #native triples with e as head + #native triples with e as tail
  degree_head(e), degree_tail(e) = the two terms separately. Entities absent from the native graph get 0.
Per evaluated triple: deg_h, deg_t (degree_total), deg_sum (primary), deg_max, deg_logsum (secondary).
Positions (not unique triples) are the unit, exactly as in the ranking panel (9/89 samples contain duplicates).

Diagnostics (no model is trained or integrated; all fits are on the fixed evaluation sample, in-sample):
  - degree baseline AUROC/NDCG@K (K = P) in both orientations (no orientation adopted);
  - Pearson / Spearman A2 vs degree (full / positives / negatives);
  - logistic Models D, A, A+D: sklearn LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100000,
    random_state=0, class_weight=None) on StandardScaler-standardised features fit on the same sample.
    C = 1.0 is the sklearn default, fixed a priori for every cell (mild L2 on standardised features), not tuned;
  - A2 residualised on [1, log1p(deg_h), log1p(deg_t)] by ordinary least squares;
  - weighted within-degree-bin AUROC on dataset-global deg_sum quartile bins (cut-points = 25/50/75th
    percentiles of deg_sum over the native triples of the dataset; Q1 <= c1 < Q2 <= c2 < Q3 <= c3 < Q4).
NDCG uses sklearn ndcg_score (ignore_ties=False: tied scores receive tie-averaged gain, which matters for
integer degree scores). Random-ranking reference: E[DCG@K]/IDCG@K = (P/N) sum_{i<=K} 1/log2(i+1) / IDCG@K.
Usage (cwd = execution root): python degree_confounding.py <execution_root> <postprocess_review_dir> <out_dir>
"""
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT, PRV, OUT = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(ROOT / "src"))
from data import DataLoader  # noqa: E402

DS = ["FB15k-237", "codex-m", "WN18RR"]
VT = ["asymmetry", "transitivity", "cardinality"]
LEVEL_N = {"FB15k-237": {5: 15505, 10: 31011, 15: 46517}, "codex-m": {5: 10310, 10: 20620, 15: 30930}, "WN18RR": {5: 4650, 10: 9300, 15: 13950}}
rk = {(r["dataset"], r["generator"], int(r["level_pct"]), r["detector"]): r for r in csv.DictReader(open(PRV / "RANKING_RECORDS_RECOMPUTED.csv"))}


def triples_sha(triples):
    return hashlib.sha256("\n".join(sorted("\t".join(t) for t in triples)).encode()).hexdigest()


def ndcg(y, s, K):
    return float(ndcg_score(y.astype(int)[None, :], s[None, :], k=K))


def ndcg_random(N, P, K):
    disc = 1 / np.log2(np.arange(2, K + 2))
    return (P / N) * disc.sum() / disc[:min(P, K)].sum()


def corr(f, x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(f(x, y)[0])


def logit_auc(X, y):
    Xs = StandardScaler().fit_transform(X)
    m = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=100000, random_state=0, class_weight=None).fit(Xs, y)
    return float(roc_auc_score(y, m.decision_function(Xs))), m.coef_[0].tolist(), int(m.n_iter_[0]), m.decision_function(Xs)


native, cuts = {}, {}
for ds in DS:
    T = DataLoader(str(ROOT / "data")).load_dataset(ds)
    dh = Counter(t.subject for t in T)
    dt = Counter(t.object for t in T)
    dtot = dh + dt
    nat_sum = np.array([dtot[t.subject] + dtot[t.object] for t in T], float)
    c = np.percentile(nat_sum, [25, 50, 75])
    native[ds] = (dh, dt, dtot)
    cuts[ds] = c.tolist()


def cell_metrics(ds, vt, lvl, T, y, a2, K):
    dh, dt, dtot = native[ds]
    c1, c2, c3 = cuts[ds]
    N, P = len(y), int(y.sum())
    srows = []
    deg_h = np.array([dtot.get(t[0], 0) for t in T], float)
    deg_t = np.array([dtot.get(t[2], 0) for t in T], float)
    hd = np.array([dh.get(t[0], 0) for t in T], float)      # head-role degree of the head entity
    td = np.array([dt.get(t[2], 0) for t in T], float)      # tail-role degree of the tail entity
    deg_sum, deg_max = deg_h + deg_t, np.maximum(deg_h, deg_t)
    deg_logsum = np.log1p(deg_h) + np.log1p(deg_t)
    er = ndcg_random(N, P, K)

    r = {"dataset": ds, "generator": vt, "level_pct": lvl, "N": N, "P": P, "K": K, "prevalence": P / N,
         "n_entities_absent_from_native": int(sum(t[0] not in dtot for t in T) + sum(t[2] not in dtot for t in T)),
         "median_deg_sum_pos": float(np.median(deg_sum[y])), "median_deg_sum_neg": float(np.median(deg_sum[~y])),
         "median_head_role_deg_pos": float(np.median(hd[y])), "median_head_role_deg_neg": float(np.median(hd[~y])),
         "median_tail_role_deg_pos": float(np.median(td[y])), "median_tail_role_deg_neg": float(np.median(td[~y])),
         "a2_auroc": float(roc_auc_score(y, a2)), "a2_ndcg": ndcg(y, a2, K), "E_ndcg_random": er}
    r["a2_ndcg_lift"] = r["a2_ndcg"] / er
    for nm, d in (("deg_sum", deg_sum), ("deg_max", deg_max), ("deg_logsum", deg_logsum)):
        r[f"auroc_{nm}_high"] = float(roc_auc_score(y, d))
        r[f"auroc_{nm}_low"] = float(roc_auc_score(y, -d))
        if nm == "deg_sum":
            r["ndcg_deg_sum_high"], r["ndcg_deg_sum_low"] = ndcg(y, d, K), ndcg(y, -d, K)
            r["ndcg_lift_deg_sum_high"], r["ndcg_lift_deg_sum_low"] = r["ndcg_deg_sum_high"] / er, r["ndcg_deg_sum_low"] / er
        for part, m in (("all", np.ones(N, bool)), ("pos", y), ("neg", ~y)):
            r[f"pearson_a2_{nm}_{part}"] = corr(pearsonr, a2[m], d[m])
            r[f"spearman_a2_{nm}_{part}"] = corr(spearmanr, a2[m], d[m])
    lh, lt = np.log1p(deg_h), np.log1p(deg_t)
    D = np.column_stack([lh, lt])
    r["auroc_model_D"], cD, _, fD = logit_auc(D, y)
    r["auroc_model_A"], cA, _, _ = logit_auc(a2[:, None], y)
    r["auroc_model_AD"], cAD, it, fAD = logit_auc(np.column_stack([a2, lh, lt]), y)
    r["ndcg_lift_model_D"], r["ndcg_lift_model_AD"] = ndcg(y, fD, K) / er, ndcg(y, fAD, K) / er
    r["delta_auroc_AD_minus_D"] = r["auroc_model_AD"] - r["auroc_model_D"]
    r["coef_a2_model_A"], r["coef_a2_model_AD"] = cA[0], cAD[0]
    r["coef_logdeg_h_model_AD"], r["coef_logdeg_t_model_AD"] = cAD[1], cAD[2]
    r["a2_enters_inverted_in_AD"] = cAD[0] < 0
    r["logit_AD_n_iter"] = it
    Xo = np.column_stack([np.ones(N), lh, lt])
    beta, *_ = np.linalg.lstsq(Xo, a2, rcond=None)
    fit = Xo @ beta
    res = a2 - fit
    r["ols_r2_a2_on_logdeg"] = float(1 - ((a2 - fit) ** 2).sum() / ((a2 - a2.mean()) ** 2).sum())
    r["a2_resid_auroc"] = float(roc_auc_score(y, res))
    r["a2_resid_ndcg"] = ndcg(y, res, K)
    r["a2_resid_ndcg_lift"] = r["a2_resid_ndcg"] / er
    # within-degree-bin AUROC (dataset-global deg_sum quartile cut-points)
    b = np.where(deg_sum <= c1, 1, np.where(deg_sum <= c2, 2, np.where(deg_sum <= c3, 3, 4)))
    wsum, wn = 0.0, 0
    for q in (1, 2, 3, 4):
        m = b == q
        npos, nneg = int(y[m].sum()), int((~y[m]).sum())
        au = float(roc_auc_score(y[m], a2[m])) if npos and nneg else float("nan")
        if npos and nneg:
            wsum += au * m.sum(); wn += int(m.sum())
        srows.append({"dataset": ds, "generator": vt, "level_pct": lvl, "degree_bin": f"Q{q}",
                      "bin_lower_exclusive": [float("-inf"), c1, c2, c3][q - 1], "bin_upper_inclusive": [c1, c2, c3, float("inf")][q - 1],
                      "n": int(m.sum()), "positives": npos, "negatives": nneg, "prevalence": npos / m.sum() if m.sum() else float("nan"),
                      "a2_auroc": au, "auroc_valid": bool(npos and nneg)})
    r["weighted_within_bin_auroc"] = wsum / wn if wn else float("nan")
    r["n_valid_bins"] = sum(1 for s in srows if s["auroc_valid"])
    r["n_in_valid_bins"] = wn
    return r, srows


rows, strat = [], []
for ds in DS:
    for vt in VT:
        for lvl, n in LEVEL_N[ds].items():
            rec = json.load(gzip.open(ROOT / f"canonical_persist/detections/{ds}/postfix/logical_{vt}_{n}_seed42__A2.json.gz", "rt"))
            sample = json.load(gzip.open(ROOT / "canonical_persist/samples" / f"{rec['sample_triples_sha256'][:24]}.json.gz", "rt"))
            T = [tuple(t) for t in sample["triples"]]
            L = {tuple(t) for t in sample["labels"]}
            z = np.load(ROOT / f"progress/20260623/ode_postfix_A2_{ds}_{vt}_{n}.npz", allow_pickle=True)
            a2 = np.asarray(z["eval_energies"], float)
            y = np.asarray(z["gt"]).astype(bool)
            # identity checks against the canonical panel inputs
            assert triples_sha(T) == rec["sample_triples_sha256"] == sample["triples_sha256"], (ds, vt, lvl, "sample sha")
            assert len(a2) == len(T) and np.isfinite(a2).all(), (ds, vt, lvl, "npz length/finite")
            assert np.array_equal(y, np.array([t in L for t in T])), (ds, vt, lvl, "labels")
            assert all(rec["scores"]["\t".join(t)] == a2[i] for i, t in enumerate(T)), (ds, vt, lvl, "npz != persisted A2 scores")
            pr = rk[(ds, vt, lvl, "ODE-A2")]
            N, P = len(y), int(y.sum())
            K = min(int(pr["K"]), N)
            assert int(pr["N_finite"]) == N and int(pr["P"]) == P, (ds, vt, lvl, "N/P vs panel")
            assert abs(ndcg(y, a2, K) - float(pr["ndcg_recomputed"])) < 1e-12 and abs(roc_auc_score(y, a2) - float(pr["auroc_stored"])) < 1e-12

            r, srows = cell_metrics(ds, vt, lvl, T, y, a2, K)
            rows.append(r); strat.extend(srows)

# Secondary scope (Q7): random entity-swap A2 detection records (seed 42 at 5/10/15 %, plus 5 % seeds 1-10).
# Not part of the ranking panel; per-position scores are mapped from the persisted record's per-triple score dict.
rnd, rstrat = [], []
for ds in DS:
    for lvl, seed in [(5, 42), (10, 42), (15, 42)] + [(5, s_) for s_ in range(1, 11)]:
        rec = json.load(gzip.open(ROOT / f"canonical_persist/detections/{ds}/postfix/random_entity_swap_{float(lvl)}_seed{seed}__A2.json.gz", "rt"))
        sample = json.load(gzip.open(ROOT / "canonical_persist/samples" / f"{rec['sample_triples_sha256'][:24]}.json.gz", "rt"))
        T = [tuple(t) for t in sample["triples"]]
        L = {tuple(t) for t in sample["labels"]}
        assert triples_sha(T) == rec["sample_triples_sha256"] == sample["triples_sha256"], (ds, lvl, seed, "sample sha")
        a2 = np.array([rec["scores"]["\t".join(t)] for t in T], float)
        y = np.array([t in L for t in T])
        assert np.isfinite(a2).all() and len(set(T)) == len(rec["scores"]), (ds, lvl, seed)
        r, srows = cell_metrics(ds, "random_entity_swap", lvl, T, y, a2, int(y.sum()))
        r["seed"] = seed
        for x in srows:
            x["seed"] = seed
        rnd.append(r); rstrat.extend(srows)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "DEGREE_CONFOUNDING_RANDOM_SECONDARY.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rnd[0])); w.writeheader(); w.writerows(rnd)
with open(OUT / "DEGREE_STRATIFIED_AUROC_RANDOM_SECONDARY.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rstrat[0])); w.writeheader(); w.writerows(rstrat)
with open(OUT / "DEGREE_CONFOUNDING_AUDIT.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(OUT / "DEGREE_STRATIFIED_AUROC.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(strat[0])); w.writeheader(); w.writerows(strat)
(OUT / "DEGREE_BIN_CUTPOINTS.json").write_text(json.dumps({"definition": "25/50/75th percentiles of deg_sum over native triples (train+valid+test)",
                                                          "cutpoints": cuts}, indent=1))
print(len(rows), "cells;", len(strat), "bin rows;", len(rnd), "random records")
