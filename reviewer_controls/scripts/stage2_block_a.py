"""Block A — checker-only, same-checker, fusion, bonus-percentile, noise and degree controls on the nine
shared ranking-panel cells (seed 42). Evaluation only: no training, no integration.

Run from the repository root:  python reviewer_controls/scripts/stage2_block_a.py reviewer_controls/frozen_inputs reviewer_controls/work

Inputs (all read-only):
  <out>/inputs/<ds>__grid_<vt>_<n>.npz          stage 1: checker count C, degree, labels, sample triples
  <frozen>/progress/20260623/ode_postfix_{A1,A2}_<ds>_<vt>_<n>.npz   A1/A2 terminal residuals (50 RK4 steps)
  <frozen>/paper_results/<ds>/ablation_audit/<vt>_<n>.json           TransE/DistMult/MuRE raw scores
  <out>/transe_epochs/<ds>__<vt>_<n>__transe_epochs.npz              N1 (optional)
All scores are oriented so that larger = more anomalous, fixed in advance (KGE: -raw, as in R2.2).
Labels are used only to compute metrics, never to build scores, calibrate b or break ties.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/experiments")
sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import ndcg_score

from compute_full_panel import metrics_for
from compute_parallel_operator_sweep import otsu_threshold
from run_s0_s4_progressive_ablation import f1_from_scores

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CELLS = [l.split() for l in (Path(__file__).resolve().parent / "panel_cells.txt").read_text().splitlines() if l.strip()]
RNG_SEED = 0
N_TIE = 200      # random tie-breaks for heavily tied scores (checker, degree)
N_BOOT = 1000    # paired bootstrap over evaluation positions (descriptive)
N_PERM = 1000    # noise control for the fusion


def pct(x):
    return rankdata(x, method="average") / (len(x) + 1.0)


def lexi(C, x):
    """Checker first; within equal C, the base score's percentile rank (scale-free)."""
    return C + pct(x)


def additive(C, x, q):
    xs = x - np.nanmin(x)
    b = float(np.quantile(xs[np.isfinite(xs)], q))
    if not (np.isfinite(b) and b > 0):
        b = float(np.nanmax(xs)) or 1.0
    return xs + b * C, b


def thresholded(gt, s):
    P, N = int(gt.sum()), len(gt)
    out = {}
    knee = f1_from_scores(s, gt, P)
    thr = otsu_threshold(s)
    det = s > thr
    tp = int((det & gt).sum()); fp = int((det & ~gt).sum()); fn = int((~det & gt).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    for name, M, f1 in (("knee", knee["n_detected"], knee["f1"]), ("otsu", int(det.sum()), 2 * p * r / max(p + r, 1e-10))):
        out[f"f1_{name}"] = f1
        out[f"n_det_{name}"] = M
        out[f"f1_{name}_chance"] = 2 * M * P / (N * (M + P)) if M + P else float("nan")
    return out


def evaluate(gt, s, tied=False, rng=None):
    P = int(gt.sum())
    if not tied:
        m = metrics_for(gt, s, P)
        m.update(thresholded(gt, s))
        return m
    acc = []
    for _ in range(N_TIE):
        acc.append(metrics_for(gt, s + rng.random(len(s)) * 1e-9, P))
    m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
    det = s > 0
    tp = int((det & gt).sum()); fp = int((det & ~gt).sum()); fn = int((~det & gt).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    M = int(det.sum())
    m.update({"f1_detector_gt0": 2 * p * r / max(p + r, 1e-10), "n_det_gt0": M,
              "f1_detector_gt0_chance": 2 * M * P / (len(gt) * (M + P)) if M + P else float("nan")})
    return m


def fast_ndcg(gt, s):
    """Same definition as compute_full_panel.metrics_for (sklearn ndcg_score, K = P, ties averaged)."""
    return float(ndcg_score(gt.astype(int)[None, :], s[None, :], k=int(gt.sum())))


def fast_auroc(gt, s):
    r = rankdata(s, method="average")
    P = gt.sum(); Nn = len(gt) - P
    return (r[gt].sum() - P * (P + 1) / 2) / (P * Nn)


def load_cell(ds, vt, n):
    inp = np.load(OUT / "inputs" / f"{ds}__grid_{vt}_{n}.npz", allow_pickle=False)
    T = [tuple(t) for t in inp["triples"]]
    gt = inp["gt"].astype(bool)
    a2 = np.load(FROZEN / "progress/20260623" / f"ode_postfix_A2_{ds}_{vt}_{n}.npz", allow_pickle=True)
    a1 = np.load(FROZEN / "progress/20260623" / f"ode_postfix_A1_{ds}_{vt}_{n}.npz", allow_pickle=True)
    audit = json.loads((FROZEN / "paper_results" / ds / "ablation_audit" / f"{vt}_{n}.json").read_text())
    rows = audit["triples"]
    assert [(r["s"], r["r"], r["o"]) for r in rows] == T, (ds, vt, n, "audit order")
    assert np.array_equal(np.array([r["is_violation"] for r in rows]), gt), (ds, vt, n, "audit labels")
    for z, name in ((a2, "A2"), (a1, "A1")):
        assert np.array_equal(np.asarray(z["gt"]).astype(bool), gt), (ds, vt, n, name, "labels")
        assert np.isfinite(np.asarray(z["eval_energies"])).all(), (ds, vt, n, name, "finite")
    S = {"A2": np.asarray(a2["eval_energies"], float), "A1": np.asarray(a1["eval_energies"], float)}
    for k in ("TransE", "DistMult", "MuRE"):
        S[k] = -np.array([r["kge_raw"][k] for r in rows], float)
    ep = OUT / "transe_epochs" / f"{ds}__{vt}_{n}__transe_epochs.npz"
    if ep.exists():
        z = np.load(ep)
        assert str(z["sha"]) == str(inp["sha"]), (ds, vt, n, "epochs sha")
        for e in (1, 2, 5, 10):
            S[f"TransE_cpu_e{e}"] = -np.asarray(z[f"raw_e{e}"], float)
    return gt, inp["C"].astype(float), inp["degree"].astype(float), S


def main():
    rng = np.random.default_rng(RNG_SEED)
    rows, boots, noise, epochs = [], [], [], []
    for ds, vt, n in CELLS:
        cell = f"{ds}/{vt}/{n}"
        gt, C, deg, S = load_cell(ds, vt, n)
        N, P = len(gt), int(gt.sum())
        configs, tied = {}, {"C"}
        configs["C"] = C
        configs["degree"] = deg
        for k in ("A2", "A1", "TransE", "DistMult", "MuRE"):
            configs[k] = S[k]
            configs[f"{k}+C"] = lexi(C, S[k])
            configs[f"{k}+C_add_q99"], _ = additive(C, S[k], 0.99)
        for k in ("A2", "TransE"):
            for q in (0.95, 0.999):
                configs[f"{k}+C_add_q{str(q)[2:]}"], _ = additive(C, S[k], q)
        configs["Fusion(A2,TransE)"] = (pct(S["A2"]) + pct(S["TransE"])) / 2
        configs["Fusion(A2,TransE)+C"] = lexi(C, configs["Fusion(A2,TransE)"])
        configs["Fusion(degree,TransE)"] = (pct(deg) + pct(S["TransE"])) / 2
        for k in [k for k in S if k.startswith("TransE_cpu_e")]:
            configs[k] = S[k]
        for name, s in configs.items():
            m = evaluate(gt, s, tied=name in tied, rng=rng)
            rows.append({"cell": cell, "config": name, "N": N, "P": P, "prevalence": P / N,
                         "C_coverage": float((C > 0).mean()), "ndcg_random": P / N, **m})
        # N2 noise control: TransE fused with A2 values permuted within the sample
        obs = metrics_for(gt, configs["Fusion(A2,TransE)"], P)
        perm_ndcg, perm_auroc = [], []
        pT = pct(S["TransE"])
        for _ in range(N_PERM):
            f = (pct(rng.permutation(S["A2"])) + pT) / 2
            perm_ndcg.append(fast_ndcg(gt, f)); perm_auroc.append(fast_auroc(gt, f))
        perm_ndcg, perm_auroc = np.array(perm_ndcg), np.array(perm_auroc)
        noise.append({"cell": cell, "fusion_ndcg": obs["ndcg_at_K"], "noise_ndcg_mean": perm_ndcg.mean(),
                      "noise_ndcg_q025": np.quantile(perm_ndcg, .025), "noise_ndcg_q975": np.quantile(perm_ndcg, .975),
                      "p_ndcg": (1 + (perm_ndcg >= obs["ndcg_at_K"]).sum()) / (N_PERM + 1),
                      "fusion_auroc": obs["auroc"], "noise_auroc_mean": perm_auroc.mean(),
                      "noise_auroc_q025": np.quantile(perm_auroc, .025), "noise_auroc_q975": np.quantile(perm_auroc, .975),
                      "p_auroc": (1 + (perm_auroc >= obs["auroc"]).sum()) / (N_PERM + 1)})
        # paired bootstrap over positions (descriptive) for the pre-specified differences
        best_single = "A2" if metrics_for(gt, S["A2"], P)["ndcg_at_K"] >= metrics_for(gt, S["TransE"], P)["ndcg_at_K"] else "TransE"
        diffs = {"(A2+C)-(TransE+C)": ("A2+C", "TransE+C"), "(A2+C)-A2": ("A2+C", "A2"),
                 "(TransE+C)-TransE": ("TransE+C", "TransE"), f"Fusion-{best_single}": ("Fusion(A2,TransE)", best_single),
                 "A2-degree": ("A2", "degree"), "C-A2": ("C", "A2")}
        idx_all = [rng.integers(0, N, N) for _ in range(N_BOOT)]
        for label, (a, b) in diffs.items():
            sa, sb = configs[a], configs[b]
            ja = rng.random(N) * 1e-9 if a in tied else 0.0
            jb = rng.random(N) * 1e-9 if b in tied else 0.0
            d_auc, d_ndcg = [], []
            for idx in idx_all:
                g = gt[idx]
                if g.sum() in (0, len(g)):
                    continue
                d_auc.append(fast_auroc(g, (sa + ja)[idx]) - fast_auroc(g, (sb + jb)[idx]))
                d_ndcg.append(fast_ndcg(g, (sa + ja)[idx]) - fast_ndcg(g, (sb + jb)[idx]))
            boots.append({"cell": cell, "difference": label,
                          "d_ndcg_point": metrics_for(gt, sa + ja, P)["ndcg_at_K"] - metrics_for(gt, sb + jb, P)["ndcg_at_K"],
                          "d_ndcg_q025": np.quantile(d_ndcg, .025), "d_ndcg_q975": np.quantile(d_ndcg, .975),
                          "d_auroc_point": fast_auroc(gt, sa + ja) - fast_auroc(gt, sb + jb),
                          "d_auroc_q025": np.quantile(d_auc, .025), "d_auroc_q975": np.quantile(d_auc, .975)})
        # N1: similarity of A2 / A1 to TransE after 1, 2, 5, 10 epochs (label-free) and to the captured TransE
        for k in [k for k in S if k.startswith("TransE")] :
            for base in ("A2", "A1", "degree"):
                ref = deg if base == "degree" else S[base]
                epochs.append({"cell": cell, "transe": k, "versus": base,
                               "spearman": float(spearmanr(ref, S[k]).statistic)})
        print(f"done {cell}", flush=True)
    res = OUT / "block_a"
    res.mkdir(exist_ok=True)
    for name, data_ in (("metrics", rows), ("bootstrap_differences", boots), ("noise_control_fusion", noise),
                        ("transe_epoch_similarity", epochs)):
        with open(res / f"{name}.csv", "w", newline="") as f:
            keys = list(dict.fromkeys(k for r in data_ for k in r))
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(data_)
    print("wrote", res)


if __name__ == "__main__":
    main()
