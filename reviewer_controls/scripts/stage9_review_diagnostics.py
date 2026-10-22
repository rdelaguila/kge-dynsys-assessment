"""Phase-1 diagnostics for the critical review (no training; reads stored scores and results only).

Run from the repository root:  python reviewer_controls/scripts/stage9_review_diagnostics.py reviewer_controls/frozen_inputs reviewer_controls/work

(a) Knee degeneration: reproduces the F1-knee computation (run_s0_s4_progressive_ablation.f1_from_scores) for stored
    scores and records every intermediate quantity, including kneed's normalised curves, for:
      - panel WN18RR asymmetry-15%, TransE+C additive (q99)   [reported F1 ~ 0]
      - panel WN18RR asymmetry-15%, TransE alone               [reported F1 0.271]
      - panel FB15k-237 transitivity-15%, TransE+C additive     [reported F1 0.257]
      - Block B WN18RR seed 1, TransE+C additive and S3+C additive
(b) Degree adjustment: the four T4 contrasts before and after adjustment (same contrast function, CI and Holm
    families of three datasets), so that intervals can be compared.
(c) Inventory of every contrast family written by stages 4, 6 and 8 (tests, Holm size, non-zero pairs).
(d) Training-seed vs perturbation-seed summaries, recomputed from the stored per-seed NDCG values.
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
from kneed import KneeLocator

from run_s0_s4_progressive_ablation import f1_from_scores, knee_threshold
from stage2_block_a import additive, fast_auroc, fast_ndcg, load_cell
from stage4_paired import contrast
from stage8_extra_analysis import adjust_degree
from statistical_tests import holm_bonferroni

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RES = OUT / "review_phase1"
RES.mkdir(exist_ok=True)


def knee_trace(name, s, gt):
    s = np.asarray(s, float)
    finite = np.isfinite(s)
    x = s[finite]
    srt = np.sort(x)[::-1]
    xs = np.arange(len(srt))
    kl = KneeLocator(xs, srt, curve="convex", direction="decreasing", interp_method="polynomial")
    thr, found = knee_threshold(x)
    res = f1_from_scores(s, gt, int(gt.sum()))
    # same knee rule with linear interpolation, only to show where the difference-curve maximum moves
    kl_lin = KneeLocator(xs, srt, curve="convex", direction="decreasing", interp_method="interp1d")
    yd = np.asarray(kl.y_difference)
    gap = np.diff(srt)
    big = int(np.argmin(gap))  # largest single drop in the descending curve
    return {"case": name, "n": int(len(x)), "n_distinct": int(len(np.unique(x))), "P": int(gt.sum()),
            "max": float(srt[0]), "min": float(srt[-1]),
            "largest_drop_index": big, "largest_drop_from": float(srt[big]), "largest_drop_to": float(srt[big + 1]),
            "values_above_largest_drop": big + 1,
            "kneed_interp": "polynomial (degree 7, kneed default)", "kneed_knee_index": None if kl.knee is None else int(kl.knee),
            "kneed_y_difference_argmax": int(np.argmax(yd)), "kneed_y_difference_max": float(yd.max()),
            "kneed_y_difference_at_largest_drop": float(yd[big]),
            "threshold": thr, "knee_found": bool(found), "fallback": "95th percentile if no knee (not triggered)" if found else "95th percentile used",
            "operator": ">= threshold, descending curve (S0-S4 convention)",
            "selected": res["n_detected"], "tp": res["tp"], "fp": res["fp"], "fn": res["fn"],
            "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
            "diagnostic_linear_interp_knee_index": None if kl_lin.knee is None else int(kl_lin.knee)}


def main():
    traces = []
    gt, C, deg, S = load_cell("WN18RR", "asymmetry", "13950")
    traces.append(knee_trace("panel WN18RR asym-15% TransE+C additive q99", additive(C, S["TransE"], .99)[0], gt))
    traces.append(knee_trace("panel WN18RR asym-15% TransE (no checker)", S["TransE"], gt))
    traces.append(knee_trace("panel WN18RR asym-15% A2+C additive q99", additive(C, S["A2"], .99)[0], gt))
    gt2, C2, _, S2 = load_cell("FB15k-237", "transitivity", "46517")
    traces.append(knee_trace("panel FB15k-237 closure-15% TransE+C additive q99", additive(C2, S2["TransE"], .99)[0], gt2))
    inp = np.load(OUT / "inputs" / "WN18RR__asym10_seed1.npz"); z = np.load(OUT / "block_b_raw" / "WN18RR__seed1.npz")
    g, c = inp["gt"].astype(bool), inp["C"].astype(float)
    traces.append(knee_trace("Block B WN18RR seed 1 TransE+C additive q99", additive(c, -np.asarray(z["TransE_raw"], float), .99)[0], g))
    traces.append(knee_trace("Block B WN18RR seed 1 S3+C additive q99", additive(c, np.asarray(z["S3"], float), .99)[0], g))
    (RES / "knee_traces.json").write_text(json.dumps(traces, indent=1))

    # (b) degree adjustment before / after
    rows = []
    for metric, f in (("NDCG", fast_ndcg), ("AUROC", fast_auroc)):
        for base in ("S3", "S1"):
            for adj in (False, True):
                items = []
                for ds in ("FB15k-237", "codex-m", "WN18RR"):
                    d = []
                    for s in range(1, 11):
                        inp = np.load(OUT / "inputs" / f"{ds}__asym10_seed{s}.npz"); z = np.load(OUT / "block_b_raw" / f"{ds}__seed{s}.npz")
                        gt, deg = inp["gt"].astype(bool), inp["degree"].astype(float)
                        a, b = np.asarray(z[base], float), -np.asarray(z["TransE_raw"], float)
                        if adj:
                            a, b = adjust_degree(a, deg), adjust_degree(b, deg)
                        d.append(f(gt, a) - f(gt, b))
                    items.append((ds, contrast(np.array(d))))
                ps = holm_bonferroni([r["p_raw"] for _, r in items])
                for (ds, r), p in zip(items, ps):
                    rows.append({"metric": metric, "contrast": f"{base} - TransE", "degree_adjusted": adj, "dataset": ds,
                                 "mean": r["mean"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
                                 "rank_biserial": r["rank_biserial"], "p_raw": r["p_raw"], "p_holm3": p,
                                 "n_pos": r["n_pos"], "n_neg": r["n_neg"], "n_zero": r["n_zero"]})
    with open(RES / "degree_before_after.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # (c) inventory of contrast families
    inv = []
    for path, famcol in ((OUT / "block_a_bis/paired_contrasts.csv", "family"), (OUT / "block_b/paired_contrasts.csv", "family"),
                         (OUT / "extra_tests/family_contrasts.csv", "family")):
        fams = {}
        for r in csv.DictReader(open(path)):
            fams.setdefault(r[famcol], []).append(r)
        for fam, lst in fams.items():
            nz = [10 - int(r["n_zero"] or 0) for r in lst]
            inv.append({"source": str(path.relative_to(OUT)), "family": fam, "tests": len(lst), "holm_size": len(lst),
                        "min_nonzero_pairs": min(nz), "max_nonzero_pairs": max(nz)})
    with open(RES / "contrast_inventory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0])); w.writeheader(); w.writerows(inv)

    # (d) training-seed vs perturbation-seed summaries
    var = []
    t3 = list(csv.DictReader(open(OUT / "extra_tests/T3_training_seeds.csv")))
    for ds in ("FB15k-237", "codex-m", "WN18RR"):
        m = np.array([[float(r[f"ndcg_TransE_seed{k}"]) for k in (42, 43, 44)] for r in t3 if r["dataset"] == ds])
        var.append({"dataset": ds, "n_perturbation_seeds": m.shape[0], "n_training_seeds": m.shape[1],
                    "within_perturbation_sd_over_3_training_seeds_mean": float(m.std(axis=1, ddof=1).mean()),
                    "within_perturbation_sd_min": float(m.std(axis=1, ddof=1).min()),
                    "within_perturbation_sd_max": float(m.std(axis=1, ddof=1).max()),
                    "sd_over_10_perturbation_seeds_at_training_seed_42": float(m[:, 0].std(ddof=1)),
                    "sd_over_10_perturbation_seeds_of_mean_over_training_seeds": float(m.mean(axis=1).std(ddof=1))})
    with open(RES / "seed_variability_summaries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(var[0])); w.writeheader(); w.writerows(var)
    print("wrote", RES)


if __name__ == "__main__":
    main()
