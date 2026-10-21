"""Ranking panel: chance-relative NDCG and AUROC orientation audit from persisted per-triple scores (read-only).

Persisted inputs (execution root):
  progress/20260623/full_panel.json                         stored metrics (compute_full_panel.py)
  progress/20260623/ode_postfix_{A1,A2,A3}_<ds>_<vt>_<n>.npz eval_energies (A1/A2 terminal energy, A3 SDE variance), gt
  paper_results/<ds>/ablation_audit/<vt>_<n>.json           per-triple kge_raw (PyKEEN plausibility), is_violation
Recomputes NDCG@K (K = P) and AUROC with the declared anomaly orientation (ODE: energy/variance as is;
KGE: -kge_raw, i.e. higher = more anomalous), the negated orientation as a diagnostic only, and the
exact random-ranking NDCG reference E[DCG@K]/IDCG@K with E[DCG@K] = (P/N) sum_{i<=K} 1/log2(i+1).
Usage: python ranking_audit.py <execution_root> <out_dir>
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
panel = json.loads((ROOT / "progress/20260623/full_panel.json").read_text())
LEVEL_N = {"FB15k-237": {5: 15505, 10: 31011, 15: 46517}, "WN18RR": {5: 4650, 10: 9300, 15: 13950}, "codex-m": {5: 10310, 10: 20620, 15: 30930}}


def arrays(det, ds, vt, n):
    if det.startswith("ODE-"):
        z = np.load(ROOT / f"progress/20260623/ode_postfix_{det[4:]}_{ds}_{vt}_{n}.npz", allow_pickle=True)
        return np.asarray(z["eval_energies"], float), np.asarray(z["gt"]).astype(bool), "energy (A1/A2) or cross-trajectory variance (A3), higher = more anomalous"
    a = json.loads((ROOT / f"paper_results/{ds}/ablation_audit/{vt}_{n}.json").read_text())["triples"]
    return (np.array([-t["kge_raw"][det] for t in a], float), np.array([t["is_violation"] for t in a], bool),
            "-kge_raw (negated PyKEEN plausibility), higher = more anomalous")


def ndcg_random(N, P, K):
    disc = 1 / np.log2(np.arange(2, K + 2))
    return (P / N) * disc.sum() / disc[:min(P, K)].sum()


rows = []
for c in panel["cells"]:
    ds, vt, lvl = c["dataset"], c["vtype"], c["level_pct"]
    n = LEVEL_N[ds][lvl]
    for det, m in c["detectors"].items():
        if "auroc" not in m:
            continue
        s, y, sem = arrays(det, ds, vt, n)
        fin = np.isfinite(s)
        s, y = s[fin], y[fin]
        N, P = len(y), int(y.sum())
        K = min(int(c["K"]), N)
        nd = float(ndcg_score(y.astype(int)[None, :], s[None, :], k=K))
        au, au_neg = float(roc_auc_score(y, s)), float(roc_auc_score(y, -s))
        er = ndcg_random(N, P, K)
        rows.append({"dataset": ds, "generator": vt, "level_pct": lvl, "detector": det, "N_finite": N, "n_nonfinite": int((~fin).sum()),
                     "P": P, "prevalence": P / N, "K": K, "ndcg_stored": m["ndcg_at_K"], "ndcg_recomputed": nd,
                     "E_ndcg_random": er, "delta_ndcg": nd - er, "lift_ndcg": nd / er if er else float("nan"),
                     "auroc_stored": m["auroc"], "auroc_recomputed_declared": au, "auroc_negated_diagnostic": au_neg,
                     "stored_matches_declared": abs(au - m["auroc"]) < 1e-12 and abs(nd - m["ndcg_at_K"]) < 1e-12,
                     "score_semantics": sem})
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "RANKING_RECORDS_RECOMPUTED.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
cnt = Counter(r["detector"] for r in rows)
print("records per detector:", dict(cnt), "total", len(rows))
print("all stored == recomputed (declared orientation):", all(r["stored_matches_declared"] for r in rows))
au = sorted(rows, key=lambda r: r["auroc_stored"])
print("min AUROC", au[0]["auroc_stored"], au[0]["dataset"], au[0]["generator"], au[0]["level_pct"], au[0]["detector"])
print("max AUROC", au[-1]["auroc_stored"], au[-1]["dataset"], au[-1]["generator"], au[-1]["level_pct"], au[-1]["detector"])
print("AUROC < 0.5:", sum(r["auroc_stored"] < 0.5 for r in rows))
print("NDCG above/below/equal random:", sum(r["delta_ndcg"] > 0 for r in rows), sum(r["delta_ndcg"] < 0 for r in rows), sum(r["delta_ndcg"] == 0 for r in rows))
