"""F1 relative to an exact fixed-M random selector, per persisted run (read-only).

For each persisted detection record (N evaluated, P positives, M detected, TP):
  X ~ Hypergeometric(N, P, M); E[F1_random] = 2*M*P / (N*(M+P)) (F1 = 2TP/(M+P), M and P fixed);
  p_enrichment = P(X >= TP), p_depletion = P(X <= TP) (diagnostic only).
  label: ABOVE if p_enrichment < 0.025, BELOW if p_depletion < 0.025, else NEAR (TP outside / inside the central
  95% of the fixed-size random-selector distribution; descriptive, no multiplicity control).
Replicated conditions are summarised per seed first, then aggregated.
Usage: python f1_chance.py <execution_root> <out_dir>
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N_LOGICAL = {"FB15k-237": (15505, 31011, 46517), "codex-m": (10310, 20620, 30930), "WN18RR": (4650, 9300, 13950)}
GEN = {"entity_swap": "random", "asymmetry": "asymmetry", "transitivity": "closure/transitivity", "cardinality": "cardinality"}


def parse_cell(ds, cell):
    # random_entity_swap_5.0_seed42 | logical_asymmetry_15505_seed42
    body, seed = cell.rsplit("_seed", 1)
    parts = body.split("_")
    if parts[0] == "random":
        return "random", float(parts[-1]), int(seed)
    vt, n = parts[1], int(parts[2])
    return GEN[vt], (5.0, 10.0, 15.0)[N_LOGICAL[ds].index(n)], int(seed)


rows = []
for p in sorted((ROOT / "canonical_persist/detections").rglob("*.json.gz")):
    d = json.load(gzip.open(p, "rt"))
    kind = d["kind"].split("/")[0]
    if kind not in ("ablation", "postfix", "bootstrap"):
        continue
    gen, lvl, seed = parse_cell(d["dataset"], d["cell"])
    N, P, M, TP = d["n"], d["n_pos"], d["n_detected"], d["tp"]
    f1 = 2 * TP / (M + P) if M + P else 0.0
    ef1 = 2 * M * P / (N * (M + P)) if M + P else 0.0
    rows.append({"dataset": d["dataset"], "generator": gen, "nominal_level_pct": lvl, "detector": d["method"],
                 "family": "random5_10seed" if kind in ("bootstrap",) or (kind == "postfix" and seed != 42) else "single_seed_42",
                 "seed": seed, "N": N, "P": P, "prevalence": P / N, "M": M, "selection_fraction": M / N,
                 "TP": TP, "FP": d["fp"], "FN": d["fn"], "F1": f1, "E_F1_random": ef1, "delta_F1": f1 - ef1,
                 "lift_F1": f1 / ef1 if ef1 else float("nan"),
                 "E_TP_random": M * P / N,
                 "p_enrichment": float(hypergeom.sf(TP - 1, N, P, M)), "p_depletion": float(hypergeom.cdf(TP, N, P, M)),
                 "source": str(p.relative_to(ROOT))})
for r in rows:
    r["label"] = "ABOVE" if r["p_enrichment"] < 0.025 else "BELOW" if r["p_depletion"] < 0.025 else "NEAR"
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "F1_CHANCE_PER_RUN.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

groups = defaultdict(list)
for r in rows:
    groups[(r["dataset"], r["generator"], r["nominal_level_pct"], r["detector"], r["family"])].append(r)
summ = []
for (ds, gen, lvl, det, fam), rs in sorted(groups.items()):
    f1 = np.array([r["F1"] for r in rs]); e = np.array([r["E_F1_random"] for r in rs])
    d_ = f1 - e; lift = f1 / e
    summ.append({"dataset": ds, "generator": gen, "nominal_level_pct": lvl, "detector": det, "family": fam, "n_seeds": len(rs),
                 "mean_F1": f1.mean(), "mean_E_F1_random": e.mean(), "mean_delta_F1": d_.mean(), "min_delta_F1": d_.min(),
                 "max_delta_F1": d_.max(), "mean_lift_F1": lift.mean(), "min_lift_F1": lift.min(), "max_lift_F1": lift.max(),
                 "n_seeds_F1_above_random_expectation": int((d_ > 0).sum()),
                 "max_p_enrichment": max(r["p_enrichment"] for r in rs), "max_p_depletion": max(r["p_depletion"] for r in rs),
                 "mean_selection_fraction": float(np.mean([r["selection_fraction"] for r in rs])),
                 "mean_prevalence": float(np.mean([r["prevalence"] for r in rs]))})
with open(OUT / "F1_CHANCE_SUMMARY.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
print(len(rows), "runs;", len(summ), "conditions")
