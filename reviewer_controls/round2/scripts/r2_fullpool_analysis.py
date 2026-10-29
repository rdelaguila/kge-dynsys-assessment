"""R2 panel 2 analysis — TransE on the full perturbed pool versus S1/S3 with the same checker (family F4) and versus
TransE trained on the sample (round-2 protocol; see ../README.md).

Run from the repository root:
    python reviewer_controls/round2/scripts/r2_fullpool_analysis.py <work> <block_b_per_seed_metrics.csv>
Inputs (read-only): <work>/inputs/<ds>__asym10_seed<s>.npz (gt, C, sha; stage1_inputs.py),
<work>/fullpool/<ds>__seed<s>.npz (r2_fullpool_transe_run.py), <work>/block_b_raw/<ds>__seed<s>.npz
(S1, S3, TransE_raw from stage5_block_b_run.py), and reviewer_controls/results/block_b/per_seed_metrics.csv
for the coherence check. Output: <work>/fullpool_analysis/. Ordering with the checker: lexicographic C + pct(score) (stage2_block_a.lexi).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from r2_common import contrast, family, lexi, metrics

WK, REF = Path(sys.argv[1]), Path(sys.argv[2])
OUT = WK / "fullpool_analysis"
DATASETS = ("FB15k-237", "codex-m", "WN18RR")
SEEDS = range(1, 11)


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {(r["dataset"], r["seed"], r["config"]): r for r in csv.DictReader(open(REF))}
    rows, coh, ND, cover = [], [], {}, []
    for ds in DATASETS:
        for s in SEEDS:
            fp = WK / "fullpool" / f"{ds}__seed{s}.npz"
            if not fp.exists():
                cover.append({"dataset": ds, "seed": s, "status": "NOT_RUN"})
                continue
            inp = np.load(WK / "inputs" / f"{ds}__asym10_seed{s}.npz")
            bb = np.load(WK / "block_b_raw" / f"{ds}__seed{s}.npz")
            fz = np.load(fp)
            assert str(inp["sha"]) == str(bb["sha"]) == str(fz["sha"]), (ds, s, "pairing by sample SHA")
            meta = json.loads(str(fz["meta"]))
            gt, C = inp["gt"].astype(bool), inp["C"].astype(float)
            base = {"S1": bb["S1"], "S3": bb["S3"], "TransE_sample": -bb["TransE_raw"], "TransE_full": -fz["TransE_full_raw"]}
            base = {k: np.asarray(v, float) for k, v in base.items()}
            assert all(np.isfinite(v).all() for v in base.values()), (ds, s, "non-finite")
            cfg = dict(base)
            for k, v in base.items():
                cfg[f"{k}+C"] = lexi(C, v)
            nd = {}
            for name, sc in cfg.items():
                m = metrics(gt, sc)
                nd[name] = m["ndcg_at_K"]
                rows.append({"dataset": ds, "seed": s, "sample_sha256": str(inp["sha"]), "config": name, "N": len(gt),
                             "P": int(gt.sum()), "C_coverage": float((C > 0).mean()), **m})
            for mine, theirs in (("S1+C", "S1+C"), ("S3+C", "S3+C"), ("TransE_sample+C", "TransE+C"), ("TransE_sample", "TransE")):
                r = ref[(ds, str(s), theirs)]
                dd = abs(nd[mine] - float(r["ndcg_at_K"]))
                coh.append({"dataset": ds, "seed": s, "config": mine, "ndcg_recomputed": nd[mine],
                            "ndcg_block_b_published": float(r["ndcg_at_K"]), "abs_diff": dd, "ok": dd <= 1e-12})
            ND[(ds, s)] = nd
            cover.append({"dataset": ds, "seed": s, "status": "DONE",
                          "spearman_full_vs_sample": float(spearmanr(base["TransE_full"], base["TransE_sample"]).statistic),
                          **{k: meta[k] for k in ("pool_size", "pool_unique", "loader_num_triples", "loader_num_entities",
                                                  "loader_num_relations", "batches_per_epoch", "updates_total",
                                                  "sample_batches_per_epoch", "train_seconds", "total_seconds", "device")}})
    write(OUT / "per_seed_metrics.csv", rows)
    write(OUT / "coherence_block_b.csv", coh)
    write(OUT / "coverage_and_cost.csv", cover)
    if any(not r["ok"] for r in coh):
        sys.exit("coherence with published Block B metrics failed; statistics not computed")
    items, desc = [], []
    for ds in DATASETS:
        seeds = [s for s in SEEDS if (ds, s) in ND]
        def d(a, b):
            return np.array([ND[(ds, s)][a] - ND[(ds, s)][b] for s in seeds])
        items.append((f"{ds}: (S1+C) - (TransE_full+C)", d("S1+C", "TransE_full+C")))
        items.append((f"{ds}: (S3+C) - (TransE_full+C)", d("S3+C", "TransE_full+C")))
        for lab, a, b in (("TransE_full - TransE_sample", "TransE_full", "TransE_sample"),
                          ("(TransE_full+C) - (TransE_sample+C)", "TransE_full+C", "TransE_sample+C"),
                          ("(S3+C) - (TransE_sample+C) [Block B B1, recomputed]", "S3+C", "TransE_sample+C"),
                          ("(S1+C) - (TransE_sample+C) [Block B B2, recomputed]", "S1+C", "TransE_sample+C")):
            if len(seeds) > 1:
                desc.append({"dataset": ds, "contrast": lab, **contrast(d(a, b)), "note": "descriptive; not part of F4"})
    fam = family("F4", items)
    complete = all(r["n_available"] == 10 for r in fam)
    for r in fam:
        r["family_complete"] = complete
    write(OUT / "paired_statistics_F4.csv", fam)
    write(OUT / "descriptive_contrasts.csv", desc)
    print("wrote", OUT, "F4 complete" if complete else "F4 INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
