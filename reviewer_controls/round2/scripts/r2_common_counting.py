"""R2 Gate C — common positional counting for the persisted detection records (round-2 protocol; see ../README.md).

Run from anywhere:  python reviewer_controls/round2/scripts/r2_common_counting.py <frozen_root> <out_dir>

For every canonical_persist/detections/**/<cell>__<method>.json.gz record, the evaluation sample is loaded by its
SHA-256 and the record's detected set and the sample's labels are projected onto the evaluation positions:
position i is detected iff its triple is in the detected set, and positive iff its triple is in the labels.
The detector is untouched: same detected set, same threshold, no score is read for the decision.
Reported per record: the record's own counts (unique-triple TP/FP/FN, with N and TN on positions), the coherent
unique-triple counts, the positional counts, and F1/precision/recall under each unit, with differences.
Ordering check: within each (dataset, cell), the F1 ranking of methods under the record's unit versus positions.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DET, SAMPLES = FROZEN / "canonical_persist/detections", FROZEN / "canonical_persist/samples"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing, cache = [], [], {}
    for path in sorted(DET.rglob("*.json.gz")):
        rec = json.load(gzip.open(path, "rt"))
        sha = rec["sample_triples_sha256"]
        sp = SAMPLES / f"{sha[:24]}.json.gz"
        if not sp.exists():
            missing.append({"record": str(path.relative_to(DET)), "reason": "sample not persisted", "sample_sha256": sha})
            continue
        if sha not in cache:
            s = json.load(gzip.open(sp, "rt"))
            assert s["triples_sha256"] == sha
            cache[sha] = ([tuple(t) for t in s["triples"]], {tuple(t) for t in s["labels"]})
        T, L = cache[sha]
        D = {tuple(t) for t in rec["detected"]}
        assert len(T) == rec["n"], (path, "n")
        assert len(L) == rec["n_pos"], (path, "labels")
        U = set(T)
        # record's own arithmetic, re-derived on unique triples
        tp_u, fp_u, fn_u = len(D & L), len(D - L), len(L - D)
        assert (tp_u, fp_u, fn_u) == (rec["tp"], rec["fp"], rec["fn"]), (path, "record counts")
        det_pos = np.array([t in D for t in T]); lab_pos = np.array([t in L for t in T])
        tp_p = int((det_pos & lab_pos).sum()); fp_p = int((det_pos & ~lab_pos).sum())
        fn_p = int((~det_pos & lab_pos).sum()); tn_p = int((~det_pos & ~lab_pos).sum())
        pu, ru, fu = prf(tp_u, fp_u, fn_u)
        pp, rp, fpos = prf(tp_p, fp_p, fn_p)
        rows.append({"dataset": rec["dataset"], "kind": rec["kind"], "cell": rec["cell"], "method": rec["method"],
                     "operator": rec.get("operator"), "device": rec.get("device"), "code_commit": rec.get("code_commit"),
                     "sample_sha256": sha, "n_positions": len(T), "n_unique": len(U), "n_repeated_positions": len(T) - len(U),
                     "P_unique": len(L), "P_positions": int(lab_pos.sum()),
                     "detected_unique": len(D), "detected_positions": int(det_pos.sum()),
                     "detected_not_in_sample": len(D - U),
                     "record_tp": rec["tp"], "record_fp": rec["fp"], "record_fn": rec["fn"], "record_tn_mixed_units": rec["tn"],
                     "tn_unique": len(U) - tp_u - fp_u - fn_u,
                     "tp_pos": tp_p, "fp_pos": fp_p, "fn_pos": fn_p, "tn_pos": tn_p,
                     "precision_unique": pu, "recall_unique": ru, "f1_unique": fu,
                     "precision_pos": pp, "recall_pos": rp, "f1_pos": fpos,
                     "d_f1": fpos - fu, "d_precision": pp - pu, "d_recall": rp - ru,
                     "record_path": str(path.relative_to(FROZEN))})
    keys = list(rows[0])
    with open(OUT / "common_counting_per_record.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT / "common_counting_missing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record", "reason", "sample_sha256"]); w.writeheader(); w.writerows(missing)
    # ordering check within (dataset, cell) across all methods recorded on the same sample
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["cell"], r["sample_sha256"])].append(r)
    order_rows = []
    for (ds, cell, sha), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        lab = [f"{r['kind']}:{r['method']}" for r in rs]
        ou = [lab[i] for i in np.argsort([-r["f1_unique"] for r in rs], kind="stable")]
        op = [lab[i] for i in np.argsort([-r["f1_pos"] for r in rs], kind="stable")]
        flips = sum(1 for i in range(len(rs)) for j in range(i + 1, len(rs))
                    if np.sign(rs[i]["f1_unique"] - rs[j]["f1_unique"]) * np.sign(rs[i]["f1_pos"] - rs[j]["f1_pos"]) < 0)
        order_rows.append({"dataset": ds, "cell": cell, "sample_sha256": sha, "n_methods": len(rs),
                           "order_unique": " > ".join(ou), "order_positions": " > ".join(op),
                           "pairwise_order_flips": flips, "max_abs_d_f1": max(abs(r["d_f1"]) for r in rs)})
    order_keys = ["dataset", "cell", "sample_sha256", "n_methods", "order_unique", "order_positions",
                  "pairwise_order_flips", "max_abs_d_f1"]  # header written even when no cell has two methods
    with open(OUT / "common_counting_ordering.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order_keys); w.writeheader(); w.writerows(order_rows)
    print(f"{len(rows)} records evaluated, {len(missing)} without persisted sample, {len(order_rows)} groups", flush=True)


if __name__ == "__main__":
    main()
