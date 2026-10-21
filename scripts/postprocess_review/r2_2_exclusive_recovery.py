"""Reviewer 2.2: exclusive / shared labelled-positive recovery at an identical review budget (read-only).

For the nine shared ranking-panel cells, A = top-K A2 triples and B = top-K KGE triples (K = P), using the
canonical panel scores and orientation:
  A2  : progress/20260623/ode_postfix_A2_<ds>_<vt>_<n>.npz  eval_energies (higher = more anomalous)
  KGE : paper_results/<ds>/ablation_audit/<vt>_<n>.json    -kge_raw[model] (higher = more anomalous)
Top-K uses np.argsort(-score, kind="stable"), exactly as compute_full_panel.metrics_for.
Before any calculation, per cell: the triples of both sources must equal the persisted canonical sample
(same SHA-256 and order), labels must agree across sources, the npz energies must equal the persisted A2
detection-record scores, N/P/K must match RANKING_RECORDS_RECOMPUTED.csv, and the recomputed NDCG must
equal the panel value. Any failure aborts (R2_2_BLOCKED_MISSING_PERSISTED_SCORES for missing artefacts).
Usage: python r2_2_exclusive_recovery.py <execution_root> <postprocess_review_dir> <out_dir>
"""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import ndcg_score

ROOT, PRV, OUT = (Path(a) for a in sys.argv[1:4])
CELLS = [("FB15k-237", "transitivity", 15, 46517), ("FB15k-237", "cardinality", 5, 15505), ("codex-m", "asymmetry", 10, 20620),
         ("codex-m", "transitivity", 5, 10310), ("codex-m", "cardinality", 5, 10310), ("codex-m", "cardinality", 15, 30930),
         ("WN18RR", "asymmetry", 15, 13950), ("WN18RR", "transitivity", 5, 4650), ("WN18RR", "cardinality", 15, 13950)]
KGE = ("TransE", "DistMult", "MuRE")


def triples_sha(triples):
    return hashlib.sha256("\n".join(sorted("\t".join(t) for t in triples)).encode()).hexdigest()


def inputs(ds, vt, n):
    return {"npz": ROOT / f"progress/20260623/ode_postfix_A2_{ds}_{vt}_{n}.npz",
            "audit": ROOT / f"paper_results/{ds}/ablation_audit/{vt}_{n}.json",
            "a2_record": ROOT / f"canonical_persist/detections/{ds}/postfix/logical_{vt}_{n}_seed42__A2.json.gz"}


missing = [str(p) for c in CELLS for p in inputs(*c[:2], c[3]).values() if not p.exists()]
if missing:
    print("R2_2_BLOCKED_MISSING_PERSISTED_SCORES"); print("\n".join(missing)); sys.exit(2)

panel = {(r["dataset"], r["generator"], int(r["level_pct"]), r["detector"]): r for r in csv.DictReader(open(PRV / "RANKING_RECORDS_RECOMPUTED.csv"))}


def region(name, idx, gt, N, P):
    n = len(idx)
    tp = int(gt[list(idx)].sum()) if n else 0
    prev = P / N
    if n == 0:
        return {f"{name}_n": 0, f"{name}_tp": 0, f"{name}_negatives": 0, f"{name}_precision": "NA", f"{name}_enrichment": "NA",
                f"{name}_delta_precision": "NA", f"{name}_expected_tp_random": 0.0, f"{name}_p_enrichment": "NA", f"{name}_p_depletion": "NA"}
    prec = tp / n
    return {f"{name}_n": n, f"{name}_tp": tp, f"{name}_negatives": n - tp, f"{name}_precision": prec, f"{name}_enrichment": prec / prev,
            f"{name}_delta_precision": prec - prev, f"{name}_expected_tp_random": n * P / N,
            f"{name}_p_enrichment": float(hypergeom.sf(tp - 1, N, P, n)), f"{name}_p_depletion": float(hypergeom.cdf(tp, N, P, n))}


rows, checks = [], []
for ds, vt, lvl, n in CELLS:
    f = inputs(ds, vt, n)
    rec = json.load(gzip.open(f["a2_record"], "rt"))
    sample = json.load(gzip.open(ROOT / "canonical_persist/samples" / f"{rec['sample_triples_sha256'][:24]}.json.gz", "rt"))
    T = [tuple(t) for t in sample["triples"]]
    L = {tuple(t) for t in sample["labels"]}
    audit = json.loads(f["audit"].read_text())["triples"]
    Ta = [(t["s"], t["r"], t["o"]) for t in audit]
    z = np.load(f["npz"], allow_pickle=True)
    e = np.asarray(z["eval_energies"], float)
    gt = np.asarray(z["gt"]).astype(bool)
    # identity checks
    assert triples_sha(T) == rec["sample_triples_sha256"] == sample["triples_sha256"], (ds, vt, "sample sha")
    assert Ta == T, (ds, vt, "audit triple order differs from canonical sample")
    assert len(e) == len(T) and np.isfinite(e).all(), (ds, vt, "npz length/finite")
    assert np.array_equal(gt, np.array([t in L for t in T])), (ds, vt, "npz labels")
    assert np.array_equal(gt, np.array([t["is_violation"] for t in audit])), (ds, vt, "audit labels")
    assert all(rec["scores"]["\t".join(t)] == e[i] for i, t in enumerate(T)), (ds, vt, "npz energies != persisted A2 scores")
    N, P = len(T), int(gt.sum())
    K = P
    for det in ("ODE-A2", *KGE):
        pr = panel[(ds, vt, lvl, det)]
        assert (int(pr["N_finite"]), int(pr["P"]), int(pr["K"])) == (N, P, K), (ds, vt, det, "N/P/K")
    oa = np.argsort(-e, kind="stable")[:K]
    nd = float(ndcg_score(gt.astype(int)[None, :], e[None, :], k=K))
    assert abs(nd - float(panel[(ds, vt, lvl, "ODE-A2")]["ndcg_stored"])) < 1e-12, (ds, vt, "A2 NDCG")
    A = set(oa.tolist())
    checks.append({"dataset": ds, "generator": vt, "level_pct": lvl, "sample_sha256": rec["sample_triples_sha256"], "N": N, "P": P, "K": K,
                   "a2_ndcg_reproduced": nd})
    for kge in KGE:
        s = np.array([-t["kge_raw"][kge] for t in audit], float)
        ndk = float(ndcg_score(gt.astype(int)[None, :], s[None, :], k=K))
        assert abs(ndk - float(panel[(ds, vt, lvl, kge)]["ndcg_stored"])) < 1e-12, (ds, vt, kge, "KGE NDCG")
        B = set(np.argsort(-s, kind="stable")[:K].tolist())
        assert len(A) == len(B) == K
        a_only, shared, b_only = A - B, A & B, B - A
        union = A | B
        assert len(a_only) + len(shared) == K and len(b_only) + len(shared) == K and len(union) == len(a_only) + len(shared) + len(b_only)
        row = {"dataset": ds, "generator": vt, "level_pct": lvl, "kge": kge, "N": N, "P": P, "K": K, "prevalence": P / N}
        row.update(region("a2_only", a_only, gt, N, P)); row.update(region("shared", shared, gt, N, P)); row.update(region("kge_only", b_only, gt, N, P))
        a_tp, k_tp = row["a2_only_tp"] + row["shared_tp"], row["kge_only_tp"] + row["shared_tp"]
        u_tp = row["a2_only_tp"] + row["shared_tp"] + row["kge_only_tp"]
        row.update({"a2_tp_total": a_tp, "kge_tp_total": k_tp,
                    "fraction_a2_tp_exclusive": row["a2_only_tp"] / a_tp if a_tp else "NA",
                    "fraction_kge_tp_exclusive": row["kge_only_tp"] / k_tp if k_tp else "NA",
                    "additional_unique_positive_count": row["a2_only_tp"] + row["kge_only_tp"],
                    "union_n": len(union), "union_tp": u_tp, "union_precision": u_tp / len(union), "union_recall": u_tp / P,
                    "union_note": "EXPANDED_BUDGET_DESCRIPTIVE_ONLY"})
        prev = P / N
        ea = row["a2_only_precision"] != "NA" and row["a2_only_precision"] > prev
        ek = row["kge_only_precision"] != "NA" and row["kge_only_precision"] > prev
        row["descriptive_pattern"] = ("BOTH_EXCLUSIVE_REGIONS_ENRICHED" if ea and ek else "A2_ONLY_ENRICHED" if ea
                                      else "KGE_ONLY_ENRICHED" if ek else "NEITHER_EXCLUSIVE_ENRICHED")
        rows.append(row)

OUT.mkdir(parents=True, exist_ok=False)
cols = ["dataset", "generator", "level_pct", "kge", "N", "P", "K", "prevalence"]
for reg in ("a2_only", "shared", "kge_only"):
    cols += [f"{reg}_{k}" for k in ("n", "tp", "negatives", "precision", "enrichment", "delta_precision", "expected_tp_random", "p_enrichment", "p_depletion")]
cols += ["a2_tp_total", "kge_tp_total", "fraction_a2_tp_exclusive", "fraction_kge_tp_exclusive", "additional_unique_positive_count",
         "union_n", "union_tp", "union_precision", "union_recall", "union_note", "descriptive_pattern"]
with open(OUT / "R2_2_EXCLUSIVE_RECOVERY.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
with open(OUT / "R2_2_INPUT_IDENTITY_CHECKS.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(checks[0])); w.writeheader(); w.writerows(checks)
print(len(rows), "comparisons written")
