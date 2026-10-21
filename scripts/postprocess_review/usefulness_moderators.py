"""Condition-level moderator matrix for A2 usefulness (read-only, descriptive).

Scope A: the 27 A2 constraint-ranking cells (seed 42). Scope B: the 27 A2-vs-KGE pure-score comparisons of the
nine shared panel cells. Inputs: persisted samples/labels, persisted A2 detection records, existing post-processing
outputs (F1_CHANCE_PER_RUN.csv, RANKING_RECORDS_RECOMPUTED.csv, r2_2 CSV), the canonical operator sweep, frozen
constraint/ODE configuration (get_constraint_config, .cache/hyperparams/<ds>_ode.json) and the native dataset files.
Force alignment (traced, scripts/milestone_postfix_section4_rerun.py::build_cc + src/ode_system.py):
  asymmetry    -> antisymmetry hinge, weight lambda_asym, relations = asymmetry_relations
  closure      -> transitivity force inside L_logic, weight lambda_logic, relations = transitivity_relations
  cardinality  -> no ODE force (no differentiable cardinality constraint; cardinality_constraints = [])
Usage: python usefulness_moderators.py <execution_root> <postprocess_review_dir> <out_dir>
"""
import csv
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT, PRV, OUT = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(ROOT / "src"))
from data import DataLoader  # noqa: E402
from run_all import get_constraint_config  # noqa: E402

DS = ["FB15k-237", "codex-m", "WN18RR"]
LEVEL_N = {"FB15k-237": {5: 15505, 10: 31011, 15: 46517}, "codex-m": {5: 10310, 10: 20620, 15: 30930}, "WN18RR": {5: 4650, 10: 9300, 15: 13950}}
GEN = {"asymmetry": "ASYMMETRY", "transitivity": "CLOSURE_TRANSITIVITY", "cardinality": "CARDINALITY"}

f1 = {(r["dataset"], r["generator"], float(r["nominal_level_pct"])): r for r in csv.DictReader(open(PRV / "F1_CHANCE_PER_RUN.csv"))
      if r["detector"] == "A2" and r["family"] == "single_seed_42"}
rk = {(r["dataset"], r["generator"], int(r["level_pct"]), r["detector"]): r for r in csv.DictReader(open(PRV / "RANKING_RECORDS_RECOMPUTED.csv"))}
r22 = list(csv.DictReader(open(PRV / "r2_2_exclusive_recovery" / "R2_2_EXCLUSIVE_RECOVERY.csv")))
sweep = {(c["dataset"], c["vtype"], int(c["level_n"])): c for c in json.loads((ROOT / "progress/parallel_operator_sweep.json").read_text())["cells"]}

native = {}
for ds in DS:
    t = DataLoader().load_dataset(ds)
    ne, nr = len({x.subject for x in t} | {x.object for x in t}), len({x.relation for x in t})
    native[ds] = {"native_triples": len(t), "native_entities": ne, "native_relations": nr,
                  "triples_per_entity": len(t) / ne, "triples_per_relation": len(t) / nr}


def force(ds, vt):
    cfg = get_constraint_config(ds)
    bp = json.loads((ROOT / f".cache/hyperparams/{ds}_ode.json").read_text())["best_params"]
    if vt == "asymmetry":
        rels = cfg["asymmetry_relations"]; lam = float(bp.get("lambda_asym", 1.0)); name = "antisymmetry hinge (L_asym)"
    elif vt == "transitivity":
        rels = cfg["transitivity_relations"]; lam = float(bp["lambda_logic"]); name = "transitivity force (in L_logic)"
    else:
        rels, lam, name = [], 0.0, "none (no ODE cardinality force)"
    return {"matching_force": name, "matching_lambda": lam, "n_configured_force_relations": len(rels),
            "force_aligned": bool(rels) and lam > 0}, set(rels)


rows = []
for ds in DS:
    for vt in ("asymmetry", "transitivity", "cardinality"):
        for lvl, n in LEVEL_N[ds].items():
            rec = json.load(gzip.open(ROOT / f"canonical_persist/detections/{ds}/postfix/logical_{vt}_{n}_seed42__A2.json.gz", "rt"))
            sample = json.load(gzip.open(ROOT / "canonical_persist/samples" / f"{rec['sample_triples_sha256'][:24]}.json.gz", "rt"))
            labels = [tuple(t) for t in sample["labels"]]
            Lset = set(labels)
            T = [tuple(t) for t in sample["triples"]]
            N = sample["n"]
            P_unique = len(labels)
            P = sum(1 for t in T if t in Lset)          # positive positions (ranking-panel convention)
            n_dup_positions = len(T) - len(set(T))
            labels = [t for t in T if t in Lset]       # relation statistics over positive positions
            fa, frels = force(ds, vt)
            rel_counts = Counter(t[1] for t in labels)
            covered = sum(c for r, c in rel_counts.items() if r in frels) if fa["force_aligned"] else 0
            probs = [c / P for c in rel_counts.values()] if P else []
            H = -sum(p * math.log(p) for p in probs) if probs else float("nan")
            Hn = H / math.log(len(probs)) if len(probs) > 1 else float("nan")
            fr = f1[(ds, "closure/transitivity" if vt == "transitivity" else vt, float(lvl))]
            rr = rk[(ds, vt, lvl, "ODE-A2")]
            # F1 records count unique labelled triples; ranking records count sample positions (duplicates possible)
            assert int(fr["N"]) == N and int(fr["P"]) == P_unique and int(rr["N_finite"]) == N and int(rr["P"]) == P, (ds, vt, lvl)
            assert P - P_unique <= n_dup_positions, (ds, vt, lvl, "duplicate accounting")
            rows.append({"dataset": ds, "generator": GEN[vt], "level_pct": lvl, "N": N, "P": P, "P_unique_labels": P_unique,
                         "n_duplicate_positions_in_sample": n_dup_positions, "prevalence": P / N, **fa,
                         "force_coverage_positives": covered / P if P else float("nan"),
                         "n_positive_relations": len(rel_counts),
                         "n_force_covered_positive_relations": sum(1 for r in rel_counts if r in frels) if fa["force_aligned"] else 0,
                         "max_positive_relation_share": max(probs) if probs else float("nan"),
                         "positive_relation_entropy": H, "positive_relation_entropy_normalized": Hn,
                         "top_positive_relation": rel_counts.most_common(1)[0][0] if rel_counts else "",
                         **native[ds],
                         "a2_f1": float(fr["F1"]), "a2_f1_chance": float(fr["E_F1_random"]), "a2_f1_lift": float(fr["lift_F1"]),
                         "a2_ndcg": float(rr["ndcg_recomputed"]), "a2_ndcg_random": float(rr["E_ndcg_random"]),
                         "a2_ndcg_lift": float(rr["lift_ndcg"]), "a2_auroc": float(rr["auroc_stored"]),
                         "source_sample_hash": rec["sample_triples_sha256"],
                         "source_score_artifact": f"progress/20260623/ode_postfix_A2_{ds}_{vt}_{n}.npz; {rec['_path'] if '_path' in rec else f'canonical_persist/detections/{ds}/postfix/logical_{vt}_{n}_seed42__A2.json.gz'}"})

OUT.mkdir(parents=True, exist_ok=False)
with open(OUT / "USEFULNESS_MODERATOR_MATRIX.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

byA = {(r["dataset"], r["generator"], r["level_pct"]): r for r in rows}
pw = []
for x in r22:
    ds, vt, lvl, kge = x["dataset"], x["generator"], int(x["level_pct"]), x["kge"]
    a = byA[(ds, GEN[vt], lvl)]
    kr = rk[(ds, vt, lvl, kge)]
    sw = sweep[(ds, vt, LEVEL_N[ds][lvl])]["detectors"]["ODE-A2"]
    pw.append({**{k: a[k] for k in ("dataset", "generator", "level_pct", "N", "P", "prevalence", "force_aligned", "matching_force",
                                     "matching_lambda", "n_configured_force_relations", "force_coverage_positives", "n_positive_relations",
                                     "max_positive_relation_share", "positive_relation_entropy_normalized", "a2_f1_lift",
                                     "a2_ndcg", "a2_ndcg_lift", "a2_auroc")},
               "kge": kge, "kge_ndcg": float(kr["ndcg_recomputed"]), "kge_ndcg_lift": float(kr["lift_ndcg"]), "kge_auroc": float(kr["auroc_stored"]),
               "kge_ndcg_above_random": float(kr["lift_ndcg"]) > 1, "kge_auroc_above_chance": float(kr["auroc_stored"]) > 0.5,
               **{k: x[k] for k in ("a2_only_n", "a2_only_tp", "a2_only_enrichment", "a2_only_p_enrichment", "shared_n", "shared_tp",
                                    "shared_enrichment", "kge_only_n", "kge_only_tp", "kge_only_enrichment", "fraction_a2_tp_exclusive",
                                    "fraction_kge_tp_exclusive", "descriptive_pattern")},
               "a2_operator_swap_delta_f1": sw["otsu"]["f1"] - sw["knee"]["f1"]})
with open(OUT / "USEFULNESS_MODERATOR_PAIRWISE.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pw[0])); w.writeheader(); w.writerows(pw)
print(len(rows), "scope-A rows;", len(pw), "scope-B rows")
