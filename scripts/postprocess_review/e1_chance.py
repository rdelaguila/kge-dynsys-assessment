"""E1 top-K overlap (A2 vs KGE) relative to independent random sets of the same sizes (read-only).

E1 (manuscript) = progress/audit_phase2_E_rerun.json["e2_jaccard"]: top-K of the native A2 energy over
the whole native pool (K = floor(0.05 N)) vs top-K of the KGE anomaly score among the KGE native detections.
Under a uniformly random A2 top-K, |A ∩ B| ~ Hypergeometric(N, K_A, K_B) for any fixed B.
Exact E[J] = sum_x pmf(x) * x / (K_A + K_B - x). Intersections are recomputed from the persisted sets
and must equal the stored counts. Secondary: full native detection sets (A2 vs each KGE).
Usage: python e1_chance.py <execution_root> <out_dir>
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DS = ["FB15k-237", "codex-m", "WN18RR"]


def overlap_stats(N, KA, KB, I):
    xs = np.arange(max(0, KA + KB - N), min(KA, KB) + 1)
    pmf = hypergeom.pmf(xs, N, KA, KB)
    ej = float(np.sum(pmf * xs / (KA + KB - xs)))
    ei = KA * KB / N
    jo = I / (KA + KB - I)
    return {"N": N, "K_A": KA, "K_B": KB, "I_obs": I, "E_I": ei, "overlap_enrichment": I / ei, "J_obs": jo, "E_J_exact": ej,
            "J_of_E_I_not_used": ei / (KA + KB - ei), "delta_J": jo - ej, "lift_J": jo / ej,
            "p_enrichment": float(hypergeom.sf(I - 1, N, KA, KB)), "p_depletion": float(hypergeom.cdf(I, N, KA, KB)),
            "pmf_mass_checked": float(pmf.sum())}


stored = {(r["dataset"], r["kge"]): r for r in json.loads((ROOT / "progress/audit_phase2_E_rerun.json").read_text())["e2_jaccard"]}
rows, sec = [], []
for ds in DS:
    a2 = json.load(gzip.open(ROOT / f"canonical_persist/detections/{ds}/native/native_rk4_50__A2.json.gz", "rt"))
    energy = {tuple(k.split("\t")): v for k, v in a2["scores"].items()}
    N = len(energy)
    K = max(1, int(0.05 * N))
    top_a2 = set(sorted(energy, key=lambda k: -energy[k])[:K])
    a2_set = {tuple(t) for t in a2["detected"]}
    for m in ("TransE", "DistMult", "MuRE", "ComplEx"):
        dets = json.loads((ROOT / f"paper_results/{ds}/{m}/detections.json").read_text())
        sc = {(d["triple"]["subject"], d["triple"]["relation"], d["triple"]["object"]): float(d["score"]) for d in dets}
        if m != "ComplEx":
            top_k = set(sorted(sc, key=lambda k: -sc[k])[:K])
            I = len(top_a2 & top_k)
            st = stored[(ds, m)]
            assert st["intersection"] == I and st["top_k"] == K, (ds, m, st, I, K)
            rows.append({"dataset": ds, "kge": m, "comparison": "E1 top-K (K = floor(0.05 N))",
                         "n_kge_native_detections": len(sc), **overlap_stats(N, K, K, I),
                         "stored_jaccard": st["jaccard"], "source": "progress/audit_phase2_E_rerun.json[e2_jaccard] + persisted sets"})
        kset = set(sc)
        sec.append({"dataset": ds, "kge": m, "comparison": "full native detection sets (secondary)",
                    **overlap_stats(N, len(a2_set), len(kset), len(a2_set & kset))})
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "E1_CHANCE_OVERLAP.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(OUT / "E1_NATIVE_SET_OVERLAP_SECONDARY.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sec[0])); w.writeheader(); w.writerows(sec)
for r in rows:
    print(f"{r['dataset']:10s} {r['kge']:8s} N={r['N']} K={r['K_A']} I={r['I_obs']} E[I]={r['E_I']:.1f} enr={r['overlap_enrichment']:.3f} "
          f"J={r['J_obs']:.4f} E[J]={r['E_J_exact']:.4f} lift={r['lift_J']:.3f} p_enr={r['p_enrichment']:.3g} p_dep={r['p_depletion']:.3g}")
print("-- secondary")
for r in sec:
    print(f"{r['dataset']:10s} {r['kge']:8s} |A2|={r['K_A']} |KGE|={r['K_B']} I={r['I_obs']} E[I]={r['E_I']:.0f} enr={r['overlap_enrichment']:.3f} "
          f"J={r['J_obs']:.4f} E[J]={r['E_J_exact']:.4f}")
