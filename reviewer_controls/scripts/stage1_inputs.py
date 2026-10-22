"""Stage 1 — regenerate perturbed graphs, verify samples, compute checker counts and degrees (no training).

Run from the repository root:  python reviewer_controls/scripts/stage1_inputs.py <dataset> reviewer_controls/frozen_inputs <out_dir>

For every evaluation sample used by the revised manuscript's grid (27 logical cells, seed 42), the paired
asymmetry-10% seeds (1-10) and the paired random-5% seeds (1-10), this script:
  V1  regenerates the perturbed graph with the producer's exact call and checks that the evaluation sample's
      SHA-256 equals the persisted sample in reviewer_controls/frozen_inputs/canonical_persist/samples/;
  C   computes the per-position symbolic-checker count exactly as S4 / the 2x2 panel do
      (symmetry + transitivity + gradient-mode antisymmetry, evaluated on the full perturbed graph);
  deg computes the label-free degree of each position in the perturbed graph (head degree + tail degree).
Labels are read from the persisted sample only to store them next to the inputs; they are not used here.
The constraint-structure joblib cache is disabled (keys are a heuristic hash that can collide across seeds).
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/experiments")

import numpy as np

import data as _data
_data._cache_load = lambda path: None  # never reuse a structure cached for another graph

from ablation_study import KGPerturbator
from data import DataLoader
from run_all import _load_hyperparam_cache, sample_triples
from canonical_persist import triples_sha
from run_s0_s4_progressive_ablation import build_constraint_checker, count_violations

DS, FROZEN, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
LEVELS = {"FB15k-237": (15505, 31011, 46517), "WN18RR": (4650, 9300, 13950), "codex-m": (10310, 20620, 30930)}


def specs(n_base):
    for vt in ("asymmetry", "transitivity", "cardinality"):
        for n in LEVELS[DS]:
            yield f"grid_{vt}_{n}", "logical", vt, n, 42
    n10 = max(10, int(n_base * 0.10))
    for s in range(1, 11):
        yield f"asym10_seed{s}", "logical", "asymmetry", n10, s
    for s in range(1, 11):
        yield f"random5_seed{s}", "random", None, 5.0, s


def main():
    base = DataLoader().load_dataset(DS)
    margin_asym = float(_load_hyperparam_cache(DS, "ode").get("margin_asym", 1.0))
    report = []
    for tag, kind, vt, level, seed in specs(len(base)):
        t0 = time.time()
        pert = KGPerturbator(base, seed=int(seed))
        if kind == "random":
            perturbed, viol = pert.perturb(percentage=level, perturbation_types=["entity_swap"])
        else:
            perturbed, viol, _ = pert.perturb_logical(violation_type=vt, n_violations=level, dataset_name=DS)
        sample, _ = sample_triples(perturbed, viol, 10000, seed=int(seed))
        sha = triples_sha(sample)
        p = FROZEN / "canonical_persist" / "samples" / f"{sha[:24]}.json.gz"
        rec = json.load(gzip.open(p, "rt")) if p.exists() else None
        v1 = bool(rec and rec["triples_sha256"] == sha and
                  [list(t) for t in rec["triples"]] == [[t.subject, t.relation, t.object] for t in sample])
        if not v1:
            report.append({"tag": tag, "V1": False, "sha": sha})
            print(f"{tag}: V1 FAILED", flush=True)
            continue
        labels = {tuple(x) for x in rec["labels"]}
        gt = np.array([(t.subject, t.relation, t.object) in labels for t in sample], dtype=bool)
        checker = build_constraint_checker(DS, margin_asym)
        C = count_violations(checker, perturbed, sample)
        deg = collections.Counter()
        for t in perturbed:
            deg[t.subject] += 1
            deg[t.object] += 1
        degree = np.array([deg[t.subject] + deg[t.object] for t in sample], dtype=float)
        np.savez_compressed(OUT / f"{DS}__{tag}.npz", C=C, degree=degree, gt=gt,
                            triples=np.array([[t.subject, t.relation, t.object] for t in sample]),
                            sha=sha, n_perturbed=len(perturbed), n_injected=len(viol))
        row = {"tag": tag, "V1": True, "sha": sha, "n_pos": int(gt.sum()), "n_C_pos": int((C > 0).sum()),
               "C_sum": int(C.sum()), "C_max": int(C.max()), "n_perturbed": len(perturbed),
               "n_injected": len(viol), "seconds": round(time.time() - t0, 1)}
        report.append(row)
        print(json.dumps(row), flush=True)
    (OUT / f"{DS}__stage1_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
