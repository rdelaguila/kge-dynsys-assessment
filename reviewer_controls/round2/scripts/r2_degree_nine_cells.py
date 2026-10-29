"""R2 Gate D, nine original conditions — head degree d(h) and tail degree d(t) as separate label-free scores
(round-2 protocol; see ../README.md). No training, no ODE.

Run from the repository root:  python reviewer_controls/round2/scripts/r2_degree_nine_cells.py <frozen_inputs> <work>

Per cell of reviewer_controls/scripts/panel_cells.txt: the perturbed graph and sample are regenerated with the
producer's call (as stage1_inputs.py, perturbation seed 42) and checked against the stage-1 SHA-256; d(h) and d(t)
are counted exactly as stage1 counts its sum (incidences over all relations and both directions of the full perturbed
graph); d(h)+d(t) must equal the stage-1 degree. A2 and TransE are read as stage2_block_a.load_cell reads them.
Coherence: degree, A2, TransE, Fusion(A2,TransE) and Fusion(degree,TransE) are recomputed and compared with
<work>/block_a/metrics.csv to 1e-12 before anything is written.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from r2_common import fusion, metrics

from ablation_study import KGPerturbator
from canonical_persist import triples_sha
from data import DataLoader
from run_all import sample_triples

FROZEN, WK = Path(sys.argv[1]), Path(sys.argv[2])
INP = WK / "inputs"
OUT = WK / "degree_nine_cells"
CELLS = [l.split() for l in Path("reviewer_controls/scripts/panel_cells.txt").read_text().splitlines() if l.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {(r["cell"], r["config"]): float(r["ndcg_at_K"]) for r in csv.DictReader(open(WK / "block_a/metrics.csv"))}
    rows, coh, bases = [], [], {}
    for ds, vt, n in CELLS:
        n = int(n)
        cell = f"{ds}/{vt}/{n}"
        if ds not in bases:
            bases[ds] = DataLoader().load_dataset(ds)
        perturbed, viol, _ = KGPerturbator(bases[ds], seed=42).perturb_logical(
            violation_type=vt, n_violations=n, dataset_name=ds)
        sample, _ = sample_triples(perturbed, viol, 10000, seed=42)
        inp = np.load(INP / f"{ds}__grid_{vt}_{n}.npz", allow_pickle=False)
        assert triples_sha(sample) == str(inp["sha"]), (cell, "sample")
        deg = collections.Counter()
        for t in perturbed:
            deg[t.subject] += 1
            deg[t.object] += 1
        dh = np.array([deg[t.subject] for t in sample], float)
        dt = np.array([deg[t.object] for t in sample], float)
        assert np.array_equal(dh + dt, inp["degree"].astype(float)), (cell, "d(h)+d(t) vs stage1 degree")
        gt = inp["gt"].astype(bool)
        a2 = np.load(FROZEN / "progress/20260623" / f"ode_postfix_A2_{ds}_{vt}_{n}.npz", allow_pickle=True)
        audit = json.loads((FROZEN / "paper_results" / ds / "ablation_audit" / f"{vt}_{n}.json").read_text())
        assert [(r["s"], r["r"], r["o"]) for r in audit["triples"]] == [tuple(t) for t in inp["triples"]], (cell, "audit order")
        A2 = np.asarray(a2["eval_energies"], float)
        TE = -np.array([r["kge_raw"]["TransE"] for r in audit["triples"]], float)
        cfg = {"deg_sum": dh + dt, "deg_h": dh, "deg_t": dt, "A2": A2, "TransE": TE,
               "Fusion(A2,TransE)": fusion(A2, TE), "Fusion(deg_sum,TransE)": fusion(dh + dt, TE),
               "Fusion(deg_h,TransE)": fusion(dh, TE), "Fusion(deg_t,TransE)": fusion(dt, TE)}
        nd = {}
        for name, s in cfg.items():
            m = metrics(gt, s)
            nd[name] = m["ndcg_at_K"]
            rows.append({"cell": cell, "config": name, "N": len(gt), "P": int(gt.sum()), "sample_sha256": str(inp["sha"]), **m})
        for mine, theirs in (("deg_sum", "degree"), ("A2", "A2"), ("TransE", "TransE"),
                             ("Fusion(A2,TransE)", "Fusion(A2,TransE)"), ("Fusion(deg_sum,TransE)", "Fusion(degree,TransE)")):
            d = abs(nd[mine] - ref[(cell, theirs)])
            coh.append({"cell": cell, "config": mine, "recomputed": nd[mine], "block_a": ref[(cell, theirs)], "abs_diff": d, "ok": d <= 1e-12})
        print(cell, "done", flush=True)
    for path, rs in ((OUT / "coherence_block_a.csv", coh), (OUT / "metrics.csv", rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    if not all(r["ok"] for r in coh):
        sys.exit("coherence with block_a metrics failed")
    print(f"coherence {sum(r['ok'] for r in coh)}/{len(coh)} ok", flush=True)


if __name__ == "__main__":
    main()
