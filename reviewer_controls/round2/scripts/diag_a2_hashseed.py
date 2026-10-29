"""Diagnostic D1/D3: A2 (cached lambda_asym) for WN18RR asymmetry-15 % seed 42, no metrics.

Run from the repository root:
    python reviewer_controls/round2/scripts/diag_a2_hashseed.py <out.npy> <nocache|cache>
Same calls as r2_asym15_run.dynamics(). Run it in separate processes with and without PYTHONHASHSEED=0 to check
cross-process reproducibility, and with <cache> to read the constraint structures from .cache/ (cache path).
"""
import sys; sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
import numpy as np
import data as _data
if sys.argv[2] == "nocache": _data._cache_load = lambda p: None
from ablation_study import KGPerturbator
from data import DataLoader
from milestone_postfix_section4_rerun import build_kgode_postfix, per_triple_energy, rk4_terminal
from run_all import sample_triples
base = DataLoader().load_dataset("WN18RR")
pert, viol, _ = KGPerturbator(base, seed=42).perturb_logical(violation_type="asymmetry", n_violations=13950, dataset_name="WN18RR")
sample, _ = sample_triples(pert, viol, 10000, seed=42)
kg = build_kgode_postfix("WN18RR", pert, seed=42, device="cpu")
e = per_triple_energy(kg, rk4_terminal(kg, 50))
idx = {(t.subject,t.relation,t.object): i for i,t in enumerate(kg.triples)}
np.save(sys.argv[1], np.array([e[idx[(t.subject,t.relation,t.object)]] for t in sample]))
