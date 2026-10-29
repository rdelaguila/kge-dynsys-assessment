"""Diagnostic D2: A1 (lambda_asym_override=0.0) for WN18RR asymmetry-15 % seed 42, no metrics.

Run from the repository root with PYTHONHASHSEED=0:
    python reviewer_controls/round2/scripts/diag_a1_order.py <out.npy> <nocache|cache> <a1_only|a2_then_a1>
Same calls as r2_asym15_run.dynamics(). Saves the A1 energies at the sample positions, to compare with the frozen
progress/20260623/ode_postfix_A1_WN18RR_asymmetry_13950.npz and check whether A1 depends on what ran before it in
the same process.
"""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
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
if sys.argv[3] == "a2_then_a1":
    kg = build_kgode_postfix("WN18RR", pert, seed=42, device="cpu")
    per_triple_energy(kg, rk4_terminal(kg, 50))
kg = build_kgode_postfix("WN18RR", pert, seed=42, device="cpu", lambda_asym_override=0.0)
e = per_triple_energy(kg, rk4_terminal(kg, 50))
idx = {(t.subject, t.relation, t.object): i for i, t in enumerate(kg.triples)}
np.save(sys.argv[1], np.array([e[idx[(t.subject, t.relation, t.object)]] for t in sample]))
