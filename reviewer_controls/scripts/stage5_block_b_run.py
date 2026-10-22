"""Block B (new execution) — per-triple S1/S2/S3 residuals and TransE scores on the paired asymmetry-10% samples.

Run from the repository root with KG_FORCE_DEVICE=cpu:
    python reviewer_controls/scripts/stage5_block_b_run.py <dataset> reviewer_controls/work

For seeds 1-10 the script regenerates the perturbed graph and evaluation sample exactly as
src/experiments/run_s0_s4_progressive_ablation.py does (checked against stage-1 inputs), then:
  * integrates S1 (logic=0, asym=0), S2 (logic=ref, asym=0) and S3 (logic=ref, asym=ref) with the same builder,
    seeding (np.random.seed / torch.manual_seed = perturbation seed before each build) and 100 RK4 steps,
    storing terminal residuals for every position and, for S1, residuals after 10, 25 and 50 steps as well;
  * trains TransE on CPU for 10 epochs on the same 10,000-triple sample (transductive, PyKEEN seed 42, cached
    hyperparameters, ABLATION_BATCH) at the cached dimension and at d = 50, storing raw scores per position.
No label enters any score. The joblib constraint-structure cache is disabled.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "src/experiments")

import numpy as np
import torch

import data as _data
_data._cache_load = lambda path: None

from ablation_study import KGPerturbator
from canonical_persist import triples_sha
from config import ABLATION_BATCH
from data import DataLoader
from pykeen_detector import PyKEENDetector
from run_all import _load_hyperparam_cache, sample_triples
from run_s0_s4_progressive_ablation import N_STEPS, build_kgode, residual_for_terminal_state

DS, OUT = sys.argv[1], Path(sys.argv[2])
CHECKPOINTS = (10, 25, 50)


def rk4_with_checkpoints(kgode, n_steps, checkpoints):
    """Identical arithmetic to run_s0_s4_progressive_ablation.rk4_integrate, recording intermediate states."""
    t0, t1 = kgode.config.t_span
    dt = (t1 - t0) / n_steps
    y = kgode.y0.clone()
    t = float(t0)
    saved = {}
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            k1 = kgode.ode_dynamics(t, y)
            k2 = kgode.ode_dynamics(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = kgode.ode_dynamics(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = kgode.ode_dynamics(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            if step in checkpoints:
                saved[step] = y.clone()
    return y, saved


def main():
    base = DataLoader().load_dataset(DS)
    n_viol = max(10, int(len(base) * 0.10))
    bp = _load_hyperparam_cache(DS, "ode")
    lam_logic, lam_asym = bp["lambda_logic"], float(bp.get("lambda_asym", 1.0))
    hp = json.loads(Path(f".cache/hyperparams/{DS}_TransE.json").read_text())["best_params"]
    out_dir = OUT / "block_b_raw"
    out_dir.mkdir(exist_ok=True)
    for seed in range(1, 11):
        target = out_dir / f"{DS}__seed{seed}.npz"
        if target.exists():
            print(f"{DS} seed {seed}: already done", flush=True)
            continue
        t0 = time.time()
        perturbed, viol, _ = KGPerturbator(base, seed=seed).perturb_logical(
            violation_type="asymmetry", n_violations=n_viol, dataset_name=DS)
        sample, _ = sample_triples(perturbed, viol, 10000, seed=seed)
        inp = np.load(OUT / "inputs" / f"{DS}__asym10_seed{seed}.npz")
        assert triples_sha(sample) == str(inp["sha"]), (DS, seed, "sample")
        res = {}
        for stage, ll, la in (("S1", 0.0, 0.0), ("S2", lam_logic, 0.0), ("S3", lam_logic, lam_asym)):
            np.random.seed(seed); torch.manual_seed(seed)
            kg = build_kgode(DS, ll, la, perturbed)
            y, saved = rk4_with_checkpoints(kg, N_STEPS, CHECKPOINTS if stage == "S1" else ())
            res[stage] = residual_for_terminal_state(kg, y, sample)
            for step, ys in saved.items():
                res[f"S1_step{step}"] = residual_for_terminal_state(kg, ys, sample)
            print(f"{DS} seed {seed} {stage} integrated ({time.time() - t0:.0f}s)", flush=True)
        for name, dim in (("TransE", hp["embedding_dim"]), ("TransE_d50", 50)):
            det = PyKEENDetector(embedding_dim=dim)
            det.train_model(sample, model_name="TransE", epochs=10, batch_size=ABLATION_BATCH,
                            learning_rate=hp.get("learning_rate", 0.001), margin=hp.get("margin", 1.0),
                            negative_sampling=hp.get("negative_sampling", 5))
            _ = det.detect_ranking_anomalies(sample)
            res[f"{name}_raw"] = np.asarray(det._last_raw_scores, float)
            print(f"{DS} seed {seed} {name} d={dim} device={det.device} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(target, sha=str(inp["sha"]), **res)
        print(f"{DS} seed {seed} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
