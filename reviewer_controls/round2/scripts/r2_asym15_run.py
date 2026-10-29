"""R2 panel 1 — WN18RR asymmetry-15% realisations with the original 15% protocol (round-2 protocol; see ../README.md).

Run from the repository root with PYTHONHASHSEED=0:
    python reviewer_controls/round2/scripts/r2_asym15_run.py <out_dir> <seed> [<seed> ...]

Per perturbation seed it composes existing functions only:
  perturbation  ablation_study.KGPerturbator(base, seed).perturb_logical("asymmetry", 13950, "WN18RR")
  sample        run_all.sample_triples(perturbed, viol, 10000, seed)
  A2 / A1       milestone_postfix_section4_rerun.build_kgode_postfix(seed=42, cpu[, lambda_asym_override=0.0])
                + rk4_terminal(50) + per_triple_energy, mapped to sample positions as detect_one_cell does
  KGE           post_analysis_capture's call: PyKEENDetector(cache dim).train_model(sample, ABLATION_EPOCHS,
                ABLATION_BATCH, cached lr/margin/neg) on the auto backend, then detect_ranking_anomalies
  degree        stage1_inputs' incidence count on the full perturbed graph, plus head and tail separately
Raw scores, triples, labels and the sample SHA are persisted per seed; no metric is computed here.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np

import data as _data
_data._cache_load = lambda path: None  # constraint-structure cache keys can collide across seeds

from ablation_study import KGPerturbator
from canonical_persist import triples_sha
from config import ABLATION_BATCH, ABLATION_EPOCHS, CACHE_DIR
from data import DataLoader
from milestone_postfix_section4_rerun import build_kgode_postfix, per_triple_energy, rk4_terminal
from pykeen_detector import PyKEENDetector
from run_all import sample_triples

DS, VT, N_VIOL, N_STEPS = "WN18RR", "asymmetry", 13950, 50
KGE_METHODS = ("TransE", "DistMult", "MuRE")
OUT = Path(sys.argv[1])
SEEDS = [int(s) for s in sys.argv[2:]]


def dynamics(perturbed, sample, lambda_asym_override):
    kgode = build_kgode_postfix(DS, perturbed, seed=42, device="cpu", lambda_asym_override=lambda_asym_override)
    energies_pool = per_triple_energy(kgode, rk4_terminal(kgode, n_steps=N_STEPS))
    triple_to_idx = {(t.subject, t.relation, t.object): i for i, t in enumerate(kgode.triples)}
    idx = [triple_to_idx.get((t.subject, t.relation, t.object)) for t in sample]
    return np.array([np.nan if i is None else float(energies_pool[i]) for i in idx]), str(kgode.device)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = DataLoader().load_dataset(DS)
    for seed in SEEDS:
        target = OUT / f"{DS}_asym15_seed{seed}.npz"
        if target.exists():
            print(f"seed {seed}: already done", flush=True)
            continue
        t0, timing, meta = time.time(), {}, {}
        perturbed, viol, _ = KGPerturbator(base, seed=seed).perturb_logical(
            violation_type=VT, n_violations=N_VIOL, dataset_name=DS)
        sample, sample_viol = sample_triples(perturbed, viol, 10000, seed=seed)
        T = [(t.subject, t.relation, t.object) for t in sample]
        gt = np.array([k in sample_viol for k in T], dtype=bool)
        res = {}
        for name, override in (("A2", None), ("A1", 0.0)):
            t1 = time.time()
            res[name], meta[f"{name}_device"] = dynamics(perturbed, sample, override)
            timing[name] = time.time() - t1
            print(f"seed {seed} {name} done ({timing[name]:.0f}s)", flush=True)
        for kge in KGE_METHODS:
            t1 = time.time()
            hp = json.loads((CACHE_DIR / f"{DS}_{kge}.json").read_text())["best_params"]
            det = PyKEENDetector(embedding_dim=hp["embedding_dim"])
            det.train_model(sample, model_name=kge, epochs=ABLATION_EPOCHS, batch_size=ABLATION_BATCH,
                            learning_rate=hp.get("learning_rate", 0.001), margin=hp.get("margin", 1.0),
                            negative_sampling=hp.get("negative_sampling", 5))
            _ = det.detect_ranking_anomalies(sample)
            res[f"{kge}_raw"] = np.asarray(det._last_raw_scores, dtype=float)
            meta[f"{kge}_device"] = str(getattr(det, "last_device", det.device))
            meta[f"{kge}_hp"] = hp
            timing[kge] = time.time() - t1
            print(f"seed {seed} {kge} device={meta[f'{kge}_device']} ({timing[kge]:.0f}s)", flush=True)
        deg = collections.Counter()
        for t in perturbed:
            deg[t.subject] += 1
            deg[t.object] += 1
        res["deg_h"] = np.array([deg[h] for h, _, _ in T], dtype=float)
        res["deg_t"] = np.array([deg[o] for _, _, o in T], dtype=float)
        res["deg_sum"] = res["deg_h"] + res["deg_t"]
        meta.update({"seed": seed, "n_perturbed": len(perturbed), "n_injected": len(viol),
                     "n_unique_perturbed": len({(t.subject, t.relation, t.object) for t in perturbed}),
                     "n_positions": len(T), "n_unique_sample": len(set(T)), "P_positions": int(gt.sum()),
                     "P_unique": len(sample_viol), "pool_sha": triples_sha(perturbed),
                     "timing_s": timing, "total_s": time.time() - t0,
                     "ablation_epochs": ABLATION_EPOCHS, "ablation_batch": ABLATION_BATCH, "n_steps_rk4": N_STEPS})
        np.savez_compressed(target, triples=np.array(T), gt=gt, sha=triples_sha(sample),
                            meta=json.dumps(meta), **res)
        print(f"seed {seed} done in {time.time() - t0:.0f}s sha={triples_sha(sample)[:16]} P={int(gt.sum())}", flush=True)


if __name__ == "__main__":
    main()
