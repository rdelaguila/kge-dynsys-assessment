"""Extra training runs (CPU, deterministic) for tests 2 and 3.

Run from the repository root with KG_FORCE_DEVICE=cpu:
    python reviewer_controls/scripts/stage7_run_extra.py random5   <dataset> reviewer_controls/work
    python reviewer_controls/scripts/stage7_run_extra.py trainseed <dataset> reviewer_controls/work

random5   : TransE (cached dimension, 10 epochs, PyKEEN seed 42) on the paired random-5% samples, seeds 1-10.
trainseed : TransE (cached dimension, 10 epochs) on the paired asymmetry-10% samples, seeds 1-10, with PyKEEN
            training seeds 43 and 44 (seed 42 is the Block B run). The published pipeline fixes random_seed=42;
            this script overrides it by wrapping pykeen.pipeline.pipeline, without editing the package code.
Each sample is regenerated with the producer's call and checked against the stage-1 SHA-256. Labels are not used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pykeen.pipeline as _pp

import data as _data
_data._cache_load = lambda path: None

from ablation_study import KGPerturbator
from canonical_persist import triples_sha
from config import ABLATION_BATCH
from data import DataLoader
from pykeen_detector import PyKEENDetector
from run_all import sample_triples

MODE, DS, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
_original_pipeline = _pp.pipeline
_SEED = {"value": 42}


def _pipeline_with_seed(*args, **kwargs):
    kwargs["random_seed"] = _SEED["value"]
    return _original_pipeline(*args, **kwargs)


_pp.pipeline = _pipeline_with_seed


def train(sample, hp, seed):
    _SEED["value"] = seed
    det = PyKEENDetector(embedding_dim=hp["embedding_dim"])
    det.train_model(sample, model_name="TransE", epochs=10, batch_size=ABLATION_BATCH,
                    learning_rate=hp.get("learning_rate", 0.001), margin=hp.get("margin", 1.0),
                    negative_sampling=hp.get("negative_sampling", 5))
    _ = det.detect_ranking_anomalies(sample)
    return np.asarray(det._last_raw_scores, float), det.device


def main():
    base = DataLoader().load_dataset(DS)
    hp = json.loads(Path(f".cache/hyperparams/{DS}_TransE.json").read_text())["best_params"]
    out_dir = OUT / "extra_raw"
    out_dir.mkdir(exist_ok=True)
    for s in range(1, 11):
        target = out_dir / f"{MODE}__{DS}__seed{s}.npz"
        if target.exists():
            continue
        t0 = time.time()
        pert = KGPerturbator(base, seed=s)
        if MODE == "random5":
            perturbed, viol = pert.perturb(percentage=5.0, perturbation_types=["entity_swap"])
            tag, seeds = f"random5_seed{s}", (42,)
        else:
            perturbed, viol, _ = pert.perturb_logical(violation_type="asymmetry",
                                                      n_violations=max(10, int(len(base) * 0.10)), dataset_name=DS)
            tag, seeds = f"asym10_seed{s}", (43, 44)
        sample, _ = sample_triples(perturbed, viol, 10000, seed=s)
        inp = np.load(OUT / "inputs" / f"{DS}__{tag}.npz")
        assert triples_sha(sample) == str(inp["sha"]), (MODE, DS, s)
        res = {}
        for seed in seeds:
            raw, dev = train(sample, hp, seed)
            res[f"TransE_seed{seed}_raw"] = raw
            print(f"{MODE} {DS} sample seed {s} pykeen seed {seed} device={dev} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(target, sha=str(inp["sha"]), **res)


if __name__ == "__main__":
    main()
