"""R2 panel 2 (Gate P) — TransE trained on the full perturbed pool, scored on the paired asymmetry-10% positions
(round-2 protocol; see ../README.md).

Run from the repository root with KG_FORCE_DEVICE=cpu PYTHONHASHSEED=0:
    python reviewer_controls/round2/scripts/r2_fullpool_transe_run.py <inputs_dir> <out_dir> <dataset> <seed> [<seed> ...]

Same call and configuration as stage5_block_b_run.py's TransE (cached dimension / lr / margin / negatives,
10 epochs, ABLATION_BATCH, PyKEEN seed 42, CPU); the only change is the training population: the complete
perturbed pool instead of the 10,000-triple sample. The sample is regenerated with the producer's call and checked
against the stage-1 SHA-256. The loader's effective population and the training budget are recorded.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

import data as _data
_data._cache_load = lambda path: None

from ablation_study import KGPerturbator
from canonical_persist import triples_sha
from config import ABLATION_BATCH
from data import DataLoader
from pykeen_detector import PyKEENDetector
from run_all import sample_triples

INP, OUT, DS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SEEDS = [int(s) for s in sys.argv[4:]]
EPOCHS = 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = DataLoader().load_dataset(DS)
    n_viol = max(10, int(len(base) * 0.10))
    hp = json.loads(Path(f".cache/hyperparams/{DS}_TransE.json").read_text())["best_params"]
    for seed in SEEDS:
        target = OUT / f"{DS}__seed{seed}.npz"
        if target.exists():
            print(f"{DS} seed {seed}: already done", flush=True)
            continue
        t0 = time.time()
        perturbed, viol, _ = KGPerturbator(base, seed=seed).perturb_logical(
            violation_type="asymmetry", n_violations=n_viol, dataset_name=DS)
        sample, _ = sample_triples(perturbed, viol, 10000, seed=seed)
        inp = np.load(INP / f"{DS}__asym10_seed{seed}.npz")
        assert triples_sha(sample) == str(inp["sha"]), (DS, seed, "sample")
        det = PyKEENDetector(embedding_dim=hp["embedding_dim"])
        t1 = time.time()
        det.train_model(perturbed, model_name="TransE", epochs=EPOCHS, batch_size=ABLATION_BATCH,
                        learning_rate=hp.get("learning_rate", 0.001), margin=hp.get("margin", 1.0),
                        negative_sampling=hp.get("negative_sampling", 5))
        t_train = time.time() - t1
        _ = det.detect_ranking_anomalies(sample)
        raw = np.asarray(det._last_raw_scores, float)
        tf = det.tf
        n_eff = int(tf.num_triples)
        meta = {"dataset": DS, "seed": seed, "device": str(det.device), "hp": hp, "epochs": EPOCHS,
                "batch_size": ABLATION_BATCH, "pool_size": len(perturbed),
                "pool_unique": len({(t.subject, t.relation, t.object) for t in perturbed}),
                "loader_num_triples": n_eff, "loader_num_entities": int(tf.num_entities),
                "loader_num_relations": int(tf.num_relations), "batches_per_epoch": math.ceil(n_eff / ABLATION_BATCH),
                "updates_total": EPOCHS * math.ceil(n_eff / ABLATION_BATCH),
                "sample_batches_per_epoch": math.ceil(10000 / ABLATION_BATCH), "train_seconds": t_train,
                "total_seconds": time.time() - t0, "pool_sha": triples_sha(perturbed), "sample_sha": str(inp["sha"])}
        np.savez_compressed(target, sha=str(inp["sha"]), TransE_full_raw=raw, meta=json.dumps(meta))
        print(json.dumps({k: v for k, v in meta.items() if k != "hp"}), flush=True)


if __name__ == "__main__":
    main()
