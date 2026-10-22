"""N1 — TransE scores after 1, 2, 5 and 10 training epochs on the nine shared ranking-panel cells.

Run from the repository root with KG_FORCE_DEVICE=cpu:
    python reviewer_controls/scripts/stage3_transe_epochs.py reviewer_controls/frozen_inputs <out_dir>

Mirrors src/experiments/post_analysis_capture.py: same perturbation (seed 42), same 10,000-triple sample, same
cached TransE hyperparameters and PyKEEN seed 42, ABLATION_BATCH; only the number of epochs varies, and training
runs on CPU so the new run is deterministic. Each epoch budget is a separate training from the same seed.
The sample is checked against the persisted one (SHA-256 and order) before training. Labels are not used.
"""
from __future__ import annotations

import gzip
import json
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

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CELLS = [l.split() for l in ((Path(__file__).resolve().parent / "panel_cells.txt").read_text().splitlines() if l.strip()]
EPOCHS = (1, 2, 5, 10)


def main():
    cache_dir = Path(".cache/hyperparams")
    for ds, vt, n in CELLS:
        base = DataLoader().load_dataset(ds)
        perturbed, viol, _ = KGPerturbator(base, seed=42).perturb_logical(
            violation_type=vt, n_violations=int(n), dataset_name=ds)
        sample, _ = sample_triples(perturbed, viol, 10000, seed=42)
        sha = triples_sha(sample)
        rec = json.load(gzip.open(FROZEN / "canonical_persist" / "samples" / f"{sha[:24]}.json.gz", "rt"))
        assert [list(t) for t in rec["triples"]] == [[t.subject, t.relation, t.object] for t in sample], (ds, vt, n)
        hp = json.loads((cache_dir / f"{ds}_TransE.json").read_text())["best_params"]
        scores = {}
        for e in EPOCHS:
            t0 = time.time()
            det = PyKEENDetector(embedding_dim=hp["embedding_dim"])
            det.train_model(sample, model_name="TransE", epochs=e, batch_size=ABLATION_BATCH,
                            learning_rate=hp.get("learning_rate", 0.001), margin=hp.get("margin", 1.0),
                            negative_sampling=hp.get("negative_sampling", 5))
            _ = det.detect_ranking_anomalies(sample)
            scores[f"raw_e{e}"] = np.asarray(det._last_raw_scores, dtype=float)
            print(f"{ds} {vt} {n} epochs={e} device={det.device} {time.time() - t0:.1f}s", flush=True)
        np.savez_compressed(OUT / f"{ds}__{vt}_{n}__transe_epochs.npz", sha=sha, **scores)


if __name__ == "__main__":
    main()
