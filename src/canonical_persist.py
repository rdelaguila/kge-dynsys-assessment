"""Persistence of evaluation samples and detections for the canonical run.

Enabled only when the environment variable ``CANONICAL_PERSIST_DIR`` is set;
otherwise every function is a no-op. Samples are stored once, keyed by the
SHA-256 of their sorted triples, so every producer that evaluates on the same
(dataset, perturbation, seed) can be checked to share the exact sample.
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path


def _root():
    d = os.environ.get("CANONICAL_PERSIST_DIR")
    return Path(d) if d else None


def triples_sha(triples) -> str:
    rows = sorted("\t".join(t) if isinstance(t, tuple) else f"{t.subject}\t{t.relation}\t{t.object}" for t in triples)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _caller():
    for f in inspect.stack()[2:]:
        if "canonical_persist" not in f.filename and "run_all.py" not in f.filename:
            return f"{Path(f.filename).name}:{f.function}"
    return "run_all.py"


def _atomic_gzip_json(path: Path, obj) -> None:
    """Write via a unique temp file and rename, so concurrent producers never interleave."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def record_sample(sampled, labels, perturbation_seed, evaluation_seed):
    root = _root()
    if root is None:
        return None
    triples = [(t.subject, t.relation, t.object) for t in sampled]
    sha = triples_sha(triples)
    lab_sha = triples_sha(sorted(labels))
    d = root / "samples"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sha[:24]}.json.gz"
    if not path.exists():
        _atomic_gzip_json(path, {"triples_sha256": sha, "labels_sha256": lab_sha, "n": len(triples), "n_pos": len(labels),
                                 "perturbation_seed": int(perturbation_seed), "evaluation_seed": int(evaluation_seed),
                                 "triples": triples, "labels": sorted(map(list, labels))})
    with open(root / "samples_index.jsonl", "a") as f:
        f.write(json.dumps({"producer": _caller(), "triples_sha256": sha, "labels_sha256": lab_sha,
                            "n": len(triples), "n_pos": len(labels), "perturbation_seed": int(perturbation_seed),
                            "evaluation_seed": int(evaluation_seed), "context": os.environ.get("CANONICAL_CONTEXT", "")}) + "\n")
    return sha


def record_detection(kind: str, dataset: str, cell: str, method: str, sample_triples, labels, detected,
                     scores=None, threshold=None, operator=None, params=None, extra=None, device=None):
    """Persist one detector's full result on one evaluation sample."""
    root = _root()
    if root is None:
        return None
    triples = [(t.subject, t.relation, t.object) if not isinstance(t, tuple) else t for t in sample_triples]
    det = set(detected)
    lab = set(labels)
    n = len(triples)
    tp, fp, fn = len(det & lab), len(det - lab), len(lab - det)
    import time
    config_fp = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16] if params else None
    rec = {"run_id": os.environ.get("CANONICAL_RUN_ID"), "code_commit": os.environ.get("CANONICAL_CODE_COMMIT"),
           "context": os.environ.get("CANONICAL_CONTEXT"), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "device": device, "config_fingerprint": config_fp,
           "kind": kind, "dataset": dataset, "cell": cell, "method": method,
           "sample_triples_sha256": triples_sha(triples), "labels_sha256": triples_sha(sorted(lab)),
           "n": n, "n_pos": len(lab), "tp": tp, "fp": fp, "fn": fn, "tn": n - tp - fp - fn,
           "n_detected": len(det), "threshold": threshold, "operator": operator, "params": params,
           "detected": sorted(map(list, det)), **(extra or {})}
    if scores is not None:
        rec["scores"] = {"\t".join(k): float(v) for k, v in scores.items()}
    d = root / "detections" / dataset / kind
    d.mkdir(parents=True, exist_ok=True)
    _atomic_gzip_json(d / f"{cell}__{method}.json.gz", rec)
    return rec
