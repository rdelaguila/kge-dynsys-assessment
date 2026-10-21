"""
Capture per-cell audit data for the 9 stratified B4.4 cells (Option 2-improved).

For each cell, deterministically reconstruct eval_triples + violation_set
and capture per-triple raw scores from every KGE method and the ODE
energy distribution. Output one JSON per cell at:

    paper_results/<dataset>/ablation_audit/<violation>_<n>.json

Schema per JSON:
{
  "dataset": str,
  "violation_type": "asymmetry" | "transitivity" | "cardinality",
  "level_n": int,                 # n_violations parameter
  "n_eval": int,                  # 10000
  "n_violations_in_eval": int,    # |violation_set ∩ eval|
  "triples": [
    {"s": str, "r": str, "o": str,
     "is_violation": bool,
     "ode_energy": float,         # pre-threshold ODE energy
     "kge_raw": {"TransE": float, "DistMult": float, "MuRE": float}}
  ]
}

Mirrors run_ablation_stage's per-cell logic with seed=42 throughout
(KGPerturbator, sample_triples, KGE training random_seed=42 via PyKEEN
pipeline, ODE deterministic on the cached lambdas). MPS-introduced
±1-3 triple noise vs B4.4's stored metrics is expected and documented.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from data import DataLoader, Triple
from ablation_study import KGPerturbator
from run_all import sample_triples, get_constraint_config
from integrated_framework import IntegratedKGDebugger
from pykeen_detector import PyKEENDetector
from config import CACHE_DIR, ABLATION_EPOCHS, ABLATION_BATCH


# 9 stratified cells: (dataset, vtype, level_n)
CELLS = [
    # ODE-WINS regime
    ("WN18RR",    "transitivity",  4650),
    ("codex-m",   "cardinality",  30930),
    ("WN18RR",    "asymmetry",   13950),
    # ODE-GREY regime
    ("FB15k-237", "transitivity", 46517),
    ("WN18RR",    "cardinality", 13950),
    ("codex-m",   "asymmetry",   20620),
    # ODE-COLLAPSED regime
    ("codex-m",   "transitivity", 10310),
    ("codex-m",   "cardinality",  10310),
    ("FB15k-237", "cardinality",  15505),
]
KGE_METHODS = ("TransE", "DistMult", "MuRE")
SAMPLE_SIZE = 10_000


def cell_already_done(ds: str, vtype: str, level_n: int) -> Path | None:
    p = Path(f"paper_results/{ds}/ablation_audit/{vtype}_{level_n}.json")
    return p if p.exists() else None


def capture_cell(ds: str, vtype: str, level_n: int) -> dict:
    print(f"\n{'='*70}\n# {ds} / {vtype} / n={level_n}\n{'='*70}")
    t_cell0 = time.time()

    loader = DataLoader()
    triples = loader.load_dataset(ds)
    print(f"  loaded {len(triples)} triples")

    # 1) Deterministic perturbation (seed=42) — matches run_ablation_stage.
    perturbator = KGPerturbator(triples, seed=42)
    perturbed_triples, violation_set, _ = perturbator.perturb_logical(
        violation_type=vtype, n_violations=level_n, dataset_name=ds,
    )
    if not violation_set:
        return {"error": "no violations generated"}
    print(f"  perturbed: {len(perturbed_triples)}; violations injected: {len(violation_set)}")

    # 2) Stratified down-sampling to 10k (seed=42).
    eval_triples, eval_viol = sample_triples(
        perturbed_triples, violation_set, SAMPLE_SIZE, seed=42,
    )
    print(f"  eval: {len(eval_triples)} triples; violations in eval: {len(eval_viol)}")

    # 3) KGE per-method raw scores via PyKEENDetector ABLATION_EPOCHS retrain.
    kge_raw_per_method: dict[str, np.ndarray] = {}
    for kge in KGE_METHODS:
        cache = json.loads((CACHE_DIR / f"{ds}_{kge}.json").read_text())["best_params"]
        det = PyKEENDetector(embedding_dim=cache["embedding_dim"])
        t0 = time.time()
        det.train_model(
            eval_triples, model_name=kge,
            epochs=ABLATION_EPOCHS, batch_size=ABLATION_BATCH,
            learning_rate=cache.get("learning_rate", 0.001),
            margin=cache.get("margin", 1.0),
            negative_sampling=cache.get("negative_sampling", 5),
        )
        # _last_raw_scores is filled by detect_ranking_anomalies; trigger it.
        _ = det.detect_ranking_anomalies(eval_triples)
        raw = det._last_raw_scores
        kge_raw_per_method[kge] = np.asarray(raw, dtype=float)
        print(f"  {kge}: {len(raw)} raw scores in {time.time() - t0:.1f}s")

    # 4) ODE solve + per-triple energy capture (pre-threshold).
    cached_ode = json.loads((CACHE_DIR / f"{ds}_ode.json").read_text())["best_params"]
    config = IntegratedKGDebugger.get_default_config()
    config["constraint_config"] = get_constraint_config(ds)
    config["use_ode"] = True
    config["use_pykeen"] = False
    config["ode_config"] = dict(cached_ode)
    if isinstance(config["ode_config"].get("t_span"), list):
        config["ode_config"]["t_span"] = tuple(config["ode_config"]["t_span"])
    n_eval_viol = len(eval_viol)
    config["ode_config"]["top_k"] = n_eval_viol * 5  # match run_ablation_stage logical sizing

    debugger = IntegratedKGDebugger(config)
    t0 = time.time()
    debugger.run_complete_analysis(eval_triples, full_triples=perturbed_triples)
    print(f"  ODE solve+detect in {time.time() - t0:.1f}s")

    # Recover per-triple raw ODE energy from the AnomalyDetector.
    # The IntegratedKGDebugger discards raw_energies after thresholding;
    # we recompute from a fresh AnomalyDetector pass on the same solution.
    from anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(
        debugger.ode_system, debugger.ode_system.solution,
        constraint_checker=debugger.constraint_checker,
        full_triples=perturbed_triples,
        symbolic_bonus=None,  # auto-calibrate (matches run_ablation_stage)
    )
    eff_bonus = float(detector.symbolic_bonus)
    reports = detector.detect_all_anomalies(top_k=None)
    # ODE energy = anomaly_score - n_viol * effective_bonus  (matches the
    # subtraction at integrated_framework.py:307-312).
    energy_by_triple: dict[tuple, float] = {}
    for r in reports:
        n_viol = sum(1 for tt in r.anomaly_types if tt.startswith("violates_"))
        e = r.anomaly_score - n_viol * eff_bonus
        key = (r.triple.subject, r.triple.relation, r.triple.object)
        energy_by_triple[key] = float(e)
    print(f"  ODE per-triple energies: {len(energy_by_triple)} (out of {len(eval_triples)} eval)")

    # 5) Assemble per-triple audit rows.
    out_triples = []
    for i, t in enumerate(eval_triples):
        key = (t.subject, t.relation, t.object)
        kge_per = {k: float(kge_raw_per_method[k][i]) for k in KGE_METHODS}
        row = {
            "s": t.subject, "r": t.relation, "o": t.object,
            "is_violation": key in eval_viol,
            "ode_energy": energy_by_triple.get(key),  # may be None if absent
            "kge_raw": kge_per,
        }
        out_triples.append(row)

    out = {
        "dataset": ds,
        "violation_type": vtype,
        "level_n": int(level_n),
        "n_eval": len(eval_triples),
        "n_violations_in_eval": len(eval_viol),
        "calibrated_bonus": eff_bonus,
        "triples": out_triples,
    }

    audit_path = Path(f"paper_results/{ds}/ablation_audit/{vtype}_{level_n}.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(out, indent=None))
    print(f"  wrote {audit_path} (cell wall-clock {time.time() - t_cell0:.1f}s)")
    return out


def main() -> None:
    print(f"# Capturing audit data for {len(CELLS)} cells")
    t0 = time.time()
    failed = []
    for i, (ds, vtype, n) in enumerate(CELLS, 1):
        existing = cell_already_done(ds, vtype, n)
        if existing is not None:
            print(f"\n[{i}/{len(CELLS)}] {ds}/{vtype}/{n} — already captured at {existing}, skipping")
            continue
        print(f"\n[{i}/{len(CELLS)}] capturing {ds}/{vtype}/{n}")
        try:
            capture_cell(ds, vtype, n)
        except Exception as e:
            import traceback
            print(f"  ❌ {ds}/{vtype}/{n} failed: {e}")
            traceback.print_exc()
            failed.append(f"{ds}/{vtype}/{n}")
            continue
    print(f"\n# All done in {time.time() - t0:.1f}s")
    if failed:  # a missing cell must not look like success to a scheduler
        sys.exit(f"capture failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
