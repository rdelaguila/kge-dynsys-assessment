"""Fase 2 del audit r0-v2 — re-run de E2/E4/E5/E6 post-fix.

Estrategia:
  - El fix está en `IntegratedKGDebugger.setup_ode_detector`. Los detection
    sets KGE no dependen del bug (PyKEEN entrena de forma independiente).
    Las únicas magnitudes que cambian post-fix son las del ODE.
  - Por tanto, re-corremos *solo* el ODE side post-fix sobre el grafo base
    de cada dataset; las KGE detections en `paper_results/<ds>/<model>/`
    se reutilizan tal cual.

E4 BC heterogeneity:
  - Compute BC sobre raw_energies post-fix por dataset.
  - Comparar contra publicado (FB=0.79, codex=0.16, WN=0.40).

E5 Pearson(ODE, −KGE) per-triple:
  - Re-align ODE scores post-fix con scores KGE existing por (s,r,o).
  - Compute Pearson per cell; agregar max|ρ|, mean ρ, cells crossing |ρ|>0.4.

E2 Jaccard arquitectural disjointness:
  - Re-define top-K (5%) ODE post-fix; intersect con top-K KGE existing.
  - Reportar Jaccard per (dataset × KGE_method).

E6 triviality_gap (focused, single-seed):
  - Re-correr random ablation @5% sobre cada dataset con ODE post-fix.
  - Compare F1_ODE vs F1_KGE (KGE F1 ya conocido de los artifacts).
  - Single seed only en este audit (la bootstrap row completa es work
    de un milestone follow-up); el objetivo es verificar dirección
    cualitativa.

Output: progress/audit_phase2_E_rerun.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from scipy.stats import pearsonr

from data import (
    DataLoader, Triple, ConstraintChecker,
    TransitivityConstraint, SymmetryConstraint, AntisymmetryConstraint,
)
from ode_system import KGODESystem, ODEConfig
from run_all import get_constraint_config, _load_hyperparam_cache


# -----------------------------------------------------------------------------
# Build helpers
# -----------------------------------------------------------------------------

def build_cc(dataset: str, present_relations: set):
    cfg = get_constraint_config(dataset)
    cc = ConstraintChecker()
    for r in cfg.get("transitivity_relations", []):
        if r in present_relations:
            cc.add_constraint(TransitivityConstraint(r, weight=1.0))
    for r in cfg.get("symmetry_relations", []):
        if r in present_relations:
            cc.add_constraint(SymmetryConstraint(r, weight=1.0))
    for r in cfg.get("asymmetry_relations", []):
        if r in present_relations:
            cc.add_constraint(AntisymmetryConstraint(r, weight=1.0, margin=1.0,
                                                      treats_as_gradient=True))
    return cc


def build_kgode_postfix(dataset: str, triples, seed: int = 42, device: str = "cpu"):
    """Build a KGODESystem POST-FIX: embeddings dict includes BOTH entities
    AND relations (matching the canonical post-fix state)."""
    bp = _load_hyperparam_cache(dataset, "ode")
    D = bp["embedding_dim"]

    entities = sorted({t.subject for t in triples} | {t.object for t in triples})
    relations = sorted({t.relation for t in triples})

    np.random.seed(seed)
    torch.manual_seed(seed)
    emb = {e: (np.random.randn(D) * 0.1).astype(np.float32) for e in entities}
    for r in relations:
        emb[r] = (np.random.randn(D) * 0.1).astype(np.float32)

    cc = build_cc(dataset, set(relations))
    odecfg = ODEConfig(
        t_span=tuple(bp["t_span"]) if isinstance(bp["t_span"], list) else bp["t_span"],
        lambda_data=bp["lambda_data"], lambda_logic=bp["lambda_logic"],
        lambda_asym=bp.get("lambda_asym", 1.0),
        margin_asym=bp.get("margin_asym", 1.0),
        lambda_reg=bp["lambda_reg"],
        rtol=bp.get("rtol", 5e-3), atol=bp.get("atol", 1e-6),
        device=device,
    )
    return KGODESystem(emb, triples, cc, odecfg), bp


def rk4_terminal(kgode, n_steps: int):
    t0, t1 = kgode.config.t_span
    dt = (t1 - t0) / n_steps
    y = kgode.y0.clone()
    t = float(t0)
    with torch.no_grad():
        for _ in range(n_steps):
            k1 = kgode.ode_dynamics(t, y)
            k2 = kgode.ode_dynamics(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = kgode.ode_dynamics(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = kgode.ode_dynamics(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
    return y


def per_triple_energy(kgode, y_T: torch.Tensor) -> torch.Tensor:
    """Per-triple ‖h + r − t‖² at terminal, vectorised."""
    H = y_T[kgode.s_indices]
    T = y_T[kgode.o_indices]
    R = kgode.r_vectors
    return ((H + R - T) ** 2).sum(dim=-1)


def knee_threshold(scores: np.ndarray) -> float:
    from kneed import KneeLocator
    n = len(scores)
    sorted_s = np.sort(scores)
    try:
        kl = KneeLocator(np.arange(n), sorted_s, curve="convex",
                         direction="increasing", interp_method="polynomial")
        knee_idx = int(kl.knee) if kl.knee is not None else int(0.95 * n)
    except Exception:
        knee_idx = int(0.95 * n)
    return float(sorted_s[knee_idx])


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient. BC > 0.555 → bimodal."""
    from scipy.stats import skew, kurtosis
    n = len(x)
    if n < 3:
        return float("nan")
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)  # excess kurtosis
    denom = (k + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))
    return (g ** 2 + 1) / denom


# -----------------------------------------------------------------------------
# Re-run native ODE post-fix per dataset
# -----------------------------------------------------------------------------

def rerun_native_ode(dataset: str, n_steps: int = 100):
    print(f"\n## {dataset} — re-running native ODE post-fix")
    loader = DataLoader()
    triples = loader.load_dataset(dataset)
    print(f"  triples={len(triples)}")
    t0 = time.time()
    kgode, bp = build_kgode_postfix(dataset, triples, seed=42, device="cpu")
    print(f"  KGODESystem built ({time.time()-t0:.1f}s); "
          f"n_entities={kgode.n_entities}, t_span={kgode.config.t_span}")

    t1 = time.time()
    y_T = rk4_terminal(kgode, n_steps=n_steps)
    print(f"  RK4 integration {n_steps} steps: {time.time()-t1:.1f}s")

    energies = per_triple_energy(kgode, y_T).cpu().numpy()
    thr = knee_threshold(energies)
    detection_mask = energies > thr
    print(f"  energies: range=[{energies.min():.4e}, {energies.max():.4e}], mean={energies.mean():.4e}")
    print(f"  knee threshold: {thr:.4e}, n_detected={int(detection_mask.sum())}/{len(energies)}")

    # Build {(s,r,o) → energy} for downstream alignment with KGE
    triple_to_energy = {}
    for i, t in enumerate(kgode.triples):
        key = (t.subject, t.relation, t.object)
        triple_to_energy[key] = float(energies[i])

    return {
        "dataset": dataset,
        "n_triples": len(kgode.triples),
        "energies": energies,
        "thr_knee": thr,
        "n_detected": int(detection_mask.sum()),
        "detected_set": set(
            (t.subject, t.relation, t.object)
            for i, t in enumerate(kgode.triples)
            if detection_mask[i]
        ),
        "triple_to_energy": triple_to_energy,
        "bc": float(bimodality_coefficient(energies)),
    }


# -----------------------------------------------------------------------------
# Load existing KGE artifacts
# -----------------------------------------------------------------------------

def load_kge_detection(dataset: str, model: str):
    """Load existing KGE detection JSON and return {(s,r,o) → score} + detected set."""
    path = REPO_ROOT / "paper_results" / dataset / model / "detections.json"
    if not path.exists():
        return None
    with open(path) as f:
        dets = json.load(f)
    triple_to_score = {}
    for d in dets:
        t = d["triple"]
        key = (t["subject"], t["relation"], t["object"])
        triple_to_score[key] = float(d.get("score", d.get("anomaly_score", 0.0)))
    return triple_to_score


# -----------------------------------------------------------------------------
# E4 BC heterogeneity post-fix
# -----------------------------------------------------------------------------

def compute_e4(per_dataset):
    print("\n" + "=" * 78)
    print("# E4 — BC heterogeneity post-fix")
    print("=" * 78)
    rows = []
    for ds, r in per_dataset.items():
        print(f"  {ds}: BC post-fix = {r['bc']:.4f}")
        rows.append({"dataset": ds, "bc_post_fix": r["bc"]})
    return rows


# -----------------------------------------------------------------------------
# E5 Pearson(ODE, -KGE) per-triple
# -----------------------------------------------------------------------------

def compute_e5(per_dataset, kge_models=("TransE", "DistMult", "MuRE")):
    print("\n" + "=" * 78)
    print("# E5 — Pearson(ODE_score, -KGE_score) per-triple post-fix")
    print("=" * 78)
    rows = []
    for ds, r in per_dataset.items():
        ode_scores = r["triple_to_energy"]
        for kge in kge_models:
            kge_scores = load_kge_detection(ds, kge)
            if not kge_scores:
                continue
            # Align on intersecting keys
            common = set(ode_scores.keys()) & set(kge_scores.keys())
            if len(common) < 50:
                continue
            keys = list(common)
            o = np.array([ode_scores[k] for k in keys])
            k = np.array([-kge_scores[k] for k in keys])  # negated KGE
            rho, p = pearsonr(o, k)
            row = {"dataset": ds, "kge": kge, "n_common": len(common),
                   "pearson_rho": float(rho), "pearson_p": float(p)}
            rows.append(row)
            print(f"  {ds:12} ODE vs -{kge:10}: ρ={rho:+.4f}, p={p:.4e}, n={len(common)}")
    return rows


# -----------------------------------------------------------------------------
# E2 Jaccard top-5% disjointness (simplified to native)
# -----------------------------------------------------------------------------

def compute_e2(per_dataset, kge_models=("TransE", "DistMult", "MuRE")):
    print("\n" + "=" * 78)
    print("# E2 — Jaccard top-5% ODE vs Best-KGE (native, post-fix)")
    print("=" * 78)
    rows = []
    for ds, r in per_dataset.items():
        ode_scores = r["triple_to_energy"]
        n = len(ode_scores)
        top_k = max(1, int(0.05 * n))
        sorted_keys_ode = sorted(ode_scores.keys(), key=lambda k: -ode_scores[k])
        top_ode = set(sorted_keys_ode[:top_k])

        for kge in kge_models:
            kge_scores = load_kge_detection(ds, kge)
            if not kge_scores:
                continue
            common_keys = sorted(set(ode_scores.keys()) & set(kge_scores.keys()),
                                 key=lambda k: kge_scores[k])  # ascending: most anomalous = lowest plausibility ~ NEGATED
            # KGE "anomaly" = lowest plausibility; existing detections JSON
            # already stores anomaly scores (higher = more anomalous).
            # Re-sort by descending score:
            sorted_keys_kge = sorted(kge_scores.keys(), key=lambda k: -kge_scores[k])
            top_kge = set(sorted_keys_kge[:top_k])

            jaccard = len(top_ode & top_kge) / max(len(top_ode | top_kge), 1)
            row = {"dataset": ds, "kge": kge, "top_k": top_k,
                   "intersection": len(top_ode & top_kge),
                   "jaccard": jaccard}
            rows.append(row)
            print(f"  {ds:12} top-5% ODE ∩ {kge:10}: ∩={len(top_ode & top_kge):6} "
                  f"Jaccard={jaccard:.4f}  (top_k={top_k})")
    return rows


# -----------------------------------------------------------------------------
# E6 triviality gap (single-seed random @5%)
# -----------------------------------------------------------------------------

def compute_e6_single_seed(dataset: str, seed: int = 1, level_pct: float = 5.0,
                           sample_size: int = 10000):
    """One-cell verification of triviality_gap post-fix for the given dataset.
    Returns F1_ODE_postfix and the existing canonical KGE F1 from artifacts."""
    print(f"\n## {dataset} — E6 single-seed random@{level_pct}%, seed={seed}")
    from ablation_study import KGPerturbator
    from run_all import sample_triples

    loader = DataLoader()
    base = loader.load_dataset(dataset)
    n_perturb = max(10, int(len(base) * (level_pct / 100.0)))

    pert = KGPerturbator(base, seed=int(seed))
    perturbed, violation_set = pert.perturb(percentage=level_pct,
                                              perturbation_types=["entity_swap"])
    eval_triples, eval_viol_set = sample_triples(perturbed, violation_set,
                                                  sample_size, seed=int(seed))
    print(f"  pool={len(perturbed)}, eval={len(eval_triples)}, eval_viol={len(eval_viol_set)}")

    # Build ODE post-fix on the perturbed pool
    kgode, bp = build_kgode_postfix(dataset, perturbed, seed=int(seed), device="cpu")
    y_T = rk4_terminal(kgode, n_steps=100)
    energies_pool = per_triple_energy(kgode, y_T).cpu().numpy()
    # Index eval_triples back into the pool
    triple_to_idx = {(t.subject, t.relation, t.object): i
                     for i, t in enumerate(kgode.triples)}
    eval_energies = []
    valid = []
    for t in eval_triples:
        k = (t.subject, t.relation, t.object)
        i = triple_to_idx.get(k)
        if i is None:
            valid.append(False); eval_energies.append(np.nan)
        else:
            valid.append(True); eval_energies.append(float(energies_pool[i]))
    eval_energies = np.array(eval_energies)
    valid = np.array(valid)
    gt = np.array([(t.subject, t.relation, t.object) in eval_viol_set
                   for t in eval_triples], dtype=bool)
    finite = np.isfinite(eval_energies)
    thr = knee_threshold(eval_energies[finite])
    detected = (eval_energies > thr) & finite
    tp = int((detected & gt).sum())
    fp = int((detected & ~gt).sum())
    fn = int((~detected & gt).sum())
    p = tp / max(tp + fp, 1); rcl = tp / max(tp + fn, 1)
    f1_ode = 2 * p * rcl / max(p + rcl, 1e-10)
    print(f"  ODE post-fix F1 = {f1_ode:.4f}  (TP={tp}, FP={fp}, FN={fn}, n_det={int(detected.sum())})")
    return {
        "dataset": dataset, "seed": seed,
        "f1_ode_postfix": float(f1_ode),
        "tp": tp, "fp": fp, "fn": fn, "n_detected": int(detected.sum()),
        "n_eval": len(eval_triples), "n_eval_viol": int(gt.sum()),
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["WN18RR", "FB15k-237", "codex-m"])
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--skip-e6", action="store_true")
    args = ap.parse_args()

    print("=" * 80)
    print("# AUDIT FASE 2 — re-run E2/E4/E5/E6 post-fix")
    print("=" * 80)

    t0 = time.time()
    per_dataset = {}
    for ds in args.datasets:
        per_dataset[ds] = rerun_native_ode(ds, n_steps=args.n_steps)
    print(f"\nNative ODE rebuilds: {time.time()-t0:.1f}s total")

    e4 = compute_e4(per_dataset)
    e5 = compute_e5(per_dataset)
    e2 = compute_e2(per_dataset)

    e6 = []
    if not args.skip_e6:
        for ds in args.datasets:
            e6.append(compute_e6_single_seed(ds, seed=1))

    out = {
        "protocol": {
            "n_steps_rk4": args.n_steps,
            "datasets": args.datasets,
            "fix_active": True,
            "wall_clock_s": float(time.time() - t0),
        },
        "e4_bc": e4,
        "e5_pearson": e5,
        "e2_jaccard": e2,
        "e6_triviality_gap_singleseed": e6,
    }
    out_path = REPO_ROOT / "progress" / "audit_phase2_E_rerun.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPersisted: {out_path}")


if __name__ == "__main__":
    main()
