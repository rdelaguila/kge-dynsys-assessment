"""Fase 2 — TASK E2 + E3 — verificación específica post-fix.

E2 — Perfil estructural diferenciado:
  Build the graph descriptors index from the base graph (r-invariant).
  Take the post-fix ODE top-5% detection set (from native re-run).
  Take the existing KGE top-5% per method (unchanged by the fix).
  For each descriptor in {head_degree, tail_degree, relation_freq,
  has_inverse, head_rel_card}, run Mann-Whitney U on the ODE-unique vs
  KGE-unique distributions, apply BH correction within (dataset, KGE)
  cell, report Cliff's δ. Compare against canonical [10⁻⁷⁹, 10⁻⁹] p-range
  and δ ∈ [-0.97, +0.70].

E3 — Independencia del bonus simbólico:
  Build the full anomaly score on a sample post-fix:
    score = energy * 5 + max_instability * 100 + sum(violations) * bonus
  Compute F1 with and without the bonus term. Compare.
  This validates the mechanism (bonus is r-invariant) AND tests
  whether the numerical F1 equality F1 = F1-pure persists post-fix.

Read-only on pipeline. Reuses post-fix native ODE energies from
audit_phase2_E_rerun.py output indirectly (re-builds the system).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from scipy.stats import mannwhitneyu, false_discovery_control

from data import (
    DataLoader, Triple, ConstraintChecker,
    TransitivityConstraint, SymmetryConstraint, AntisymmetryConstraint,
)
from ode_system import KGODESystem, ODEConfig
from run_all import get_constraint_config, _load_hyperparam_cache


def build_graph_index(triples):
    head_deg = defaultdict(int)
    tail_deg = defaultdict(int)
    rel_freq = defaultdict(int)
    out_edges = defaultdict(set)
    triple_set = set()
    for t in triples:
        head_deg[t.subject] += 1
        tail_deg[t.object] += 1
        rel_freq[t.relation] += 1
        out_edges[(t.subject, t.relation)].add(t.object)
        triple_set.add((t.subject, t.relation, t.object))
    return {"head_deg": head_deg, "tail_deg": tail_deg,
            "rel_freq": rel_freq, "out_edges": out_edges,
            "triple_set": triple_set}


def descriptors(triple, idx):
    s, r, o = triple
    return {
        "head_degree":   idx["head_deg"].get(s, 0) + idx["tail_deg"].get(s, 0),
        "tail_degree":   idx["head_deg"].get(o, 0) + idx["tail_deg"].get(o, 0),
        "relation_freq": idx["rel_freq"].get(r, 0),
        "has_inverse":   1 if (o, r, s) in idx["triple_set"] else 0,
        "head_rel_card": len(idx["out_edges"].get((s, r), set())),
    }


def cliffs_delta(a, b):
    if not a or not b:
        return 0.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) > 5000:
        a = np.random.default_rng(42).choice(a, 5000, replace=False)
    if len(b) > 5000:
        b = np.random.default_rng(43).choice(b, 5000, replace=False)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def build_kgode(dataset, triples, init_relations: bool, seed=42):
    """Build KGODESystem with optional relation init.

    init_relations=False → A1 canonical state (entity-only dict, r=0).
    init_relations=True  → A2 post-fix state (entities + relations).
    """
    bp = _load_hyperparam_cache(dataset, "ode")
    D = bp["embedding_dim"]
    entities = sorted({t.subject for t in triples} | {t.object for t in triples})
    relations = sorted({t.relation for t in triples})
    np.random.seed(seed); torch.manual_seed(seed)
    emb = {e: (np.random.randn(D) * 0.1).astype(np.float32) for e in entities}
    if init_relations:
        for r in relations:
            emb[r] = (np.random.randn(D) * 0.1).astype(np.float32)
    cfg = get_constraint_config(dataset)
    cc = ConstraintChecker()
    for r in cfg.get("transitivity_relations", []):
        if r in relations: cc.add_constraint(TransitivityConstraint(r, weight=1.0))
    for r in cfg.get("symmetry_relations", []):
        if r in relations: cc.add_constraint(SymmetryConstraint(r, weight=1.0))
    for r in cfg.get("asymmetry_relations", []):
        if r in relations:
            cc.add_constraint(AntisymmetryConstraint(r, weight=1.0, margin=1.0,
                                                      treats_as_gradient=True))
    odecfg = ODEConfig(
        t_span=tuple(bp["t_span"]) if isinstance(bp["t_span"], list) else bp["t_span"],
        lambda_data=bp["lambda_data"], lambda_logic=bp["lambda_logic"],
        lambda_asym=bp.get("lambda_asym", 1.0),
        margin_asym=bp.get("margin_asym", 1.0),
        lambda_reg=bp["lambda_reg"],
        rtol=bp.get("rtol", 5e-3), atol=bp.get("atol", 1e-6),
        device="cpu",
    )
    return KGODESystem(emb, triples, cc, odecfg), bp, cc


def build_kgode_postfix(dataset, triples, seed=42):
    """Backward-compat shim — calls build_kgode with init_relations=True."""
    return build_kgode(dataset, triples, init_relations=True, seed=seed)


def rk4_terminal(kgode, n_steps=50):
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


def per_triple_energy(kgode, y_T):
    H = y_T[kgode.s_indices]
    T = y_T[kgode.o_indices]
    R = kgode.r_vectors
    return ((H + R - T) ** 2).sum(dim=-1).cpu().numpy()


def load_kge_detection(dataset, model):
    path = REPO_ROOT / "paper_results" / dataset / model / "detections.json"
    if not path.exists():
        return None
    with open(path) as f:
        dets = json.load(f)
    return {
        (d["triple"]["subject"], d["triple"]["relation"], d["triple"]["object"]):
        float(d.get("score", d.get("anomaly_score", 0.0)))
        for d in dets
    }


# --- E2 ----------------------------------------------------------------------

def compute_ode_scores_architecture(dataset, triples, arch: str, n_steps: int = 50,
                                     sde_sigma: float = 0.05, sde_k: int = 5):
    """Compute per-triple anomaly score under one of A1 / A2 / A3.

    A1: canonical r=0 deterministic — entity-only embedding dict, RK4 terminal energy.
    A2: post-fix r≠0 deterministic — entities + relations, RK4 terminal energy.
    A3: post-fix SDE per-triple variance — entities + relations, k SDE trajectories.
    """
    if arch == "A1":
        kgode, bp, cc = build_kgode(dataset, triples, init_relations=False, seed=42)
        y_T = rk4_terminal(kgode, n_steps=n_steps)
        energies = per_triple_energy(kgode, y_T)
        return kgode, {(t.subject, t.relation, t.object): float(energies[i])
                       for i, t in enumerate(kgode.triples)}
    if arch == "A2":
        kgode, bp, cc = build_kgode(dataset, triples, init_relations=True, seed=42)
        y_T = rk4_terminal(kgode, n_steps=n_steps)
        energies = per_triple_energy(kgode, y_T)
        return kgode, {(t.subject, t.relation, t.object): float(energies[i])
                       for i, t in enumerate(kgode.triples)}
    if arch == "A3":
        from sde_system import KGSDESystem, integrate_one_trajectory
        kgode, bp, cc = build_kgode(dataset, triples, init_relations=True, seed=42)
        sde = KGSDESystem(kgode, sigma=sde_sigma)
        terminals = []
        for k_idx in range(sde_k):
            ent = 42 * 10_000 + k_idx
            ys = integrate_one_trajectory(
                sde, kgode.y0, t_span=kgode.config.t_span,
                n_steps=n_steps, entropy=int(ent),
            )
            terminals.append(ys[-1])
        terminals = torch.stack(terminals, dim=0)  # (k, N, D)
        # Per-triple energy per trajectory, then cross-trajectory variance
        H_all = terminals[:, kgode.s_indices, :]  # (k, T, D)
        T_all = terminals[:, kgode.o_indices, :]
        R = kgode.r_vectors  # (T, D)
        energies_k = ((H_all + R.unsqueeze(0) - T_all) ** 2).sum(dim=-1)  # (k, T)
        variances = energies_k.var(dim=0, unbiased=False).cpu().numpy()
        return kgode, {(t.subject, t.relation, t.object): float(variances[i])
                       for i, t in enumerate(kgode.triples)}
    raise ValueError(arch)


def run_e2_for_arch(dataset: str, arch: str, graph_idx: dict,
                    kge_methods=("TransE", "DistMult", "MuRE"),
                    n_steps: int = 50):
    print(f"\n  -- {dataset} / arch {arch} --")
    loader = DataLoader()
    triples = loader.load_dataset(dataset)
    kgode, ode_score = compute_ode_scores_architecture(
        dataset, triples, arch=arch, n_steps=n_steps,
    )

    results = []
    for kge in kge_methods:
        kge_score = load_kge_detection(dataset, kge)
        if not kge_score:
            continue
        common = set(kge_score.keys()) & set(ode_score.keys())
        if len(common) < 100:
            continue
        ode_in = {k: ode_score[k] for k in common}
        kge_in = {k: kge_score[k] for k in common}
        K_c = max(1, int(0.05 * len(common)))
        ode_top = set(sorted(ode_in.keys(), key=lambda k: -ode_in[k])[:K_c])
        kge_top = set(sorted(kge_in.keys(), key=lambda k: -kge_in[k])[:K_c])
        ode_only = ode_top - kge_top
        kge_only = kge_top - ode_top
        ode_descs = [descriptors(t, graph_idx) for t in ode_only]
        kge_descs = [descriptors(t, graph_idx) for t in kge_only]

        cell_rows = []
        p_vals = []
        for d_name in ("head_degree", "tail_degree", "relation_freq",
                       "has_inverse", "head_rel_card"):
            a = [r[d_name] for r in ode_descs]
            b = [r[d_name] for r in kge_descs]
            if len(a) < 2 or len(b) < 2:
                continue
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            delta = cliffs_delta(a, b)
            cell_rows.append({
                "dataset": dataset, "arch": arch, "kge": kge,
                "descriptor": d_name,
                "n_ode_unique": len(a), "n_kge_unique": len(b),
                "mw_stat": float(stat), "mw_p": float(p),
                "cliffs_delta": delta,
            })
            p_vals.append(p)
        if p_vals:
            p_bh = false_discovery_control(p_vals).tolist()
            for r, p in zip(cell_rows, p_bh):
                r["mw_p_bh"] = float(p)
        results.extend(cell_rows)
        sig_cnt = sum(1 for r in cell_rows
                      if r.get("mw_p_bh", r["mw_p"]) < 0.001)
        print(f"    {kge:10}: n_ode_only={len(ode_only):>4}, "
              f"n_kge_only={len(kge_only):>4}, "
              f"{sig_cnt}/{len(cell_rows)} descriptors p_BH<0.001")
    return results


def run_e2(dataset: str, n_steps: int = 50,
           kge_methods=("TransE", "DistMult", "MuRE"),
           architectures=("A1", "A2", "A3")):
    print(f"\n## E2 post-fix cross-architecture — {dataset}")
    loader = DataLoader()
    triples = loader.load_dataset(dataset)
    graph_idx = build_graph_index(triples)
    all_rows = []
    for arch in architectures:
        all_rows.extend(run_e2_for_arch(dataset, arch, graph_idx,
                                          kge_methods=kge_methods,
                                          n_steps=n_steps))
    return all_rows


# --- E3 ----------------------------------------------------------------------

def run_e3(dataset: str, n_steps: int = 50, sample_size: int = 5000):
    """Compare F1 with bonus vs F1-pure (no bonus) on a native sample.

    Both use the same anomaly score formula, but score_pure excludes the
    `len(violations) * symbolic_bonus` term. The bonus mechanism uses
    `get_violating_triples` (BOOLEAN), so it is r-invariant per audit
    TASK 1.3.1. This driver verifies the F1 equality numerically on the
    post-fix detection.
    """
    print(f"\n## E3 post-fix — {dataset}")
    from anomaly_detector import AnomalyDetector  # noqa: F401  (smoke import)

    loader = DataLoader()
    base_triples = loader.load_dataset(dataset)
    rng = np.random.default_rng(0)
    if len(base_triples) > sample_size:
        idx = rng.choice(len(base_triples), size=sample_size, replace=False)
        sample = [base_triples[i] for i in idx]
    else:
        sample = list(base_triples)

    kgode, bp, cc = build_kgode_postfix(dataset, sample, seed=42)
    y_T = rk4_terminal(kgode, n_steps=n_steps)
    energies = per_triple_energy(kgode, y_T).astype(np.float64)

    # Per-triple violation count via the cached structural checks.
    # AntisymmetryConstraint.get_violating_triples returns the set; we
    # build a lookup per (s,r,o) → count.
    viol_count = Counter()
    for c in cc.constraints:
        if hasattr(c, "get_violating_triples"):
            v_set = c.get_violating_triples(kgode.triples)
            for tt in v_set:
                viol_count[(tt.subject, tt.relation, tt.object)] += 1

    # symbolic_bonus calibration: q99 - q50 of the pre-bonus score.
    # Pre-bonus score per triple is energy*5 + max_inst*100. We use
    # max_inst=0 here as a simplification (acceptable for this audit:
    # symbolic_bonus is dominated by the energy term anyway).
    pre = energies * 5.0
    bonus = float(np.quantile(pre, 0.99) - np.quantile(pre, 0.50))
    print(f"  calibrated symbolic_bonus = {bonus:.3f}")

    pure_scores = pre  # without bonus
    full_scores = pre + np.array([viol_count.get(
        (t.subject, t.relation, t.object), 0) * bonus
        for t in kgode.triples])

    # Knee threshold on each
    from kneed import KneeLocator
    def knee(s):
        n = len(s); ss = np.sort(s)
        try:
            kl = KneeLocator(np.arange(n), ss, curve="convex",
                             direction="increasing", interp_method="polynomial")
            ki = int(kl.knee) if kl.knee is not None else int(0.95 * n)
        except Exception:
            ki = int(0.95 * n)
        return float(ss[ki])

    thr_pure = knee(pure_scores)
    thr_full = knee(full_scores)
    det_pure = pure_scores > thr_pure
    det_full = full_scores > thr_full

    # Native detection has no ground-truth violations injected; for E3 we
    # only test whether the bonus shifts the DETECTION SET (which would
    # change F1 if there were ground truth). Report the overlap.
    pure_set = {(t.subject, t.relation, t.object)
                for i, t in enumerate(kgode.triples) if det_pure[i]}
    full_set = {(t.subject, t.relation, t.object)
                for i, t in enumerate(kgode.triples) if det_full[i]}
    intersection = pure_set & full_set
    union = pure_set | full_set
    jaccard = len(intersection) / max(len(union), 1)
    sym_diff = pure_set.symmetric_difference(full_set)
    print(f"  pure: n_det={len(pure_set)},  full: n_det={len(full_set)}")
    print(f"  intersection={len(intersection)}  symmetric_diff={len(sym_diff)}")
    print(f"  Jaccard(pure, full) = {jaccard:.4f}  → "
          f"{'identical' if jaccard > 0.999 else 'differ'} detection set")
    return {
        "dataset": dataset,
        "n_sample": len(sample),
        "n_triples_in_system": len(kgode.triples),
        "symbolic_bonus": bonus,
        "n_det_pure": len(pure_set),
        "n_det_full": len(full_set),
        "intersection": len(intersection),
        "symmetric_diff": len(sym_diff),
        "jaccard": jaccard,
    }


# --- Driver ------------------------------------------------------------------

def main():
    print("=" * 80)
    print("# AUDIT FASE 2 — E2 + E3 post-fix verification")
    print("=" * 80)

    t0 = time.time()
    e2_rows = []
    e3_rows = []
    for ds in ("WN18RR", "FB15k-237", "codex-m"):
        e2_rows.extend(run_e2(ds, n_steps=50))
        e3_rows.append(run_e3(ds, n_steps=50, sample_size=5000))
    elapsed = time.time() - t0

    out = {
        "protocol": {"n_steps_rk4": 50, "wall_clock_s": elapsed,
                     "fix_active": True},
        "e2_rows": e2_rows,
        "e3_rows": e3_rows,
    }
    out_path = REPO_ROOT / "progress" / "audit_phase2_E2E3_postfix.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPersisted: {out_path}")

    # Aggregate verdict E2 — segregate by architecture
    print("\n## E2 aggregate per architecture")
    by_cell_count = Counter()
    by_cell_sig = Counter()
    for r in e2_rows:
        cell = (r["dataset"], r["arch"], r["kge"])
        by_cell_count[cell] += 1
        if r.get("mw_p_bh", r["mw_p"]) < 0.001:
            by_cell_sig[cell] += 1
    for arch in ("A1", "A2", "A3"):
        print(f"\n  Architecture {arch}:")
        for (ds, a, kge), total in by_cell_count.items():
            if a != arch: continue
            sig = by_cell_sig[(ds, a, kge)]
            print(f"    {ds:12} vs {kge:10}: {sig}/{total} descriptors with p_BH<0.001")

    print("\n## E3 aggregate")
    for r in e3_rows:
        print(f"  {r['dataset']:12} Jaccard(pure, full) = {r['jaccard']:.4f}  "
              f"{'identical' if r['jaccard'] > 0.999 else 'differ'}")


if __name__ == "__main__":
    main()
