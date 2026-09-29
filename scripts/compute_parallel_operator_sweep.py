"""Parallel-operator sweep — F1 under the four (ODE-operator × KGE-operator) combinations.

Read-only over persisted per-triple arrays. No model invoked, no re-integration.

For each of the 9 stratified ablation_audit cells:
  - ODE per-triple energy: A2 from progress/20260623/ode_postfix_A2_<ds>_<vt>_<n>.npz
  - KGE per-triple scores (TransE / DistMult / MuRE): from
      paper_results/<ds>/ablation_audit/<vt>_<n>.json
  - Ground truth: is_violation column of the same JSON

For each detector (ODE-A2 + 3 KGEs) we compute F1 against is_violation under
two thresholding operators:
  - knee  : KneeLocator(convex, decreasing) on the sorted score curve;
            fallback = 95th percentile if kneed returns None.
  - Otsu  : pure-numpy Otsu on the (KGE-negated for anomaly-high) score.

The claim in §3.6 is that swapping the operators (ODE-Otsu / KGE-knee) is
"uniformly worse". This script produces the table that either confirms or
refutes that.

Output:
  progress/parallel_operator_sweep.json  (per-cell per-detector per-operator F1)
  progress/parallel_operator_sweep.md    (rendered as Supplementary §S2.3)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from kneed import KneeLocator

REPO_ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("WN18RR",    "transitivity",  4650),
    ("codex-m",   "cardinality",  30930),
    ("WN18RR",    "asymmetry",   13950),
    ("FB15k-237", "transitivity", 46517),
    ("WN18RR",    "cardinality", 13950),
    ("codex-m",   "asymmetry",   20620),
    ("codex-m",   "transitivity", 10310),
    ("codex-m",   "cardinality",  10310),
    ("FB15k-237", "cardinality",  15505),
]
KGE_METHODS = ("TransE", "DistMult", "MuRE")


def otsu_threshold(values: np.ndarray) -> float:
    """Pure-numpy Otsu (mirrors src/pykeen_detector.py::_otsu_threshold)."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    lo, hi = float(values.min()), float(values.max())
    if hi == lo:
        return lo
    n_bins = 256
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(values, bins=bin_edges)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return (lo + hi) / 2
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(n_bins))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2 = np.nan_to_num(sigma_b2, nan=-np.inf)
    k = int(np.argmax(sigma_b2))
    return float(bin_edges[k])


def knee_threshold(values: np.ndarray) -> float:
    """Knee on descending sorted values; fallback = 95th percentile."""
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    sorted_e = np.sort(values)[::-1]
    xs = np.arange(len(sorted_e))
    try:
        knl = KneeLocator(xs, sorted_e, curve="convex", direction="decreasing",
                          interp_method="polynomial")
        if knl.knee is not None:
            return float(sorted_e[int(knl.knee)])
    except Exception:
        pass
    return float(np.percentile(values, 95))


def f1_score(scores: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    """F1 for detection = score > thr (higher score = more anomalous)."""
    finite = np.isfinite(scores)
    s = scores[finite]
    y = gt[finite].astype(bool)
    det = s > thr
    tp = int((det & y).sum())
    fp = int((det & ~y).sum())
    fn = int((~det & y).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
            "n_detected": int(det.sum()), "threshold": thr}


def load_cell(ds: str, vtype: str, n: int) -> dict:
    """Load ODE A2 per-triple (post-fix) + KGE per-triple + GT for one cell."""
    # KGE + GT from ablation_audit JSON
    p = REPO_ROOT / "paper_results" / ds / "ablation_audit" / f"{vtype}_{n}.json"
    d = json.loads(p.read_text())
    triples = d["triples"]
    gt = np.array([t["is_violation"] for t in triples], dtype=bool)
    kge_scores = {
        kge: -np.array([t["kge_raw"][kge] for t in triples], dtype=float)   # negate: higher = anomalous
        for kge in KGE_METHODS
    }

    # ODE A2 from post-fix npz (aligned to same eval order by construction — both use seed=42)
    npz_path = REPO_ROOT / "progress" / "20260623" / f"ode_postfix_A2_{ds}_{vtype}_{n}.npz"
    z = np.load(npz_path, allow_pickle=True)
    ode_energies = np.asarray(z["eval_energies"], dtype=float)
    ode_gt = np.asarray(z["gt"], dtype=bool)

    # Sanity: same GT ordering
    if len(ode_gt) != len(gt):
        raise RuntimeError(f"length mismatch: audit={len(gt)} vs npz={len(ode_gt)}")
    if not np.array_equal(ode_gt, gt):
        raise RuntimeError(f"GT mismatch on {ds}/{vtype}/{n}")

    return {"gt": gt, "ode_A2": ode_energies, "kge_scores": kge_scores,
            "n_eval": len(triples), "n_pos": int(gt.sum())}


def sweep_cell(cell_data: dict) -> dict:
    """Compute F1 under both operators for ODE-A2 and each KGE."""
    out = {}
    # ODE-A2 under knee (canonical) and Otsu (parallel-swap)
    ode = cell_data["ode_A2"]
    ode_valid = ode[np.isfinite(ode)]
    thr_knee = knee_threshold(ode)
    thr_otsu = otsu_threshold(ode_valid)
    out["ODE-A2"] = {
        "knee":  f1_score(ode, cell_data["gt"], thr_knee),
        "otsu":  f1_score(ode, cell_data["gt"], thr_otsu),
    }
    # KGE under Otsu (canonical) and knee (parallel-swap)
    for kge, scores in cell_data["kge_scores"].items():
        s_valid = scores[np.isfinite(scores)]
        thr_o = otsu_threshold(s_valid)
        thr_k = knee_threshold(scores)
        out[kge] = {
            "otsu": f1_score(scores, cell_data["gt"], thr_o),
            "knee": f1_score(scores, cell_data["gt"], thr_k),
        }
    return out


def main():
    print("Parallel-operator sweep on 9 ablation_audit cells")
    print("=" * 70)
    rows = []
    for ds, vt, n in CELLS:
        try:
            cell = load_cell(ds, vt, n)
        except Exception as e:
            print(f"  SKIP {ds}/{vt}/{n}: {e}")
            continue
        sweep = sweep_cell(cell)
        row = {"dataset": ds, "vtype": vt, "level_n": n,
               "n_eval": cell["n_eval"], "n_pos": cell["n_pos"],
               "prevalence": cell["n_pos"] / cell["n_eval"],
               "detectors": sweep}
        rows.append(row)
        pct = 100 * cell["n_pos"] / cell["n_eval"]
        print(f"  {ds:12s}/{vt:13s}/n={n:>5d} prev={pct:5.1f}%")
        for det, d in sweep.items():
            canonical_op = "knee" if det == "ODE-A2" else "otsu"
            parallel_op  = "otsu" if det == "ODE-A2" else "knee"
            f1_c = d[canonical_op]["f1"]
            f1_p = d[parallel_op]["f1"]
            delta = f1_p - f1_c
            marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"    {det:>8s}: canonical({canonical_op}) F1={f1_c:.3f} | parallel({parallel_op}) F1={f1_p:.3f}  Δ={delta:+.3f} {marker}")

    out_path = REPO_ROOT / "progress" / "parallel_operator_sweep.json"
    out_path.write_text(json.dumps({
        "protocol": {
            "description": "F1 under canonical (ODE-knee, KGE-Otsu) vs parallel-swap (ODE-Otsu, KGE-knee) operators.",
            "cells": len(rows), "seed": 42,
            "notes": "ODE-A2 per-triple energies from progress/20260623/ode_postfix_A2_*.npz. "
                     "KGE per-triple scores (TransE/DistMult/MuRE, negated) from "
                     "paper_results/<ds>/ablation_audit/<vt>_<n>.json. "
                     "ComplEx excluded (not persisted per §3.5).",
        },
        "cells": rows,
    }, indent=2))
    print(f"\nwrote {out_path}")

    # Also emit a markdown table for direct inclusion in Supplementary §S2.3
    md_path = REPO_ROOT / "progress" / "parallel_operator_sweep.md"
    md = ["# Parallel-operator sweep\n",
          "F1 under two operator pairs on the 9 stratified ablation cells.\n",
          "- **Canonical**: ODE under knee; KGE under Otsu (both on the negated KGE score).",
          "- **Parallel-swap**: ODE under Otsu; KGE under knee.",
          "\n**Δ column** = F1(parallel-swap) − F1(canonical) per (cell, detector). Negative Δ means the canonical operator wins on that cell.\n"]
    md.append("| Cell | prev | ODE-A2 knee | ODE-A2 Otsu | Δ | TransE Otsu | TransE knee | Δ | DistMult Otsu | DistMult knee | Δ | MuRE Otsu | MuRE knee | Δ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["detectors"]
        line = f"| {r['dataset']}/{r['vtype']}@n={r['level_n']} | {r['prevalence']:.3f} "
        for det, canonical_op, parallel_op in [("ODE-A2", "knee", "otsu"),
                                                 ("TransE",  "otsu", "knee"),
                                                 ("DistMult","otsu", "knee"),
                                                 ("MuRE",    "otsu", "knee")]:
            f1_c = d[det][canonical_op]["f1"]
            f1_p = d[det][parallel_op]["f1"]
            line += f"| {f1_c:.3f} | {f1_p:.3f} | {f1_p - f1_c:+.3f} "
        line += "|"
        md.append(line)
    # Aggregate summary
    md.append("\n## Aggregate\n")
    n_wins_canon = 0
    n_wins_parallel = 0
    n_tie = 0
    for r in rows:
        d = r["detectors"]
        for det, cop, pop in [("ODE-A2","knee","otsu"),("TransE","otsu","knee"),
                              ("DistMult","otsu","knee"),("MuRE","otsu","knee")]:
            delta = d[det][pop]["f1"] - d[det][cop]["f1"]
            if abs(delta) < 1e-6: n_tie += 1
            elif delta > 0: n_wins_parallel += 1
            else: n_wins_canon += 1
    total = n_wins_canon + n_wins_parallel + n_tie
    md.append(f"Across {len(rows)} cells × 4 detectors = {total} (cell, detector) comparisons:\n")
    md.append(f"- **Canonical operator wins**: {n_wins_canon}/{total} ({100*n_wins_canon/total:.1f}%)")
    md.append(f"- **Parallel-swap wins**: {n_wins_parallel}/{total} ({100*n_wins_parallel/total:.1f}%)")
    md.append(f"- **Tie (Δ < 10⁻⁶)**: {n_tie}/{total}\n")
    md_path.write_text("\n".join(md))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
