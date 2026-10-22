"""Knee audit — does kneed stop at an early local maximum in any dynamical result? (analysis only)

Run from the repository root:  python reviewer_controls/scripts/stage10_knee_audit.py reviewer_controls/frozen_inputs reviewer_controls/work

For every stored dynamical score vector the knee is recomputed with the convention of the family that produced the
reported result, and kneed's internals are recorded:
  grid A1/A2 (Tables 4-8, ranking panel, random-5% seeds)  progress/20260623/ode_postfix_*.npz      ascending, '>'
  native A2 (Table 14, consensus)                           canonical_persist/.../native_rk4_50__A2  ascending, '>'
  paired asymmetry-10% S1/S2/S3 (S0-S4: descending, '>=';  H_asym / 2x2: ascending, '>')  Block B residuals
A vector is flagged when kneed returns a knee at a local maximum of its difference curve that is not the curve's
global maximum (the failure documented in the reviewer-controls appendix), or when no knee is found (fallback).
The reported threshold is also checked against the persisted one where available.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from kneed import KneeLocator

FROZEN, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def audit(family, item, scores, direction, persisted_thr=None):
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if direction == "ascending":
        srt = np.sort(s)
        kl = KneeLocator(np.arange(n), srt, curve="convex", direction="increasing", interp_method="polynomial")
    else:
        srt = np.sort(s)[::-1]
        kl = KneeLocator(np.arange(n), srt, curve="convex", direction="decreasing", interp_method="polynomial")
    yd = np.asarray(kl.y_difference)
    gmax = int(np.argmax(yd))
    if direction == "ascending":
        # kneed reverses x for convex-increasing curves; the difference curve is indexed from the other end
        yd = yd[::-1]
        gmax = n - 1 - gmax
    knee = None if kl.knee is None else int(kl.knee)
    if knee is None:
        thr = float(srt[int(0.95 * n)]) if direction == "ascending" else float(np.percentile(s, 95))
    else:
        thr = float(srt[knee])
    sel = int((s > thr).sum()) if direction == "ascending" else int((s >= thr).sum())
    return {"family": family, "item": item, "convention": direction, "n": n,
            "n_local_maxima": int(len(kl.maxima_indices)), "knee_index": knee, "global_max_index": gmax,
            "knee_is_global_max": knee == gmax, "fallback_used": knee is None,
            "y_diff_at_knee": None if knee is None else float(yd[knee]), "y_diff_global_max": float(yd[gmax]),
            "selected": sel, "selected_fraction": sel / n, "threshold": thr,
            "persisted_threshold": persisted_thr,
            "matches_persisted": None if persisted_thr is None else bool(np.isclose(thr, persisted_thr, rtol=1e-6))}


def main():
    rows = []
    for p in sorted((FROZEN / "progress/20260623").glob("ode_postfix_*.npz")):
        z = np.load(p, allow_pickle=True)
        e = np.asarray(z["eval_energies"], float)
        fin = np.asarray(z["finite"]).astype(bool)
        rows.append(audit("grid " + str(z["arch"]), p.stem.replace("ode_postfix_", ""), e[fin], "ascending"))
    for ds in ("FB15k-237", "codex-m", "WN18RR"):
        d = json.load(gzip.open(FROZEN / "canonical_persist/detections" / ds / "native/native_rk4_50__A2.json.gz", "rt"))
        rows.append(audit("native A2", ds, list(d["scores"].values()), "ascending", d["threshold"]))
        for s in range(1, 11):
            z = np.load(OUT / "block_b_raw" / f"{ds}__seed{s}.npz")
            for st in ("S1", "S2", "S3"):
                rows.append(audit("paired asym-10% (S0-S4 convention)", f"{ds} seed{s} {st}", z[st], "descending"))
                if st in ("S2", "S3"):  # A1 = S2, A2 = S3 in the H_asym / 2x2 families
                    rows.append(audit("paired asym-10% (H_asym/2x2 convention)", f"{ds} seed{s} {st}", z[st], "ascending"))
    res = OUT / "review_phase1"
    res.mkdir(exist_ok=True)
    with open(res / "knee_audit_dynamical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", res / "knee_audit_dynamical.csv")


if __name__ == "__main__":
    main()
