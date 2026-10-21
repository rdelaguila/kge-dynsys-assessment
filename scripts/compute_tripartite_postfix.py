#!/usr/bin/env python
"""Recompute the §4.8 tripartite cross-method overlap (Table 9) under the
post-§3.8.6-fix Neural ODE dynamics.

Reuses `audit_phase2_E_rerun::rerun_native_ode` to regenerate the ODE
detection set on the unperturbed graph, then loads the KGE detection sets
from `paper_results/<ds>/<model>/detections.json` (those are produced by
the canonical native pipeline of `src/run_all.py` and are unaffected by the
§3.8.6 fix because KGE training does not depend on the embedding-dict path).

Outputs:
  progress/tripartite_postfix.json — per-dataset counts and the Universal
  Consensus triple lists.
  progress/tripartite_postfix.md   — human-readable summary table (Table 9).

Usage (from repo root):
  .venv/bin/python scripts/compute_tripartite_postfix.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse the audit helpers — they already implement the post-fix native ODE
# re-run and the KGE detection loader. See scripts/audit_phase2_E_rerun.py.
from audit_phase2_E_rerun import rerun_native_ode, load_kge_detection  # noqa: E402

DATASETS = ("FB15k-237", "codex-m", "WN18RR")
KGE_MODELS = ("TransE", "DistMult", "MuRE", "ComplEx")


def load_persisted_native_a2(ds: str):
    """Native A2 set persisted by rerun_native_ode in the canonical run, or None."""
    import gzip, os
    root = os.environ.get("CANONICAL_PERSIST_DIR")
    path = Path(root) / "detections" / ds / "native" / "native_rk4_50__A2.json.gz" if root else None
    if path is None or not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        return {tuple(t) for t in json.load(f)["detected"]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    out_rows: list[dict] = []
    universal_consensus: dict[str, list[tuple[str, str, str]]] = {}

    for ds in args.datasets:
        t0 = time.time()
        ode_set: set[tuple[str, str, str]] = load_persisted_native_a2(ds)
        if ode_set is None:
            if os.environ.get("CANONICAL_PERSIST_DIR"):  # canonical run: the persisted native A2 set is mandatory
                raise SystemExit(f"{ds}: persisted native A2 set missing; run audit_phase2_E_rerun.py first")
            ode_set = rerun_native_ode(ds, n_steps=50)["detected_set"]

        kge_sets: dict[str, set[tuple[str, str, str]]] = {}
        for kge in KGE_MODELS:
            kd = load_kge_detection(ds, kge)
            if kd is None:
                if os.environ.get("CANONICAL_PERSIST_DIR"):  # canonical run: all four KGE sets are mandatory
                    raise SystemExit(f"{ds}/{kge}/detections.json missing")
                print(f"  [warn] {ds}/{kge}/detections.json missing — skipped")
                continue
            # load_kge_detection returns a dict {(s,r,o): score} — every key
            # is, by construction, a detection; cast to set.
            kge_sets[kge] = set(kd.keys())

        union_kge: set[tuple[str, str, str]] = set().union(*kge_sets.values()) if kge_sets else set()

        confirmed = ode_set & union_kge
        fp_risk = union_kge - ode_set
        fn_risk = ode_set - union_kge

        # Universal Consensus = ODE ∩ TransE ∩ DistMult ∩ MuRE ∩ ComplEx
        # NB: copy() is essential — `&=` mutates the left operand in place.
        uc: set[tuple[str, str, str]] = ode_set.copy()
        for kge in KGE_MODELS:
            if kge in kge_sets:
                uc &= kge_sets[kge]
        universal_consensus[ds] = sorted(uc)
        from canonical_persist import record_detection
        for name, members in (("confirmed", confirmed), ("kge_only", fp_risk), ("a2_only", fn_risk),
                              ("union_kge", union_kge), ("all_method", uc)):
            record_detection("consensus", ds, "native_rk4_50", name, [], set(), members)

        row = {
            "dataset": ds,
            "ode_n": len(ode_set),
            **{f"{kge}_n": len(kge_sets[kge]) for kge in kge_sets},
            "union_kge_n": len(union_kge),
            "confirmed_n": len(confirmed),
            "fp_risk_n": len(fp_risk),
            "fn_risk_n": len(fn_risk),
            "universal_consensus_n": len(uc),
            "wall_clock_s": time.time() - t0,
        }
        out_rows.append(row)
        print(f"\n  {ds}: ODE={row['ode_n']}, ⋃KGE={row['union_kge_n']}, "
              f"Confirmed={row['confirmed_n']}, FP Risk={row['fp_risk_n']}, "
              f"FN Risk={row['fn_risk_n']}, UC={row['universal_consensus_n']}")

    out = {
        "protocol": {
            "n_steps_rk4": 50,
            "fix_active": True,
            "kge_consensus": list(KGE_MODELS),
            "universal_consensus_def": "ODE ∩ TransE ∩ DistMult ∩ MuRE ∩ ComplEx",
        },
        "rows": out_rows,
        "universal_consensus_triples": {
            ds: [list(t) for t in triples] for ds, triples in universal_consensus.items()
        },
    }

    out_json = REPO_ROOT / "progress" / "tripartite_postfix.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPersisted: {out_json}")

    # Render Markdown summary
    out_md_lines = [
        "# Table 9 — Tripartite Classification (post-§3.8.6 fix)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')} by `scripts/compute_tripartite_postfix.py`",
        "",
        "**Definitions**",
        "- Confirmed Errors = ODE ∩ ⋃KGE",
        "- FP Risk = ⋃KGE \\ ODE",
        "- FN Risk = ODE \\ ⋃KGE",
        "- Universal Consensus = ODE ∩ TransE ∩ DistMult ∩ MuRE ∩ ComplEx",
        "",
        "| Dataset | ODE | ⋃KGE | Confirmed | FP Risk | FN Risk | Universal Consensus |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out_rows:
        out_md_lines.append(
            f"| {r['dataset']} | {r['ode_n']:,} | {r['union_kge_n']:,} | "
            f"{r['confirmed_n']:,} | {r['fp_risk_n']:,} | {r['fn_risk_n']:,} | "
            f"{r['universal_consensus_n']:,} |"
        )
    total_confirmed = sum(r["confirmed_n"] for r in out_rows)
    total_fp = sum(r["fp_risk_n"] for r in out_rows)
    total_fn = sum(r["fn_risk_n"] for r in out_rows)
    out_md_lines.append(
        f"| **TOTAL** | — | — | **{total_confirmed:,}** | **{total_fp:,}** | **{total_fn:,}** | — |"
    )

    out_md = REPO_ROOT / "progress" / "tripartite_postfix.md"
    with out_md.open("w") as f:
        f.write("\n".join(out_md_lines) + "\n")
    print(f"Persisted: {out_md}")


if __name__ == "__main__":
    main()
