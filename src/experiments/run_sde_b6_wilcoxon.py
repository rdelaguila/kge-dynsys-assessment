"""B.6 — 10-seed paired Wilcoxon panel for the SDE per-triple variance observable.

Runs run_sde_pivot_v2.py for seeds 1..n_seeds with ``initialise_relations=True``
(the post-r=0 protocol), collects per-seed (F1_ref, F1_zero, Δ F1), and reports
the paired Wilcoxon signed-rank statistic with rank-biserial effect size, plus
the canonical descriptive statistics.

Output:
  - progress/sde_b6_wilcoxon_WN18RR.json — full per-seed + aggregate result.
  - progress/sde_b6_wilcoxon_WN18RR.md — human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_one_seed(seed: int, dataset: str, k: int, sigma: float, n_steps: int,
                 sample_size: int, level_pct: float) -> dict:
    """Run the v2 driver for one seed and read back its JSON summary."""
    log_path = REPO_ROOT / "logs" / f"sde_b6_seed_{seed}.log"
    summary_path = REPO_ROOT / "progress" / "sde_pivot_WN18RR_asymmetry_v2.json"

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(REPO_ROOT / "src" / "experiments" / "run_sde_pivot_v2.py"),
        "--dataset", dataset,
        "--seed", str(seed),
        "--k", str(k),
        "--sigma", str(sigma),
        "--n-steps", str(n_steps),
        "--sample-size", str(sample_size),
        "--level-pct", str(level_pct),
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"seed": seed, "ok": False, "error": f"return code {proc.returncode}", "elapsed_s": elapsed}

    with open(summary_path) as f:
        summary = json.load(f)

    # Archive the per-seed JSON so the next iteration's write doesn't overwrite it.
    seed_summary = REPO_ROOT / "progress" / f"sde_b6_WN18RR_seed_{seed}.json"
    shutil.copy(summary_path, seed_summary)

    f1_ref = summary["f1"]["ref"]["f1"]
    f1_zero = summary["f1"]["zero"]["f1"]
    var_mean_ref = summary["configs"]["ref"]["variance_mean"]
    var_mean_zero = summary["configs"]["zero"]["variance_mean"]
    delta = f1_ref - f1_zero
    return {
        "seed": seed, "ok": True, "elapsed_s": elapsed,
        "f1_ref": f1_ref, "f1_zero": f1_zero, "delta_f1": delta,
        "variance_mean_ref": var_mean_ref, "variance_mean_zero": var_mean_zero,
        "n_detected_ref": summary["f1"]["ref"]["n_detected"],
        "n_detected_zero": summary["f1"]["zero"]["n_detected"],
        "threshold_ref": summary["f1"]["ref"]["threshold"],
        "threshold_zero": summary["f1"]["zero"]["threshold"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="WN18RR")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--level-pct", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 72)
    print(f"# B.6 Wilcoxon panel — {args.dataset} asym @{args.level_pct}% | "
          f"seeds 1..{args.n_seeds} | σ={args.sigma}, k={args.k}, n_steps={args.n_steps}")
    print("=" * 72)

    results = []
    t_panel = time.time()
    for seed in range(1, args.n_seeds + 1):
        print(f"\n--- seed {seed}/{args.n_seeds} ---", flush=True)
        r = run_one_seed(seed, args.dataset, args.k, args.sigma, args.n_steps,
                         args.sample_size, args.level_pct)
        results.append(r)
        if r["ok"]:
            print(f"  done in {r['elapsed_s']:.1f}s | "
                  f"F1 ref={r['f1_ref']:.4f}, zero={r['f1_zero']:.4f}, Δ={r['delta_f1']:+.4f}",
                  flush=True)
        else:
            print(f"  FAILED: {r.get('error')}", flush=True)
    panel_elapsed = time.time() - t_panel

    ok = [r for r in results if r["ok"]]
    if len(ok) < 2:
        print(f"\nNot enough successful seeds ({len(ok)}); aborting Wilcoxon.")
        return

    deltas = np.array([r["delta_f1"] for r in ok])
    f1_refs = np.array([r["f1_ref"] for r in ok])
    f1_zeros = np.array([r["f1_zero"] for r in ok])

    # Wilcoxon paired signed-rank (two-sided)
    try:
        wstat, pval = wilcoxon(f1_refs, f1_zeros, alternative="two-sided", zero_method="pratt")
    except ValueError as e:
        wstat, pval = float("nan"), float("nan")
        print(f"Wilcoxon failed: {e}")

    # Rank-biserial effect size: r = (#positive_diffs - #negative_diffs) / n_nonzero
    n_pos = int((deltas > 0).sum())
    n_neg = int((deltas < 0).sum())
    n_zero = int((deltas == 0).sum())
    n_nz = n_pos + n_neg
    rb = (n_pos - n_neg) / max(n_nz, 1)

    # Holm-Bonferroni — single comparison here, so p_holm = p.
    p_holm = pval

    summary = {
        "protocol": {
            "dataset": args.dataset,
            "sigma": args.sigma,
            "k_trajectories": args.k,
            "n_steps": args.n_steps,
            "sample_size": args.sample_size,
            "level_pct": args.level_pct,
            "n_seeds": args.n_seeds,
            "panel_wall_clock_s": panel_elapsed,
        },
        "per_seed": ok,
        "stats": {
            "mean_delta_f1": float(deltas.mean()),
            "median_delta_f1": float(np.median(deltas)),
            "stddev_delta_f1": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_zero": n_zero,
            "rank_biserial": float(rb),
            "wilcoxon_W": float(wstat) if not np.isnan(wstat) else None,
            "wilcoxon_p_two_sided": float(pval) if not np.isnan(pval) else None,
            "p_holm": float(p_holm) if not np.isnan(p_holm) else None,
            "supported_at_alpha_005": bool((not np.isnan(pval)) and (pval < 0.05)),
        },
    }

    out_json = REPO_ROOT / "progress" / f"sde_b6_wilcoxon_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown summary
    out_md = REPO_ROOT / "progress" / f"sde_b6_wilcoxon_{args.dataset}.md"
    lines = [
        f"# B.6 Wilcoxon panel — {args.dataset} asym @{args.level_pct}%",
        "",
        f"**Protocol**: σ={args.sigma}, k={args.k}, n_steps={args.n_steps}, "
        f"sample_size={args.sample_size}, seeds 1..{args.n_seeds}.",
        f"**Panel wall-clock**: {panel_elapsed:.1f} s ({panel_elapsed/60:.1f} min).",
        f"**Driver**: `src/experiments/run_sde_b6_wilcoxon.py` (initialise_relations=True).",
        "",
        "## Per-seed",
        "",
        "| seed | F1 ref | F1 zero | Δ F1 | var_mean ref | var_mean zero | wall-clock (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ok:
        lines.append(
            f"| {r['seed']} | {r['f1_ref']:.5f} | {r['f1_zero']:.5f} | "
            f"{r['delta_f1']:+.5f} | {r['variance_mean_ref']:.3f} | "
            f"{r['variance_mean_zero']:.3f} | {r['elapsed_s']:.1f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- Mean Δ F1: **{summary['stats']['mean_delta_f1']:+.5f}**",
        f"- Median Δ F1: **{summary['stats']['median_delta_f1']:+.5f}**",
        f"- Stddev: {summary['stats']['stddev_delta_f1']:.5f}",
        f"- Sign distribution: + {n_pos}, − {n_neg}, 0 {n_zero}",
        f"- Rank-biserial: **{rb:+.3f}**",
        f"- Wilcoxon W: {wstat if not np.isnan(wstat) else 'NaN'}",
        f"- p (two-sided): **{pval if not np.isnan(pval) else 'NaN'}**",
        f"- p_Holm (1 test): same as raw p here",
        f"- Supported at α=0.05: **{summary['stats']['supported_at_alpha_005']}**",
        "",
    ]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 72)
    print("PANEL SUMMARY")
    print("=" * 72)
    print(f"  successful seeds: {len(ok)}/{args.n_seeds}")
    print(f"  mean Δ F1:        {summary['stats']['mean_delta_f1']:+.5f}")
    print(f"  median Δ F1:      {summary['stats']['median_delta_f1']:+.5f}")
    print(f"  sign +/-/0:       {n_pos}/{n_neg}/{n_zero}")
    print(f"  rank-biserial:    {rb:+.3f}")
    print(f"  Wilcoxon p:       {pval}")
    print(f"  supported α=0.05: {summary['stats']['supported_at_alpha_005']}")
    print(f"  wall-clock:       {panel_elapsed:.1f} s")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
