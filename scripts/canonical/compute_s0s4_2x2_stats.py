"""Section 4.5 paired statistics: S0-S4 sequential contrasts (Holm-12) and the
WN18RR 2x2 simple effects (Holm-4) plus difference-of-differences.

Reads the per-seed outputs of run_s0_s4_progressive_ablation.py and
run_c2_dynamics_symbolic_2x2.py; writes
progress/r2_1_progressive_ablation/stats_summary.json and
progress/milestone_c2_dynamics_symbolic_2x2_WN18RR_stats.json.
Two-sided exact Wilcoxon (zero differences dropped), r_rb = (W+ - W-)/(W+ + W-),
95% percentile bootstrap CI of the mean difference (10,000 resamples, seed recorded).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from statistical_tests import bootstrap_ci, holm_bonferroni, paired_wilcoxon  # noqa: E402

BOOT_SEED = 12345
STAGES = ["S0_zero_integration", "S1_data_only", "S2_data_sym_trans", "S3_plus_antisymmetry", "S4_plus_checker_bonus"]
DATASETS = ["FB15k-237", "codex-m", "WN18RR"]


def contrast(diff):
    w = paired_wilcoxon(list(diff), [0.0] * len(diff))
    ci = bootstrap_ci(list(diff), rng_seed=BOOT_SEED)
    return {"mean": float(np.mean(diff)), "sd": float(np.std(diff, ddof=1)), "ci95_lo": ci["ci_lo"], "ci95_hi": ci["ci_hi"],
            "rank_biserial": w.get("rank_biserial", 0.0), "p_raw": w.get("p_value", 1.0), "n_negative": w.get("n_negative"),
            "n_positive": w.get("n_positive"), "n_zero": w.get("n_zero"), "per_seed": [float(d) for d in diff]}


def progressive():
    base = REPO_ROOT / "progress" / "r2_1_progressive_ablation"
    rows = []
    for ds in DATASETS:
        seeds = sorted((json.loads(p.read_text()) for p in base.glob(f"{ds}_seed_*.json")), key=lambda r: r["seed"])
        assert len(seeds) == 10 and all(s.get("ok", True) for s in seeds), ds
        f1 = {st: np.array([s["stages"][st]["f1"] for s in seeds]) for st in STAGES}
        for a, b in zip(STAGES[:-1], STAGES[1:]):
            rows.append({"dataset": ds, "contrast": f"{b[:2]}-{a[:2]}", "seeds": [s["seed"] for s in seeds], **contrast(f1[b] - f1[a])})
    for r, p in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
        r["p_holm_12"] = p
    out = {"family": "S0-S4 sequential contrasts, 3 datasets x 4, Holm-12", "bootstrap_seed": BOOT_SEED, "rows": rows}
    (base / "stats_summary.json").write_text(json.dumps(out, indent=2))
    return out


def two_by_two():
    src = REPO_ROOT / "progress" / "milestone_c2_dynamics_symbolic_2x2_WN18RR.json"
    x = json.loads(src.read_text())
    col = lambda n: np.array([p[n] for p in x["per_seed"]])  # noqa: E731
    ro, rf, zo, zf = col("f1_ref_bonus_on"), col("f1_ref_bonus_off"), col("f1_zero_bonus_on"), col("f1_zero_bonus_off")
    simple = {"force effect, bonus off": rf - zf, "force effect, bonus on": ro - zo,
              "bonus effect, force inactive": zo - zf, "bonus effect, force active": ro - rf}
    rows = [{"contrast": k, **contrast(v)} for k, v in simple.items()]
    for r, p in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
        r["p_holm_4"] = p
    out = {"source": str(src.relative_to(REPO_ROOT)), "bootstrap_seed": BOOT_SEED,
           "cell_means": {k: float(v.mean()) for k, v in (("f1_zero_bonus_off", zf), ("f1_zero_bonus_on", zo),
                                                          ("f1_ref_bonus_off", rf), ("f1_ref_bonus_on", ro))},
           "simple_effects_holm4": rows, "difference_of_differences": contrast((ro - zo) - (rf - zf))}
    (REPO_ROOT / "progress" / "milestone_c2_dynamics_symbolic_2x2_WN18RR_stats.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    progressive()
    two_by_two()
    print("written stats_summary.json and milestone_c2_dynamics_symbolic_2x2_WN18RR_stats.json")
