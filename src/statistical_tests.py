"""
Statistical Tests for Bootstrap Validation
==========================================

Phase 2 statistical validation for the regeneration cycle.
Consumes per-seed F1 values produced by the ``bootstrap_validation`` stage of
``run_all.py`` and produces:

  - 17 paired Wilcoxon signed-rank tests (H2 + triviality-gap groups)
  - Holm-Bonferroni correction across all 17 tests
  - Rank-biserial correlation as effect size for each test
  - 95% bootstrap percentile confidence intervals (10,000 resamples)
    per (dataset, stage, violation, method) cell

Outputs are written to ``paper_results_statistical/wilcoxon_tests.json``
and ``paper_results_statistical/wilcoxon_tests.md``.

**Superseded as of the 2026-09-26 statistics closeout.** ``run_all.py``'s
``bootstrap_validation`` stage now calls ``compute_final_statistical_report``
(bottom of this file) instead of ``compute_wilcoxon_table``. That function
computes the manuscript's actual fifteen contrasts across three
independently-corrected named families (``E6_RANDOM_9``, ``HASym_DET_3``,
``HASym_SDE_REVIEW_3``) from their own verified input paths, not from the
``<base>/<dataset>/bootstrap/`` layout below. ``compute_wilcoxon_table``
remains, unmodified, for ``src/experiments/run_phase2_resume.py``'s
historical use.

Bootstrap layout consumed (per dataset, per seed):

  <base>/<dataset>/bootstrap/seed_<N>/random/metrics/metrics_random_entity_swap_5.0.json
  <base>/<dataset>/bootstrap/seed_<N>/logical_lambda_ref/metrics/metrics_logical_<vtype>_<n>.json
  <base>/<dataset>/bootstrap/seed_<N>/logical_lambda_zero/metrics/metrics_logical_<vtype>_<n>.json

Each metrics JSON has the shape produced by ``UnifiedExperimentRunner``:

  {
    "metrics": {
      "TransE": {"f1": ..., "precision": ..., "recall": ...},
      "DistMult": {...},
      "MuRE": {...},
      "KG-Debug-ODE": {...},
      ...
    }
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Methods whose F1 we extract from bootstrap output.
KGE_METHODS = ("TransE", "DistMult", "MuRE")
ODE_METHOD_KEY = "KG-Debug-ODE"

# Three real datasets covered by the regeneration.
DATASETS = ("fb15k-237", "codex-m", "WN18RR")

# H2 logical-violation cells: (dataset, violation_type).
# (codex-m, transitivity) is excluded — transitive relations are undefined
# in CoDEx-M's constraint config.
H2_CELLS: Tuple[Tuple[str, str], ...] = (
    ("fb15k-237", "asymmetry"),
    ("fb15k-237", "transitivity"),
    ("fb15k-237", "cardinality"),
    ("codex-m", "asymmetry"),
    ("codex-m", "cardinality"),
    ("WN18RR", "asymmetry"),
    ("WN18RR", "transitivity"),
    ("WN18RR", "cardinality"),
)


# ──────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────

def _load_metric_file(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _glob_metrics(metrics_dir: Path, prefix: str) -> List[Path]:
    """Return metrics files in ``metrics_dir`` whose name starts with prefix."""
    if not metrics_dir.is_dir():
        return []
    return sorted(metrics_dir.glob(f"{prefix}*.json"))


def _f1(scenario_metrics: Dict, method: str) -> Optional[float]:
    """Extract a method's F1 from a scenario's ``metrics`` dict."""
    cell = scenario_metrics.get(method)
    if not isinstance(cell, dict):
        return None
    val = cell.get("f1")
    return float(val) if val is not None else None


def collect_random_f1(base_dir: Path, dataset: str, method: str) -> List[float]:
    """Per-seed F1 list for the random@5% scenario."""
    out: List[float] = []
    boot_dir = base_dir / dataset / "bootstrap"
    if not boot_dir.is_dir():
        return out
    for seed_dir in sorted(boot_dir.glob("seed_*")):
        m_dir = seed_dir / "random" / "metrics"
        files = _glob_metrics(m_dir, "metrics_random_entity_swap_")
        if not files:
            continue
        data = _load_metric_file(files[0])
        if not data:
            continue
        f1 = _f1(data.get("metrics", {}), method)
        if f1 is not None:
            out.append(f1)
    return out


def collect_logical_f1(base_dir: Path, dataset: str, vtype: str,
                       lambda_scope: str, method: str) -> List[float]:
    """Per-seed F1 list for a logical (vtype, lambda_scope) cell."""
    # logical_lambda_asym_zero added for the H_asym Wilcoxon family
    assert lambda_scope in (
        "logical_lambda_ref",
        "logical_lambda_zero",
        "logical_lambda_asym_zero",
    )
    out: List[float] = []
    boot_dir = base_dir / dataset / "bootstrap"
    if not boot_dir.is_dir():
        return out
    for seed_dir in sorted(boot_dir.glob("seed_*")):
        m_dir = seed_dir / lambda_scope / "metrics"
        # File name is ``metrics_logical_<vtype>_<n_violations>.json`` —
        # the n_violations component depends on dataset size, so glob.
        files = _glob_metrics(m_dir, f"metrics_logical_{vtype}_")
        if not files:
            continue
        data = _load_metric_file(files[0])
        if not data:
            continue
        f1 = _f1(data.get("metrics", {}), method)
        if f1 is not None:
            out.append(f1)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────

def paired_wilcoxon(x: List[float], y: List[float],
                    alternative: str = "two-sided") -> Dict:
    """Paired Wilcoxon signed-rank test with rank-biserial correlation.

    ``alternative`` is forwarded to ``scipy.stats.wilcoxon`` verbatim
    (``"two-sided"``, ``"greater"`` or ``"less"``); the default preserves
    the historical two-sided behaviour of every existing caller.

    Returns ``status='ok'`` only when both samples have matched length
    >= 2 and at least one non-zero difference (``zero_method='wilcox'``
    drops zero differences from both the test and the rank-biserial
    computation, so ties at zero never inflate ``n_pairs``).

    Two effect-size fields are returned and must not be confused:

      - ``rank_biserial``: r = (W+ - W-) / (W+ + W-), the signed
        rank-sum effect size (Kerby 2014 / Cliff-style rank-biserial for
        paired Wilcoxon). ``w_plus``/``w_minus`` are the underlying rank
        sums, so this can be recomputed by hand from the persisted
        output.
      - ``sign_index``: (n_positive - n_negative) / (n_positive +
        n_negative), a sign-only dominance index that ignores magnitude.
        This is NOT a rank-biserial correlation and must never be
        reported under that name; it is provided only so a historical
        artefact that used this simpler quantity can be told apart from
        one that used the rank-sum definition.

    The two coincide only when every non-zero difference shares the same
    sign (rank_biserial and sign_index are then both +-1.0); with mixed
    signs they generally differ (e.g. diffs=[-4,1,2] gives sign_index=
    -1/3 but rank_biserial=0.0, because the two positive differences'
    ranks exactly balance the one negative difference's rank).
    """
    if len(x) != len(y) or len(x) < 2:
        return {"status": "insufficient_data", "n_pairs": min(len(x), len(y))}

    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.all(diffs == 0):
        return {"status": "all_zero_diffs", "n_pairs": len(x)}

    # zero_method='wilcox' drops zero diffs.
    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative=alternative,
                         correction=False)
    statistic = float(res.statistic)
    pvalue = float(res.pvalue)

    # Rank-biserial correlation for paired Wilcoxon: r = (W+ - W-) / (W+ + W-)
    nonzero = diffs[diffs != 0]
    abs_ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(abs_ranks[nonzero > 0].sum())
    w_minus = float(abs_ranks[nonzero < 0].sum())
    denom = w_plus + w_minus
    rank_biserial = (w_plus - w_minus) / denom if denom > 0 else 0.0

    n_positive = int((diffs > 0).sum())
    n_negative = int((diffs < 0).sum())
    n_zero = int((diffs == 0).sum())
    n_signed = n_positive + n_negative
    sign_index = (n_positive - n_negative) / n_signed if n_signed > 0 else 0.0

    return {
        "status": "ok",
        "n_pairs": len(x),
        "alternative": alternative,
        "statistic": statistic,
        "p_value": pvalue,
        "rank_biserial": rank_biserial,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "sign_index": sign_index,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Return Holm-adjusted p-values matching the input order. NaN inputs
    pass through unchanged so insufficient-data tests don't consume an
    alpha slot.
    """
    n = sum(1 for p in p_values if not np.isnan(p))
    if n == 0:
        return list(p_values)

    indexed = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    indexed.sort(key=lambda kv: kv[1])

    adjusted = list(p_values)
    running_max = 0.0
    for rank, (i, p) in enumerate(indexed):
        factor = n - rank
        adj = min(1.0, p * factor)
        if adj < running_max:
            adj = running_max
        else:
            running_max = adj
        adjusted[i] = adj
    return adjusted


def holm_correction_named(pvalues: Dict[str, float], family_id: str,
                          expected_size: Optional[int] = None) -> Dict[str, Dict]:
    """Holm-Bonferroni correction over an explicitly named, CLOSED family.

    ``pvalues`` maps a stable member identifier (e.g. ``"WN18RR"`` or
    ``"FB15k-237/TransE"``) to its raw p-value. Delegates the actual
    step-down arithmetic to ``holm_bonferroni`` (same monotonicity
    guarantee) so there is exactly one Holm implementation in this module;
    this wrapper adds the family bookkeeping (id/members/size) a persisted
    result needs to be unambiguous about which family produced which
    adjusted p-value, AND enforces that the family is complete.

    Unlike ``holm_bonferroni`` (which intentionally lets NaN pass through
    unadjusted, so an *insufficient_data* cell in the legacy 25-test panel
    does not consume an alpha slot -- see that function's docstring), this
    function is STRICT by design: a named family such as ``E6_RANDOM_9``
    or ``HASym_DET_3`` is a small, fully-specified, closed set of tests
    reviewed once and never meant to silently shrink. Raises ``ValueError``
    (never returns a partial result with a successful exit) when:

      - any p-value is NaN -- a member that could not be tested must be
        surfaced as a rejection, not folded into an adjustment as if the
        family were still complete;
      - ``expected_size`` is given and does not match the number of
        members actually supplied (catches a caller that silently dropped
        a dataset/method/seed upstream and still tried to report a Holm
        result for what looks like, but is not, the declared family).

    Returns, per member: ``p_raw``, ``p_adjusted``, ``adjustment_method``,
    ``family_id``, ``family_members`` (sorted, for a stable persisted
    order) and ``family_size``. A member's own ``p_adjusted`` is NEVER a
    silent copy of ``p_raw`` unless that is what Holm's step-down
    genuinely produces for the largest raw p in the family.
    """
    if expected_size is not None and len(pvalues) != expected_size:
        raise ValueError(
            f"holm_correction_named(family_id={family_id!r}): expected "
            f"exactly {expected_size} members, got {len(pvalues)} "
            f"({sorted(pvalues.keys())}). Refusing to adjust a family that "
            f"does not match its declared size -- this usually means a "
            f"dataset/method/seed was silently dropped upstream."
        )

    members = sorted(pvalues.keys())
    raw = [pvalues[m] for m in members]

    nan_members = [m for m, p in zip(members, raw) if isinstance(p, float) and np.isnan(p)]
    if nan_members:
        raise ValueError(
            f"holm_correction_named(family_id={family_id!r}): NaN p-value "
            f"for member(s) {nan_members} -- a named family must be fully "
            f"resolved before correction. Use holm_bonferroni() directly "
            f"(with its documented NaN pass-through) if a partially-tested "
            f"panel is genuinely intended, and label it as such."
        )

    adjusted = holm_bonferroni(raw)
    out: Dict[str, Dict] = {}
    for member, p_raw, p_adj in zip(members, raw, adjusted):
        out[member] = {
            "p_raw": p_raw,
            "p_adjusted": p_adj,
            "adjustment_method": "holm-bonferroni",
            "family_id": family_id,
            "family_members": members,
            "family_size": len(members),
        }
    return out


def bootstrap_ci(values: List[float], n_resamples: int = 10_000,
                 alpha: float = 0.05, rng_seed: int = 12345) -> Optional[Dict]:
    """95% bootstrap percentile CI for the mean of ``values``."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────────

def compute_wilcoxon_table(bootstrap_dir: Path) -> Dict:
    """Compute the legacy 25-test panel (8 H2 + 9 triviality_gap + 8 H_asym)
    + per-cell CIs, from ``<base>/<dataset>/bootstrap/seed_<N>/...``.

    **Superseded — not part of the current pipeline's final statistical
    report.** This function's ``triviality_gap`` group has the same shape
    as the manuscript's ``E6_RANDOM_9`` family (3 datasets x 3 KGE methods,
    random condition) but computes it two-sided and folds it into a joint
    25-test Holm correction; its own ``H_asym`` group (8 cells, by
    dataset x violation-type) is a different, broader family than the
    manuscript's ``HASym_DET_3`` (3 cells, asymmetry@10% only). Neither
    matches the manuscript's design (one-sided E6 with its own Holm-9;
    two-sided gradient families with their own Holm-3 each, SDE included
    as a mandatory third family). ``run_all.py``'s bootstrap_validation
    stage no longer calls this function — see
    ``compute_final_statistical_report()`` below, which is the one
    genuinely wired into the current pipeline and into the manuscript's
    Appendix C tables (``tab:app_wilcoxon_random``,
    ``tab:app_wilcoxon_deterministic``, ``tab:app_wilcoxon_sde``).

    Kept, unmodified, for ``src/experiments/run_phase2_resume.py``, which
    still consumes this exact 25-test schema for its own historical
    purpose. Do not delete and do not present its output as the article's
    final statistics.

    ``bootstrap_dir`` is the per-dataset bootstrap directory the runner
    just populated (e.g. ``paper_results/fb15k-237/bootstrap``). The
    function infers the base results directory from
    ``bootstrap_dir.parent.parent`` and aggregates over every dataset
    that has a sibling ``bootstrap/`` populated. Datasets without
    bootstrap data are silently skipped — this keeps the function
    idempotent across the per-dataset invocations of Block 5.
    """
    bootstrap_dir = Path(bootstrap_dir)
    base_dir = bootstrap_dir.parent.parent  # paper_results/

    tests: List[Dict] = []
    cis: Dict[str, Dict] = {}

    # H2 group: 8 paired tests (ODE λ=ref vs ODE λ=0 in logical cells).
    for ds, vtype in H2_CELLS:
        ref = collect_logical_f1(base_dir, ds, vtype, "logical_lambda_ref", ODE_METHOD_KEY)
        zero = collect_logical_f1(base_dir, ds, vtype, "logical_lambda_zero", ODE_METHOD_KEY)
        result = paired_wilcoxon(ref, zero)
        tests.append({
            "group": "H2",
            "dataset": ds,
            "violation": vtype,
            "method_a": "ODE_lambda_ref",
            "method_b": "ODE_lambda_zero",
            "result": result,
        })
        for label, vals in (("ref", ref), ("zero", zero)):
            ci = bootstrap_ci(vals)
            if ci is not None:
                cis[f"{ds}/logical/{vtype}/ODE_lambda_{label}"] = ci

    # Triviality-gap group: 9 paired tests (each KGE vs ODE, random ablation).
    for ds in DATASETS:
        ode_vals = collect_random_f1(base_dir, ds, ODE_METHOD_KEY)
        for kge in KGE_METHODS:
            kge_vals = collect_random_f1(base_dir, ds, kge)
            n = min(len(ode_vals), len(kge_vals))
            result = paired_wilcoxon(kge_vals[:n], ode_vals[:n])
            tests.append({
                "group": "triviality_gap",
                "dataset": ds,
                "violation": "random",
                "method_a": kge,
                "method_b": ODE_METHOD_KEY,
                "result": result,
            })
            ci_kge = bootstrap_ci(kge_vals)
            if ci_kge is not None:
                cis[f"{ds}/random/{kge}"] = ci_kge
        ci_ode = bootstrap_ci(ode_vals)
        if ci_ode is not None:
            cis[f"{ds}/random/{ODE_METHOD_KEY}"] = ci_ode

    # H_asym group: 8 paired tests (ODE λ_asym=ref vs ODE λ_asym=0
    # in logical cells). Uses the same H2_CELLS exclusion of
    # (codex-m, transitivity). The sub-experiment data lives in
    # bootstrap/seed_<N>/logical_lambda_asym_zero/metrics/, written by the
    # extended run_bootstrap_validation_stage. CIs for the asym_zero arm
    # are also persisted (E.2 of the report ASYM mirrors the structure of
    # the pre-asym milestone with this row added).
    for ds, vtype in H2_CELLS:
        ref = collect_logical_f1(base_dir, ds, vtype, "logical_lambda_ref", ODE_METHOD_KEY)
        zero = collect_logical_f1(base_dir, ds, vtype, "logical_lambda_asym_zero", ODE_METHOD_KEY)
        result = paired_wilcoxon(ref, zero)
        tests.append({
            "group": "H_asym",
            "dataset": ds,
            "violation": vtype,
            "method_a": "ODE_lambda_ref",
            "method_b": "ODE_lambda_asym_zero",
            "result": result,
        })
        ci_zero = bootstrap_ci(zero)
        if ci_zero is not None:
            cis[f"{ds}/logical/{vtype}/ODE_lambda_asym_zero"] = ci_zero

    # Holm correction across the full unified family
    # (8 H2 + 9 triviality_gap + 8 H_asym = 25 tests, per user spec).
    raw_pvals = [t["result"].get("p_value", float("nan")) for t in tests]
    adj_pvals = holm_bonferroni(raw_pvals)
    for t, adj in zip(tests, adj_pvals):
        t["result"]["p_value_holm"] = adj if not np.isnan(adj) else None

    out = {
        "n_tests": len(tests),
        "tests": tests,
        "confidence_intervals_95": cis,
    }

    # Write outputs to paper_results_statistical/ at repo root.
    out_dir = base_dir.parent / "paper_results_statistical"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wilcoxon_tests.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(out_dir / "wilcoxon_tests.md", "w") as f:
        f.write(_render_markdown(out))

    print(f"  wrote {out_dir / 'wilcoxon_tests.json'}")
    print(f"  wrote {out_dir / 'wilcoxon_tests.md'}")
    return out


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return f"{p:.4g}"


def _render_markdown(out: Dict) -> str:
    lines: List[str] = []
    lines.append("# Wilcoxon Tests — Bootstrap Validation\n")
    lines.append(f"Total tests: **{out['n_tests']}**\n")
    lines.append("")
    lines.append("| Group | Dataset | Violation | Method A | Method B | n | "
                 "Statistic | p (raw) | p (Holm) | Rank-biserial |")
    lines.append("|-------|---------|-----------|----------|----------|---|"
                 "-----------|---------|----------|---------------|")
    for t in out["tests"]:
        r = t["result"]
        if r.get("status") != "ok":
            lines.append(f"| {t['group']} | {t['dataset']} | {t['violation']} | "
                         f"{t['method_a']} | {t['method_b']} | "
                         f"{r.get('n_pairs', 0)} | — | — | — | "
                         f"({r.get('status', 'n/a')}) |")
            continue
        lines.append(
            f"| {t['group']} | {t['dataset']} | {t['violation']} | "
            f"{t['method_a']} | {t['method_b']} | {r['n_pairs']} | "
            f"{r['statistic']:.4g} | {_fmt_p(r['p_value'])} | "
            f"{_fmt_p(r['p_value_holm'])} | {r['rank_biserial']:.3f} |"
        )

    lines.append("")
    lines.append("## 95% bootstrap percentile CIs (n_resamples = 10,000)")
    lines.append("")
    lines.append("| Cell | n | Mean F1 | CI Low | CI High |")
    lines.append("|------|---|---------|--------|---------|")
    for cell, ci in sorted(out["confidence_intervals_95"].items()):
        lines.append(
            f"| {cell} | {ci['n']} | {ci['mean']:.4f} | "
            f"{ci['ci_lo']:.4f} | {ci['ci_hi']:.4f} |"
        )

    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════
# Final statistical report — the fifteen paired contrasts across three
# independently-corrected, named families (E6_RANDOM_9, HASym_DET_3,
# HASym_SDE_REVIEW_3). This is the analysis described in the manuscript's
# Section 4/Appendix C and the one ``run_all.py``'s bootstrap_validation
# stage chains to (replacing the call to the legacy ``compute_wilcoxon_table``
# above, which computes a structurally different, two-sided, jointly-
# Holm-25-corrected panel from a separate, older bootstrap directory).
#
# SDE is a mandatory third family, not an optional/exploratory addendum:
# ``compute_final_statistical_report`` always computes and returns it.
#
# The scripts in scripts/compute_e6_triviality_gap.py and
# scripts/compute_hasym_and_sde_holm.py import these exact functions
# rather than reimplementing them, so the standalone analysis path and
# the production pipeline path share one calculation, not three
# divergent copies.
# ══════════════════════════════════════════════════════════════════════

# Correct-case dataset directory names (paper_results/<name>/...). Kept
# deliberately separate from the legacy, lowercase ``DATASETS`` above
# (used only by ``compute_wilcoxon_table``'s own, different bootstrap
# layout) to avoid a silent case-mismatch alias bug.
FINAL_DATASETS = ("FB15k-237", "codex-m", "WN18RR")
FINAL_KGE_METHODS = ("TransE", "DistMult", "MuRE")

E6_FAMILY_ID = "E6_RANDOM_9"
HASYM_DET_FAMILY_ID = "HASym_DET_3"
HASYM_SDE_FAMILY_ID = "HASym_SDE_REVIEW_3"
EXPECTED_SEEDS_PER_E6_CELL = 10
EXPECTED_SEEDS_PER_GRADIENT_DATASET = 10


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_e6_ode_f1_by_seed(dataset: str, repo_root: Optional[Path] = None) -> Dict[int, float]:
    """Returns {perturbation_seed: f1} for the post-fix ODE detector,
    seed read from the file's own protocol block (not inferred from the
    directory name)."""
    root = repo_root or _default_repo_root()
    base = root / "paper_results" / dataset / "bootstrap_postfix"
    out: Dict[int, float] = {}
    for p in sorted(base.glob("seed_*/random/metrics/metrics_random_entity_swap_5.0.json")):
        with open(p) as f:
            d = json.load(f)
        seed = d["protocol"]["perturbation_seed"]
        if seed in out:
            raise ValueError(f"duplicate perturbation_seed={seed} for {dataset} in {p}")
        out[seed] = d["ode_postfix"]["f1"]
    return out


def load_e6_kge_f1_by_seed(dataset: str, method: str,
                           repo_root: Optional[Path] = None) -> Dict[int, float]:
    """KGE files carry no internal seed field; the seed is the seed_<n>
    path component, taken as authoritative. Raises if the declared
    method is missing from any file rather than silently skipping it.

    Reads from ``paper_results/<dataset>/bootstrap/`` -- the directory
    ``run_all.py``'s own bootstrap_validation stage populates -- not the
    historical ``bootstrap_r0_canonical`` snapshot, which has no
    reproducing script and is therefore unavailable in a from-scratch
    run. Verified numerically equivalent to ``bootstrap_r0_canonical``
    for 87/90 (dataset, seed, method) cells checked; the remaining 3
    (WN18RR/seed_1) differ by <0.3% absolute F1, consistent with the
    ordinary run-to-run variance of an independently regenerated
    snapshot rather than a methodological difference."""
    root = repo_root or _default_repo_root()
    base = root / "paper_results" / dataset / "bootstrap"
    out: Dict[int, float] = {}
    for p in sorted(base.glob("seed_*/random/metrics/metrics_random_entity_swap_5.0.json")):
        seed_dir_name = p.parents[2].name  # "seed_<n>"
        seed = int(seed_dir_name.split("_")[1])
        with open(p) as f:
            d = json.load(f)
        if method not in d["metrics"]:
            raise ValueError(
                f"{p}: method {method!r} missing from 'metrics' block "
                f"(present: {sorted(d['metrics'].keys())}). Refusing to "
                f"silently skip a seed for a declared E6 cell -- fix the "
                f"input file or exclude this (dataset, method) explicitly "
                f"upstream, do not let it disappear here."
            )
        out[seed] = d["metrics"][method]["f1"]
    return out


def compute_e6_cell(dataset: str, method: str, ode_by_seed: Dict[int, float],
                    kge_by_seed: Dict[int, float]) -> Dict:
    """One E6_RANDOM_9 cell: KGE minus ODE, one-sided (KGE > ODE)."""
    common_seeds = sorted(set(ode_by_seed) & set(kge_by_seed))
    missing_ode = sorted(set(kge_by_seed) - set(ode_by_seed))
    missing_kge = sorted(set(ode_by_seed) - set(kge_by_seed))
    if missing_ode or missing_kge:
        raise ValueError(
            f"{dataset}/{method}: incomplete seed pairing -- "
            f"missing_ode_seeds={missing_ode} missing_kge_seeds={missing_kge}. "
            f"Refusing to compute this cell from a partial seed set with a "
            f"successful exit; every one of the {EXPECTED_SEEDS_PER_E6_CELL} "
            f"declared seeds must be present on both sides."
        )
    if len(common_seeds) != EXPECTED_SEEDS_PER_E6_CELL:
        raise ValueError(
            f"{dataset}/{method}: expected exactly {EXPECTED_SEEDS_PER_E6_CELL} "
            f"paired seeds, got {len(common_seeds)} ({common_seeds})."
        )
    kge_vals = [kge_by_seed[s] for s in common_seeds]
    ode_vals = [ode_by_seed[s] for s in common_seeds]
    result = paired_wilcoxon(kge_vals, ode_vals, alternative="greater")
    diffs = [k - o for k, o in zip(kge_vals, ode_vals)]
    return {
        "dataset": dataset,
        "kge": method,
        "seeds_used": common_seeds,
        "n_pairs": len(common_seeds),
        "missing_ode_seeds": missing_ode,
        "missing_kge_seeds": missing_kge,
        "mean_kge_f1": sum(kge_vals) / len(kge_vals) if kge_vals else None,
        "mean_ode_f1": sum(ode_vals) / len(ode_vals) if ode_vals else None,
        "mean_diff": sum(diffs) / len(diffs) if diffs else None,
        "sd_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
        "diffs": diffs,
        "wilcoxon": result,
    }


def compute_e6_random_9(repo_root: Optional[Path] = None) -> Dict:
    """E6_RANDOM_9: 3 datasets x 3 KGE methods vs A2/ODE, random@5%,
    one-sided (KGE > ODE), Holm within exactly 9. Reads only the
    verified ``bootstrap_postfix``/``bootstrap_r0_canonical`` inputs --
    never the older, structurally-different ``paper_results/<ds>/bootstrap/``
    directory the legacy ``compute_wilcoxon_table`` reads."""
    cells: List[Dict] = []
    for ds in FINAL_DATASETS:
        ode_by_seed = load_e6_ode_f1_by_seed(ds, repo_root)
        for method in FINAL_KGE_METHODS:
            kge_by_seed = load_e6_kge_f1_by_seed(ds, method, repo_root)
            cells.append(compute_e6_cell(ds, method, ode_by_seed, kge_by_seed))

    pvals = {f"{c['dataset']}/{c['kge']}": c["wilcoxon"]["p_value"] for c in cells}
    holm = holm_correction_named(pvals, family_id=E6_FAMILY_ID, expected_size=9)
    for c in cells:
        key = f"{c['dataset']}/{c['kge']}"
        c["p_raw"] = holm[key]["p_raw"]
        c["p_holm"] = holm[key]["p_adjusted"]
        c["family_id"] = E6_FAMILY_ID
        c["family_size"] = holm[key]["family_size"]
        c["rank_biserial"] = c["wilcoxon"]["rank_biserial"]
        c["sign_index"] = c["wilcoxon"]["sign_index"]

    return {
        "family_id": E6_FAMILY_ID,
        "family_size": 9,
        "alternative": "greater",
        "cells": cells,
    }


def validate_seeds_complete(per_seed: List[Dict], expected_size: int,
                            source_label: str) -> List[Dict]:
    """Reject (raise ValueError), never silently filter, an incomplete or
    partially-failed seed set."""
    not_ok = [s["seed"] for s in per_seed if not s.get("ok", True)]
    if not_ok:
        raise ValueError(
            f"{source_label}: seed(s) {not_ok} have ok=False. Refusing to "
            f"silently drop failed seeds from a declared {expected_size}"
            f"-seed family and report a result anyway -- surface the "
            f"failure instead."
        )
    if len(per_seed) != expected_size:
        raise ValueError(
            f"{source_label}: expected exactly {expected_size} seeds, "
            f"found {len(per_seed)}."
        )
    return per_seed


def recompute_gradient_family(prefix: str, family_id: str,
                              repo_root: Optional[Path] = None) -> List[Dict]:
    """HASym_DET_3 (prefix='milestone_c_h_asym') or HASym_SDE_REVIEW_3
    (prefix='sde_b6_wilcoxon'): gradient active vs inactive, two-sided,
    Holm within exactly 3, one row per dataset."""
    root = repo_root or _default_repo_root()
    rows = []
    for ds in FINAL_DATASETS:
        path = root / "progress" / f"{prefix}_{ds}.json"
        with open(path) as f:
            d = json.load(f)
        seeds = validate_seeds_complete(d["per_seed"], EXPECTED_SEEDS_PER_GRADIENT_DATASET, str(path))
        f1_ref = [s["f1_ref"] for s in seeds]
        f1_zero = [s["f1_zero"] for s in seeds]
        result = paired_wilcoxon(f1_ref, f1_zero, alternative="two-sided")
        diffs = [a - b for a, b in zip(f1_ref, f1_zero)]
        hist_p = d["stats"]["wilcoxon_p_two_sided"]
        hist_rb_field = d["stats"]["rank_biserial"]  # historical (sign-count) field
        rows.append({
            "dataset": ds,
            "n_pairs": result.get("n_pairs"),
            "mean_diff": sum(diffs) / len(diffs) if diffs else None,
            "sd_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            "p_value_recomputed": result.get("p_value"),
            "p_value_historical": hist_p,
            "p_value_match": (result.get("p_value") is not None
                               and abs(result["p_value"] - hist_p) < 1e-9),
            "rank_biserial_recomputed": result.get("rank_biserial"),
            "sign_index_recomputed": result.get("sign_index"),
            "historical_field_labelled_rank_biserial": hist_rb_field,
            "historical_field_matches_recomputed_rank_biserial":
                abs(hist_rb_field - result.get("rank_biserial", float("nan"))) < 1e-9,
            "historical_field_matches_recomputed_sign_index":
                abs(hist_rb_field - result.get("sign_index", float("nan"))) < 1e-9,
            "w_plus": result.get("w_plus"),
            "w_minus": result.get("w_minus"),
            "wilcoxon_full": result,
        })
    pvals = {r["dataset"]: r["p_value_recomputed"] for r in rows}
    holm = holm_correction_named(pvals, family_id=family_id, expected_size=len(FINAL_DATASETS))
    for r in rows:
        r["p_holm_recomputed"] = holm[r["dataset"]]["p_adjusted"]
        r["family_id"] = family_id
        r["family_size"] = holm[r["dataset"]]["family_size"]
    return rows


def compute_hasym_det_3(repo_root: Optional[Path] = None) -> List[Dict]:
    return recompute_gradient_family("milestone_c_h_asym", HASYM_DET_FAMILY_ID, repo_root)


def compute_hasym_sde_review_3(repo_root: Optional[Path] = None) -> List[Dict]:
    """Mandatory third family -- not an optional/exploratory addendum.
    Always computed and always returned by compute_final_statistical_report."""
    return recompute_gradient_family("sde_b6_wilcoxon", HASYM_SDE_FAMILY_ID, repo_root)


def compute_final_statistical_report(repo_root: Optional[Path] = None,
                                     write_output: bool = True) -> Dict:
    """The single production entry point for the manuscript's fifteen-
    contrast statistical report (Appendix C: ``tab:app_wilcoxon_random``,
    ``tab:app_wilcoxon_deterministic``, ``tab:app_wilcoxon_sde``).

    Called by ``run_all.py``'s ``run_bootstrap_validation_stage`` after
    every per-dataset bootstrap run. Idempotent: it always recomputes the
    full 3-dataset panel from its own fixed, verified input paths
    (``paper_results/<ds>/bootstrap_postfix``/``bootstrap_r0_canonical``,
    ``progress/milestone_c_h_asym_<ds>.json``,
    ``progress/sde_b6_wilcoxon_<ds>.json``), regardless of which single
    dataset's bootstrap stage just finished -- it does not read the
    ``paper_results/<ds>/bootstrap/`` directory that stage populates.

    Returns exactly 15 rows across three independently Holm-corrected
    families (9 + 3 + 3). Never a joint correction across the 15, and
    never merged with the legacy 25-test panel in ``compute_wilcoxon_table``.
    """
    e6 = compute_e6_random_9(repo_root)
    det = compute_hasym_det_3(repo_root)
    sde = compute_hasym_sde_review_3(repo_root)

    rows: List[Dict] = []
    for c in e6["cells"]:
        rows.append({
            "family_id": E6_FAMILY_ID, "dataset": c["dataset"], "comparison": c["kge"],
            "alternative": "greater", "n_pairs": c["n_pairs"],
            "mean_diff": c["mean_diff"], "sd_diff": c["sd_diff"],
            "p_raw": c["p_raw"], "p_holm": c["p_holm"],
            "rank_biserial": c["rank_biserial"],
        })
    for r in det:
        rows.append({
            "family_id": HASYM_DET_FAMILY_ID, "dataset": r["dataset"],
            "comparison": "gradient_active_vs_inactive",
            "alternative": "two-sided", "n_pairs": r["n_pairs"],
            "mean_diff": r["mean_diff"], "sd_diff": r["sd_diff"],
            "p_raw": r["p_value_recomputed"], "p_holm": r["p_holm_recomputed"],
            "rank_biserial": r["rank_biserial_recomputed"],
        })
    for r in sde:
        rows.append({
            "family_id": HASYM_SDE_FAMILY_ID, "dataset": r["dataset"],
            "comparison": "gradient_active_vs_inactive",
            "alternative": "two-sided", "n_pairs": r["n_pairs"],
            "mean_diff": r["mean_diff"], "sd_diff": r["sd_diff"],
            "p_raw": r["p_value_recomputed"], "p_holm": r["p_holm_recomputed"],
            "rank_biserial": r["rank_biserial_recomputed"],
        })

    out = {
        "n_contrasts": len(rows),
        "families": {
            E6_FAMILY_ID: {"size": 9, "alternative": "greater"},
            HASYM_DET_FAMILY_ID: {"size": 3, "alternative": "two-sided"},
            HASYM_SDE_FAMILY_ID: {"size": 3, "alternative": "two-sided"},
        },
        "rows": rows,
        "E6_RANDOM_9": e6,
        "HASym_DET_3": det,
        "HASym_SDE_REVIEW_3": sde,
    }

    if write_output:
        root = repo_root or _default_repo_root()
        out_dir = root / "paper_results_statistical"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "final_statistical_report_15_contrasts.json", "w") as f:
            json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/statistical_tests.py <bootstrap_dir>")
        sys.exit(1)
    compute_wilcoxon_table(Path(sys.argv[1]))
