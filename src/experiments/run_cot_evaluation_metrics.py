"""Fase 3 — TASK 3.4 — CoT human-in-the-loop evaluation metrics scaffolding.

Computes the agreement metrics for the CoT validation protocol:

  - Cohen's κ pairwise (3 pairs of annotators)
  - Krippendorff's α (global multi-annotator agreement)
  - Per-category confusion matrix (annotator-vs-CoT majority)

The driver is intended to be re-run AFTER the human annotation is collected.
For now it includes:

  (a) the metric implementations (NO external dependency on `krippendorff`
      package; we inline a nominal-data implementation since the
      taxonomy is categorical).
  (b) a synthetic-annotation generator + self-test that confirms the
      metrics behave correctly on a known agreement pattern.

When real human annotations land in
``paper_results/cot_evaluation/annotation_sheet_annotator_{1,2,3}.json``
with the ``annotator_judgment`` field populated, this driver consumes
them and produces ``paper_results/cot_evaluation/agreement_report.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

EVAL_DIR = REPO_ROOT / "paper_results" / "cot_evaluation"

CATEGORIES = ["A", "B", "C", "D"]


# ===== Agreement metrics =====================================================


def cohen_kappa(coder_a: List[str], coder_b: List[str],
                categories: List[str] = CATEGORIES) -> float:
    """Cohen's κ for two coders with categorical labels.

    κ = (p_o − p_e) / (1 − p_e), where:
      p_o = observed agreement = #(a_i == b_i) / N
      p_e = expected agreement = Σ_c (P_a(c) × P_b(c))
    """
    assert len(coder_a) == len(coder_b)
    n = len(coder_a)
    if n == 0:
        return float("nan")
    p_o = sum(1 for a, b in zip(coder_a, coder_b) if a == b) / n
    counts_a = Counter(coder_a)
    counts_b = Counter(coder_b)
    p_e = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n)
              for c in categories)
    if (1 - p_e) < 1e-12:
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha(annotations: Dict[int, List[Optional[str]]],
                       categories: List[str] = CATEGORIES) -> float:
    """Krippendorff's α for nominal data (multiple coders × items).

    `annotations` is {item_index: [label1, label2, label3, ...]} where each
    inner list has one entry per annotator, possibly with None for missing
    values.

    Formula (nominal, see Krippendorff 2004 / 2018):
      α = 1 − Do / De
      Do = (1/n) × Σ_units Σ_{coder pairs (c,c')} δ(label_c, label_c')
      De = (1/(N×(N−1))) × Σ_categories (n_cat × (N − n_cat)) where N is total
        observed values across all units/coders, n_cat is total observed of that category.

    We implement the formulation valid when each unit has ≥ 2 valid codings
    (skip-missing).
    """
    # Strip missing
    cleaned = {idx: [v for v in vals if v is not None]
               for idx, vals in annotations.items()}
    cleaned = {idx: vals for idx, vals in cleaned.items() if len(vals) >= 2}
    if not cleaned:
        return float("nan")

    # Disagreement Do
    Do = 0.0
    units_n = 0
    for idx, vals in cleaned.items():
        m = len(vals)
        if m < 2:
            continue
        # Number of pairs in this unit
        n_pairs = m * (m - 1)
        # Disagreements
        d = 0
        for a, b in combinations(range(m), 2):
            if vals[a] != vals[b]:
                d += 2  # ordered pairs
        # Per-unit average disagreement, normalised by (m × (m − 1))
        Do += d / max(n_pairs, 1)
        units_n += 1
    if units_n == 0:
        return float("nan")
    Do /= units_n

    # Expected disagreement De — built from marginal distribution
    all_vals = [v for vals in cleaned.values() for v in vals]
    N = len(all_vals)
    if N < 2:
        return float("nan")
    counts = Counter(all_vals)
    De = 0.0
    for c1 in categories:
        for c2 in categories:
            if c1 == c2:
                continue
            De += (counts.get(c1, 0) / N) * (counts.get(c2, 0) / (N - 1)) \
                  * (N / (N - 1))  # correction for finite N
    De = 2 * sum((counts.get(c1, 0) / N) * (counts.get(c2, 0) / N)
                 for c1 in categories for c2 in categories if c1 < c2)

    if De < 1e-12:
        return float("nan")
    return 1.0 - Do / De


def confusion_matrix(annotator: List[str], cot: List[str],
                     categories: List[str] = CATEGORIES) -> np.ndarray:
    """Confusion matrix rows=cot, cols=annotator."""
    n_cat = len(categories)
    cm = np.zeros((n_cat, n_cat), dtype=int)
    cat_idx = {c: i for i, c in enumerate(categories)}
    for a, c in zip(annotator, cot):
        if a in cat_idx and c in cat_idx:
            cm[cat_idx[c], cat_idx[a]] += 1
    return cm


# ===== Loading + reporting ===================================================


def load_annotations(eval_dir: Path) -> Optional[Dict]:
    """Load the canonical sample + three annotation sheets, align by triple_id.

    Returns None if any sheet has unpopulated annotator_judgment fields.
    """
    canon_path = eval_dir / "sample_100_stratified.json"
    if not canon_path.exists():
        return None
    with open(canon_path) as f:
        canon = json.load(f)

    by_id = {rec["triple_id"]: rec for rec in canon["records"]}

    annot_by_id = {tid: {} for tid in by_id}
    any_unpopulated = False
    for ann_id in (1, 2, 3):
        sheet_path = eval_dir / f"annotation_sheet_annotator_{ann_id}.json"
        if not sheet_path.exists():
            print(f"Missing sheet: {sheet_path}")
            return None
        with open(sheet_path) as f:
            sheet = json.load(f)
        for rec in sheet["records"]:
            tid = rec["triple_id"]
            judgment = rec.get("annotator_judgment")
            if judgment is None or judgment not in CATEGORIES:
                any_unpopulated = True
            annot_by_id[tid][ann_id] = judgment
    return {
        "canonical": by_id,
        "annotations": annot_by_id,
        "fully_populated": not any_unpopulated,
    }


def compute_report(loaded: Dict) -> Dict:
    """Aggregate the agreement metrics from a fully-populated loaded set."""
    canon = loaded["canonical"]
    annots = loaded["annotations"]
    triple_ids = list(canon.keys())

    cot_labels = [canon[tid]["cot_judgment"]["category"] for tid in triple_ids]
    annot_1 = [annots[tid].get(1) for tid in triple_ids]
    annot_2 = [annots[tid].get(2) for tid in triple_ids]
    annot_3 = [annots[tid].get(3) for tid in triple_ids]

    # Pairwise Cohen's κ
    kappas = {
        "1_vs_2": cohen_kappa(annot_1, annot_2),
        "1_vs_3": cohen_kappa(annot_1, annot_3),
        "2_vs_3": cohen_kappa(annot_2, annot_3),
    }
    # Krippendorff α (multi-annotator)
    k_alpha = krippendorff_alpha({
        i: [annot_1[i], annot_2[i], annot_3[i]] for i in range(len(triple_ids))
    })

    # Annotator majority vs CoT
    majority = []
    for i in range(len(triple_ids)):
        votes = [annot_1[i], annot_2[i], annot_3[i]]
        counts = Counter(votes)
        winner = counts.most_common(1)[0][0]
        majority.append(winner)
    cm_majority = confusion_matrix(majority, cot_labels)
    # Accuracy CoT vs majority
    acc = sum(1 for m, c in zip(majority, cot_labels) if m == c) / len(majority)

    # Per-category breakdown
    per_cat = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        cot_n = sum(1 for c in cot_labels if c == cat)
        if cot_n == 0:
            per_cat[cat] = {"cot_n": 0, "agreement_with_majority": None}
        else:
            agreement = sum(1 for m, c in zip(majority, cot_labels)
                            if c == cat and m == c) / cot_n
            per_cat[cat] = {"cot_n": cot_n,
                            "agreement_with_majority": agreement}

    return {
        "n_items": len(triple_ids),
        "cohen_kappa": kappas,
        "mean_pairwise_kappa": float(np.mean(list(kappas.values()))),
        "krippendorff_alpha": k_alpha,
        "cot_vs_majority_accuracy": acc,
        "confusion_matrix": {
            "rows": "cot",
            "cols": "annotator_majority",
            "categories": CATEGORIES,
            "matrix": cm_majority.tolist(),
        },
        "per_category": per_cat,
    }


def write_report(report: Dict, eval_dir: Path):
    out_json = eval_dir / "agreement_report.json"
    out_md = eval_dir / "agreement_report.md"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# CoT validation protocol — agreement report",
        "",
        f"**n = {report['n_items']} triples × 3 annotators**",
        "",
        "## Inter-annotator agreement",
        "",
        f"  - Cohen's κ pairwise: 1↔2 = {report['cohen_kappa']['1_vs_2']:.3f}, "
        f"1↔3 = {report['cohen_kappa']['1_vs_3']:.3f}, "
        f"2↔3 = {report['cohen_kappa']['2_vs_3']:.3f}",
        f"  - Mean pairwise κ: {report['mean_pairwise_kappa']:.3f}",
        f"  - Krippendorff's α (nominal, 3 annotators): {report['krippendorff_alpha']:.3f}",
        "",
        "## CoT vs annotator majority",
        "",
        f"  - Accuracy of CoT against majority vote: {report['cot_vs_majority_accuracy']:.3f}",
        "",
        "## Confusion matrix (rows = CoT, cols = annotator majority)",
        "",
        "| CoT \\\\ Annot | A | B | C | D |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, row in zip(CATEGORIES, report["confusion_matrix"]["matrix"]):
        lines.append(f"| {cat} | " + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "## Per-category agreement",
        "",
        "| CoT category | n | agreement with majority |",
        "|---|---:|---:|",
    ]
    for cat in CATEGORIES:
        pc = report["per_category"][cat]
        if pc["cot_n"] == 0:
            lines.append(f"| {cat} | 0 | n/a |")
        else:
            lines.append(f"| {cat} | {pc['cot_n']} | "
                         f"{pc['agreement_with_majority']:.3f} |")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))


# ===== Self-test on synthetic annotations ====================================


def self_test(verbose: bool = True):
    """Generate synthetic annotations with known agreement pattern and verify
    the metrics behave correctly."""
    rng = np.random.default_rng(42)
    # 100 items, true labels with the actual sample's distribution
    quotas = {"A": 27, "B": 26, "C": 26, "D": 21}
    true_labels = []
    for cat, n in quotas.items():
        true_labels += [cat] * n
    rng.shuffle(true_labels)

    def noisy(labels, p_correct):
        out = []
        for lab in labels:
            if rng.random() < p_correct:
                out.append(lab)
            else:
                alts = [c for c in CATEGORIES if c != lab]
                out.append(alts[int(rng.integers(0, 3))])
        return out

    # Three annotators with varying agreement
    a1 = noisy(true_labels, 0.85)
    a2 = noisy(true_labels, 0.80)
    a3 = noisy(true_labels, 0.75)

    if verbose:
        print("== Self-test ==")
    k12 = cohen_kappa(a1, a2)
    k13 = cohen_kappa(a1, a3)
    k23 = cohen_kappa(a2, a3)
    if verbose:
        print(f"  Cohen κ:  1-2={k12:.3f}  1-3={k13:.3f}  2-3={k23:.3f}")
    assert 0.55 < k12 < 1.0, f"κ_12 out of expected range: {k12}"
    assert 0.50 < k13 < 1.0, f"κ_13 out of expected range: {k13}"

    annots = {i: [a1[i], a2[i], a3[i]] for i in range(len(true_labels))}
    alpha = krippendorff_alpha(annots)
    if verbose:
        print(f"  Krippendorff α: {alpha:.3f}")
    assert 0.4 < alpha < 1.0, f"α out of expected range: {alpha}"

    cm = confusion_matrix(a1, true_labels)
    if verbose:
        print(f"  Confusion (rows true, cols a1):\n{cm}")
    diag = np.trace(cm)
    assert diag / cm.sum() > 0.7, "diagonal mass too low"
    if verbose:
        print("  Self-test PASSED")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="Run the synthetic self-test of the metrics.")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    loaded = load_annotations(EVAL_DIR)
    if loaded is None:
        print(f"Annotation sheets missing or canonical sample absent at {EVAL_DIR}")
        return
    if not loaded["fully_populated"]:
        print("Annotation sheets present but some `annotator_judgment` fields "
              "are None — human annotation pending.")
        print("Run with --self-test to verify the metrics on synthetic data.")
        return

    report = compute_report(loaded)
    write_report(report, EVAL_DIR)
    print("Wrote agreement report:")
    print(f"  {EVAL_DIR / 'agreement_report.json'}")
    print(f"  {EVAL_DIR / 'agreement_report.md'}")
    print()
    print(f"Cohen κ mean: {report['mean_pairwise_kappa']:.3f}")
    print(f"Krippendorff α: {report['krippendorff_alpha']:.3f}")
    print(f"CoT vs majority accuracy: {report['cot_vs_majority_accuracy']:.3f}")


if __name__ == "__main__":
    main()
