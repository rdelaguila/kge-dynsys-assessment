"""Fase 3 — TASK 3.2 + 3.3 — generate stratified evaluation sample for the
human-in-the-loop validation protocol of the CoT explainability module.

Source: paper_results/cot_consensus_4047.json (Universal Consensus,
flagged by ODE and all 4 KGEs simultaneously).

Stratification: dataset × judgment × constraint_context. Over-represent
the rare categories (C and D) to give the protocol statistical power on
minority strata.

Target output: 100 triples per `paper_results/cot_evaluation/sample_100_stratified.json`
plus a blind annotation sheet (no CoT judgment visible) for each of 3 annotators.

Read-only.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SOURCE = REPO_ROOT / "paper_results" / "cot_consensus_4047.json"
OUT_DIR = REPO_ROOT / "paper_results" / "cot_evaluation"
OUT_DIR.mkdir(exist_ok=True, parents=True)

TARGET_TOTAL = 100
# Stratum quotas: balance judgement, slight over-representation of C/D.
# A=27, B=26, C=26, D=21 (all 21 D available; 79 split across A/B/C as 27/26/26).
QUOTAS_PER_JUDGMENT = {"A": 27, "B": 26, "C": 26, "D": 21}


def constraint_context(rel: str, dataset: str) -> str:
    """Map relation to a coarse constraint context label.

    The classification mirrors the constraint_config used by
    `KGPerturbator` and `IntegratedKGDebugger`. We label as
    `asymmetry`, `transitivity`, `symmetry`, `cardinality`, or `none`.

    The list below is a static map curated from
    `src/run_all.py:get_constraint_config` (block7 era).
    """
    asym_wn = {"_hypernym", "_instance_hypernym", "_has_part", "_member_meronym",
               "_member_of_domain_region", "_member_of_domain_usage",
               "_synset_domain_topic_of"}
    sym_wn = {"_also_see", "_similar_to", "_verb_group",
              "_derivationally_related_form"}
    trans_wn = {"_hypernym", "_instance_hypernym", "_has_part", "_member_meronym"}

    if dataset == "wn18rr":
        # asym ∩ trans no vacío en WN; preferimos asym si está en ambos
        if rel in asym_wn:
            return "asymmetry"
        if rel in trans_wn:
            return "transitivity"
        if rel in sym_wn:
            return "symmetry"
        return "none"

    if dataset == "fb15k-237":
        sym_fb = {"/people/person/spouse_s./people/marriage/spouse",
                  "/people/person/sibling_s./people/sibling_relationship/sibling"}
        if rel in sym_fb:
            return "symmetry"
        if "/location/contains" in rel:
            return "transitivity"
        if "subPartOf" in rel or "/part_of" in rel:
            return "transitivity"
        return "none"

    if dataset == "codex-m":
        sym_cx = {"P3373", "P26"}
        if rel in sym_cx:
            return "symmetry"
        # asym P-numbers (per get_constraint_config codex-m)
        asym_cx = {"P131", "P361", "P527", "P31", "P279", "P171", "P749",
                   "P155", "P156", "P457"}
        if rel in asym_cx:
            return "asymmetry"
        return "none"

    return "none"


def main():
    with open(SOURCE) as f:
        data = json.load(f)
    reports = data["reports"]
    print(f"Source: {SOURCE.name}, total={len(reports)}")

    # Augment with constraint_context
    for r in reports:
        r["constraint_context"] = constraint_context(r["triple"]["relation"],
                                                     r["dataset"])

    by_j = defaultdict(list)
    for r in reports:
        by_j[r["chosen_judgment"]].append(r)
    print("\nAvailable per judgment:")
    for j in "ABCD":
        print(f"  J={j}: {len(by_j[j])}")

    # Within-judgment stratification: dataset × constraint_context
    rng = random.Random(20260518)

    selected = []
    used_keys = set()
    for j, quota in QUOTAS_PER_JUDGMENT.items():
        pool = by_j[j]
        # If quota > available, take all
        if quota >= len(pool):
            for r in pool:
                key = (r["dataset"], r["triple"]["relation"], r["triple"]["subject"],
                       r["triple"]["object"])
                if key in used_keys: continue
                used_keys.add(key)
                selected.append(r)
            continue
        # Build sub-strata: dataset × constraint_context
        sub = defaultdict(list)
        for r in pool:
            sub[(r["dataset"], r["constraint_context"])].append(r)

        # Allocate proportional to sub-stratum size, with a floor of 1 for
        # any non-empty sub-stratum.
        sub_keys = list(sub.keys())
        sub_sizes = {k: len(sub[k]) for k in sub_keys}
        total_pool = sum(sub_sizes.values())
        # Allocations: max(1, round(quota * size/total))
        alloc = {}
        for k in sub_keys:
            alloc[k] = max(1, round(quota * sub_sizes[k] / total_pool))
        # Adjust to match quota exactly
        diff = quota - sum(alloc.values())
        while diff != 0:
            # Find largest pool (or smallest if diff < 0)
            target = max(sub_keys, key=lambda k: sub_sizes[k] if diff > 0
                         else -alloc[k])
            if diff > 0:
                alloc[target] += 1
                diff -= 1
            else:
                if alloc[target] > 1:
                    alloc[target] -= 1
                    diff += 1
                else:
                    # Pick another to reduce
                    cands = [k for k in sub_keys if alloc[k] > 1]
                    if not cands: break
                    target = cands[0]
                    alloc[target] -= 1
                    diff += 1

        # Sample
        for k in sub_keys:
            n = min(alloc[k], len(sub[k]))
            if n <= 0: continue
            picks = rng.sample(sub[k], n)
            for r in picks:
                key = (r["dataset"], r["triple"]["relation"],
                       r["triple"]["subject"], r["triple"]["object"])
                if key in used_keys: continue
                used_keys.add(key)
                selected.append(r)

    # Top up if under target due to overlap removal
    if len(selected) < TARGET_TOTAL:
        # Fill with extra from B (largest pool)
        extras = [r for r in by_j["B"]
                  if (r["dataset"], r["triple"]["relation"],
                      r["triple"]["subject"], r["triple"]["object"]) not in used_keys]
        rng.shuffle(extras)
        deficit = TARGET_TOTAL - len(selected)
        for r in extras[:deficit]:
            selected.append(r)
            used_keys.add((r["dataset"], r["triple"]["relation"],
                           r["triple"]["subject"], r["triple"]["object"]))

    print(f"\nSelected {len(selected)} triples")
    # Stats
    by_judgment = Counter(r["chosen_judgment"] for r in selected)
    by_dataset = Counter(r["dataset"] for r in selected)
    by_constraint = Counter(r["constraint_context"] for r in selected)
    print(f"  by judgment: {dict(by_judgment)}")
    print(f"  by dataset: {dict(by_dataset)}")
    print(f"  by constraint: {dict(by_constraint)}")

    # Build the structured sample
    sample_id = 0
    sample_records = []
    for r in selected:
        sample_id += 1
        rec = {
            "triple_id": f"S{sample_id:03d}",
            "head": r["triple"]["subject"],
            "relation": r["triple"]["relation"],
            "tail": r["triple"]["object"],
            "dataset": r["dataset"],
            "detection_signals": {
                "ode_energy": r["ode_energy"],
                "kge_methods_flagging": r["kge_methods_flagging"],
            },
            "constraint_context": {
                "violated_constraint": r["constraint_context"],
                "from_universal_consensus_set": True,
                "consensus_set_size": 4047,
            },
            "cot_judgment": {
                "category": r["chosen_judgment"],
                "scores": r["judgment_scores"],
                "reasoning": r["reasoning"],
                "risk_level": r["risk_level"],
                "recommended_action": r["recommended_action"],
            },
            "evaluation_metadata": {
                "stratum_dataset": r["dataset"],
                "stratum_judgment": r["chosen_judgment"],
                "stratum_constraint": r["constraint_context"],
                "for_human_annotation": True,
                "annotation_status": "pending",
            },
        }
        sample_records.append(rec)

    # Persist canonical (with CoT judgment visible)
    out_canonical = OUT_DIR / "sample_100_stratified.json"
    payload = {
        "protocol": {
            "source": "paper_results/cot_consensus_4047.json",
            "source_set": "Universal Consensus (ODE ∩ TransE ∩ DistMult ∩ MuRE ∩ ComplEx)",
            "source_total": 4047,
            "target_sample_size": TARGET_TOTAL,
            "actual_sample_size": len(sample_records),
            "stratification": "dataset × judgment × constraint_context",
            "quotas_per_judgment": QUOTAS_PER_JUDGMENT,
            "random_seed": 20260518,
            "audit_r0_scope_decision": (
                "Universal Consensus drawn from canonical (pre-fix r=0) pipeline. "
                "A prior audit determined E1 (Jaccard architectural "
                "disjointness) is INMUNE qualitatively to r=0 → Universal Consensus is "
                "structurally stable; sample can be used without regenerating from post-fix."
            ),
        },
        "stats": {
            "by_judgment": dict(by_judgment),
            "by_dataset": dict(by_dataset),
            "by_constraint": dict(by_constraint),
            "by_dataset_x_judgment": {
                f"{ds}|{j}": c
                for (ds, j), c in Counter(
                    (r["dataset"], r["chosen_judgment"]) for r in selected
                ).items()
            },
        },
        "records": sample_records,
    }
    with open(out_canonical, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote canonical sample: {out_canonical}")

    # Persist blind annotation sheets (3 annotators, no CoT judgment visible)
    for annotator_id in (1, 2, 3):
        blind_records = []
        for rec in sample_records:
            blind = {
                "triple_id": rec["triple_id"],
                "head": rec["head"],
                "relation": rec["relation"],
                "tail": rec["tail"],
                "dataset": rec["dataset"],
                "detection_signals": rec["detection_signals"],
                "annotator_judgment": None,
                "annotator_rationale": "",
                "annotator_id": annotator_id,
            }
            blind_records.append(blind)
        # Shuffle independently per annotator to mitigate order bias
        rng2 = random.Random(20260518 + annotator_id)
        rng2.shuffle(blind_records)
        out_blind = OUT_DIR / f"annotation_sheet_annotator_{annotator_id}.json"
        with open(out_blind, "w") as f:
            json.dump({
                "annotator_id": annotator_id,
                "protocol": "Assign judgment in {A, B, C, D} per the categorical "
                            "taxonomy of Appendix B; provide a 1-2 line rationale. "
                            "Do NOT consult the CoT canonical sample.",
                "judgment_taxonomy": {
                    "A": "Constraint Violation (logical contradictions, hierarchy/transitivity)",
                    "B": "Data Quality Issue (typos, canonicalization, misalignment)",
                    "C": "Contextual / Polysemy (technically valid in obscure sense)",
                    "D": "Rare Exception (genuine fact violating norms)",
                },
                "records": blind_records,
            }, f, indent=2)
        print(f"Wrote blind sheet (annotator {annotator_id}): {out_blind}")


if __name__ == "__main__":
    main()
