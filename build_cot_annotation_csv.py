"""Build annotator-friendly CSVs + instructions from the stratified
sample 100. Adds per-triple graph context, relation semantics, and
lookup hints to make annotation tractable without external tools.

Outputs into paper_results/cot_evaluation/:
  - sample_100_canonical.csv               (full context, CoT judgment visible)
  - sample_100_blind_annotator_{1,2,3}.csv (shuffle per annotator, no CoT)
  - per_triple_context_pack.json           (machine-readable context)
  - ANNOTATION_INSTRUCTIONS.md             (protocol for the human annotator)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import DataLoader

EVAL_DIR = REPO_ROOT / "paper_results" / "cot_evaluation"
CANONICAL_JSON = EVAL_DIR / "sample_100_stratified.json"

# Human-readable semantics for the most common relations.
RELATION_SEMANTICS = {
    # WN18RR
    "_hypernym": "<head> is a hyponym of <tail> (i.e. <head> is a more specific type of <tail>)",
    "_instance_hypernym": "<head> is a specific instance of the category <tail>",
    "_has_part": "<head> has <tail> as one of its parts (meronymy)",
    "_member_meronym": "<head> has <tail> as a member (collective relation)",
    "_member_of_domain_region": "<head> is a member of the regional domain <tail>",
    "_member_of_domain_usage": "<head> is a usage-domain member of <tail>",
    "_synset_domain_topic_of": "<head> is a synset belonging to the topical domain <tail>",
    "_also_see": "<head> is conceptually related to <tail> (cross-reference; symmetric)",
    "_similar_to": "<head> has a similar meaning to <tail> (symmetric)",
    "_verb_group": "<head> and <tail> belong to the same verb group (symmetric)",
    "_derivationally_related_form": "<head> and <tail> are derivationally related forms (symmetric)",
    # codex-m (Wikidata P-codes — top relations in consensus set)
    "P161": "cast member: <tail> performs as a member of the cast of work <head>",
    "P463": "member of: <head> is a member of organisation/group <tail>",
    "P106": "occupation: <head> has occupation <tail>",
    "P1303": "instrument: <head> plays <tail> (musical instrument)",
    "P264": "record label: <head> is signed to label <tail>",
    "P108": "employer: <head> is employed by <tail>",
    "P131": "located in: <head> is located in administrative entity <tail>",
    "P361": "part of: <head> is part of <tail>",
    "P527": "has part: <head> has part <tail>",
    "P31":  "instance of: <head> is an instance of class <tail>",
    "P279": "subclass of: <head> is a subclass of <tail>",
    "P171": "parent taxon: <head> has parent taxon <tail>",
    "P3373": "sibling: <head> and <tail> are siblings (symmetric)",
    "P26":  "spouse: <head> and <tail> are spouses (symmetric)",
}


def relation_description(rel: str) -> str:
    if rel in RELATION_SEMANTICS:
        return RELATION_SEMANTICS[rel]
    # Best-effort for Freebase paths
    if rel.startswith("/"):
        parts = rel.strip("/").split("./")
        last = parts[-1] if parts else rel
        return f"Freebase relation path: {rel} — semantics derivable from path tokens '{last}'"
    return f"<no canonical description — refer to dataset documentation>"


def lookup_url(entity: str, dataset: str) -> str:
    """Return a URL pattern the annotator can use to look up the entity."""
    if dataset == "codex-m" and entity.startswith("Q"):
        return f"https://www.wikidata.org/wiki/{entity}"
    if dataset == "fb15k-237" and entity.startswith("/m/"):
        # Freebase is deprecated; try Wikidata via mapping or Google KG
        return f"https://www.google.com/search?q=%22{entity}%22"
    if dataset == "wn18rr":
        return f"http://wordnet-rdf.princeton.edu/id/{entity}-n"
    return ""


def build_graph_index(triples):
    head_deg = defaultdict(int)
    tail_deg = defaultdict(int)
    rel_freq = defaultdict(int)
    out_edges_h = defaultdict(set)
    in_edges_t = defaultdict(set)
    triple_set = set()
    head_relations = defaultdict(set)
    tail_relations = defaultdict(set)
    for t in triples:
        head_deg[t.subject] += 1
        tail_deg[t.object] += 1
        rel_freq[t.relation] += 1
        out_edges_h[t.subject].add((t.relation, t.object))
        in_edges_t[t.object].add((t.subject, t.relation))
        triple_set.add((t.subject, t.relation, t.object))
        head_relations[t.subject].add(t.relation)
        tail_relations[t.object].add(t.relation)
    return {
        "head_deg": head_deg, "tail_deg": tail_deg,
        "rel_freq": rel_freq,
        "out_edges_h": out_edges_h,
        "in_edges_t": in_edges_t,
        "triple_set": triple_set,
        "head_relations": head_relations,
        "tail_relations": tail_relations,
    }


def context_for(record, idx):
    s = record["head"]; r = record["relation"]; o = record["tail"]
    head_deg = idx["head_deg"].get(s, 0) + idx["tail_deg"].get(s, 0)
    tail_deg = idx["head_deg"].get(o, 0) + idx["tail_deg"].get(o, 0)
    rel_freq = idx["rel_freq"].get(r, 0)
    inverse_present = (o, r, s) in idx["triple_set"]
    # Sample of co-occurring relations involving head and tail
    head_rels = sorted(idx["head_relations"].get(s, set()))
    tail_rels = sorted(idx["tail_relations"].get(o, set()))
    # Sample neighbours (limit to 3 each for compactness)
    head_neighbours = sorted(idx["out_edges_h"].get(s, set()))[:3]
    tail_neighbours = sorted(idx["in_edges_t"].get(o, set()))[:3]
    return {
        "head_degree": head_deg,
        "tail_degree": tail_deg,
        "relation_freq_in_dataset": rel_freq,
        "inverse_triple_in_graph": inverse_present,
        "head_other_relations": head_rels[:8],
        "tail_other_relations": tail_rels[:8],
        "head_sample_outgoing": [f"(r={r2}, o={o2})" for r2, o2 in head_neighbours],
        "tail_sample_incoming": [f"(s={s2}, r={r2})" for s2, r2 in tail_neighbours],
    }


def build_canonical_csv(records, contexts):
    out_path = EVAL_DIR / "sample_100_canonical.csv"
    fields = [
        "triple_id", "dataset", "stratum_judgment", "stratum_constraint",
        "head", "relation", "tail", "relation_description",
        "head_lookup_url", "tail_lookup_url",
        "head_degree_in_graph", "tail_degree_in_graph",
        "relation_freq_in_dataset", "inverse_triple_in_graph",
        "head_other_relations", "tail_other_relations",
        "head_sample_outgoing", "tail_sample_incoming",
        "ode_energy", "kge_methods_flagging",
        "cot_score_A", "cot_score_B", "cot_score_C", "cot_score_D",
        "cot_chosen_judgment", "cot_risk_level", "cot_recommended_action",
        "cot_reasoning",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            ctx = contexts[rec["triple_id"]]
            ds = rec["dataset"]
            row = {
                "triple_id": rec["triple_id"],
                "dataset": ds,
                "stratum_judgment": rec["evaluation_metadata"]["stratum_judgment"],
                "stratum_constraint": rec["evaluation_metadata"]["stratum_constraint"],
                "head": rec["head"], "relation": rec["relation"], "tail": rec["tail"],
                "relation_description": relation_description(rec["relation"]),
                "head_lookup_url": lookup_url(rec["head"], ds),
                "tail_lookup_url": lookup_url(rec["tail"], ds),
                "head_degree_in_graph": ctx["head_degree"],
                "tail_degree_in_graph": ctx["tail_degree"],
                "relation_freq_in_dataset": ctx["relation_freq_in_dataset"],
                "inverse_triple_in_graph": ctx["inverse_triple_in_graph"],
                "head_other_relations": "; ".join(ctx["head_other_relations"]),
                "tail_other_relations": "; ".join(ctx["tail_other_relations"]),
                "head_sample_outgoing": "; ".join(ctx["head_sample_outgoing"]),
                "tail_sample_incoming": "; ".join(ctx["tail_sample_incoming"]),
                "ode_energy": rec["detection_signals"]["ode_energy"],
                "kge_methods_flagging": ", ".join(rec["detection_signals"]["kge_methods_flagging"]),
                "cot_score_A": rec["cot_judgment"]["scores"]["A"],
                "cot_score_B": rec["cot_judgment"]["scores"]["B"],
                "cot_score_C": rec["cot_judgment"]["scores"]["C"],
                "cot_score_D": rec["cot_judgment"]["scores"]["D"],
                "cot_chosen_judgment": rec["cot_judgment"]["category"],
                "cot_risk_level": rec["cot_judgment"]["risk_level"],
                "cot_recommended_action": rec["cot_judgment"]["recommended_action"],
                "cot_reasoning": rec["cot_judgment"]["reasoning"],
            }
            w.writerow(row)
    return out_path


def build_blind_csv(records, contexts, annotator_id: int):
    out_path = EVAL_DIR / f"sample_100_blind_annotator_{annotator_id}.csv"
    # Shuffle per annotator
    items = list(records)
    rng = random.Random(20260518 + annotator_id)
    rng.shuffle(items)
    fields = [
        "presentation_order", "triple_id", "dataset",
        "head", "relation", "tail", "relation_description",
        "head_lookup_url", "tail_lookup_url",
        "head_degree_in_graph", "tail_degree_in_graph",
        "relation_freq_in_dataset", "inverse_triple_in_graph",
        "head_other_relations", "tail_other_relations",
        "head_sample_outgoing", "tail_sample_incoming",
        "ode_energy", "kge_methods_flagging",
        "annotator_judgment_A_B_C_D",
        "annotator_risk_low_med_high",
        "annotator_recommended_action",
        "annotator_rationale",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for order, rec in enumerate(items, start=1):
            ctx = contexts[rec["triple_id"]]
            ds = rec["dataset"]
            row = {
                "presentation_order": order,
                "triple_id": rec["triple_id"],
                "dataset": ds,
                "head": rec["head"], "relation": rec["relation"], "tail": rec["tail"],
                "relation_description": relation_description(rec["relation"]),
                "head_lookup_url": lookup_url(rec["head"], ds),
                "tail_lookup_url": lookup_url(rec["tail"], ds),
                "head_degree_in_graph": ctx["head_degree"],
                "tail_degree_in_graph": ctx["tail_degree"],
                "relation_freq_in_dataset": ctx["relation_freq_in_dataset"],
                "inverse_triple_in_graph": ctx["inverse_triple_in_graph"],
                "head_other_relations": "; ".join(ctx["head_other_relations"]),
                "tail_other_relations": "; ".join(ctx["tail_other_relations"]),
                "head_sample_outgoing": "; ".join(ctx["head_sample_outgoing"]),
                "tail_sample_incoming": "; ".join(ctx["tail_sample_incoming"]),
                "ode_energy": rec["detection_signals"]["ode_energy"],
                "kge_methods_flagging": ", ".join(rec["detection_signals"]["kge_methods_flagging"]),
                "annotator_judgment_A_B_C_D": "",
                "annotator_risk_low_med_high": "",
                "annotator_recommended_action": "",
                "annotator_rationale": "",
            }
            w.writerow(row)
    return out_path


def build_context_pack(records, contexts):
    out_path = EVAL_DIR / "per_triple_context_pack.json"
    pack = []
    for rec in records:
        ctx = contexts[rec["triple_id"]]
        ds = rec["dataset"]
        pack.append({
            "triple_id": rec["triple_id"],
            "dataset": ds,
            "triple": {"head": rec["head"], "relation": rec["relation"], "tail": rec["tail"]},
            "relation_description": relation_description(rec["relation"]),
            "lookup_urls": {
                "head": lookup_url(rec["head"], ds),
                "tail": lookup_url(rec["tail"], ds),
            },
            "graph_context": ctx,
            "detection_signals": rec["detection_signals"],
            "stratum": rec["evaluation_metadata"],
        })
    with open(out_path, "w") as f:
        json.dump(pack, f, indent=2)
    return out_path


INSTRUCTIONS_MD = """# Annotation instructions — CoT validation protocol

**Sample**: `sample_100_blind_annotator_{N}.csv` (your assigned sheet, N = 1, 2, or 3).
**Sample size**: 100 triples drawn from the Universal Consensus set (anomalies flagged simultaneously by the Neural ODE and all four KGE architectures: TransE, DistMult, MuRE, ComplEx).
**Estimated time**: ~3-5 hours per annotator (≈ 2-3 min per triple).
**Independence**: do NOT consult other annotators or the canonical CoT judgement at any point.

## 1. What is the goal

We are validating the LLM-based Chain-of-Thought (CoT) explainability module. The CoT assigns each anomalous triple to one of four diagnostic categories (A/B/C/D). Your task is to assign the same taxonomy *independently*, so we can measure agreement (Cohen's κ pairwise, Krippendorff's α).

## 2. The four-category taxonomy

| Code | Name | When to use |
|---|---|---|
| **A** | **Constraint Violation** | The triple breaks a structural / logical / ontological rule of the dataset's domain: e.g. hierarchy depth, reversed transitivity, asymmetry violated, type mismatch with parent class. Examples: *(`person_X`, `_hypernym`, `person_Y`)* in WordNet, where `_hypernym` is supposed to be asymmetric and *(person_Y, _hypernym, person_X)* also exists; *(`actor_A`, `cast_member`, `actor_B`)* where the cast-member relation is directionally swapped. |
| **B** | **Data Quality Issue** | The triple is factually wrong because of extraction noise, canonicalization failure, typo, or misaligned entities. The triple is wrong in a way that human knowledge or simple lookup can confirm immediately. |
| **C** | **Contextual / Polysemy** | The triple is *technically valid* but relies on an obscure or secondary sense of the relation; or on a non-standard reading of the entity. A reviewer with domain expertise could justify keeping the triple, but a generalist reviewer would flag it. |
| **D** | **Rare Exception** | The triple is a *genuine real-world fact* that violates a general ontological norm but is historically or contextually accurate. Example: a person occupying a normally single-tenure position twice in non-consecutive periods. |

If you are stuck between two categories, choose the one **closer to the source of the anomaly**: A and B describe what is *wrong* with the triple; C and D describe why the triple may *still be acceptable*.

## 3. Risk level

| Level | Meaning |
|---|---|
| **High** | Triple is almost certainly an error in the graph; warrants deletion. |
| **Medium-High** | Likely error; warrants verification and probable deletion. |
| **Medium** | Ambiguous; needs subject-matter expert review. |
| **Low** | Probably valid; warrants caution but no immediate action. |

A and B typically map to High / Medium-High; C and D typically to Medium / Low. Use your judgement.

## 4. Recommended action

| Action | When |
|---|---|
| **delete** | Triple is an error; remove from graph |
| **verify** | Needs external verification before action |
| **defer** | Pass to subject-matter expert; do not block on this |

## 5. How to use the context columns

Each row of your CSV includes:

  - `dataset` — `wn18rr`, `fb15k-237`, or `codex-m`. Different datasets have different conventions; expect different patterns of anomalies.
  - `head`, `relation`, `tail` — the triple identifiers. They are typically opaque IDs (Q-codes for codex-m / Wikidata; synset IDs for WN18RR; `/m/...` codes for FB15k-237).
  - `relation_description` — a one-line semantic gloss of the relation (e.g. `_hypernym` → "X is a hyponym of Y"). Use this to decide whether the triple's direction or membership is plausible.
  - `head_lookup_url` and `tail_lookup_url` — direct URLs you can visit to identify the entities. For codex-m (Wikidata) the URLs are stable. For WN18RR the URLs may return a 404 for some senses; in that case, search the synset ID on a WordNet browser.
  - `head_degree_in_graph` and `tail_degree_in_graph` — number of triples involving each entity. High-degree entities (>100) are usually well-known; low-degree entities (1-2) are often peripheral.
  - `relation_freq_in_dataset` — how common this relation is across the dataset. Low frequency may indicate a rare / specialised relation.
  - `inverse_triple_in_graph` — whether `(tail, relation, head)` also exists in the graph. For asymmetric relations (e.g. `_hypernym`), this being TRUE is a strong A signal.
  - `head_other_relations`, `tail_other_relations` — top relations involving each endpoint. Helps you sense-check whether the entity types are plausible.
  - `head_sample_outgoing`, `tail_sample_incoming` — a few example neighbours. If the head is a person and its outgoing edges are typical for an organisation, that is a type-mismatch signal (often B).
  - `ode_energy` — the ODE anomaly score (higher = more anomalous). Above ~8 is in the upper tail.
  - `kge_methods_flagging` — which of {TransE, DistMult, MuRE, ComplEx} also flagged this triple. All four flagged it (it is by definition in the Universal Consensus).

## 6. The sample's structure (informational)

The 100 triples are stratified by judgement category to give statistical power across the taxonomy. Quotas:

| Quota | Why this number |
|---|---|
| **A: 27** | The natural distribution in the Universal Consensus (4047 triples) is A = 10.8%. The sample over-represents A (27% vs natural 11%) so the agreement measurement on the most operationally important category has high statistical power. |
| **B: 26** | The natural distribution is B = 81.0%. The sample under-represents B (26% vs natural 81%) so the other categories are not crowded out. Even at 26%, B is the largest absolute quota. |
| **C: 26** | The natural distribution is C = 7.7%. The sample over-represents C (26% vs natural 8%) to allow meaningful agreement measurement on the polysemy category, which is the most cognitively demanding for the annotator. |
| **D: 21** | The natural distribution is D = 0.5% (only 21 triples in the entire Universal Consensus). All 21 are included so the annotation covers the full rare-exception population. |

Within each judgement category, sub-strata are dataset × constraint_context to spread the sample across the dataset diversity:

  - **Datasets**: wn18rr (45 triples), codex-m (53 triples), fb15k-237 (2 triples). FB is sparse because the Universal Consensus contains only 11 FB triples total.
  - **Constraint context**: asymmetry (45 triples) and none (55 triples). The asymmetry stratum covers the dataset's declared asymmetric relations; the "none" stratum is everything else.

## 7. Submission format

Fill the four columns in the CSV:

  - `annotator_judgment_A_B_C_D` — exactly one of `A`, `B`, `C`, `D`.
  - `annotator_risk_low_med_high` — exactly one of `Low`, `Medium`, `Medium-High`, `High`.
  - `annotator_recommended_action` — exactly one of `delete`, `verify`, `defer`.
  - `annotator_rationale` — 1-2 sentences in English or Spanish (your choice; we will translate as needed). State the evidence: which property of the triple drove the verdict.

When all 100 rows are filled, save and return the CSV file.

## 8. Ethics + reproducibility

  - Annotators are not informed of each other's verdicts during the protocol.
  - Annotators may use external sources (Wikidata, WordNet, web search) but must not consult LLMs.
  - The CoT canonical judgement is held aside until all 3 sheets are returned. The agreement report runs deterministically over the 3 returned sheets (`src/experiments/run_cot_evaluation_metrics.py`).

If you have questions about the protocol that don't change your verdict — note them in `annotator_rationale`. If you encounter a triple whose IDs cannot be resolved to entities even after URL lookup, mark it with `annotator_judgment_A_B_C_D = ?` and explain in rationale; we will treat those as missing-at-random.

---

**Files in your kit**:

  - `sample_100_blind_annotator_{N}.csv` — your CSV to fill (no CoT judgement visible).
  - `ANNOTATION_INSTRUCTIONS.md` — this file.
  - `per_triple_context_pack.json` — same context per triple but in machine-readable form (use only if you need to script lookups).
"""


def main():
    with open(CANONICAL_JSON) as f:
        payload = json.load(f)
    records = payload["records"]
    print(f"Loaded {len(records)} records")

    # Build per-dataset graph indices once
    loader = DataLoader()
    datasets_seen = sorted({r["dataset"] for r in records})
    indices = {}
    for ds in datasets_seen:
        # The records use lowercase dataset names; DataLoader takes the
        # canonical name. Map them.
        ds_load = {
            "wn18rr": "WN18RR",
            "fb15k-237": "FB15k-237",
            "codex-m": "codex-m",
        }.get(ds, ds)
        print(f"  Indexing {ds_load}...")
        triples = loader.load_dataset(ds_load)
        indices[ds] = build_graph_index(triples)

    # Per-triple context
    contexts = {}
    for rec in records:
        ds = rec["dataset"]
        contexts[rec["triple_id"]] = context_for(rec, indices[ds])

    p1 = build_canonical_csv(records, contexts)
    print(f"Wrote canonical CSV: {p1}")
    for ann_id in (1, 2, 3):
        p = build_blind_csv(records, contexts, ann_id)
        print(f"Wrote blind CSV (annotator {ann_id}): {p}")
    p_pack = build_context_pack(records, contexts)
    print(f"Wrote context pack: {p_pack}")
    instr_path = EVAL_DIR / "ANNOTATION_INSTRUCTIONS.md"
    with open(instr_path, "w") as f:
        f.write(INSTRUCTIONS_MD)
    print(f"Wrote instructions: {instr_path}")


if __name__ == "__main__":
    main()
