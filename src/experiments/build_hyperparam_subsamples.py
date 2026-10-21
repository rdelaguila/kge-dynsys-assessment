"""
Build stratified-by-relation hyperparameter subsamples (Block 4.1.a).

For each of the three real datasets, draw a subsample whose relation
distribution mirrors the full graph. Save the triple indices and a
metadata JSON (sample size, original size, seed, relation
distribution before/after).

Output:
    paper_results/<dataset>/hyperparam_subsample.npy
    paper_results/<dataset>/hyperparam_subsample_meta.json

Reproducibility: numpy seed = 42.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from data import DataLoader, Triple

DATASETS = {
    "FB15k-237": 30_000,
    "codex-m":   25_000,
    "WN18RR":    20_000,
}
SEED = 42


def stratified_indices(triples: list[Triple], target: int, seed: int) -> np.ndarray:
    """Return stratified-by-relation indices preserving relative frequency.

    Per relation r with n_r triples, allocate
        k_r = round(n_r * target / N)
    then top up or trim by ±1 to hit exactly ``target`` total — the
    adjustment is applied to the relations with largest fractional
    remainders so the resulting distribution stays closest to the
    original frequencies.
    """
    rng = np.random.default_rng(seed)
    N = len(triples)
    by_rel: dict[str, list[int]] = {}
    for i, t in enumerate(triples):
        by_rel.setdefault(t.relation, []).append(i)

    raw_alloc: dict[str, float] = {r: len(idxs) * target / N for r, idxs in by_rel.items()}
    base_alloc = {r: int(np.floor(v)) for r, v in raw_alloc.items()}
    remainders = {r: raw_alloc[r] - base_alloc[r] for r in by_rel}

    deficit = target - sum(base_alloc.values())
    # Distribute the rounding deficit (positive or zero by construction)
    # to the relations with the largest fractional remainder.
    if deficit > 0:
        order = sorted(remainders.items(), key=lambda kv: -kv[1])
        for r, _ in order[:deficit]:
            base_alloc[r] += 1

    # Trim allocations that exceed available triples.
    for r in base_alloc:
        base_alloc[r] = min(base_alloc[r], len(by_rel[r]))

    # If trimming created a deficit, redistribute among relations with
    # remaining headroom.
    while sum(base_alloc.values()) < target:
        slack = [r for r in base_alloc if base_alloc[r] < len(by_rel[r])]
        if not slack:
            break  # All relations exhausted — sample size below target.
        bumped = False
        for r in slack:
            if sum(base_alloc.values()) >= target:
                break
            base_alloc[r] += 1
            bumped = True
        if not bumped:
            break

    sampled: list[int] = []
    for r, k in base_alloc.items():
        if k <= 0:
            continue
        idxs = by_rel[r]
        if k >= len(idxs):
            sampled.extend(idxs)
        else:
            sampled.extend(rng.choice(idxs, size=k, replace=False).tolist())

    sampled_arr = np.array(sorted(sampled), dtype=np.int64)
    return sampled_arr


def relation_distribution(triples: list[Triple]) -> dict[str, int]:
    return dict(Counter(t.relation for t in triples))


def ks_stat(p_full: dict[str, int], p_sample: dict[str, int]) -> float:
    """Discrete KS-style statistic on the sorted relation-frequency CDFs."""
    relations = sorted(set(p_full.keys()) | set(p_sample.keys()),
                       key=lambda r: -p_full.get(r, 0))
    n_full = sum(p_full.values()) or 1
    n_sample = sum(p_sample.values()) or 1
    cdf_full = 0.0
    cdf_sample = 0.0
    max_diff = 0.0
    for r in relations:
        cdf_full += p_full.get(r, 0) / n_full
        cdf_sample += p_sample.get(r, 0) / n_sample
        max_diff = max(max_diff, abs(cdf_full - cdf_sample))
    return max_diff


def main() -> None:
    loader = DataLoader()
    for ds, target in DATASETS.items():
        print(f"\n=== {ds} → target {target} ===")
        triples = loader.load_dataset(ds)
        print(f"  loaded {len(triples)} triples, {len(set(t.relation for t in triples))} relations")

        idxs = stratified_indices(triples, target, SEED)
        sampled = [triples[i] for i in idxs]
        print(f"  sampled {len(sampled)} triples ({len(set(t.relation for t in sampled))} relations)")

        out_dir = Path("paper_results") / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "hyperparam_subsample.npy", idxs)

        full_dist = relation_distribution(triples)
        samp_dist = relation_distribution(sampled)
        ks = ks_stat(full_dist, samp_dist)

        meta = {
            "dataset": ds,
            "original_size": len(triples),
            "sample_size": int(len(idxs)),
            "target": target,
            "sampling_seed": SEED,
            "stratification": "by-relation, proportional-with-largest-remainders",
            "ks_relation_cdf": float(ks),
            "n_relations_full": len(full_dist),
            "n_relations_sample": len(samp_dist),
            "relation_distribution_full":   full_dist,
            "relation_distribution_sample": samp_dist,
        }
        with open(out_dir / "hyperparam_subsample_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  KS(full vs sample) = {ks:.4f}")
        print(f"  wrote {out_dir / 'hyperparam_subsample.npy'}")
        print(f"  wrote {out_dir / 'hyperparam_subsample_meta.json'}")


if __name__ == "__main__":
    main()
