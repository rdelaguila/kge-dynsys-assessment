# Second-round reviewer controls

This folder contains the code, per-triple scores and result tables of the analyses run for the second review round.
Most of the requests come from Reviewer 4:

- a ten-seed replication of WN18RR asymmetry at 15 %;
- degree controls;
- A1 in fusion and in exclusive recovery;
- TransE trained on the full perturbed graph;
- a common counting convention.

The changes requested by Reviewer 3 concern presentation, and none of the experiments here is attributed to them.

The scripts reuse the functions of `src/`, `scripts/` and `reviewer_controls/scripts/` unchanged. No force,
perturbation generator, training routine, score, fusion rule, hyperparameter or seed was modified.

## Analyses

| Script | Question | Trains or integrates |
|---|---|---|
| `r2_asym15_run.py` | Ten new perturbation realisations of WN18RR asymmetry-15 % (13,950 injected triples), with the original 15 % protocol. Per seed it computes A2, A1, TransE/DistMult/MuRE and the head, tail and summed degree at the sample positions | Yes: ODE integration on CPU and KGE training |
| `r2_asym15_analysis.py` | Does the gain of Fusion(A2, TransE) over TransE hold across realisations? Does it survive a degree-only fusion? What happens without the antisymmetry force (A1) and with DistMult/MuRE? It also covers exclusive top-K recovery, the permuted-A2 control, the seed-42 runner check, and families F1–F3 | No |
| `r2_degree_nine_cells.py` | Head degree d(h) and tail degree d(t) as separate label-free scores, and their fusion with TransE, in the nine original panel conditions | No |
| `r2_fullpool_transe_run.py` | TransE trained on the full perturbed pool (not the 10,000-triple sample), scored on the paired asymmetry-10 % positions | Yes: TransE training on CPU |
| `r2_fullpool_analysis.py` | S1+C and S3+C compared with TransE_full+C under the same checker and lexicographic ordering (family F4), and the change from sample-trained to pool-trained TransE | No |
| `r2_common_counting.py` | Projects every persisted detection set onto the evaluation positions and recomputes TP/FP/FN/TN and F1 per position against the unique-triple convention | No |
| `diag_a2_hashseed.py`, `diag_a1_order.py` | Numerical diagnostics for the seed-42 reference: cross-process reproducibility (`PYTHONHASHSEED`) and the constraint-structure cache path | Yes: ODE integration, seed 42 only |
| `r2_common.py` | Shared helpers: the percentile fusion and lexicographic ordering (verbatim copies from `stage2_block_a.py`), the exclusive regions (verbatim from `scripts/postprocess_review/r2_2_exclusive_recovery.py`), metrics via `compute_full_panel.metrics_for`, and contrasts and families via `src/statistical_tests.py` | — |

## Effective protocol

The protocol was fixed before any new result was inspected.

### Asymmetry-15 % panel (WN18RR)

| Element | Setting |
|---|---|
| Perturbation | `KGPerturbator(base, seed=s).perturb_logical("asymmetry", 13950, "WN18RR")` with perturbation seeds **1–10** |
| Sample | `run_all.sample_triples(perturbed, viol, 10000, seed=s)`, which draws its stream from `stable_seed(s, "evaluation")` |
| Reference | **Seed 42** is a separate reference: it is used to check the runner against the frozen artefacts, never as an eleventh panel seed |
| A2 | `milestone_postfix_section4_rerun.build_kgode_postfix(seed=42)`. Initialisation is fixed (`randn·0.1`, float32, seed 42) with cached λ, followed by `rk4_terminal(50 steps)` and `per_triple_energy` on CPU. Energies are mapped to the sample positions as in `detect_one_cell` |
| A1 | As A2, with `lambda_asym_override=0.0` |
| KGE | The call of `post_analysis_capture.py`: `PyKEENDetector(cached dim).train_model(sample, ABLATION_EPOCHS=10, ABLATION_BATCH=128, cached lr/margin/negatives)`, PyKEEN seed 42, automatic backend (MPS on Apple Silicon). Score = −raw plausibility |
| Degree | Incidence counts on the full perturbed graph, over all relations and both directions, as in `stage1_inputs.py`. `deg_sum = d(h)+d(t)` is the pre-specified control; `deg_h` and `deg_t` are descriptive. Higher degree ranks first |
| Fusion | `(pct(a) + pct(b)) / 2`, with `pct = rankdata(average) / (n+1)` |
| Metrics | NDCG@K with K = P (sklearn); also AUROC and PR-AUC |

### Full-pool TransE (asymmetry-10 %)

| Element | Setting |
|---|---|
| Panel | FB15k-237, CoDEx-M and WN18RR × perturbation seeds 1–10. The samples and the checker count C are those of `stage1_inputs.py`, and each sample's SHA-256 is checked |
| Training | Same call as `stage5_block_b_run.py` (cached hyperparameters, 10 epochs, batch 128, PyKEEN seed 42, CPU). The only change is the training population: the full perturbed pool instead of the sample. The loader does not deduplicate |
| Training budget | Equal epochs mean different update counts: 25,690 for FB15k-237, 17,730 for CoDEx-M and 8,000 for WN18RR, against 790 on the sample. The meta field of every npz records this budget |

### Statistics

| Element | Setting |
|---|---|
| Unit | The perturbation realisation (n = 10) |
| Test | `statistical_tests.paired_wilcoxon`: two-sided, `zero_method="wilcox"`, signed rank-biserial (W+ − W−)/(W+ + W−) |
| Intervals | `bootstrap_ci`: percentile interval for the mean, 50,000 resamples, seed 12345 |
| Multiplicity | `holm_bonferroni` within each family |

| Family | Contrasts |
|---|---|
| F1 | Fus(A2,TransE)−TransE; Fus(deg_sum,TransE)−TransE; Fus(A2,TransE)−Fus(deg_sum,TransE) |
| F2 | Fus(A1,TransE)−TransE; Fus(A1,TransE)−Fus(A2,TransE) |
| F3 | Fusions of A2 and A1 with DistMult and MuRE, minus the KGE alone |
| F4 | Per dataset: (S1+C)−(TransE_full+C) and (S3+C)−(TransE_full+C) |

Contrasts with `deg_h`/`deg_t`, enrichment summaries and the permuted-A2 control (1,000 permutations,
`default_rng(0)`) are descriptive.

### Environment

`PYTHONHASHSEED=0` is required, as in the frozen execution manifest. Without it, the ODE integration is not
reproducible across processes (diagnostic D1). The runners disable reading the constraint-structure cache
(`data._cache_load`), as `stage1_inputs.py` and `stage5_block_b_run.py` do.

## Inputs and availability

| Material | Used by | Where |
|---|---|---|
| Benchmark datasets, `.cache/hyperparams/` | runners | Published: `data/`, `.cache/hyperparams/` |
| Frozen panel scores (A1/A2 npz, KGE `ablation_audit`), samples, A2 records for random-5 % and native | analysis, nine cells, counting subset | Published: `reviewer_controls/frozen_inputs/` |
| Stage-1 inputs (perturbed-sample triples, labels, checker count C, degree) | asym15 analysis, nine cells, full-pool run and analysis | Regenerated by `reviewer_controls/scripts/stage1_inputs.py` (no training) |
| `block_a/metrics.csv` (coherence reference) | asym15 analysis, nine cells | Regenerated by `stage2_block_a.py`; the published copy is `reviewer_controls/results/block_a/` |
| S1/S3/TransE-on-sample scores (Block B) | full-pool analysis | Published: `reviewer_controls/raw_outputs/block_b_raw/` |
| Block B published metrics | full-pool coherence check | Published: `reviewer_controls/results/block_b/per_seed_metrics.csv` |
| Per-triple scores of the new runs | analyses | **Published here:** `raw_outputs/asym15/` (11 realisations: seeds 1–10 and reference 42) and `raw_outputs/fullpool/` (30 realisations) |
| Seed-42 diagnostic vectors | D2/D3 | Published here: `raw_outputs/diag/` |
| `R2_2_EXCLUSIVE_RECOVERY.csv` (coherence reference for exclusive regions) | asym15 analysis | Published here: `results/r0_reference/`. It was produced by `scripts/postprocess_review/r2_2_exclusive_recovery.py` from the complete canonical execution; its other inputs (logical A2 detection records, `RANKING_RECORDS_RECOMPUTED.csv`) are **kept locally only** |
| All 354 canonical detection records (336 with a persisted sample) | full common-counting run | **Kept locally only** (≈ 56 MB of records, with per-triple scores). The published `frozen_inputs` subset (30 random-5 % A2 records) reproduces its 30 rows exactly |
| ODE 30-trial search per trial; S0–S4 and 2×2 per-triple energies; per-triple scores of the 18-05-2026 native run; the June-2026 consensus lists | — | **Not available** (never persisted) |

## Reproducing the results

Run all commands from the repository root.

```bash
export PYTHONHASHSEED=0
FI=reviewer_controls/frozen_inputs
WK=reviewer_controls/work
R2=reviewer_controls/round2
```

### 1. From stored scores (no training)

```bash
mkdir -p $WK/inputs $WK/r2_2
for ds in FB15k-237 codex-m WN18RR; do
  python reviewer_controls/scripts/stage1_inputs.py $ds $FI $WK/inputs      # about 12 min in total
done
python reviewer_controls/scripts/stage2_block_a.py $FI $WK                   # writes $WK/block_a
cp -R reviewer_controls/raw_outputs/block_b_raw $WK/
cp -R $R2/raw_outputs/asym15 $R2/raw_outputs/fullpool $WK/
cp $R2/results/r0_reference/R2_2_EXCLUSIVE_RECOVERY.csv $WK/r2_2/

python $R2/scripts/r2_asym15_analysis.py $FI $WK      # -> $WK/asym15_analysis    (a few minutes)
python $R2/scripts/r2_degree_nine_cells.py $FI $WK    # -> $WK/degree_nine_cells
python $R2/scripts/r2_fullpool_analysis.py $WK reviewer_controls/results/block_b/per_seed_metrics.csv
                                                      # -> $WK/fullpool_analysis
python $R2/scripts/r2_common_counting.py $FI $WK/counting_subset   # published subset only
```

The regenerated tables can be compared with `results/`. Every analysis checks itself against its reference before it
writes any statistics, and stops if a check fails:

| Analysis | Coherence check |
|---|---|
| asym15 | 162 checks against `block_a` and R2.2, at 1e-12 |
| nine cells | 45 checks against `block_a` |
| full pool | 120 checks against the published Block B metrics |

**The full counting run** needs the complete canonical execution root, which is kept locally only:
`python $R2/scripts/r2_common_counting.py <execution_root> $WK/counting`.

### 2. Re-running the training and integration

```bash
# Asymmetry-15 % panel: about 8 min per seed. ODE on CPU; KGE on the automatic backend (MPS on Apple Silicon)
python $R2/scripts/r2_asym15_run.py $WK/asym15_rerun 42 1 2 3 4 5 6 7 8 9 10

# Full-pool TransE: about 190 / 230 / 550 s per realisation for FB15k-237 / CoDEx-M / WN18RR on CPU
for ds in FB15k-237 codex-m WN18RR; do
  KG_FORCE_DEVICE=cpu python $R2/scripts/r2_fullpool_transe_run.py $WK/inputs $WK/fullpool_rerun $ds 1 2 3 4 5 6 7 8 9 10
done

# Seed-42 diagnostics (D1–D3)
python $R2/scripts/diag_a2_hashseed.py $WK/a2_nocache.npy nocache
python $R2/scripts/diag_a2_hashseed.py $WK/a2_cache.npy cache          # run after a nocache run has written .cache/
python $R2/scripts/diag_a1_order.py $WK/a1_nocache.npy nocache a1_only
python $R2/scripts/diag_a1_order.py $WK/a1_cache.npy cache a1_only
```

These runs write to `work/`, which `.gitignore` excludes, and do not overwrite `raw_outputs/`. They may write
constraint-structure caches to `.cache/`, which is also ignored.

### What a re-run reproduces

**Verified bitwise** with `PYTHONHASHSEED=0`:
- A2 at seed 42, which equals the frozen vector and is identical across processes (D1);
- the samples, the labels and the summed degree at seed 42 against the frozen artefacts.

Cross-process identity of the CPU TransE trainings on the full pool was not tested.

**Not bitwise:**
- **KGE scores on MPS.** At seed 42 the maximum absolute difference against the frozen scores is 0.383 for TransE,
  2.4·10⁻⁷ for DistMult and 0.080 for MuRE, with Spearman ≥ 0.992. The fusion gain at seed 42 is +0.0339, against
  +0.0275 in the frozen panel. No parameter was adjusted.
- **A1 against the frozen A1.** The frozen A1 read cached transitivity structures that the frozen A2 run had just
  written, while the panel computes both A2 and A1 from scratch. Through the cache path, A1 at seed 42 is bitwise equal
  to the frozen vector. The effect on NDCG@K is 0.000776 for A1 alone and 0.001831 for A2 alone; with fusion it is
  0.001682 for A1 and 0.002179 for A2 (`results/diag/D2_D3_cache_path_seed42.csv`).

## Outputs

| Folder | Content |
|---|---|
| `results/asym15_analysis/` | `metrics_per_seed`, `exclusive_recovery`, `fusion_comparisons` (with `deg_h`/`deg_t` rows), `a1_a2_comparisons`, `permutation_control`, `paired_statistics_F1_F3`, `descriptive_summaries`, `seed42_runner_check`, `evaluator_coherence` |
| `results/degree_nine_cells/` | `metrics`, `coherence_block_a` |
| `results/fullpool_analysis/` | `per_seed_metrics`, `paired_statistics_F4`, `descriptive_contrasts`, `coverage_and_cost`, `coherence_block_b` |
| `results/counting/` | `common_counting_per_record` (336 records), `common_counting_ordering`, `common_counting_missing` (the 18 native and consensus records without labels; they enter no supervised metric) |
| `results/r0_reference/` | R2.2 exclusive-recovery reference and its identity checks |
| `results/diag/` | Seed-42 cache-path comparison |
| `docs/` | `R2_CODE_CONFIG_RECONCILIATION_DOCUMENTED.csv` (first submission against the revision: what changed, with evidence labels) and `R2_REPRODUCTION_MAP_DOCUMENTED.csv` (table and figure → producer, inputs, availability and verification status). Both are in Spanish |

## Adaptations made when moving the scripts into the repository

- **Paths.** The scripts originally ran from a copy of this repository in a separate working folder. In this folder:
  - `r2_asym15_analysis.py`, `r2_degree_nine_cells.py` and `r2_fullpool_analysis.py` read and write under a working
    directory `<work>` with the layout of `reviewer_controls/README.md` (`inputs/`, `block_a/`, `block_b_raw/`);
  - `r2_fullpool_analysis.py` takes the Block B metrics file as an argument instead of the repository root.
- **Docstrings.** Usage lines and references to the protocol were updated, and a docstring was added to
  `diag_a2_hashseed.py`.
- **`r2_common_counting.py`.** It now writes the ordering table header even when no cell has two methods, which happens
  with the published subset. With the complete execution root the three output files are byte-identical to the
  previous version.

No algorithm, hyperparameter, seed, decision rule, metric or result was changed. The result files are the original
outputs.
