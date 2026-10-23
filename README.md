# An Empirical Analysis of Constraint-Guided Embedding Dynamics for Unsupervised Anomaly Detection and Diagnosis in Knowledge Graphs

## Abstract

Knowledge graph quality can be examined through statistical plausibility and structural constraints. This study investigates when embedding models and constraint-guided continuous-time dynamics provide useful complementary information for detecting anomalous triples in static knowledge graphs. Experiments on FB15k-237, CoDEx-Medium and WN18RR compare four knowledge graph embedding (KGE) models with deterministic and stochastic dynamics under random substitutions and constraint-oriented perturbations. KGE pipelines provide stronger recovery under random substitution; other comparisons depend on the dataset, configuration and decision rule, with dynamical scores frequently near or below chance. Complementary recovery is supported in the single-seed WN18RR asymmetry-15\% condition: exclusive-recovery analysis identifies additional injected positives at a common review budget, and fusion with TransE improves normalised discounted cumulative gain (NDCG) by 0.027. Separate ten-seed controls using the same symbolic checker favour TransE in NDCG. Component analyses estimate path-dependent contributions of data evolution, constraint forces and symbolic checking. A diagnostic procedure organises residual, variability and symbolic evidence for review; an exploratory expert study finds limited taxonomy agreement and does not evaluate generated explanations. The findings characterise the conditions and limits of complementary detection and support its use for within-graph triage

### Authors 

* Raul del Aguila Escobar @: raul.aguilaescobar@ceu.es
* Mariano Fernandez Lopez @: mariano.fernandez@educacion.gob.es
* Boris Villazon Terrazas @: boris.marcelo.villazon.terrazas@es.ey.com

## Package contents

This is a minimal, reproducible release: the code and data needed to
train the models, run every experiment reported in the manuscript,
regenerate its tables and statistical tests, and run the CoT diagnostic
procedure. Apart from Figure 1, it does not include figures or
visualization code.

- `data/` -- the three benchmark datasets used in the study
- `src/` -- the modules needed to load data, train KGE, run the ODE/SDE
  dynamics, apply constraints and perturbations, score, threshold,
  compute metrics and statistics, run the CoT module, and execute
  `run_all.py`
- `src/experiments/` -- the dedicated experiments of the manuscript:
  paired antisymmetry contrasts (deterministic and SDE), the S0--S4
  progressive ablation, the 2x2 antisymmetry-force x symbolic-bonus
  analysis, the score-weight sensitivity diagnostic and the per-triple
  score capture used by the ranking panel
- `scripts/` -- the producers of the results reported in the manuscript
  - `scripts/canonical/` -- KGE random-5% seeds, S0--S4/2x2 statistics and
    the table builder
  - `scripts/postprocess_review/` -- chance-reference, ranking, exclusive
    recovery, moderator and degree-confounding analyses
- `.cache/hyperparams/` -- the tuned hyperparameters used to produce the
  published results
- `figures/` -- Figure 1 (SDE diffusion-scale sweep), PDF and PNG
- `results/ranking_panel/` -- the complete multimetric ranking panel (Appendix B.1)
- `reviewer_controls/` -- data, scripts and results of the additional checker,
  training and fusion controls (see `reviewer_controls/README.md`)
- `tests/` -- tests verifying table generation
- `generate_tables.py` -- thin wrapper that regenerates the statistical
  families from the outputs of `run_all.py`
- `requirements.txt`, `LICENSE`

## Folder structure

```text
data/
src/
  experiments/
scripts/
  canonical/
  postprocess_review/
tests/
figures/
results/ranking_panel/
reviewer_controls/
.cache/hyperparams/
generate_tables.py
README.md
requirements.txt

```

## Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS/Apple Silicon: the optional `PyClause` dependency needs OpenMP
(`brew install llvm libomp` before `pip install`). Every other dependency
installs without it. PyTorch runs on CPU, MPS or CUDA.

## Main execution

All commands run from the repository root. Setting `CANONICAL_PERSIST_DIR`
stores the evaluation samples and detection sets that the consensus and
table-building steps read; set it for the whole session.

```bash
export CANONICAL_PERSIST_DIR="$PWD/canonical_persist" PYTHONHASHSEED=0

# KGE (TransE, DistMult, MuRE, ComplEx): native detection and the
# controlled random and constraint-oriented ablations, per dataset
for ds in FB15k-237 codex-m WN18RR; do
  python3 src/run_all.py --methods TransE DistMult MuRE ComplEx \
      --no-interactive --skip-postprocess --datasets $ds --stages native random_ablation logical_ablation
done

# Check the command and stage graph without running anything:
python3 src/run_all.py --methods TransE DistMult MuRE ComplEx --datasets WN18RR --stages paper --dry-run
```

Hyperparameters in `.cache/hyperparams/` are the ones used to produce the
published results. `tune` is not part of any stage alias and the controlled
stages never fall back to default KGE parameters. To force re-tuning:
`python3 src/run_all.py --datasets <ds> --stages tune --force-tune`
(requires typing `regenerate`; non-deterministic, will diverge from the
published numbers). Re-tuning first needs the tuning subsets:
`python3 src/experiments/build_hyperparam_subsamples.py`.

## Pipeline stages

| Stage | Depends on |
|---|---|
| `tune` | -- (explicit only) |
| `native` | -- |
| `random_ablation` | `native` |
| `logical_ablation` | `native` |
| `explanations` | `random_ablation`, `logical_ablation` |
| `visualizations` | `native` |
| `paper` | `native`, `random_ablation`, `logical_ablation` |
| `all` | `paper` plus `explanations` and `visualizations` |

Figure/visualization code is not part of this release. The `visualizations`
stage prints a one-line skip notice and continues; every metrics/detections
JSON that tables and statistics read is written regardless.

## Reproducing the manuscript experiments

After the main execution, run the producers below (same session, same
`CANONICAL_PERSIST_DIR`). Each writes its outputs under `paper_results/`,
`progress/` or `canonical_persist/`. The order respects their dependencies.

```bash
DS="FB15k-237 codex-m WN18RR"
for ds in $DS; do
  # Random-5% KGE seeds (Table 5)
  python3 scripts/canonical/run_kge_random5_seeds.py --dataset $ds
  # Single-seed A2 and A1 grids (Tables 5-8, ranking panel)
  python3 scripts/milestone_postfix_section4_rerun.py --datasets $ds
  python3 scripts/milestone_postfix_section4_rerun.py --datasets $ds --lambda-asym-override 0.0 --skip-bootstrap
  # Paired antisymmetry contrasts: deterministic (A1/A2) and SDE (A3)
  python3 src/experiments/run_c_h_asym_deterministic.py --dataset $ds
  python3 src/experiments/run_sde_b6_wilcoxon.py --dataset $ds
  # S0-S4 progressive ablation (Tables progressive_f1/ndcg/stats)
  python3 src/experiments/run_s0_s4_progressive_ablation.py --datasets $ds
done

# A3 (SDE) ranking cells of the nine shared panel conditions
while read ds vt n; do
  python3 scripts/run_sde_per_cell.py --dataset $ds --vtype $vt --n-violations $n
done <<'EOF'
WN18RR transitivity 4650
codex-m cardinality 30930
WN18RR asymmetry 13950
FB15k-237 transitivity 46517
WN18RR cardinality 13950
codex-m asymmetry 20620
codex-m transitivity 10310
codex-m cardinality 10310
FB15k-237 cardinality 15505
EOF

# 2x2 antisymmetry force x symbolic bonus (WN18RR) and score-weight sensitivity
python3 src/experiments/run_c2_dynamics_symbolic_2x2.py --dataset WN18RR
python3 src/experiments/run_b_weight_sensitivity.py

# Native A2, structural/overlap/correlation evidence (E1-E5) and consensus (Table 12)
python3 scripts/audit_phase2_E_rerun.py --datasets $DS --n-steps 50 --skip-e6
python3 scripts/audit_phase2_E2E3_postfix.py
python3 scripts/extract_e1_e5_evidence.py
python3 scripts/compute_tripartite_postfix.py --datasets $DS

# Ranking panel (Table 10) and operator swap (Appendix)
python3 src/experiments/post_analysis_capture.py
python3 scripts/compute_full_panel.py
python3 scripts/compute_parallel_operator_sweep.py
python3 scripts/compute_r02_swap_means.py
```

| Manuscript result | Producer |
|---|---|
| Tables 4--8, 12 | `src/run_all.py`, `scripts/canonical/run_kge_random5_seeds.py`, `scripts/milestone_postfix_section4_rerun.py`, `scripts/audit_phase2_E_rerun.py`, `scripts/compute_tripartite_postfix.py` |
| Table 9, paired antisymmetry contrasts | `src/experiments/run_c_h_asym_deterministic.py`, `src/experiments/run_sde_b6_wilcoxon.py`, `scripts/run_sde_per_cell.py` |
| Table 10, ranking panel | `src/experiments/post_analysis_capture.py`, `scripts/compute_full_panel.py` |
| S0--S4 progressive ablation | `src/experiments/run_s0_s4_progressive_ablation.py`, `scripts/canonical/compute_s0s4_2x2_stats.py` |
| 2x2 force x symbolic bonus | `src/experiments/run_c2_dynamics_symbolic_2x2.py`, `scripts/canonical/compute_s0s4_2x2_stats.py` |
| Score-weight sensitivity | `src/experiments/run_b_weight_sensitivity.py` |
| Operator swap | `scripts/compute_parallel_operator_sweep.py`, `scripts/compute_r02_swap_means.py` |
| Chance references, exclusive recovery, moderators, degree confounding | `scripts/postprocess_review/` |

## Generate manuscript tables

```bash
python3 scripts/canonical/build_canonical_tables.py --public . tables_out
```

Writes one CSV per manuscript table (`tables_out/csv/`) and
`tables_out/TABLE_VALUES.json`, with the source files and producer of
every value. The human-study tables are not recomputed (the expert
annotations are not part of this release).

```bash
python3 generate_tables.py
```

Requires `src/run_all.py` to have been run first for the datasets you
need. It calls the existing producer scripts under `scripts/` and
`src/experiments/`, in the order needed for each to read the previous
one's output, and writes a summary to `GENERATE_TABLES_REPORT.json`. Each
producer writes its own output files under `paper_results/` and
`progress/`.

By default it only runs the producers verified, by actually running them
during packaging, to reproduce their expected table structure from real
data: the E6/HASym/SDE/R02 statistical families and the ODE-side native
re-run. A further set of producers (E1-E5 evidence, tripartite consensus,
the ranking panel) is available behind `--include-not-verified`, with a
printed warning -- running them was observed, during packaging, to
overwrite their own reference output with a different or incomplete
result, so check their output against the expected row/cell counts
before trusting it.

## Statistical analysis

```bash
python3 scripts/compute_e6_triviality_gap.py
python3 scripts/compute_hasym_and_sde_holm.py
python3 scripts/compute_r02_swap_means.py
python3 scripts/canonical/compute_s0s4_2x2_stats.py
```

The first three produce the manuscript's paired statistical tests: the
E6_RANDOM_9, HASym_DET_3 and HASym_SDE_REVIEW_3 families (paired Wilcoxon,
Holm-Bonferroni within each family, rank-biserial effect sizes), and the
operator-swap comparison. `compute_s0s4_2x2_stats.py` produces the twelve
S0--S4 sequential contrasts (one Holm family) and the four 2x2 simple
effects with the difference-of-differences. `generate_tables.py` runs the
first three.

`scripts/compute_e6_triviality_gap.py` reads the ODE side of E6 from
`scripts/milestone_postfix_section4_rerun.py`'s output
(`paper_results/<ds>/bootstrap_postfix/`), not from `run_all.py`'s own
bootstrap-stage output directly -- run that script first if it has not
already run as part of `generate_tables.py`.

## Post-processing analyses

```bash
P=postprocess_review
python3 scripts/postprocess_review/f1_chance.py . $P
python3 scripts/postprocess_review/e1_chance.py . $P
python3 scripts/postprocess_review/ranking_audit.py . $P
python3 scripts/postprocess_review/r2_2_exclusive_recovery.py . $P $P/r2_2_exclusive_recovery
python3 scripts/postprocess_review/usefulness_moderators.py . $P $P/usefulness_moderators
python3 scripts/postprocess_review/usefulness_patterns.py $P/usefulness_moderators
python3 scripts/postprocess_review/final_closeout/degree_confounding.py . $P $P/final_closeout
```

These produce the chance-reference diagnostics (F1 lift, NDCG@K and E1
overlap against random references), the equal-budget exclusive-recovery
analysis, the counterexample-first moderator stress test and the
degree-confounding diagnostic. They read persisted scores and samples only.

## Figure 1

`figures/figure1_sde_calibration.{pdf,png}` shows the SDE diffusion-scale
sweep on 50 WN18RR triples with ten trajectories per triple: per-triple
variance (mean and SD as separate series; since mean - SD <= 0 at every
sigma, no +/-SD bars are drawn on the logarithmic axis) and the coefficient
of variation. The dotted line marks the fixed A3 configuration sigma = 0.05.

## Ranking panel and additional controls

`results/ranking_panel/full_panel.json` contains the complete multimetric
ranking panel of the canonical execution: Hit@K, NDCG@K, BEDROC, EF@1%,
AUROC, pAUC@5% and PR-AUC for every populated detector--condition record.

`reviewer_controls/` contains the additional controls: the symbolic
checker on its own, conventionally trained TransE with the same checker,
common decision rules, rank fusion with a permuted-score control, and
the training-seed, dimension, degree and checker-scope analyses. It
includes the subset of the canonical execution they read, the per-triple
scores of the new runs, the result tables and the commands to reproduce
them; see `reviewer_controls/README.md`.

## CoT / diagnostic module

```bash
python3 scripts/build_cot_evaluation_sample.py
python3 scripts/build_cot_annotation_csv.py
python3 src/experiments/run_cot_evaluation_metrics.py
```

Uses `Qwen/Qwen2.5-3B-Instruct` (downloaded automatically on first use).
Set `ENABLE_LLM=False` to use the deterministic rule-based fallback
instead, which requires no download. Inputs: the Universal Consensus
candidate set produced by the main pipeline. Output: per-triple diagnostic
judgments and the human-agreement metrics (Cohen's kappa, Krippendorff's
alpha) over the annotated sample.

## Minimal validation

```bash
python3 smoke_test.py
# After running the main pipeline (see "Main execution") and
# generate_tables.py's VERIFIED_STEPS:
pytest tests/ -v
```

`tests/` verifies table generation and needs the outputs of the main
pipeline to exist first -- it is not meant to pass on a bare clone with
no experiments run yet.

`smoke_test.py` runs a fast, reduced-scale pass through the pipeline.
`tests/` verifies table generation: that the producers run, produce the
expected number of tables/rows, and never treat a missing or placeholder
value as a valid result.

## License


[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This project and its contents are distributed under the terms of the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

You are free to:
* **Share:** copy and redistribute the material in any medium or format.
* **Adapt:** remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
