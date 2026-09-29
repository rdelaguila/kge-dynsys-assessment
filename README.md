# An Empirical Analysis of Constraint-Guided Embedding Dynamics for Unsupervised Anomaly Detection and Diagnosis in Knowledge Graphs

## Abstract

Knowledge graph quality can be examined through statistical plausibility and structural constraints. This study investigates whether established embedding models and a continuous-time dynamical formulation provide different information for detecting anomalous triples in static knowledge graphs. The dynamical formulation combines data-fidelity and constraint-derived forces with a score decomposition into terminal energy, trajectory variance and symbolic-check contributions. Experiments on FB15k-237, CoDEx-Medium and WN18RR compare four embedding models with three dynamical configurations under random substitutions and constraint-oriented perturbations. The embedding baselines achieve higher F1 in the reported random conditions, whereas the remaining comparisons depend on dataset, configuration and decision rule. Paired contrasts on WN18RR show changes in F1 of -0.042 and +0.016 when activating the antisymmetry gradient in deterministic and stochastic configurations, respectively. Cross-method agreement identifies 3,136 candidate triples. A three-annotator study did not establish reliable agreement in applying the diagnostic taxonomy; its labels do not validate the full candidate set. The findings characterise differences between geometric and dynamical detection signals without establishing universal superiority or validating every consensus-selected candidate. The evaluation distinguishes recovery of injected perturbations from formal ontological inconsistency, and the diagnostic procedure organises score components rather than demonstrating causal explanations.

### Authors 

* Raul del Aguila Escobar @: raul.aguilaescobar@ceu.es
* Mariano Fernandez Lopez @: mariano.fernandez@educacion.gob.es
* Boris Villazon Terrazas @: boris.marcelo.villazon.terrazas@es.ey.com

## Package contents

This is a minimal, reproducible release: the code and data needed to
train the models, regenerate the manuscript's tables, run the final
statistical tests, and run the CoT diagnostic procedure. It does not
include figures, visualization code, or documentation beyond this file.

- `data/` -- the three benchmark datasets used in the study
- `src/` -- the modules needed to load data, train KGE, run the ODE/SDE
  dynamics, apply constraints and perturbations, score, threshold,
  compute metrics and statistics, run the CoT module, and execute
  `run_all.py`
- `scripts/` -- the producers of the results and tables reported in the
  manuscript
- `.cache/hyperparams/` -- the tuned hyperparameters used to produce the
  published results
- `tests/` -- tests verifying table generation
- `generate_tables.py` -- thin wrapper that regenerates every table from
  the outputs of `run_all.py`
- `requirements.txt`, `LICENSE`

## Folder structure

```text
data/
src/
scripts/
tests/
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

`src/run_all.py` is the main entry point. It trains KGE and the
ODE and runs the full detection/evaluation protocol.

```bash
python3 src/run_all.py --datasets fb15k-237 --stages all --no-interactive
python3 src/run_all.py --datasets codex-m --stages all --no-interactive
python3 src/run_all.py --datasets WN18RR --stages all --no-interactive

# Check the command and stage graph without running anything:
python3 src/run_all.py --datasets WN18RR --stages all --dry-run
```

Hyperparameters in `.cache/hyperparams/` are the ones used to produce the
published results; `tune` is skipped automatically while they are
present. To force re-tuning: `python3 src/run_all.py --datasets <ds>
--stages tune --force-tune` (non-deterministic, will diverge from the
published numbers).

## Pipeline stages

| Stage | Depends on |
|---|---|
| `tune` | -- |
| `native` | `tune` |
| `random_ablation` | `native` |
| `logical_ablation` | `native` |
| `explanations` | `random_ablation`, `logical_ablation` |
| `visualizations` | `native` |
| `all` | every stage above, in order |

Figure/visualization code is not part of this release. The `visualizations`
stage and the visualization step inside `native` print a one-line skip
notice and continue; every metrics/detections JSON that tables and
statistics read is written regardless.

## Generate manuscript tables

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
```

These produce the manuscript's final statistical tests: the E6_RANDOM_9,
HASym_DET_3 and HASym_SDE_REVIEW_3 families (paired Wilcoxon,
Holm-Bonferroni within each family, rank-biserial effect sizes), and the
operator-swap comparison. `generate_tables.py` runs the same scripts.

`scripts/compute_e6_triviality_gap.py` reads the ODE side of E6 from
`scripts/milestone_postfix_section4_rerun.py`'s output
(`paper_results/<ds>/bootstrap_postfix/`), not from `run_all.py`'s own
bootstrap-stage output directly -- run that script first if it has not
already run as part of `generate_tables.py`.

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
