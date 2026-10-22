# Additional checker, training and fusion controls

This folder contains the data, scripts and results of the additional controls reported in the appendix
"Additional Checker, Training and Fusion Controls" of the manuscript:

- the symbolic checker on its own;
- conventionally trained TransE with the same checker as the dynamical analysis;
- common decision rules;
- equal-weight rank fusion with a permuted-score control;
- bonus-percentile sensitivity;
- TransE training-length, dimension and training-seed controls;
- degree adjustment;
- checker scope;
- an audit of the knee thresholds of the dynamical results.

The complete multimetric ranking panel referred to in Appendix B.1 is in `results/ranking_panel/full_panel.json`
at the repository root.

## Contents

| Path | Content |
|---|---|
| `scripts/` | One script per stage (`stage1` … `stage10`) and `panel_cells.txt` (the nine shared ranking-panel cells) |
| `frozen_inputs/` | The subset of the canonical execution read by the controls, with the same layout as the execution root. It contains the evaluation samples (`canonical_persist/samples`), A1/A2/A3 per-triple scores (`progress/20260623`), per-triple KGE scores of the panel (`paper_results/<ds>/ablation_audit`), persisted A2 detections (random-5% seeds and native), and the S0–S4 per-seed summaries |
| `raw_outputs/` | Per-triple scores produced by the new training/integration runs: S1–S3 residuals and TransE scores on the paired asymmetry-10% samples (`block_b_raw`); TransE on the paired random-5% samples and with training seeds 43/44 (`extra_raw`); TransE after 1/2/5/10 epochs on the panel cells (`transe_epochs`) |
| `results/` | The result tables written by the analysis stages (CSV/JSON) |

## Stages

| Stage | What it does | Trains or integrates |
|---|---|---|
| `stage1_inputs.py` | Regenerates every perturbed graph and evaluation sample, checks it against the persisted sample (SHA-256 and order), and computes the checker count C (full perturbed graph) and node degree | No |
| `stage2_block_a.py` | Nine panel cells: checker alone, same-checker comparisons, rank fusion and permuted-score control, bonus percentile, common knee/Otsu rules, degree, TransE epoch similarity | No |
| `stage3_transe_epochs.py` | TransE after 1, 2, 5 and 10 epochs on the panel cells (CPU) | Yes |
| `stage4_paired.py` | Checker alone and A2 + checker on the paired seeds (random-5%, asymmetry-10%) | No |
| `stage5_block_b_run.py` | S1/S2/S3 residuals and TransE (d = 128 and d = 50) on the paired asymmetry-10% samples (CPU) | Yes |
| `stage6_block_b_analysis.py` | Primary and secondary same-checker families on the paired asymmetry-10% samples | No |
| `stage7_run_extra.py` | TransE on the paired random-5% samples; TransE with training seeds 43 and 44 (CPU) | Yes |
| `stage8_extra_analysis.py` | Fusion replication (T1), random-5% common criterion (T2), training seeds (T3), degree adjustment (T4), checker scope (T5) | No |
| `stage9_review_diagnostics.py` | Knee-degeneration traces, degree adjustment before/after, inventory of contrast families, seed-variability summaries | No |
| `stage10_knee_audit.py` | Recomputes the knee of every stored dynamical score vector with its family's convention and checks it against the persisted thresholds | No |

## Reproducing the results

Run all commands from the repository root. `PYTHONHASHSEED=0` is used throughout.

**From the stored per-triple scores (no training, about 15 minutes on a CPU):**

```bash
export PYTHONHASHSEED=0
FI=reviewer_controls/frozen_inputs
WK=reviewer_controls/work
mkdir -p $WK/inputs
cp -R reviewer_controls/raw_outputs/* $WK/
for ds in FB15k-237 codex-m WN18RR; do
  python reviewer_controls/scripts/stage1_inputs.py $ds $FI $WK/inputs
done
for s in stage2_block_a stage4_paired stage6_block_b_analysis stage8_extra_analysis stage9_review_diagnostics stage10_knee_audit; do
  python reviewer_controls/scripts/$s.py $FI $WK
done
```

The regenerated tables in `reviewer_controls/work/` can then be compared with `reviewer_controls/results/`.

**Re-running the training and integration stages (CPU, a few hours):**

```bash
export KG_FORCE_DEVICE=cpu PYTHONHASHSEED=0
python reviewer_controls/scripts/stage3_transe_epochs.py $FI $WK/transe_epochs
for ds in FB15k-237 codex-m WN18RR; do
  python reviewer_controls/scripts/stage5_block_b_run.py $ds $WK
  python reviewer_controls/scripts/stage7_run_extra.py random5 $ds $WK
  python reviewer_controls/scripts/stage7_run_extra.py trainseed $ds $WK
done
```

These runs use CPU and fixed seeds. They write to `work/` and do not overwrite `raw_outputs/`.

## Notes

- **Labels.** Labels are used only to compute metrics. They never enter scores, the bonus calibration, tie-breaking
  or the fusion weights. K = P is used only as an evaluation budget.
- **Which runs are new.** The S1–S3 integrations and all TransE trainings in this folder are new runs. They are
  distinct from the frozen canonical execution, whose relevant inputs are in `frozen_inputs/`.
- **Caches.** The scripts disable reading the constraint-structure cache. They may still write cache files to
  `.cache/`, which `.gitignore` excludes.
