"""
Pipeline Configuration Constants
=================================
Single source of truth for protocol-level constants shared across
run_all.py, ablation_study.py, and hyperparameter_tuning.py.

These values are NOT hyperparameters (not tuned by HyperparameterTuner).
They represent experimental protocol decisions:
- NATIVE_EPOCHS: full-graph training budget (convergence matters).
- ABLATION_EPOCHS: re-training budget on each perturbed subgraph
  (calibration only; full convergence not needed for ablation eval).
"""

from pathlib import Path

# ── Training budgets ──────────────────────────────────────────────────────────
# epochs for full-graph KGE training (native stage)
NATIVE_EPOCHS: int = 50

# epochs for KGE re-training on each perturbed subgraph (ablation stages)
# Intentionally lower than NATIVE_EPOCHS: hyperparams are already optimised,
# so 10 epochs is enough to calibrate embedding geometry for detection.
ABLATION_EPOCHS: int = 10

# Batch sizes (not tuned — chosen for GPU memory / training speed trade-off)
NATIVE_BATCH: int = 256
ABLATION_BATCH: int = 128

# ── Hyperparameter cache ──────────────────────────────────────────────────────
# Root of the hyperparameter cache directory (relative to repo root).
# This directory is tracked in git; the _archive/ subdirectory is not.
CACHE_DIR: Path = Path(".cache/hyperparams")
CACHE_ARCHIVE_DIR: Path = CACHE_DIR / "_archive"

# ── Stage dependency graph ────────────────────────────────────────────────────
# Maps each stage to the stages that must be satisfied before it can run.
# Used by run_all.py to emit actionable error messages.
STAGE_DEPENDENCIES: dict = {
    "tune":                 [],
    "native":               ["tune"],
    "random_ablation":      ["native"],
    "logical_ablation":     ["native"],
    "explanations":         ["random_ablation", "logical_ablation"],
    "visualizations":       ["native"],
    # bootstrap_validation: re-runs random/logical ablations across n seeds
    # and runs paired Wilcoxon tests. Depends only on native (it re-derives
    # ablation outputs internally per seed under bootstrap/seed_<N>/...).
    "bootstrap_validation": ["native"],
}

# Stage aliases — expanded before dependency checking.
# Note: bootstrap_validation is intentionally NOT in any alias group; it is
# expensive and must be invoked explicitly.
STAGE_ALIASES: dict = {
    "all":      ["tune", "native", "random_ablation", "logical_ablation",
                 "explanations", "visualizations"],
    "ablation": ["random_ablation", "logical_ablation"],
    "paper":    ["tune", "native", "random_ablation", "logical_ablation"],
}

# Canonical stage order (used for dry-run output and alias expansion ordering)
STAGE_ORDER: list = [
    "tune", "native", "random_ablation", "logical_ablation",
    "bootstrap_validation",
    "explanations", "visualizations",
]
