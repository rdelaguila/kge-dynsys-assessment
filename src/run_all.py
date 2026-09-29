"""
Unified Experimental Runner (run_all.py)
========================================
Consolidates all experiments for MDPI publication:
1. Native baseline evaluation (0% perturbation)
2. Random entity-swap ablation (e.g., 5%, 10%, 15%)
3. Logical pattern ablation (Asymmetry, Transitivity, Cardinality)

Includes standalone models (KGEs, ODE) and
hybrid models (KGE + ODE) evaluated fairly against each other.
Saves all metrics (Precision, Recall, F1), detections, and triggers
visualizations and concluding analysis scripts.
`visualization_generator is not part of this release.`
"""

import argparse
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Internal imports
from data import DataLoader, Triple
from integrated_framework import IntegratedKGDebugger, IntegratedReport
from ablation_study import KGPerturbator


try:
    from visualization_generator import generate_dataset_visualizations
except ImportError:
    generate_dataset_visualizations = None

from config import (
    NATIVE_EPOCHS, NATIVE_BATCH,
    ABLATION_EPOCHS, ABLATION_BATCH,
    CACHE_DIR, CACHE_ARCHIVE_DIR,
    STAGE_ALIASES, STAGE_ORDER, STAGE_DEPENDENCIES,
)

# ── Extraction from run_comparison ─────────────────────────────────────────────

def get_constraint_config(dataset_name: str) -> Dict:
    """Return constraint_config for IntegratedKGDebugger."""
    if dataset_name == "WN18RR":
        asymmetric_rels   = ["_hypernym", "_instance_hypernym", "_has_part",
                             "_member_meronym", "_member_of_domain_region",
                             "_member_of_domain_usage", "_synset_domain_topic_of"]
        transitive_rels   = ["_hypernym", "_instance_hypernym", "_has_part",
                             "_member_meronym"]
        return {
            "asymmetry_relations":    asymmetric_rels,
            "transitivity_relations": transitive_rels,
            "symmetry_relations":     ["_also_see", "_similar_to", "_verb_group",
                                       "_derivationally_related_form"],
            "cardinality_constraints": [],
        }
    elif dataset_name in ("FB15k-237", "FB15K237", "fb15k237"):
        # [2026-05-11] Synced with FB15K237_RELATION_PROPERTIES injection table
        # and extended with user-supplied candidates (data-verified compound forms
        # where applicable). See memory/project_asymmetry_config_audit.md.
        return {
            "asymmetry_relations": [
                "/location/location/contains",
                "/location/country/capital",
                "/people/person/nationality",
                "/people/person/gender",
                "/organization/organization/headquarters./location/mailing_address/country",
                "/music/artist/origin",
                "/organization/organization/child./organization/organization_relationship/child",
                "/award/award_winner/awards_won./award/award_honor/award_winner",
                "/business/business_operation/industry",
            ],
            "transitivity_relations": [
                "/location/location/contains",
                "/organization/organization/child./organization/organization_relationship/child",
            ],
            "symmetry_relations": [
                "/people/person/spouse_s./people/marriage/spouse",
                "/people/person/sibling_s./people/sibling_relationship/sibling",
            ],
            "cardinality_constraints": [],
        }
    elif dataset_name == 'codex-m':
        return {
            "asymmetry_relations":    ["P131", "P361", "P17", "P19", "P20", "P27", "P50", "P69", "P106", "P108"],
            "transitivity_relations": ["P131", "P361"],
            "symmetry_relations":     ["P3373", "P26"],
            "cardinality_constraints": [],
        }
    else:
        return {
            "asymmetry_relations":    ["parent", "child", "grandparent"],
            "transitivity_relations": ["ancestor"],
            "symmetry_relations":     ["marriedTo", "siblingOf"],
            "cardinality_constraints": [],
        }


def sample_triples(
    perturbed_triples: List[Triple],
    violation_set: Set[Tuple[str, str, str]],
    sample_size: int,
    seed: int = 42,
) -> Tuple[List[Triple], Set[Tuple[str, str, str]]]:
    import random as _random
    rng = _random.Random(seed)

    if len(perturbed_triples) <= sample_size:
        return perturbed_triples, violation_set
        
    sampled = rng.sample(perturbed_triples, sample_size)
    sampled_set = {(t.subject, t.relation, t.object) for t in sampled}
    
    # Keep only the anomalies that actually appear in the sample
    sample_viol = violation_set.intersection(sampled_set)
    
    return sampled, sample_viol


def compute_metrics(
    detected: Set[Tuple[str, str, str]],
    violations: Set[Tuple[str, str, str]],
    total_triples: int,
) -> Dict:
    tp = len(detected & violations)
    fp = len(detected - violations)
    fn = len(violations - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "n_detected":      len(detected),
        "n_violations":    len(violations),
        "n_total":         total_triples,
    }


def print_results_table(results: Dict[str, Dict], title: str = ""):
    width = 80
    print("\n" + "=" * width)
    if title: print(f"  {title}\n" + "=" * width)
    print(f"{'Method':<26} {'Detected':>9} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * width)
    for method, m in results.items():
        if method.startswith('_'): continue
        print(f"{method:<26} {m['n_detected']:>9} {m['true_positives']:>6} "
              f"{m['false_positives']:>6} {m['false_negatives']:>6} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter cache helpers
# ─────────────────────────────────────────────────────────────────────────────

KGE_REQUIRED_KEYS = {"embedding_dim", "learning_rate", "margin", "negative_sampling"}
ODE_REQUIRED_KEYS = {"lambda_data", "lambda_logic", "lambda_reg"}


def _load_hyperparam_cache(dataset: str, model: str) -> Dict:
    """Load best_params from cache. Supports both new {_meta,best_params} and legacy flat format.
    Aborts with actionable error if cache is missing."""
    cache_path = CACHE_DIR / f"{dataset}_{model}.json"
    if not cache_path.exists():
        print(f"\n[native] ERROR: No hyperparameter cache found for {dataset}/{model}.")
        print(f"         Expected: {cache_path}")
        print(f"         Run: python src/run_all.py --datasets {dataset} --stages tune")
        print(f"         Or:  python src/run_all.py --datasets {dataset} --stages all")
        sys.exit(1)
    raw = json.loads(cache_path.read_text())
    params = raw.get("best_params", raw)  # support legacy flat format
    # Validate required keys
    required = ODE_REQUIRED_KEYS if model == "ode" else KGE_REQUIRED_KEYS
    missing = required - params.keys()
    if missing:
        print(f"[native] ERROR: Cache for {dataset}/{model} missing keys: {missing}")
        print(f"         Delete {cache_path} and re-run --stages tune to regenerate.")
        sys.exit(1)
    return params


def _write_hyperparam_cache(cache_path: Path, best_params: Dict, meta: Dict):
    """Write cache in {_meta, best_params} format. Archives old file first."""
    CACHE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        old = json.loads(cache_path.read_text())
        ts = old.get("_meta", {}).get("tuned_at", "unknown").replace(":", "-")
        archive_path = CACHE_ARCHIVE_DIR / f"{cache_path.stem}_{ts}.json"
        shutil.copy(cache_path, archive_path)
        print(f"[tune] Archived previous cache → {archive_path}")
    data = {"_meta": meta, "best_params": best_params}
    cache_path.write_text(json.dumps(data, indent=2))


def _print_cache_summary(dataset: str, model: str, data: Dict):
    meta = data.get("_meta", {})
    bp = data.get("best_params", data)
    params_str = ", ".join(f"{k}={v}" for k, v in list(bp.items())[:4])
    print(f"[tune] Cache found for {dataset}/{model}:")
    print(f"       tuned_at:    {meta.get('tuned_at', 'historical')}")
    if meta.get("best_metric_value"):
        print(f"       best_metric: MRR = {meta['best_metric_value']:.4f}")
    print(f"       best_params: {params_str}")


def _diff_params(label: str, old: Dict, new: Dict):
    """Print diff between old and new best_params. Returns True if any changed."""
    changed = False
    for k in sorted(set(old) | set(new)):
        o, n = old.get(k, "<missing>"), new.get(k, "<missing>")
        if o != n:
            print(f"  ⚠ {label}/{k}: {o} → {n}  (CHANGED)")
            changed = True
        else:
            print(f"    {label}/{k}: {o} → {n}  (unchanged)")
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# Stage: tune
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_tune(
    datasets: List[str],
    models: List[str],
    force_tune: bool = False,
    no_interactive: bool = False,
):
    """Interactive hyperparameter search stage.
    - If cache exists: asks user before re-running (grouped prompt).
    - If cache missing: runs automatically.
    - --force-tune: requires typing 'regenerate' to confirm.
    - --no-interactive / non-TTY: keeps existing cache silently.
    """
    from hyperparameter_tuning import HyperparameterTuner
    from data import DataLoader

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    auto_batch = not sys.stdin.isatty()
    tuner = HyperparameterTuner(cache_dir=str(CACHE_DIR))

    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"[tune] Dataset: {dataset}")
        print(f"{'='*70}")

        # Load triples once per dataset (needed for tuner)
        triples_cache: Dict[str, list] = {}

        # Categorise each model: missing vs cached
        missing, cached_models = [], []
        for model in models:
            cp = CACHE_DIR / f"{dataset}_{model}.json"
            (missing if not cp.exists() else cached_models).append(model)

        # Run tuner immediately for missing models (no prompt)
        for model in missing:
            print(f"[tune] No cache for {dataset}/{model}. Running search...")
            if dataset not in triples_cache:
                loader = DataLoader()
                triples_cache[dataset] = loader.load_dataset(dataset)
            triples = triples_cache[dataset]
            if model == "ode":
                bp = tuner.tune_ode(dataset, triples)
            else:
                bp = tuner.tune_for_dataset(dataset, triples, model)
            meta = {
                "dataset": dataset, "model": model,
                "tuned_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": 42, "n_trials": 8,
                "metric": "MRR_held_out", "best_metric_value": None,
            }
            _write_hyperparam_cache(CACHE_DIR / f"{dataset}_{model}.json", bp, meta)
            print(f"[tune] Written {dataset}_{model}.json")

        if not cached_models:
            continue

        # Print summary of existing caches
        print(f"\n[tune] Existing caches for {dataset}:")
        for i, model in enumerate(cached_models):
            data = json.loads((CACHE_DIR / f"{dataset}_{model}.json").read_text())
            _print_cache_summary(dataset, model, data)

        # Determine whether to retune any cached models
        retune_models: List[str] = []

        if force_tune:
            # Require explicit confirmation
            print("\n" + "╔" + "═"*72 + "╗")
            for line in [
                "  --force-tune will overwrite the cached hyperparameters used to",
                "  produce the published paper results.",
                "",
                "  Non-determinism in the tuner may cause the new values to differ",
                "  from the originals; downstream results would then diverge from",
                "  the paper. To restore: git checkout .cache/hyperparams/",
                "",
                "  Type 'regenerate' and press Enter to proceed.",
                "  Any other input aborts.",
            ]:
                print(f"║ {line:<72} ║")
            print("╚" + "═"*72 + "╝")
            if auto_batch or no_interactive:
                print("[tune] Aborted: --force-tune requires interactive confirmation. "
                      "Cannot combine with --no-interactive or non-TTY.")
                sys.exit(1)
            resp = input("> ").strip()
            if resp != "regenerate":
                print("[tune] Aborted. Cache unchanged.")
                return
            retune_models = cached_models

        elif auto_batch or no_interactive:
            print(f"[tune] Batch/non-interactive mode: keeping all existing caches.")
            retune_models = []

        else:
            # Grouped prompt
            print(f"\nRe-run tuning for {dataset}? [a]ll, [s]ome, [n]one (default n): ", end="")
            resp = input().strip().lower()
            if resp in ("a", "all"):
                retune_models = cached_models
            elif resp in ("s", "some"):
                for i, m in enumerate(cached_models):
                    print(f"  [{i}] {m}")
                sel = input("Enter indices (space-separated): ").strip()
                try:
                    indices = [int(x) for x in sel.split()]
                    retune_models = [cached_models[i] for i in indices if 0 <= i < len(cached_models)]
                except (ValueError, IndexError):
                    print("[tune] Invalid selection. Keeping all caches.")
                    retune_models = []
            else:
                print(f"[tune] Keeping all existing caches for {dataset}.")
                retune_models = []

        for model in retune_models:
            if dataset not in triples_cache:
                loader = DataLoader()
                triples_cache[dataset] = loader.load_dataset(dataset)
            triples = triples_cache[dataset]
            cache_path = CACHE_DIR / f"{dataset}_{model}.json"
            old_data = json.loads(cache_path.read_text())
            old_bp = old_data.get("best_params", old_data)

            print(f"[tune] Running search for {dataset}/{model}...")
            if model == "ode":
                new_bp = tuner.tune_ode(dataset, triples)
            else:
                new_bp = tuner.tune_for_dataset(dataset, triples, model)

            meta = {
                "dataset": dataset, "model": model,
                "tuned_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": 42, "n_trials": 8,
                "metric": "MRR_held_out", "best_metric_value": None,
            }
            _write_hyperparam_cache(cache_path, new_bp, meta)

            # Post-tuning diff
            print(f"\n[tune] Diff against previous cache for {dataset}/{model}:")
            changed = _diff_params(f"{dataset}/{model}", old_bp, new_bp)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            diff_file = CACHE_DIR / f"_diff_{ts}_{dataset}_{model}.txt"
            lines = [f"Diff {dataset}/{model} at {ts}\n"]
            for k in sorted(set(old_bp) | set(new_bp)):
                o, n = old_bp.get(k), new_bp.get(k)
                lines.append(f"  {'CHANGED' if o!=n else 'ok':8} {k}: {o} → {n}\n")
            diff_file.write_text("".join(lines))
            print(f"[tune] Diff saved to {diff_file}")
            if changed:
                print(f"[tune] ⚠ Values changed. Downstream results may differ from paper.")
                print(f"[tune]   To restore: git checkout {cache_path}")
            else:
                print(f"[tune] ✓ All values match historical cache. Tuner is deterministic.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage: explanations
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_explanations(datasets: List[str], base_dir: str = "paper_results"):
    """Run CoT explanations + static analysis reports for each dataset."""
    print(f"\n{'='*70}")
    print("STAGE: EXPLANATIONS (CoT + Analysis Reports)")
    print(f"{'='*70}")
    from explanation import ChainOfThoughtExplainer
    from explanation_generator import generate_dataset_explanations

    base = Path(base_dir)
    for dataset in datasets:
        ds_dir = base / dataset
        expl_dir = ds_dir / "explanations"
        expl_dir.mkdir(parents=True, exist_ok=True)

        # Static analysis report (no LLM required)
        print(f"\n[explanations] Generating analysis report for {dataset}...")
        try:
            generate_dataset_explanations(ds_dir, expl_dir)
        except Exception as e:
            print(f"[explanations] Warning: analysis report failed: {e}")

        # CoT over Universal Consensus triples (requires LLM)
        consensus_file = ds_dir / "cot_consensus.json"
        if consensus_file.exists():
            print(f"[explanations] Running CoT over consensus triples...")
            try:
                explainer = ChainOfThoughtExplainer()
                with open(consensus_file) as f:
                    triples_data = json.load(f)
                results = []
                for item in triples_data[:70]:  # cap at 70
                    expl = explainer.explain_violation(
                        item.get("triple", item),
                        violation_type=item.get("violation_type", "unknown"),
                        constraint_info=item.get("constraint_info", ""),
                    )
                    results.append({**item, "cot_explanation": expl})
                out_file = expl_dir / "cot_consensus_explained.json"
                out_file.write_text(json.dumps(results, indent=2))
                print(f"[explanations] CoT written to {out_file}")
            except Exception as e:
                print(f"[explanations] Warning: CoT failed: {e}")
        else:
            print(f"[explanations] No consensus file found at {consensus_file}; skipping CoT.")


class UnifiedExperimentRunner:
    def __init__(self, dataset_name: str, methods: List[str], base_dir: str = "paper_results", resume: bool = False,
                 lambda_logic_override: float = None, symbolic_bonus=None,
                 lambda_asym_override: float = None):
        self.dataset_name = dataset_name
        self.methods = methods
        self.base_dir = Path(base_dir) / dataset_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        # [phase0] In-memory lambda_logic override — does NOT modify any cache file on disk.
        self.lambda_logic_override = lambda_logic_override
        # In-memory lambda_asym override — same contract as
        # lambda_logic_override but for the asymmetry-as-gradient force.
        # Used by run_bootstrap_validation_stage's logical_lambda_asym_zero
        # sub-experiment (TASK 5) to neutralise the asymmetry contribution
        # without touching any cache file. Default None = use cache value.
        self.lambda_asym_override = lambda_asym_override
        # symbolic_bonus=None → AnomalyDetector
        # auto-calibrates per-dataset (q99 - q50 of pre-bonus score). Numeric
        # value pins the bonus and skips calibration (backward-compat).
        self.symbolic_bonus = symbolic_bonus
        
        self.trained_params = {}
        self.models_cache = {} # Used to avoid reloading/reinstantiating PyKEEN if possible (optional)
        
        # Determine available methods
        self.use_ode = 'ode' in self.methods
        self.kge_models = [m for m in self.methods if m != 'ode']
        
        # To store results across stages
        self.results_summary = {
            'dataset': dataset_name,
            'native': {},
            'random_ablation': {},
            'logical_ablation': {}
        }
        
    def _save_native_results(self, method_name: str, detections: List[Dict], execution_time: float, extra_params: Dict = None):
        """Save results in the format expected by visualization_generator.py"""
        out_dir = self.base_dir / method_name
        out_dir.mkdir(exist_ok=True)
        
        scores = [float(d['score']) for d in detections]

        res = {
            'method': method_name,
            'dataset': self.dataset_name,
            'execution_time': execution_time,
            'parameters': extra_params or {},
            'detections': {
                'n_detected': len(detections),
                'scores': scores,
                'mean_score': float(np.mean(scores)) if scores else 0.0,
                # Otsu metadata (populated by KGE detector; None for ODE)
                'otsu_threshold': extra_params.pop('_otsu_threshold', None) if extra_params else None,
                'raw_scores_all': extra_params.pop('_raw_scores_all', None) if extra_params else None,
            }
        }
        
        with open(out_dir / 'results.json', 'w') as f:
            json.dump(res, f, indent=2)
            
        serialized_detections = []
        for d in detections:
            trip = d['triple']
            # If trip is a dict (hybrid), or a Triple object (native)
            if hasattr(trip, 'subject'):
                t_dict = {'subject': trip.subject, 'relation': trip.relation, 'object': trip.object}
            else:
                t_dict = trip
                
            serialized_detections.append({
                'triple': t_dict,
                'score': float(d['score']),
                'types': d.get('types', []),
                'explanation': d.get('explanation', ''),
                'action': d.get('action', 'verify')
            })
            
        with open(out_dir / 'detections.json', 'w') as f:
            json.dump(serialized_detections, f, indent=2)
            
        return res

    def run_native_stage(self, triples: List[Triple]):
        """Run all requested methods on the clean, unperturbed graph."""
        print(f"\n{'='*80}")
        print(f"STAGE 1: NATIVE EVALUATION (0% perturbation) - {self.dataset_name}")
        print(f"{'='*80}")
        
        # Provide base configuration
        config = IntegratedKGDebugger.get_default_config()
        config['constraint_config'] = get_constraint_config(self.dataset_name)
        
        # 1. Train and Run PyKEEN models
        all_pykeen_detections = {}
        for kge in self.kge_models:
            print(f"\n--- Running Native KGE: {kge} ---")
            
            # Check for resume
            existing_results_file = self.base_dir / kge / 'results.json'
            if self.resume and existing_results_file.exists():
                print(f"  [Resume] Skipping {kge} native evaluation (results.json already exists).")
                with open(existing_results_file, 'r') as f:
                    cached_results = json.load(f)
                
                # We still need the detections to populate `all_pykeen_detections` for hybrids
                existing_detections_file = self.base_dir / kge / 'detections.json'
                if existing_detections_file.exists():
                    with open(existing_detections_file, 'r') as f:
                        cached_dets = json.load(f)
                        all_pykeen_detections[kge] = {
                            (d['triple']['subject'], d['triple']['relation'], d['triple']['object']): d['score']
                            for d in cached_dets
                        }
                
                # Recover trained params if they exist
                self.trained_params[kge] = cached_results.get('parameters', {})
                self.results_summary['native'][kge] = cached_results
                continue
                
            kge_config = config.copy()
            kge_config['use_ode'] = False
            kge_config['use_pykeen'] = True

            # Load hyperparameters from cache (no hardcoded values).
            # epochs is a protocol constant, not a tuned hyperparameter.
            cached_params = _load_hyperparam_cache(self.dataset_name, kge)
            kge_config['pykeen_config']['model'] = kge
            kge_config['pykeen_config']['embedding_dim']       = cached_params['embedding_dim']
            kge_config['pykeen_config']['learning_rate']       = cached_params['learning_rate']
            kge_config['pykeen_config']['margin']              = cached_params['margin']
            kge_config['pykeen_config']['negative_sampling']   = cached_params['negative_sampling']
            # epochs is not tuned — it is a protocol constant defined in src/config.py.
            kge_config['pykeen_config']['epochs'] = NATIVE_EPOCHS
            kge_config['pykeen_config']['batch_size'] = NATIVE_BATCH
            
            t0 = time.time()
            # Native stage: instantiate PyKEENDetector directly and use
            # ranking-only detection (Otsu threshold on full score distribution).
            # This avoids neighbourhood/constraint sub-detectors that inflate
            # counts to tens of thousands and bypasses IntegratedKGDebugger's
            # run_complete_analysis which calls detect_all_anomalies.
            from pykeen_detector import PyKEENDetector
            pykeen_detector = PyKEENDetector(
                embedding_dim=kge_config['pykeen_config']['embedding_dim']
            )
            pykeen_detector.train_model(
                triples,
                model_name=kge,
                epochs=kge_config['pykeen_config']['epochs'],
                batch_size=kge_config['pykeen_config']['batch_size'],
                learning_rate=kge_config['pykeen_config'].get('learning_rate', 0.01),
                margin=kge_config['pykeen_config'].get('margin', 1.0),
                negative_sampling=kge_config['pykeen_config'].get('negative_sampling', 5),
            )
            raw_reports = pykeen_detector.detect_ranking_anomalies(triples)
            t_exec = time.time() - t0

            raw_detections = [
                {'triple': r.triple, 'score': r.anomaly_score,
                 'types': r.anomaly_types, 'explanation': r.explanation,
                 'action': r.suggested_action}
                for r in raw_reports
            ]
            all_pykeen_detections[kge] = {
                (d['triple'].subject, d['triple'].relation, d['triple'].object): d['score']
                for d in raw_detections
            }

            # Persist Otsu metadata alongside params
            params = {
                'embedding_dim': kge_config['pykeen_config']['embedding_dim'],
                'learning_rate': kge_config['pykeen_config'].get('learning_rate', 0.01),
                'margin': kge_config['pykeen_config'].get('margin', 1.0),
                'negative_sampling': kge_config['pykeen_config'].get('negative_sampling', 5),
                'epochs': kge_config['pykeen_config'].get('epochs', NATIVE_EPOCHS),
                'batch_size': kge_config['pykeen_config'].get('batch_size', NATIVE_BATCH),
                # Otsu metadata injected into detections block by _save_native_results
                '_otsu_threshold': getattr(pykeen_detector, '_last_otsu_threshold', None),
                '_raw_scores_all': (
                    getattr(pykeen_detector, '_last_raw_scores', None).tolist()
                    if getattr(pykeen_detector, '_last_raw_scores', None) is not None
                    else None
                ),
            }
            self.trained_params[kge] = {k: v for k, v in params.items() if not k.startswith('_')}
            self.results_summary['native'][kge] = self._save_native_results(kge, raw_detections, t_exec, params)

        # 2. Train and Run ODE
        ode_detections_map = {}
        if self.use_ode:
            print(f"\n--- Running Native: KG-Debug-ODE ---")
            
            existing_results_file = self.base_dir / 'ode' / 'results.json'
            if self.resume and existing_results_file.exists():
                print(f"  [Resume] Skipping ODE native evaluation (results.json already exists).")
                with open(existing_results_file, 'r') as f:
                    cached_results = json.load(f)
                
                existing_detections_file = self.base_dir / 'ode' / 'detections.json'
                if existing_detections_file.exists():
                    with open(existing_detections_file, 'r') as f:
                        cached_dets = json.load(f)
                        ode_detections_map = {
                            (d['triple']['subject'], d['triple']['relation'], d['triple']['object']): d['score']
                            for d in cached_dets
                        }
                        
                self.trained_params['ode'] = cached_results.get('parameters', {})
            else:
                ode_config = config.copy()
                ode_config['use_ode'] = True
                ode_config['use_pykeen'] = False

                # Load tuned ODE hyperparameters from
                # .cache/hyperparams/<dataset>_ode.json — same cache contract
                # the KGE branch above already honours via _load_hyperparam_cache.
                # Without this, the native ODE silently uses
                # IntegratedKGDebugger.get_default_config() defaults
                # (lambda_data=1.0, lambda_logic=1.0, lambda_reg=0.1,
                # embedding_dim=50, t_span=(0,5), rtol=None) and the
                # deterministic-tuner caches written by Block 4.1 are unused.
                cached_ode = _load_hyperparam_cache(self.dataset_name, 'ode')
                ode_config.setdefault('ode_config', {})
                ode_config['ode_config']['embedding_dim'] = cached_ode['embedding_dim']
                ode_config['ode_config']['lambda_data']   = cached_ode['lambda_data']
                ode_config['ode_config']['lambda_logic']  = cached_ode['lambda_logic']
                ode_config['ode_config']['lambda_reg']    = cached_ode['lambda_reg']
                # lambda_asym + margin_asym — new keys for the
                # asymmetry-as-gradient force. Default to 1.0 if the cache
                # predates block7 (backward-compat).
                ode_config['ode_config']['lambda_asym']   = float(cached_ode.get('lambda_asym', 1.0))
                ode_config['ode_config']['margin_asym']   = float(cached_ode.get('margin_asym', 1.0))
                if 'rtol' in cached_ode:
                    ode_config['ode_config']['rtol'] = cached_ode['rtol']
                if 'atol' in cached_ode:
                    ode_config['ode_config']['atol'] = cached_ode['atol']
                if 't_span' in cached_ode:
                    ts = cached_ode['t_span']
                    ode_config['ode_config']['t_span'] = tuple(ts) if isinstance(ts, list) else ts
                if 'device' in cached_ode:
                    ode_config['ode_config']['device'] = cached_ode['device']
                # [phase0] in-memory override still wins (e.g. H2 condition).
                if self.lambda_logic_override is not None:
                    ode_config['ode_config']['lambda_logic'] = self.lambda_logic_override
                    print(f"  [lambda-override] lambda_logic overridden to {self.lambda_logic_override} (in-memory only)")
                # same pattern for lambda_asym (logical_lambda_asym_zero condition)
                if self.lambda_asym_override is not None:
                    ode_config['ode_config']['lambda_asym'] = self.lambda_asym_override
                    print(f"  [lambda-override] lambda_asym overridden to {self.lambda_asym_override} (in-memory only)")

                # Standard ODE window
                ode_config['ode_config']['top_k_ratio'] = 0.05

                t0 = time.time()
                debugger = IntegratedKGDebugger(ode_config)
                res = debugger.run_complete_analysis(triples)
                t_exec = time.time() - t0
                
                raw_detections = res.get('ode_results', [])
                ode_detections_map = {
                    (d['triple'].subject, d['triple'].relation, d['triple'].object): d['score']
                    for d in raw_detections
                }
                
                self.trained_params['ode'] = ode_config['ode_config']
                self.results_summary['native']['ode'] = self._save_native_results('ode', raw_detections, t_exec, ode_config['ode_config'])
            
            # Create Hybrid native combinations saving
            for kge in self.kge_models:
                combo_name = f"{kge}+ODE"
                print(f"\n--- Generating Hybrid Native: {combo_name} ---")
                
                # Manual hybrid scoring: mean(ode_score, kge_score) - simulating IntegratedKGDebugger consensus
                hybrid_detections = []
                for t_tuple, k_score in all_pykeen_detections[kge].items():
                    o_score = ode_detections_map.get(t_tuple, 0.0)
                    consensus = np.mean([k_score, o_score]) if o_score > 0 else k_score
                    hybrid_detections.append({
                        'triple': {'subject': t_tuple[0], 'relation': t_tuple[1], 'object': t_tuple[2]},
                        'score': consensus,
                        'types': ['hybrid_consensus'],
                        'explanation': f'Hybrid score {consensus:.3f}',
                        'action': 'verify'
                    })
                
                # We save hybrids basically instantly
                self.results_summary['native'][combo_name] = self._save_native_results(combo_name, hybrid_detections, 0.1, {})

        print("\nNative stage complete.")
        if generate_dataset_visualizations is not None:
            print("Generating native visualizations...")
            generate_dataset_visualizations(self.base_dir, self.base_dir / 'visualizations')
        else:
            print("  [skipped] visualization_generator not present in this distribution "
                  "-- all metrics/detections JSON above are already persisted.")

    def run_ablation_stage(self, triples: List[Triple], stage_type: str, ablation_levels: List[float], sample_size: int = None,
                           perturbation_seed: int = 42, output_subdir: str = None):
        """Run either 'random_ablation' or 'logical_ablation'.

        Optional parameters (default behaviour preserved when omitted):
            perturbation_seed: seed forwarded to ``KGPerturbator`` and
                ``sample_triples``. Default 42 = the canonical paper seed.
            output_subdir: when provided, ablation metrics are written to
                ``self.base_dir / output_subdir / metrics/`` instead of the
                default ``self.base_dir / metrics/``. Used by
                ``run_bootstrap_validation_stage`` to write per-seed results
                under ``bootstrap/seed_<N>/<scope>/`` without colliding with
                the canonical ablation outputs.

        ─── Path A vs Path B (logical-violation evaluation) ──────────────────────
        This method is **Path A** of the two non-comparable evaluation paths in this
        codebase. The other is **Path B** in `src/ablation_study.py::run_logical`,
        invoked indirectly by `run_full_rerun.py`.

        - **Path A (this method)** — produced **manuscript v4 Tables 7/8 / D1–D3**
          (logical ablation, canonical commit ``879a39b``). The KGE branch leaves
          the default ``use_constraints=True`` of ``IntegratedKGDebugger``, so
          ``pykeen_detector.detect_all_anomalies()`` folds the symbolic
          constraint-checker output into the KGE detection sets. Every triple
          flagged by the symbolic checker is added to *every* KGE method's
          detection set — that is why the manuscript reports identical
          TransE / DistMult / MuRE F1 within each cell.

        - **Path B (`src/ablation_study.py::run_logical`)** — used by
          `run_full_rerun.py`, canonical for **random ablation only** (Tables
          C.1–C.3). Its KGE branch explicitly sets ``use_constraints=False`` at
          ``src/ablation_study.py:678``, so the symbolic checker is NOT folded
          into KGE outputs. KGE detection sets are pure ranking (Otsu) results.

        The two paths produce **non-comparable** KGE columns when applied to
        logical violations. See ``PROVENANCE.md`` (repo root) for the full
        artefact split, ``progress/diagnostic_phase3/step1_provenance.md`` for
        the audit, and ``progress/diagnostic_phase3/step2_fixes.md`` for the
        documentation rationale of this docstring.
        """
        stage_name = 'Logical Ablation' if stage_type == 'logical' else 'Random Ablation'
        print(f"\n{'='*80}")
        print(f"STAGE 2/3: {stage_name.upper()} - Levels: {ablation_levels}")
        print(f"{'='*80}")

        # Ensure trained_params are loaded from disk if native stage wasn't run in this session
        for method_name in self.kge_models + (['ode'] if self.use_ode else []):
            if method_name not in self.trained_params:
                dir_name = 'ode' if method_name == 'ode' else method_name
                results_file = self.base_dir / dir_name / 'results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        cached = json.load(f)
                    self.trained_params[method_name] = cached.get('parameters', {})

        summary_key = f"{stage_type}_ablation"
        
        # Depending on stage type, loop logic is different
        if stage_type == 'random':
            # Levels mean standard percentages: [5, 10, 15]
            scenarios = [{'type': 'entity_swap', 'val': pct} for pct in ablation_levels]
        else:
            # Logical: Run asymmetry, transitivity, cardinality using the levels as scaling for n_violations
            n_base = len(triples)
            scenarios = []
            for vtype in ['asymmetry', 'transitivity', 'cardinality']:
                for pct in ablation_levels:
                    # e.g., 5% logical violations = 5% of graph size
                    n_viol = max(10, int(n_base * (pct / 100)))
                    scenarios.append({'type': vtype, 'val': n_viol, 'pct': pct})
                    
        # Resolve output root once (shared by resume check and final write).
        # output_subdir is an opt-in nesting used by bootstrap_validation;
        # default None preserves the canonical layout exactly.
        out_root = self.base_dir / output_subdir if output_subdir else self.base_dir

        for scenario in scenarios:
            v_type = scenario['type']
            val = scenario['val']

            # Early skip check for ablation stage
            metrics_dir = out_root / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            out_file = metrics_dir / f"metrics_{stage_type}_{v_type}_{val}.json"
            
            print(f"\n>>> Scenario: {v_type} | {'pct=' if stage_type=='random' else 'n='}{val}")
            
            scenario_results = {}
            if self.resume and out_file.exists():
                try:
                    with open(out_file, 'r') as f:
                        scenario_results = json.load(f).get('metrics', {})
                except Exception:
                    pass
                
                missing_methods = False
                if self.kge_models and not all(kge in scenario_results for kge in self.kge_models): missing_methods = True
                if self.use_ode and 'KG-Debug-ODE' not in scenario_results: missing_methods = True
                if self.use_ode and self.kge_models and 'Best_KGE+ODE' not in scenario_results: missing_methods = True

                if not missing_methods:
                    print(f"  [Resume] Skipping ablation {stage_type}_{v_type}_{val} (all requested metrics already generated).")
                    scen_key = f"{v_type}_{val}"
                    self.results_summary[summary_key][scen_key] = scenario_results
                    continue
                else:
                    print(f"  [Resume] Loading existing metrics for {stage_type}_{v_type}_{val} and computing missing methods...")
                
            perturbator = KGPerturbator(triples, seed=perturbation_seed)
            if stage_type == 'random':
                perturbed_triples, violation_set = perturbator.perturb(percentage=val, perturbation_types=['entity_swap'])
            else:
                perturbed_triples, violation_set, _ = perturbator.perturb_logical(
                    violation_type=v_type, n_violations=val, dataset_name=self.dataset_name
                )

            if not violation_set:
                print(f"  ⚠ No violations could be generated for {v_type}. Skipping.")
                continue

            n_actual_viol = len(violation_set)

            # Stratified Sampling (essential for large datasets like FB15k-237 to speed up evaluation)
            eval_triples, eval_viol = perturbed_triples, violation_set
            if sample_size and sample_size < len(perturbed_triples):
                eval_triples, eval_viol = sample_triples(perturbed_triples, violation_set, sample_size, seed=perturbation_seed)
            
            n_eval_viol = len(eval_viol)
            
            # scenario_results is preserved from the resume check above
            
            # --- Evaluation ---
            kge_detections = {}
            best_kge = None
            if self.kge_models:
                # Evaluate ALL KGE models; select best by ablation F1 afterwards
                for kge in self.kge_models:
                    print(f"  Evaluating KGE: {kge}...")
                    params = self.trained_params.get(kge, {})
                    config = IntegratedKGDebugger.get_default_config()
                    config['constraint_config'] = get_constraint_config(self.dataset_name)
                    config['use_ode'] = False
                    config['use_pykeen'] = True
                    config['pykeen_config'] = {
                        'model': kge,
                        'embedding_dim': params.get('embedding_dim', 50),
                        'learning_rate': params.get('learning_rate') or 0.01,
                        'margin': params.get('margin') or 1.0,
                        'negative_sampling': params.get('negative_sampling') or 5,
                        # ABLATION_EPOCHS < NATIVE_EPOCHS: hyperparams are already
                        # optimised; 10 epochs calibrates embeddings for detection.
                        'epochs': ABLATION_EPOCHS,
                        'batch_size': ABLATION_BATCH,
                    }

                    debugger = IntegratedKGDebugger(config)
                    res = debugger.run_complete_analysis(eval_triples)
                    raw_d = res.get('pykeen_results', [])
                    det_set = {(r['triple'].subject, r['triple'].relation, r['triple'].object) for r in raw_d}
                    kge_detections[kge] = {t: d['score'] for d in raw_d for t in [(d['triple'].subject, d['triple'].relation, d['triple'].object)]}
                    scenario_results[kge] = compute_metrics(det_set, eval_viol, len(eval_triples))

                # Select best KGE by ablation F1 (recall as tiebreaker, then list order)
                best_kge = max(
                    self.kge_models,
                    key=lambda m: (
                        scenario_results.get(m, {}).get('f1', 0.0),
                        scenario_results.get(m, {}).get('recall', 0.0)
                    )
                )
                print(f"  Best KGE selected by ablation F1: {best_kge} (f1={scenario_results[best_kge]['f1']:.4f})")
                
            ode_detections = {}
            if self.use_ode:
                if self.resume and 'KG-Debug-ODE' in scenario_results:
                    print(f"  [Resume] Skipping KG-Debug-ODE evaluation (already exists).")
                    ode_detections = {}
                else:
                    print(f"  Evaluating KG-Debug-ODE...")
                params = self.trained_params.get('ode', {})
                config = IntegratedKGDebugger.get_default_config()
                config['constraint_config'] = get_constraint_config(self.dataset_name)
                config['use_ode'] = True
                config['use_pykeen'] = False
                config['ode_config'] = {
                    'embedding_dim': params.get('embedding_dim', 50),
                    't_span': tuple(params.get('t_span', [0.0, 5.0])),
                    'lambda_data': params.get('lambda_data', 5.0),
                    'lambda_logic': params.get('lambda_logic', 1.0),
                    'lambda_reg': params.get('lambda_reg', 0.1),
                    # lambda_asym + margin_asym propagate the new
                    # asymmetry-as-gradient knobs through ablation re-runs.
                    'lambda_asym': params.get('lambda_asym', 1.0),
                    'margin_asym': params.get('margin_asym', 1.0),
                    'rtol': params.get('rtol', 0.005),
                    'atol': params.get('atol', 1e-6),
                    'device': params.get('device', 'mps')
                }
                # [phase0] --lambda-logic-override: apply in memory, never written to disk.
                if self.lambda_logic_override is not None:
                    config['ode_config']['lambda_logic'] = self.lambda_logic_override
                    print(f"  [lambda-override] lambda_logic overridden to {self.lambda_logic_override} (in-memory only)")
                # same pattern for lambda_asym (logical_lambda_asym_zero condition)
                if self.lambda_asym_override is not None:
                    config['ode_config']['lambda_asym'] = self.lambda_asym_override
                    print(f"  [lambda-override] lambda_asym overridden to {self.lambda_asym_override} (in-memory only)")
                # Pass symbolic_bonus to
                # AnomalyDetector via config dict. None → auto-calibrate per
                # dataset; numeric → pin (user override). Logged at the
                # detector level (anomaly_detector.py:__init__).
                config['symbolic_bonus'] = self.symbolic_bonus
                
                # To be fair, give ODE a detection window equal to the number of anomalies * scaling buffer
                # E.g., top_k = n_eval_viol * 5
                config['ode_config']['top_k'] = n_eval_viol * 5 if stage_type == 'logical' else n_eval_viol * 2
                
                debugger = IntegratedKGDebugger(config)
                res = debugger.run_complete_analysis(eval_triples, full_triples=perturbed_triples)
                raw_d = res.get('ode_results', [])
                det_set = {(r['triple'].subject, r['triple'].relation, r['triple'].object) for r in raw_d}
                ode_detections = {t: d['score'] for d in raw_d for t in [(d['triple'].subject, d['triple'].relation, d['triple'].object)]}
                scenario_results['KG-Debug-ODE'] = compute_metrics(det_set, eval_viol, len(eval_triples))

                # [phase3-step2-fix2] ODE-pure: subset of ODE detections whose anomaly_score
                # received a contribution from the dynamical signal (energy or instability),
                # not solely from the symbolic constraint penalty.
                #
                # Source of the filter: src/anomaly_detector.py:152-170 appends "high_energy"
                # iff energy > 0.05 and "unstable_embeddings" iff max_instability > 0.005;
                # "violates_*" tags are appended at line 176 when the symbolic checker fires.
                # A triple is therefore "ODE-pure" iff its types include at least one
                # dynamical tag, even if it ALSO violates a constraint. Triples whose only
                # types are "violates_*" (constraint-only) are excluded.
                #
                # CAVEAT: this discriminates *which signal contributed evidence*; it does
                # NOT measure how much each signal contributed quantitatively. A triple
                # with energy=0.06 (just barely tagged "high_energy") and a +500 bonus
                # would still pass through, even though the bonus dominates the score.
                # See progress/diagnostic_phase3/step2_fixes.md for the full discussion.
                _DYNAMICAL_TAGS = ("high_energy", "unstable_embeddings")
                det_set_pure = {
                    (r['triple'].subject, r['triple'].relation, r['triple'].object)
                    for r in raw_d
                    if any(t in _DYNAMICAL_TAGS for t in (r.get('types') or []))
                }
                scenario_results['KG-Debug-ODE-pure'] = compute_metrics(
                    det_set_pure, eval_viol, len(eval_triples)
                )
                
            # Hybrid evaluation (Dynamic Best KGE selection)
            if self.use_ode and self.kge_models and best_kge:
                prior_best = scenario_results.get('_best_kge_name')
                if self.resume and 'Best_KGE+ODE' in scenario_results and prior_best == best_kge:
                    print(f"  [Resume] Skipping Hybrid {best_kge}+ODE evaluation (already exists with correct best_kge).")
                else:
                    print(f"  Evaluating Hybrid {best_kge}+ODE...")
                scored_triples = []
                for t_tuple, k_score in kge_detections[best_kge].items():
                    o_score = ode_detections.get(t_tuple, 0.0)
                    consensus = np.mean([k_score, o_score]) if o_score > 0 else k_score
                    scored_triples.append((consensus, t_tuple))
                    
                scored_triples.sort(key=lambda x: x[0], reverse=True)
                
                # Match detection window sizing
                top_k_count = n_eval_viol * 5 if stage_type == 'logical' else n_eval_viol * 2
                hybrid_set = {t_tuple for score, t_tuple in scored_triples[:top_k_count]}
                
                scenario_results['Best_KGE'] = scenario_results[best_kge]
                scenario_results['Best_KGE+ODE'] = compute_metrics(hybrid_set, eval_viol, len(eval_triples))
                scenario_results['_best_kge_name'] = best_kge
                    
            # Print and Save
            # Unique identifier for the scenario
            scen_key = f"{v_type}_{val}"
            self.results_summary[summary_key][scen_key] = scenario_results
            
            title = f"{stage_name} | {v_type} | {val}"
            filtered_results = {k: v for k,v in scenario_results.items() if not k.startswith('_')}
            print_results_table(filtered_results, title)
            
            # Persistent fine-grained metrics (uses out_root resolved above —
            # default = self.base_dir, bootstrap = self.base_dir / output_subdir)
            metrics_dir = out_root / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            out_file = metrics_dir / f"metrics_{stage_type}_{v_type}_{val}.json"
            
            with open(out_file, 'w') as f:
                json.dump({
                    "dataset": self.dataset_name,
                    "stage": stage_type,
                    "scenario": v_type,
                    "level": val,
                    "n_clean": len(triples),
                    "n_perturbed_graph": len(perturbed_triples),
                    "n_eval_graph": len(eval_triples),
                    "metrics": scenario_results
                }, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    # bootstrap_validation stage — Phase 2 statistical validation
    # ──────────────────────────────────────────────────────────────────────
    def run_bootstrap_validation_stage(self, triples: List[Triple], n_seeds: int = 10,
                                       sample_size: int = 10000):
        """Re-run random + logical ablations across n seeds to support paired
        Wilcoxon tests. Output layout:

            <base_dir>/bootstrap/seed_<N>/random/metrics/...
            <base_dir>/bootstrap/seed_<N>/logical_lambda_ref/metrics/...
            <base_dir>/bootstrap/seed_<N>/logical_lambda_zero/metrics/...
            <base_dir>/bootstrap/seed_<N>/logical_lambda_asym_zero/metrics/... 

        Methods bootstrapped: ODE, TransE, DistMult, MuRE. Any other KGE
        (e.g. ComplEx) is excluded for this stage only — usage flags are
        restored on exit.

        Adds the fourth sub-experiment ``logical_lambda_asym_zero``,
        which overrides ``lambda_asym=0`` at inference (keeping
        ``lambda_logic`` at its tuned ref value). This is the H_asym
        causal counterfactual: F1[ODE | lambda_asym=ref] vs
        F1[ODE | lambda_asym=0] across the same 8 logical cells used by H2.

        After all seeds, ``compute_final_statistical_report`` runs the
        manuscript's fifteen-contrast statistical report (B2.4 — user
        never invokes statistical_tests.py manually). This reads its own
        verified input paths (``bootstrap_postfix``/``bootstrap_r0_canonical``,
        ``progress/milestone_c_h_asym_*.json``, ``progress/sde_b6_wilcoxon_*.json``),
        not the ``bootstrap/`` directory this stage just populated above --
        see ``statistical_tests.py::compute_final_statistical_report``'s
        docstring. The legacy ``compute_wilcoxon_table`` (which read that
        directory) is no longer called here; it remains only for
        ``src/experiments/run_phase2_resume.py``'s historical use.
        """
        from statistical_tests import compute_final_statistical_report

        print(f"\n{'='*80}")
        print(f"STAGE 4: BOOTSTRAP VALIDATION ({n_seeds} seeds) - {self.dataset_name}")
        print(f"{'='*80}")

        bootstrap_methods = {"ode", "TransE", "DistMult", "MuRE"}
        # Snapshot runner state to restore on exit; bootstrap mutates these
        # in-place so the inner ``run_ablation_stage`` call honours the
        # restricted method set.
        saved_methods = list(self.methods)
        saved_kge = list(self.kge_models)
        saved_use_ode = self.use_ode
        saved_lambda_override = self.lambda_logic_override
        saved_lambda_asym_override = self.lambda_asym_override

        self.methods = [m for m in self.methods if m in bootstrap_methods]
        self.kge_models = [m for m in self.kge_models if m in bootstrap_methods]
        self.use_ode = "ode" in bootstrap_methods and saved_use_ode

        try:
            for seed in range(1, n_seeds + 1):
                seed_root = f"bootstrap/seed_{seed}"
                print(f"\n--- bootstrap seed {seed}/{n_seeds} ---")

                # Random ablation @5%
                self.lambda_logic_override = saved_lambda_override
                self.lambda_asym_override = saved_lambda_asym_override
                self.run_ablation_stage(
                    triples, "random", [5.0], sample_size,
                    perturbation_seed=seed,
                    output_subdir=f"{seed_root}/random",
                )

                # Logical ablation @10%, λ=ref (cached / pre-existing override)
                self.lambda_logic_override = saved_lambda_override
                self.lambda_asym_override = saved_lambda_asym_override
                self.run_ablation_stage(
                    triples, "logical", [10.0], sample_size,
                    perturbation_seed=seed,
                    output_subdir=f"{seed_root}/logical_lambda_ref",
                )

                # Logical ablation @10%, λ_logic=0 (H2 condition)
                self.lambda_logic_override = 0.0
                self.lambda_asym_override = saved_lambda_asym_override
                self.run_ablation_stage(
                    triples, "logical", [10.0], sample_size,
                    perturbation_seed=seed,
                    output_subdir=f"{seed_root}/logical_lambda_zero",
                )

                # Logical ablation @10%, λ_asym=0 (H_asym condition)
                self.lambda_logic_override = saved_lambda_override
                self.lambda_asym_override = 0.0
                self.run_ablation_stage(
                    triples, "logical", [10.0], sample_size,
                    perturbation_seed=seed,
                    output_subdir=f"{seed_root}/logical_lambda_asym_zero",
                )
        finally:
            # Restore runner state regardless of whether the loop completed.
            self.methods = saved_methods
            self.kge_models = saved_kge
            self.use_ode = saved_use_ode
            self.lambda_logic_override = saved_lambda_override
            self.lambda_asym_override = saved_lambda_asym_override

        # B2.4 — final statistical report (15 contrasts, 3 named families:
        # E6_RANDOM_9, HASym_DET_3, HASym_SDE_REVIEW_3) runs automatically.
        # Reads its own fixed, verified input paths (see docstring above),
        # independent of the ``self.base_dir / "bootstrap"`` directory this
        # stage just populated.
        print(f"\n→ Computing final statistical report (15 contrasts, 3 families)")
        compute_final_statistical_report()


def trigger_visualizations(datasets: List[str]):
    """Trigger the generation of line chart visualizations for ablations"""
    print(f"\n{'='*80}")
    print(f"STAGE 3.5: GENERATING ABLATION VISUALIZATIONS")
    print(f"{'='*80}")
    try:
        from plot_ablations import generate_ablation_plots
        base_dir = Path(__file__).parent.parent / "paper_results"
        generate_ablation_plots(str(base_dir))
    except Exception as e:
        print(f"Error generating ablation visualizations: {e}")

def trigger_conclusions(datasets: List[str]):
    """Trigger the LLM conclusions generation"""
    import sys
    print(f"\n{'='*80}")
    print(f"STAGE 4: GENERATING CONCLUSIONS (MDPI DRAFTING)")
    print(f"{'='*80}")
    for ds in datasets:
        try:
            cmd = [sys.executable, str(Path(__file__).parent / "run_conclusions.py"), "--dataset", ds]
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error running conclusions for {ds}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Unified Experimental Orchestrator — KG-Debug-ODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stage dependency graph:
  tune → native → {random_ablation, logical_ablation} → explanations → visualizations

Stage aliases:
  all      = tune native random_ablation logical_ablation explanations visualizations
  ablation = random_ablation logical_ablation
  paper    = tune native random_ablation logical_ablation

Examples:
  python src/run_all.py --datasets fb15k-237 --stages all
  python src/run_all.py --datasets fb15k-237 --stages all --no-interactive
  python src/run_all.py --datasets fb15k-237 --stages tune --force-tune
  python src/run_all.py --datasets fb15k-237 --stages all --dry-run

New flags (phase 0 — all defaults reproduce pre-patch behaviour):
  --seed N                    Set global random seed (default: 42)
  --lambda-logic-override F   Override ODE lambda_logic in memory (default: None = use cache value)
  --output-suffix STR         Suffix appended to the output base directory (default: "")
"""
    )
    parser.add_argument("--datasets", nargs="+", default=["WN18RR"],
                        help="Datasets to run (fb15k-237, codex-m, WN18RR, ...)")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run all three paper datasets")
    parser.add_argument("--methods", nargs="+",
                        default=["ode", "TransE", "DistMult", "MuRE", "TuckER"],
                        help="Methods to run")
    parser.add_argument("--stages", nargs="+",
                        default=["native", "random_ablation", "logical_ablation"],
                        help="Stages to run (or aliases: all, ablation, paper)")
    parser.add_argument("--ablation_levels", nargs="+", type=float,
                        default=[5.0, 10.0, 15.0],
                        help="Perturbation levels (%%) for ablation")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="Graph size limit for ablation. 0 for full graph.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run fast with small subsets")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models/stages that already have result files")
    parser.add_argument("--force-tune", action="store_true",
                        help="Re-run hyperparameter search (requires typing 'regenerate')")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Batch mode: keep existing cache, skip all prompts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing anything")
    # ── New flags (phase 0) ─────────────────────────────────────────────────
    # All defaults preserve pre-patch behaviour exactly.
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42). Propagates to numpy/torch/random and KGPerturbator."
    )
    parser.add_argument(
        "--lambda-logic-override", type=float, default=None,
        dest="lambda_logic_override",
        help="Override lambda_logic for the ODE in memory (not written to cache). Default: None (use cached value)."
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        dest="output_suffix",
        help="Suffix appended to the output base directory (default: empty string)."
    )
    parser.add_argument(
        "--symbolic-bonus", type=float, default=None,
        dest="symbolic_bonus",
        help="Score added per symbolic constraint violation in AnomalyDetector. "
             "Default (flag omitted): dataset-adaptive calibration "
             "(q99 - q50 of the pre-bonus score). Set to 0 to disable the penalty, "
             "or pin a numeric value (e.g. 500.0) for backward-compat with prior runs."
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        dest="n_seeds",
        help="Number of bootstrap seeds for the bootstrap_validation stage (default: 10). "
             "Ignored unless --stages includes bootstrap_validation."
    )

    args = parser.parse_args()

    if args.smoke_test:
        args.ablation_levels = [5.0]
        args.sample_size = 2000

    # ── Propagate random seed (phase 0) ────────────────────────────────────────
    import random as _random
    import torch as _torch
    _random.seed(args.seed)
    np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    if args.seed != 42:
        print(f"[seed] Global seed set to {args.seed}")

    # ── Compute output base directory (phase 0) ─────────────────────────────────
    # Default seed=42 and empty suffix → "paper_results" (unchanged behaviour).
    _seed_suffix = f"_seed{args.seed}" if args.seed != 42 else ""
    base_dir = "paper_results" + _seed_suffix + args.output_suffix
    if base_dir != "paper_results":
        print(f"[output] Base directory: {base_dir}")

    # Dataset selection
    if args.all_datasets:
        datasets = ["fb15k-237", "codex-m", "WN18RR"]
    else:
        datasets = args.datasets

    # Expand stage aliases
    raw_stages: List[str] = args.stages
    expanded: List[str] = []
    for s in raw_stages:
        if s in STAGE_ALIASES:
            for sub in STAGE_ALIASES[s]:
                if sub not in expanded:
                    expanded.append(sub)
        else:
            if s not in expanded:
                expanded.append(s)
    # Preserve canonical ordering
    stages = [s for s in STAGE_ORDER if s in expanded]

    # --dry-run: print and exit
    if args.dry_run:
        print("[dry-run] Would execute the following stages in order:")
        for s in stages:
            deps = STAGE_DEPENDENCIES.get(s, [])
            dep_str = f" (requires: {', '.join(deps)})" if deps else ""
            print(f"  {s}{dep_str}")
        print(f"[dry-run] Datasets: {datasets}")
        print(f"[dry-run] Methods:  {args.methods}")
        sys.exit(0)

    # Dependency check helper
    def _check_prereq(stage: str, dataset: str):
        base = Path("paper_results") / dataset
        kge_methods = [m for m in args.methods if m != "ode"]
        # [phase0] ODE-only runs load params from .cache/hyperparams — no KGE model dirs needed.
        ode_only = len(kge_methods) == 0
        checks = {
            "native": lambda: (CACHE_DIR / f"{dataset}_{args.methods[0]}.json").exists(),
            "random_ablation": lambda: ode_only or any((base / m).exists() for m in kge_methods),
            "logical_ablation": lambda: ode_only or any((base / m).exists() for m in kge_methods),
            # bootstrap_validation re-uses cached native trained_params just
            # like the ablation stages — same prerequisite check.
            "bootstrap_validation": lambda: ode_only or any((base / m).exists() for m in kge_methods),
            "explanations": lambda: (base / "metrics").exists(),
            "visualizations": lambda: (base / "ode").exists() or any((base / m).exists() for m in args.methods),
        }
        checker = checks.get(stage)
        if checker and not checker():
            prereqs = STAGE_DEPENDENCIES.get(stage, [])
            print(f"\n[{stage}] Missing prerequisite. Have you run: {' '.join(prereqs)}?")
            print(f"  Resume with: python src/run_all.py --datasets {dataset} "
                  f"--stages {' '.join(prereqs + [stage])}")
            return False
        return True

    sample_size = args.sample_size if args.sample_size > 0 else None
    no_interactive = args.no_interactive
    force_tune = args.force_tune
    kge_methods = [m for m in args.methods if m != "ode"]

    # Execute stages
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"STARTING COMPREHENSIVE EXPERIMENT: {dataset}")
        print(f"{'#'*70}")

        loader = DataLoader()
        triples = (loader.load_dataset(dataset)[:5000]
                   if args.smoke_test else loader.load_dataset(dataset))

        runner = UnifiedExperimentRunner(
            dataset_name=dataset, methods=args.methods, base_dir=base_dir,
            resume=args.resume, lambda_logic_override=args.lambda_logic_override,
            symbolic_bonus=args.symbolic_bonus,
        )

        for stage in stages:
            try:
                if stage == "tune":
                    run_stage_tune(
                        datasets=[dataset],
                        models=args.methods,
                        force_tune=force_tune,
                        no_interactive=no_interactive,
                    )

                elif stage == "native":
                    if not _check_prereq("native", dataset):
                        sys.exit(1)
                    runner.run_native_stage(triples)

                elif stage == "random_ablation":
                    if not _check_prereq("random_ablation", dataset):
                        sys.exit(1)
                    runner.run_ablation_stage(
                        triples, "random", args.ablation_levels, sample_size
                    )

                elif stage == "logical_ablation":
                    if not _check_prereq("logical_ablation", dataset):
                        sys.exit(1)
                    runner.run_ablation_stage(
                        triples, "logical", args.ablation_levels, sample_size
                    )

                elif stage == "bootstrap_validation":
                    if not _check_prereq("bootstrap_validation", dataset):
                        sys.exit(1)
                    runner.run_bootstrap_validation_stage(
                        triples, n_seeds=args.n_seeds, sample_size=sample_size
                    )

                elif stage == "explanations":
                    if not _check_prereq("explanations", dataset):
                        sys.exit(1)
                    run_stage_explanations([dataset])

                elif stage == "visualizations":
                    trigger_visualizations([dataset])

            except SystemExit:
                raise
            except Exception as exc:
                remaining = stages[stages.index(stage) + 1:]
                resume_cmd = (
                    f"python src/run_all.py --datasets {dataset} "
                    f"--stages {stage} {' '.join(remaining)}"
                ) if remaining else ""
                print(f"\n\u274c Stage '{stage}' failed for {dataset}: {exc}")
                if resume_cmd:
                    print(f"   To resume: {resume_cmd}")
                sys.exit(1)

    # Visualizations & conclusions are always attempted at the end
    if "visualizations" not in stages:
        trigger_visualizations(datasets)
    trigger_conclusions(datasets)


if __name__ == "__main__":
    main()
