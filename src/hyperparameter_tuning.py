"""
Hyperparameter Tuning Module
=============================
Lightweight grid search for optimal KGE parameters
"""

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
from pykeen_detector import PyKEENDetector
from data import DataLoader, Triple


class HyperparameterTuner:
    """Performs lightweight grid search for KGE models"""
    
    def __init__(self, cache_dir: str = ".cache/hyperparams"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_param_grid(self, dataset_size: int) -> Dict:
        """
        Get parameter grid based on dataset size
        Small grid for fast tuning
        """
        if dataset_size < 1000:
            return {
                'embedding_dim': [32, 50],
                'learning_rate': [0.01, 0.001],
                'margin': [1.0, 2.0],
                'negative_sampling': [1, 5]
            }
        elif dataset_size < 10000:
            return {
                'embedding_dim': [50, 100],
                'learning_rate': [0.01, 0.001],
                'margin': [1.0, 2.0],
                'negative_sampling': [5, 10]
            }
        else:
            return {
                'embedding_dim': [100, 128],
                'learning_rate': [0.001, 0.0001],
                'margin': [1.0, 2.0],
                'negative_sampling': [10, 20]
            }
    
    def tune_for_dataset(self, 
                        dataset_name: str,
                        triples: List[Triple],
                        model_name: str,
                        max_trials: int = 8) -> Dict:
        """
        Tune hyperparameters for a specific dataset and model
        
        Returns:
            Best hyperparameters found
        """
        cache_file = self.cache_dir / f"{dataset_name}_{model_name}_best_params.json"
        
        # Check cache
        if cache_file.exists():
            print(f"Loading cached hyperparameters for {model_name} on {dataset_name}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"\nTuning hyperparameters for {model_name} on {dataset_name}...")
        print(f"Dataset size: {len(triples)} triples")
        
        param_grid = self.get_param_grid(len(triples))
        
        # Random search (faster than full grid)
        best_score = -np.inf
        best_params = None
        
        trials = 0
        np.random.seed(42)
        
        while trials < max_trials:
            # Sample parameters
            params = {
                'embedding_dim': int(np.random.choice(param_grid['embedding_dim'])),
                'learning_rate': float(np.random.choice(param_grid['learning_rate'])),
                'margin': float(np.random.choice(param_grid['margin'])),
                'negative_sampling': int(np.random.choice(param_grid['negative_sampling']))
            }
            
            # Train and evaluate
            detector = PyKEENDetector(embedding_dim=params['embedding_dim'])
            
            metrics = detector.train_model(
                triples,
                model_name=model_name,
                epochs=50,  # Fast evaluation
                batch_size=128,
                learning_rate=params['learning_rate']
            )
            
            score = metrics.get('mrr', 0.0)
            
            print(f"  Trial {trials+1}/{max_trials}: MRR={score:.4f} | {params}")
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            trials += 1
        
        print(f"✓ Best params: {best_params} (MRR={best_score:.4f})")
        
        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_params
    
    def tune_ode(self,
                 dataset_name: str,
                 triples: List[Triple],
                 max_trials: int = 5,
                 param_grid_override: Dict = None) -> Dict:
        """
        Tune hyperparameters for Neural ODE method
        Uses a validation split to evaluate MRR

        Three sources of non-determinism in
        the original implementation made the chosen lambdas non-reproducible
        across runs of the tuner:
          1. ``np.random.permutation`` for the train/valid split inherits
             whatever global numpy state the caller left behind.
          2. ``np.random.choice`` for trial param sampling has the same
             dependency on global state.
          3. ``joblib`` workers run in fresh processes that do NOT inherit
             the parent's ``np.random`` state, and ``_evaluate_embeddings_mrr``
             samples random negatives without seeding.
        We pin all three: explicit ``np.random.seed(42)`` here, and a per-
        trial seed forwarded to ``_run_ode_trial`` (which seeds numpy,
        torch, and the negative-sampler before doing any work).

        After tuning, the full trial history
        and the best MRR are stored on the tuner instance as
        ``self.last_trial_history`` and ``self.last_best_score`` so that
        drivers can persist them into the canonical cache (the
        legacy short-circuit cache file ``<ds>_ode_best_params.json`` is
        kept untouched). Driver flow:

            tuner.tune_ode(ds, sub_triples, max_trials=N)
            history = tuner.last_trial_history   # [{trial_id, params, mrr, status}]
            best_mrr = tuner.last_best_score

        ``param_grid_override`` lets drivers supply a wider/narrower
        grid without monkey-patching this method. If None, the canonical
        block7 grid is used (back-compat).
        """
        cache_file = self.cache_dir / f"{dataset_name}_ode_best_params.json"

        if cache_file.exists():
            print(f"Loading cached ODE parameters for {dataset_name}")
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            # Back-compat: short-circuit cache may be flat params dict OR
            # the new {best_params, ...} schema. The block7 driver writes
            # the canonical cache elsewhere, so this short-circuit is
            # only hit by legacy callers.
            return cached.get("best_params", cached) if isinstance(cached, dict) and "best_params" in cached else cached

        print(f"\nTuning Neural ODE parameters for {dataset_name}...")

        # Pin global np.random for the train/valid split and the trial
        # param sampling that follow.
        np.random.seed(42)

        # Split train/valid (simulated for tuning)
        # We use a small validation set for speed
        n_total = len(triples)
        indices = np.random.permutation(n_total)
        split_idx = int(n_total * 0.9)
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

        train_triples = [triples[i] for i in train_indices]
        valid_triples = [triples[i] for i in valid_indices]

        print(f"Split: {len(train_triples)} train, {len(valid_triples)} validation triples")

        from joblib import Parallel, delayed

        print(f"Split: {len(train_triples)} train, {len(valid_triples)} validation triples")
        print(f"Running {max_trials} trials in parallel...")

        # Parameter grid
        # Extended per user spec D7+D11:
        # - lambda_data widened with 0.5.
        # - lambda_logic widened to 8 values across two orders of magnitude
        #   (0.05..20.0) for better tuning quality on the post-block7
        #   asymmetry-as-gradient regime where lambda_logic now governs
        #   ONLY transitivity + symmetry.
        # - lambda_asym + margin_asym are NEW (TASK 1 introduces them).
        # - t_span sampled from {(0,5), (0,10)} per prompt TASK 2 (vs
        #   the pre-block7 hardcoded (0,3)-then-restore-to-(0,5) shortcut).
        # Drivers can override the grid wholesale via the
        # ``param_grid_override`` arg. Required keys: lambda_data,
        # lambda_logic, lambda_asym, margin_asym, lambda_reg, rtol,
        # t_span, embedding_dim. Used by ``run_block8_landscape_search``
        # to sample lambda_logic=0 and lambda_asym=0 explicitly.
        param_grid_default = {
            'lambda_data':  [0.5, 1.0, 5.0, 10.0],
            'lambda_logic': [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            'lambda_asym':  [0.1, 0.5, 1.0, 5.0],
            'margin_asym':  [0.5, 1.0, 2.0],
            'lambda_reg':   [0.01, 0.1],
            'rtol':         [1e-2, 5e-3],
            't_span':       [(0.0, 5.0), (0.0, 10.0)],
            'embedding_dim': [50],
        }
        param_grid = param_grid_override if param_grid_override is not None else param_grid_default
        required_keys = set(param_grid_default.keys())
        missing = required_keys - set(param_grid.keys())
        assert not missing, f"param_grid_override missing required keys: {missing}"

        # Random search over the joint grid. Each trial draws an
        # independent assignment (seeded via np.random.seed(42) above
        # for reproducibility).
        trials_params = []
        for _ in range(max_trials):
            t_span_choice = param_grid['t_span'][int(np.random.choice(len(param_grid['t_span'])))]
            trials_params.append({
                'embedding_dim': int(np.random.choice(param_grid['embedding_dim'])),
                't_span': t_span_choice,
                'lambda_data':  float(np.random.choice(param_grid['lambda_data'])),
                'lambda_logic': float(np.random.choice(param_grid['lambda_logic'])),
                'lambda_asym':  float(np.random.choice(param_grid['lambda_asym'])),
                'margin_asym':  float(np.random.choice(param_grid['margin_asym'])),
                'lambda_reg':   float(np.random.choice(param_grid['lambda_reg'])),
                'rtol':         float(np.random.choice(param_grid['rtol'])),
                'atol':         1e-6,
                'device':       'mps',
            })

        # Run in parallel. Each worker gets a deterministic seed = 42 + i
        # so its (numpy, torch, negative-sampler) state is reproducible.
        results = Parallel(n_jobs=-1)(
            delayed(self._run_ode_trial)(params, train_triples, valid_triples,
                                         trial_seed=42 + i)
            for i, params in enumerate(trials_params)
        )

        # Find best + build trial_history (block8 persistence patch).
        # ``self.last_trial_history`` and ``self.last_best_score`` are
        # populated so downstream drivers (e.g. run_block8_landscape_search)
        # can persist the full random-search trajectory into the canonical
        # cache. The legacy short-circuit cache <ds>_ode_best_params.json
        # below stays minimal (just best_params) for back-compat with the
        # legacy reader at line 142.
        best_score = -np.inf
        best_params = None
        trial_history: List[Dict] = []

        for i, (score, params) in enumerate(results):
            trial_entry = {
                "trial_id": i + 1,
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in params.items()},
                "mrr": float(score) if score is not None else None,
                "status": "ok" if score is not None else "failed",
            }
            trial_history.append(trial_entry)
            if score is not None:
                print(f"  Trial {i+1}: MRR={score:.4f} | {params}")
                if score > best_score:
                    best_score = score
                    best_params = params
            else:
                print(f"  Trial {i+1}: Failed")

        if best_params is None:
            # Fallback: minimal config that respects the new schema.
            best_params = self.get_ode_params(len(triples))
            best_params.setdefault('t_span', (0.0, 5.0))
            best_params.setdefault('lambda_asym', 1.0)
            best_params.setdefault('margin_asym', 1.0)
            best_score = None  # no successful trial; surface as null in cache
        # Otherwise leave best_params exactly as the winning trial: the
        # tuned t_span and rtol are part of the deliberate operating point.

        # Expose trial history on the instance for downstream drivers.
        self.last_trial_history = trial_history
        self.last_best_score = (float(best_score)
                                if isinstance(best_score, (int, float)) and best_score != -np.inf
                                else None)

        print(f"✓ Best ODE params: {best_params} (MRR={best_score})")

        # Cache results (legacy short-circuit cache; canonical cache is
        # written by the driver).
        with open(cache_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        return best_params

    def _run_ode_trial(self, params, train_triples, valid_triples,
                       trial_seed: int = 42):
        """Helper to run a single ODE trial (Picklable).

        ``trial_seed`` pins numpy / torch / random in the worker process
        so the ODE setup, the ODE solve, and the MRR evaluation are
        reproducible across runs of the tuner. Without this seed, joblib
        spawns workers with whatever default state they happen to have,
        producing different MRR estimates on repeated invocations.
        """
        import random as _random
        try:
            import torch as _torch
        except ImportError:
            _torch = None

        np.random.seed(trial_seed)
        _random.seed(trial_seed)
        if _torch is not None:
            _torch.manual_seed(trial_seed)

        try:
            # We need to import here to avoid pickling issues with the class if it wasn't valid
            # but since self is passed, we rely on installed package imports
            from integrated_framework import IntegratedKGDebugger

            # Configure
            config = IntegratedKGDebugger.get_default_config()
            config['use_pykeen'] = False
            config['use_ode'] = True
            config['ode_config'] = params

            # Initialize
            debugger = IntegratedKGDebugger(config)
            debugger.setup_ode_detector(train_triples)

            if debugger.ode_system is None:
                return None, params

            # Solve
            solution = debugger.ode_system.solve()
            final_embeddings = debugger.ode_system.get_final_embeddings(solution)

            # Evaluate (re-seed so MRR sampling is independent of the
            # post-ODE-solve numpy state; uses trial_seed + 1 to avoid
            # collision with the pre-solve seed).
            np.random.seed(trial_seed + 1)
            mrr = self._evaluate_embeddings_mrr(final_embeddings, valid_triples)
            return mrr, params

        except Exception as e:
            # print(f"Trial failed: {e}") # validation might noise up stdout
            return None, params

    def _evaluate_embeddings_mrr(self, embeddings: Dict[str, np.ndarray], triples: List[Triple]) -> float:
        """Calculate MRR for embeddings using TransE scoring function"""
        ranks = []
        
        # Create entity matrix for fast scoring
        entities = sorted(list(embeddings.keys()))
        ent_to_idx = {e: i for i, e in enumerate(entities)}
        
        # Filter entities that have embeddings (relations might be missing in dict if implicit)
        valid_entities_mask = [e in embeddings for e in entities]
        if not any(valid_entities_mask):
            return 0.0
            
        # We only evaluate on implicit relations handled by the ODE (often embedded in the system)
        # But here we need relation embeddings. 
        # The ODE system stores relations in the same embedding dict usually?
        # Let's check ODE system structure. It passes 'embeddings' dict.
        
        for t in triples:
            if t.subject not in embeddings or t.object not in embeddings:
                continue
                
            h = embeddings[t.subject]
            target_t = embeddings[t.object]
            
            # If relation embedding exists, use it. If not, assume 0 (identity)
            r = embeddings.get(t.relation, np.zeros_like(h))
            
            # Predict t: target = h + r
            pred_vec = h + r
            
            # Calculate distances to all entities
            # Optimization: Just do a small sample if dataset is huge in future
            # For now, iterate all
            dists = []
            target_dist = np.linalg.norm(pred_vec - target_t)
            
            # Compare against 100 random negatives + target to estimate rank (Standard approx)
            neg_samples = np.random.choice(entities, 50)
            
            rank = 1
            for neg in neg_samples:
                if neg == t.object: continue
                if neg not in embeddings: continue
                
                neg_emb = embeddings[neg]
                neg_dist = np.linalg.norm(pred_vec - neg_emb)
                
                if neg_dist < target_dist:
                    rank += 1
            
            ranks.append(1.0 / rank)
            
        return np.mean(ranks) if ranks else 0.0

    def get_ode_params(self, dataset_size: int) -> Dict:
        """Get default ODE parameters based on dataset size (Fallback)"""
        if dataset_size < 1000:
            return {
                'embedding_dim': 32,
                't_span': (0.0, 3.0),
                'lambda_data': 1.0,
                'lambda_logic': 1.0,
                'lambda_reg': 0.1,
                'device': 'mps'
            }
        elif dataset_size < 10000:
            return {
                'embedding_dim': 50,
                't_span': (0.0, 5.0),
                'lambda_data': 1.0,
                'lambda_logic': 1.0,
                'lambda_reg': 0.1,
                'device': 'mps'
            }
        else:
            return {
                'embedding_dim': 100,
                't_span': (0.0, 5.0),
                'lambda_data': 1.0,
                'lambda_logic': 1.0,
                'lambda_reg': 0.1,
                'device': 'mps'
            }
