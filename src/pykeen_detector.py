"""
PyKEEN-based Anomaly Detection
===============================
Detection using Knowledge Graph Embeddings via PyKEEN
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from tqdm import tqdm

from data import Triple

@dataclass 
class PyKEENAnomalyReport:
    triple: Triple
    anomaly_score: float
    anomaly_types: List[str]
    explanation: str
    confidence: float
    suggested_action: str

class PyKEENDetector:
    """
    PyKEEN-based anomaly detector with full functionality
    matching ODE detector capabilities
    """
    
    def __init__(self, embedding_dim: int = 50, device: str = None):
        self.embedding_dim = embedding_dim
        
        # Auto-detect best available device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
                print(f"🚀 Using MPS (Apple Silicon GPU) for PyKEEN")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 Using CUDA (NVIDIA GPU) for PyKEEN")
            else:
                self.device = 'cpu'
                print(f"💻 Using CPU for PyKEEN")
        else:
            self.device = device
            print(f"🔧 Using specified device: {device} for PyKEEN")
        
        self.model = None
        self.tf = None  # TriplesFactory
        self.embeddings = None
        
    def train_model(self, triples: List[Triple], 
                   model_name: str = "TransE",
                   epochs: int = 100,
                   batch_size: int = 32,
                   **kwargs) -> Dict:
        """Train PyKEEN model on triples"""
        try:
            from pykeen.pipeline import pipeline
            from pykeen.triples import TriplesFactory
            
            print(f"\nTraining {model_name} model...")
            
            # Convert to PyKEEN format
            triple_list = [(t.subject, t.relation, t.object) for t in triples]
            self.tf = TriplesFactory.from_labeled_triples(
                np.array(triple_list, dtype=str)
            )
            
            # Check for complex models on MPS
            device_to_use = self.device
            if self.device == 'mps' and model_name in ['RotatE', 'ComplEx']:
                print(f"⚠️ Switching to CPU for {model_name} (Complex operations not fully supported on MPS)")
                device_to_use = 'cpu'

            # Train model
            result = pipeline(
                training=self.tf,
                testing=self.tf,  # Use same set for testing in this context
                model=model_name,
                model_kwargs=dict(
                    embedding_dim=int(self.embedding_dim),
                ),
                training_kwargs=dict(
                    num_epochs=epochs,
                    batch_size=batch_size,
                ),
                optimizer_kwargs=dict(
                    lr=kwargs.get('learning_rate', 0.001)
                ),
                negative_sampler_kwargs=dict(
                    num_negs_per_pos=kwargs.get('negative_sampling', 5)
                ) if 'negative_sampling' in kwargs else None,
                loss_kwargs=dict(
                    margin=kwargs.get('margin', 1.0)
                ) if 'margin' in kwargs else None,
                random_seed=42,
                device=device_to_use
            )
            
            self.model = result.model
            self.extract_embeddings()
            
            # Get metrics
            metrics = {
                'model': model_name,
                'epochs': epochs,
                'final_loss': result.losses[-1] if hasattr(result, 'losses') else 0.0
            }
            
            # Evaluate if possible
            try:
                from pykeen.evaluation import RankBasedEvaluator
                evaluator = RankBasedEvaluator()
                eval_results = evaluator.evaluate(
                    model=self.model,
                    mapped_triples=self.tf.mapped_triples,
                    batch_size=batch_size
                )
                metrics['mrr'] = eval_results.get_metric('mrr')
                metrics['hits@10'] = eval_results.get_metric('hits@10')
            except:
                metrics['mrr'] = 0.0
                metrics['hits@10'] = 0.0
            
            print(f"Training complete. MRR: {metrics.get('mrr', 0.0):.4f}")
            return metrics
            
        except ImportError:
            print("PyKEEN not installed, using mock training")
            return self._mock_train(triples)
    
    def _mock_train(self, triples: List[Triple]) -> Dict:
        """Mock training for when PyKEEN is not available"""
        # Create random embeddings
        entities = set()
        relations = set()
        for t in triples:
            entities.add(t.subject)
            entities.add(t.object)
            relations.add(t.relation)
        
        self.embeddings = {
            'entities': {e: np.random.randn(self.embedding_dim) for e in entities},
            'relations': {r: np.random.randn(self.embedding_dim) for r in relations}
        }
        
        return {'model': 'mock', 'mrr': 0.5}
    
    def extract_embeddings(self) -> Dict[str, np.ndarray]:
        """Extract embeddings from trained model"""
        if self.model is None:
            return {}
        
        embeddings = {}
        
        try:
            # Get entity embeddings
            if hasattr(self.model, 'entity_embeddings'):
                entity_emb = self.model.entity_embeddings.weight.detach().cpu().numpy()
            elif hasattr(self.model, 'entity_representations'):
                # PyKEEN 1.9+
                entity_emb = self.model.entity_representations[0](indices=None).detach().cpu().numpy()
            else:
                entity_emb = None

            if entity_emb is not None:
                for i, entity in enumerate(self.tf.entity_to_id.keys()):
                    embeddings[entity] = entity_emb[i]
            
            # Get relation embeddings
            if hasattr(self.model, 'relation_embeddings'):
                rel_emb = self.model.relation_embeddings.weight.detach().cpu().numpy()
            elif hasattr(self.model, 'relation_representations'):
                rel_emb = self.model.relation_representations[0](indices=None).detach().cpu().numpy()
            else:
                rel_emb = None
                
            if rel_emb is not None:
                for i, relation in enumerate(self.tf.relation_to_id.keys()):
                    embeddings[relation] = rel_emb[i]
        except:
            pass
        
        self.embeddings = embeddings
        return embeddings
    
    def compute_triple_score(self, triple: Triple) -> float:
        """Compute plausibility score for a triple using the trained model.

        BUG FIX history:
          - Bug 1: score_hrt() expects a single (batch, 3) tensor, not 3 separate tensors.
          - Bug 2: embeddings fallback used 'entities' key that never exists (flat dict).
          - Bug 3: silent np.random.random() fallback was returning identical scores for
                   all models when seed was fixed by pipeline(random_seed=42), making the
                   'KGE convergence' result an artefact rather than an empirical finding.
        """
        if self.model is None:
            raise RuntimeError(
                "compute_triple_score called but model is not trained. "
                "Call train_model() first."
            )

        h_id = self.tf.entity_to_id.get(triple.subject)
        r_id = self.tf.relation_to_id.get(triple.relation)
        t_id = self.tf.entity_to_id.get(triple.object)

        if None in [h_id, r_id, t_id]:
            # Triple contains entities/relations unseen during training.
            return 0.0

        # BUG 1 FIX: PyKEEN score_hrt expects a single tensor of shape (batch, 3).
        # Also ensure tensor is on the same device as the model.
        hrt_tensor = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long).to(self.model.device)
        with torch.no_grad():
            score = self.model.score_hrt(hrt_tensor).item()
        return score
    
    def compute_embedding_variance(self, entity: str) -> float:
        """Compute variance in entity's neighborhood (similar to ODE instability).

        BUG FIX: extract_embeddings() stores a FLAT dict {name: array}.
        The old code checked 'entities' in self.embeddings which was always False,
        so it accidentally fell through to use the flat dict anyway via the else branch.
        Simplified to always use the flat dict directly.
        """
        if not self.embeddings:
            return 0.0

        # BUG 2 FIX: embeddings is a flat dict {name: array}, no 'entities' sub-key.
        emb_dict = self.embeddings

        if entity not in emb_dict:
            return 0.0

        entity_emb = emb_dict[entity]

        # Find neighbors (entities with similar embeddings)
        distances = [
            np.linalg.norm(entity_emb - other_emb)
            for other, other_emb in emb_dict.items()
            if other != entity and isinstance(other_emb, np.ndarray)
        ]

        # Variance of distances indicates how isolated the entity is
        if distances:
            return np.var(distances)
        return 0.0

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_all_triples_batch(
        self, triples: List[Triple], batch_size: int = 4096
    ) -> np.ndarray:
        """Score every triple in *triples* using a single batched forward pass.

        Returns raw model scores as a float32 ndarray of shape (len(triples),).
        Triples that contain unknown entities/relations receive score 0.0.

        Args:
            triples:    Triples to score.
            batch_size: Maximum number of triples per GPU/CPU forward call.
                        Lower this if you hit OOM on large graphs.
        """
        if self.model is None:
            raise RuntimeError(
                "score_all_triples_batch called but model is not trained. "
                "Call train_model() first."
            )

        scores = np.zeros(len(triples), dtype=np.float32)
        valid_idx, valid_ids = [], []

        for i, t in enumerate(triples):
            h = self.tf.entity_to_id.get(t.subject)
            r = self.tf.relation_to_id.get(t.relation)
            o = self.tf.entity_to_id.get(t.object)
            if None not in (h, r, o):
                valid_idx.append(i)
                valid_ids.append([h, r, o])

        if not valid_ids:
            return scores

        device = self.model.device
        with torch.no_grad():
            for start in range(0, len(valid_ids), batch_size):
                chunk = valid_ids[start:start + batch_size]
                hrt = torch.tensor(chunk, dtype=torch.long).to(device)
                raw = self.model.score_hrt(hrt).cpu().numpy().flatten()
                # Sanitize NaN/Inf that can occur on MPS with ComplEx/RotatE.
                # Replace non-finite values with 0.0 so downstream Otsu
                # threshold and histogram calls never receive invalid data.
                raw = np.where(np.isfinite(raw), raw, 0.0)
                for pos, global_idx in enumerate(valid_idx[start:start + batch_size]):
                    scores[global_idx] = raw[pos]

        return scores

    @staticmethod
    def _otsu_threshold(values: np.ndarray) -> float:
        """Compute Otsu's binarisation threshold using pure numpy.

        Finds the threshold t that maximises inter-class variance:
            σ²_B(t) = w₀(t)·w₁(t)·[μ₀(t) − μ₁(t)]²

        Equivalent to skimage.filters.threshold_otsu but with no extra dependency.

        Robustness:
        - Strips NaN/Inf before histogram (should already be absent after
          score_all_triples_batch sanitisation, but kept as a safety net).
        - Returns the median when the distribution is constant (all values equal),
          avoiding the [nan, nan] range error from np.histogram.
        """
        # Strip non-finite values defensively
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            return 0.0

        # Degenerate case: all values identical → histogram range collapses
        if float(finite_vals.max()) == float(finite_vals.min()):
            return float(finite_vals[0])

        hist, bin_edges = np.histogram(finite_vals, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = float(hist.sum())
        if total == 0:
            return float(bin_centers[len(bin_centers) // 2])

        w0 = np.cumsum(hist) / total                         # weight class 0
        w1 = 1.0 - w0                                        # weight class 1
        mu_cumsum = np.cumsum(hist * bin_centers)
        mu0 = mu_cumsum / (np.cumsum(hist) + 1e-12)          # mean class 0
        mu_total = mu_cumsum[-1] / total
        with np.errstate(invalid="ignore", divide="ignore"):
            mu1 = np.where(w1 > 1e-12, (mu_total - w0 * mu0) / w1, 0.0)

        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
        return float(bin_centers[np.argmax(sigma_b)])

    def detect_ranking_anomalies(
        self, triples: List[Triple]
    ) -> List[PyKEENAnomalyReport]:
        """Detect anomalies by scoring all triples in batch and binarising
        with Otsu's threshold on the full score distribution.

        Uses a pure-numpy Otsu implementation — no extra dependencies.
        Side effects:
          self._last_raw_scores     : float32 ndarray of shape (N,)
          self._last_otsu_threshold : float, the chosen threshold
        """
        print(f"    Batch-scoring {len(triples)} triples...")
        raw_scores = self.score_all_triples_batch(triples)   # shape (N,)
        self._last_raw_scores = raw_scores

        # PyKEEN: lower score = more anomalous.
        # Negate so anomalous triples have HIGH values → Otsu splits correctly.
        neg_scores = -raw_scores
        threshold = self._otsu_threshold(neg_scores)
        self._last_otsu_threshold = float(threshold)
        anomaly_mask = neg_scores >= threshold

        n_anomalies = int(anomaly_mask.sum())
        print(
            f"    Otsu threshold: {threshold:.4f}  →  "
            f"{n_anomalies} anomalies ({n_anomalies / len(triples) * 100:.1f}%)"
        )

        # Pre-compute embedding variances for detected triples only
        variance_cache: Dict[str, float] = {}
        if self.embeddings:
            relevant_entities: set = set()
            for i, triple in enumerate(triples):
                if anomaly_mask[i]:
                    relevant_entities.add(triple.subject)
                    relevant_entities.add(triple.object)
            for entity in relevant_entities:
                variance_cache[entity] = self.compute_embedding_variance(entity)

        variance_p90 = (
            float(np.percentile(list(variance_cache.values()), 90))
            if variance_cache else 0.0
        )

        # Normalise the anomalous neg_scores to [0, 1]
        anom_neg = neg_scores[anomaly_mask]
        lo, hi = anom_neg.min(), anom_neg.max()
        span = float(hi - lo) if hi > lo else 1.0

        reports: List[PyKEENAnomalyReport] = []
        for i, triple in enumerate(triples):
            if not anomaly_mask[i]:
                continue

            norm_score = float((neg_scores[i] - lo) / span)
            anomaly_types = ["low_plausibility"]

            subj_var = variance_cache.get(triple.subject, 0.0)
            obj_var = variance_cache.get(triple.object, 0.0)
            if subj_var > variance_p90:
                anomaly_types.append("isolated_subject")
            if obj_var > variance_p90:
                anomaly_types.append("isolated_object")

            explanation = (
                f"Raw score {raw_scores[i]:.4f} — below Otsu threshold "
                f"(neg_score {neg_scores[i]:.4f} ≥ {threshold:.4f})."
            )
            if "isolated_subject" in anomaly_types:
                explanation += f" Subject '{triple.subject}' is isolated."
            if "isolated_object" in anomaly_types:
                explanation += f" Object '{triple.object}' is isolated."

            action = (
                "remove" if len(anomaly_types) >= 2
                else "verify" if norm_score > 0.8
                else "correct"
            )

            reports.append(PyKEENAnomalyReport(
                triple=triple,
                anomaly_score=norm_score,
                anomaly_types=anomaly_types,
                explanation=explanation,
                confidence=norm_score,
                suggested_action=action,
            ))

        return reports

    def detect_neighborhood_anomalies(self, triples: List[Triple]) -> List[PyKEENAnomalyReport]:
        """Detect anomalies based on neighborhood-embedding variance.

        The historical implementation hardcoded
        ``avg_variance > 5.0`` as the inconsistency threshold. KGE methods
        produce embeddings on very different magnitude scales — empirically:

            TransE / DistMult / MuRE  →  norms ~0.02 (variance never reaches 5.0)
            ComplEx                    →  norms ~14   (variance crosses 5.0 often)

        On real ablation runs this meant TransE/DistMult/MuRE silently returned
        the empty list while ComplEx returned thousands of reports — fully
        consistent with what the KGE-AUDIT decomposition observed (constraint=0,
        neighborhood=0 for 3/4 methods; only ComplEx had neighborhood>0).

        ENABLE-FOR-ALL fix: replace the magic 5.0 with a per-call adaptive
        threshold equal to ``q99`` of the empirical avg_variance distribution
        across the triples in this call. This calibrates the cutoff to the
        embedding magnitude regime of whichever KGE method was trained, so
        the detector fires consistently on the top ~1 % most-heterogeneous
        neighborhoods regardless of model. Two-pass implementation: first
        pass computes the variances, derives the threshold; second pass
        emits reports.

        Backward-compat note: on ComplEx the q99 cutoff lands well above 5.0
        (typical neighborhood variance there is around 14), so the new
        threshold is more selective than the old one — the tail-fraction is
        what we actually want, not the absolute magnitude.
        """
        reports = []

        if not self.embeddings:
            return reports

        # Build graph structure
        graph = {}
        for triple in triples:
            if triple.subject not in graph:
                graph[triple.subject] = {'out': [], 'in': []}
            if triple.object not in graph:
                graph[triple.object] = {'out': [], 'in': []}

            graph[triple.subject]['out'].append((triple.relation, triple.object))
            graph[triple.object]['in'].append((triple.relation, triple.subject))

        # Pass 1 — compute avg_variance for every triple (no thresholding yet).
        per_triple_var = []
        for triple in triples:
            v = None
            if triple.subject in graph:
                subj_neighbors = [o for _, o in graph[triple.subject]['out']]
                if subj_neighbors:
                    neighbor_embeddings = []
                    for n in subj_neighbors:
                        # BUG 2 FIX: embeddings is a flat dict {name: array}
                        emb = self.embeddings.get(n)
                        if emb is not None and isinstance(emb, np.ndarray):
                            neighbor_embeddings.append(emb)
                    if len(neighbor_embeddings) > 1:
                        mean_emb = np.mean(neighbor_embeddings, axis=0)
                        variances = [np.linalg.norm(e - mean_emb)
                                     for e in neighbor_embeddings]
                        v = float(np.mean(variances))
            per_triple_var.append(v)

        # Adaptive threshold: q99 of the non-None avg_variance distribution.
        # Falls back to a no-op when too few triples have neighborhoods.
        defined_vars = [v for v in per_triple_var if v is not None]
        if len(defined_vars) < 10:
            return reports
        adaptive_threshold = float(np.quantile(defined_vars, 0.99))
        # Defend against degenerate distributions (all variances equal) — when
        # everything is equal there is no meaningful "tail" to flag.
        if not np.isfinite(adaptive_threshold) or adaptive_threshold <= 0.0:
            return reports
        print(f"    [neighborhood] adaptive threshold = q99(avg_variance) = "
              f"{adaptive_threshold:.4f} over {len(defined_vars)} triples-with-neighbors")

        # Pass 2 — emit reports for triples whose avg_variance exceeds q99.
        for triple, v in zip(triples, per_triple_var):
            if v is None or v <= adaptive_threshold:
                continue
            report = PyKEENAnomalyReport(
                triple=triple,
                anomaly_score=min(0.3, 1.0),
                anomaly_types=["neighborhood_inconsistency"],
                explanation=(f"Triple's neighborhood variance "
                             f"{v:.4f} > q99 = {adaptive_threshold:.4f}"),
                confidence=0.5,
                suggested_action="verify",
            )
            reports.append(report)

        return reports

    
    def detect_constraint_violations(self, triples: List[Triple], 
                                    constraint_checker) -> List[PyKEENAnomalyReport]:
        """
        Detect constraint violations (matching ODE capability)
        
        Args:
            triples: List of triples to check
            constraint_checker: ConstraintChecker with defined constraints
        """
        reports = []
        
        if constraint_checker is None:
            return reports
        
        # Get all violating triples
        violations = constraint_checker.get_all_violating_triples(triples)
        
        for constraint_name, violating_triples in violations.items():
            for triple in violating_triples:
                # Also check PyKEEN score for this triple
                pykeen_score = self.compute_triple_score(triple)
                
                # Combine constraint violation (weight 0.6) with low plausibility (weight 0.4)
                # This gives PyKEEN's perspective on constraint violations
                combined_score = 0.6 * 1.0 + 0.4 * (1.0 - pykeen_score)
                
                report = PyKEENAnomalyReport(
                    triple=triple,
                    anomaly_score=combined_score,
                    anomaly_types=['constraint_violation', f'violates_{constraint_name}'],
                    explanation=f'Violates {constraint_name} constraint. PyKEEN plausibility: {pykeen_score:.3f}',
                    confidence=0.9,  # High confidence for constraint violations
                    suggested_action='remove' if combined_score > 0.8 else 'verify'
                )
                reports.append(report)
        
        return reports
    
    def detect_all_anomalies(self, triples: List[Triple],
                            methods: List[str] = None,
                            constraint_checker=None) -> List[PyKEENAnomalyReport]:
        """
        Run all PyKEEN-based anomaly detection methods INCLUDING constraints

        Args:
            triples: List of triples to check
            methods: Which methods to use (default: all)
            constraint_checker: Optional constraint checker for logical violations
        """
        if methods is None:
            methods = ['ranking', 'neighborhood', 'constraints']

        all_reports = {}
        # Per-detector contribution sets, kept
        # as Python sets so we can compute pairwise/triple intersections after
        # the merges complete. Purely additive instrumentation — does not
        # affect detect_all_anomalies's return value or merge semantics.
        ranking_set: set = set()
        neighborhood_set: set = set()
        constraint_set: set = set()
        latent_set: set = set()

        print("\nRunning PyKEEN-based anomaly detection...")

        # Ranking-based detection (statistical anomalies)
        if 'ranking' in methods:
            print("  - Detecting ranking anomalies...")
            ranking_reports = self.detect_ranking_anomalies(triples)
            for report in ranking_reports:
                key = (report.triple.subject, report.triple.relation, report.triple.object)
                ranking_set.add(key)
                all_reports[key] = report
        
        # Neighborhood-based detection (structural anomalies)
        if 'neighborhood' in methods:
            print("  - Detecting neighborhood anomalies...")
            neighborhood_reports = self.detect_neighborhood_anomalies(triples)
            for report in neighborhood_reports:
                key = (report.triple.subject, report.triple.relation, report.triple.object)
                neighborhood_set.add(key)
                if key in all_reports:
                    # Merge reports
                    all_reports[key].anomaly_score = max(
                        all_reports[key].anomaly_score,
                        report.anomaly_score
                    )
                    all_reports[key].anomaly_types.extend(report.anomaly_types)
                else:
                    all_reports[key] = report
        
        # Constraint-based detection (logical violations) - NEW!
        if 'constraints' in methods and constraint_checker is not None:
            print("  - Detecting constraint violations...")
            constraint_reports = self.detect_constraint_violations(triples, constraint_checker)
            for report in constraint_reports:
                key = (report.triple.subject, report.triple.relation, report.triple.object)
                constraint_set.add(key)
                if key in all_reports:
                    # Merge with existing report - take maximum score
                    all_reports[key].anomaly_score = max(
                        all_reports[key].anomaly_score,
                        report.anomaly_score
                    )
                    all_reports[key].anomaly_types.extend(report.anomaly_types)
                    all_reports[key].explanation += " | " + report.explanation
                    # Upgrade action if constraint violation
                    if 'constraint_violation' in report.anomaly_types:
                        all_reports[key].suggested_action = 'remove'
                else:
                    all_reports[key] = report
            
            # Latent Violation Detection (Inference-based) - NEW!
            print("  - Detecting LATENT violations (inference)...")
            latent_reports = self.detect_latent_anomalies(triples, constraint_checker)
            for report in latent_reports:
                # These are NEW triples (not in original list), so we add them directly
                # We key them by their content
                key = (report.triple.subject, report.triple.relation, report.triple.object)
                latent_set.add(key)
                if key not in all_reports:
                    all_reports[key] = report
                else:
                    # If it already exists (e.g. it was in the graph), we merge
                    all_reports[key].anomaly_types.append("latent_violation")
                    all_reports[key].explanation += " | " + report.explanation
        
        # Sort by anomaly score
        final_reports = sorted(all_reports.values(), 
                              key=lambda x: x.anomaly_score, 
                              reverse=True)
        
        print(f"PyKEEN detected {len(final_reports)} total anomalies:")
        print(f"  - Statistical: {sum(1 for r in final_reports if 'low_plausibility' in str(r.anomaly_types))}")
        print(f"  - Structural: {sum(1 for r in final_reports if 'neighborhood' in str(r.anomaly_types))}")
        print(f"  - Constraints: {sum(1 for r in final_reports if 'constraint_violation' in r.anomaly_types)}")
        print(f"  - Latent: {sum(1 for r in final_reports if 'latent_violation' in str(r.anomaly_types))}")

        # Per-detector contribution breakdown.
        # ranking_set / neighborhood_set / constraint_set / latent_set were
        # populated above. Print exclusive and pairwise/triple intersection
        # counts so the audit can decompose which detector drives the
        # final detection set on each ablation cell. Purely additive.
        rk, nb, ct, lt = ranking_set, neighborhood_set, constraint_set, latent_set
        union_all = rk | nb | ct | lt
        print(f"  [KGE-AUDIT decomp] |ranking|={len(rk)} |neighborhood|={len(nb)} "
              f"|constraints|={len(ct)} |latent|={len(lt)} |union|={len(union_all)}")
        print(f"  [KGE-AUDIT decomp] ranking_only={len(rk - nb - ct - lt)} "
              f"neighborhood_only={len(nb - rk - ct - lt)} "
              f"constraints_only={len(ct - rk - nb - lt)} "
              f"latent_only={len(lt - rk - nb - ct)}")
        print(f"  [KGE-AUDIT decomp] rank∩nb={len(rk & nb)} rank∩ct={len(rk & ct)} "
              f"nb∩ct={len(nb & ct)} rank∩nb∩ct={len(rk & nb & ct)}")
        # Side-effect: expose the decomposition for any caller that wants it.
        self._last_audit_decomp = {
            "ranking": rk, "neighborhood": nb, "constraints": ct, "latent": lt,
            "union": union_all,
        }

        return final_reports

    def detect_latent_anomalies(self, triples: List[Triple], constraint_checker) -> List[PyKEENAnomalyReport]:
        """
        Detect LATENT anomalies: Triples that are NOT in the graph but are predicted
        with high probability and violate constraints.
        
        Focuses on:
        1. Latent Cardinality Violations: Predicting extra objects for max-1 relations
        2. Latent Disjointness: Predicting relations that conflict with existing ones
        """
        reports = []
        if self.model is None or constraint_checker is None:
            return reports
            
        # 1. Check Cardinality Constraints
        # If (s, r, o) exists and r is max-1, check if model predicts (s, r, o') with high score
        for constraint in constraint_checker.constraints:
            if hasattr(constraint, 'max_cardinality') and constraint.max_cardinality == 1:
                relation = constraint.relation
                # Get subjects that have this relation
                subjects = {t.subject for t in triples if t.relation == relation}
                
                for s in subjects:
                    # Get existing objects
                    existing_objects = {t.object for t in triples if t.subject == s and t.relation == relation}
                    
                    # Predict top candidates for (s, r, ?)
                    # We scan all entities (expensive, so we limit to a subset or use top-k if available)
                    # For efficiency in this demo, we check top 10 entities by embedding similarity
                    # or just check a random sample if model doesn't support efficient top-k
                    
                    # Simplified approach: Check top 5 closest entities in embedding space to s+r
                    candidates = self._suggest_top_k_objects(s, relation, k=5)
                    
                    for cand_obj, score in candidates:
                        if cand_obj not in existing_objects:
                            # This is a predicted triple (s, r, cand_obj)
                            # It violates cardinality because s already has an object (existing_objects)
                            
                            # If score is high enough, it's a latent violation
                            if score > 0.8: # Threshold
                                report = PyKEENAnomalyReport(
                                    triple=Triple(s, relation, cand_obj),
                                    anomaly_score=score,
                                    anomaly_types=["latent_violation", "latent_cardinality"],
                                    explanation=f"Model predicts ({s}, {relation}, {cand_obj}) with high confidence ({score:.2f}), but {s} already has {relation} {list(existing_objects)[0]}.",
                                    confidence=score,
                                    suggested_action="verify" # Verify if the NEW fact is true (and old one wrong) or if model is hallucinating
                                )
                                reports.append(report)

        return reports

    def _suggest_top_k_objects(self, subject: str, relation: str, k: int = 5) -> List[Tuple[str, float]]:
        """Helper to find top-k predicted objects using embeddings"""
        if not self.embeddings:
            return []
            
        try:
            # BUG 2 FIX: embeddings is a flat dict {name: array} — no 'entities' sub-key.
            h = self.embeddings.get(subject)
            r = self.embeddings.get(relation)

            if h is None or r is None:
                return []

            target = h + r

            candidates = [
                (ent, 1.0 / (1.0 + np.linalg.norm(target - emb)))
                for ent, emb in self.embeddings.items()
                if ent != subject and isinstance(emb, np.ndarray)
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:k]

        except Exception:
            return []

    
    def get_embedding_statistics(self) -> Dict:
        """Get statistics about the embeddings"""
        if not self.embeddings:
            return {}
        
        if 'entities' in self.embeddings:
            entity_emb = list(self.embeddings['entities'].values())
        else:
            entity_emb = list(self.embeddings.values())
        
        if not entity_emb:
            return {}
        
        entity_emb = np.array(entity_emb)
        
        stats = {
            'n_embeddings': len(entity_emb),
            'embedding_dim': entity_emb.shape[1] if len(entity_emb) > 0 else 0,
            'mean_norm': np.mean([np.linalg.norm(e) for e in entity_emb]),
            'std_norm': np.std([np.linalg.norm(e) for e in entity_emb]),
            'max_norm': np.max([np.linalg.norm(e) for e in entity_emb]),
            'min_norm': np.min([np.linalg.norm(e) for e in entity_emb])
        }
        
        return stats

# Utility function for easy use
def detect_pykeen_anomalies(triples: List[Tuple[str, str, str]],
                           model_name: str = "TransE",
                           epochs: int = 100) -> List[PyKEENAnomalyReport]:
    """
    Quick function to detect anomalies using PyKEEN
    
    Args:
        triples: List of (subject, relation, object) tuples
        model_name: Which PyKEEN model to use
        epochs: Training epochs
    
    Returns:
        List of anomaly reports
    """
    # Convert to Triple objects
    triple_objects = [Triple(s, r, o) for s, r, o in triples]
    
    # Initialize detector
    detector = PyKEENDetector()
    
    # Train model
    detector.train_model(triple_objects, model_name=model_name, epochs=epochs)
    
    # Detect anomalies
    reports = detector.detect_all_anomalies(triple_objects)
    
    return reports

if __name__ == "__main__":
    # Test the detector
    test_triples = [
        ('Paris', 'capitalOf', 'France'),
        ('France', 'locatedIn', 'Europe'),
        ('Paris', 'locatedIn', 'Asia'),  # Anomaly
        ('Bob', 'marriedTo', 'Alice'),
        ('Alice', 'marriedTo', 'Bob'),
    ]
    
    reports = detect_pykeen_anomalies(test_triples, epochs=50)
    
    print("\nTop anomalies detected:")
    for i, report in enumerate(reports[:5], 1):
        print(f"{i}. {report.triple.subject} - {report.triple.relation} - {report.triple.object}")
        print(f"   Score: {report.anomaly_score:.3f} | Action: {report.suggested_action}")
