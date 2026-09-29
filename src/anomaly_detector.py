"""
Anomaly Detection Module
Analyzes ODE trajectories to identify problematic triples
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from constraints import Triple, ConstraintChecker
from ode_system import ODESolution
from explanation import ChainOfThoughtExplainer


@dataclass
class AnomalyReport:
    """Report for a single anomalous triple"""
    triple: Triple
    anomaly_score: float
    anomaly_types: List[str]
    explanation: str
    confidence: float
    suggested_action: str  # "explain", "verify"
    cot_analysis: Dict = None
    plausibility_score: float = 0.0  # New: Logic vs Embedding agreement
    risk_assessment: str = "low"     # New: "false_positive_risk", "false_negative_risk", "confirmed"


class AnomalyDetector:
    """
    Detects anomalies in KG by analyzing ODE system behavior
    """
    
    def __init__(self, ode_system, solution, constraint_checker=None, full_triples=None,
                 symbolic_bonus=None):
        """
        Initialize the anomaly detector

        Args:
            ode_system: The solved ODE system
            solution: The solution object
            constraint_checker: Optional Constraints checking system
            full_triples: Optional complete list of triples (for accurate constraint evaluation when sampling)
            symbolic_bonus: Score added per symbolic constraint violation.
                            - None (default): dataset-adaptive calibration. The bonus is
                              computed lazily on first use as ``q99(score)`` of the
                              per-triple pre-bonus score
                              (energy * 5 + max_instability * 100). A violation lifts
                              the triple's score by the 99th-percentile value of the
                              dataset's empirical pre-bonus distribution, putting any
                              violator in the "deep tail" zone regardless of where its
                              own dynamical signal sits. Calibration runs once per
                              AnomalyDetector instance and the result is cached on
                              ``self._calibrated_bonus``.
                            - float (e.g. 0.0, 500.0): explicit override; calibration is
                              skipped. Backward-compatible with prior runs that pinned the
                              hardcoded +500 magic number.
        """
        self.ode_system = ode_system
        self.solution = solution
        self.triples = ode_system.triples # The triples that were part of the ODE system
        self.full_triples = full_triples if full_triples is not None else self.triples # All triples for constraint checking
        self.constraint_checker = constraint_checker
        # Store the user-provided value (may be None for auto-calibration).
        # The effective bonus used at scoring time is exposed via the
        # ``effective_symbolic_bonus`` property below, which lazy-calibrates
        # if the user did not pin a value.
        self._user_symbolic_bonus = symbolic_bonus
        self._calibrated_bonus = None  # filled by _compute_calibrated_bonus on first use
        self.explainer = ChainOfThoughtExplainer()  # Initialize explainer

        print(f"\nInitialized Anomaly Detector")
        print(f"  - Analyzing {len(self.triples)} triples")
        print(f"  - Solution has {len(solution.t)} time points")
        if symbolic_bonus is None:
            print(f"  - Symbolic bonus: AUTO-CALIBRATE (q99 - q50 of pre-bonus score)")
        else:
            print(f"  - Symbolic bonus: {float(symbolic_bonus):.4f} (user-pinned)")

    # ------------------------------------------------------------------ #
    # Calibrated symbolic bonus
    # ------------------------------------------------------------------ #
    @property
    def symbolic_bonus(self):
        """Effective symbolic bonus used by ``compute_triple_anomaly_score``.

        Returns the user-pinned value when one was given; otherwise returns
        the dataset-adaptive calibrated value, computing it on first access.
        """
        if self._user_symbolic_bonus is not None:
            return float(self._user_symbolic_bonus)
        if self._calibrated_bonus is None:
            self._calibrated_bonus = self._compute_calibrated_bonus()
        return float(self._calibrated_bonus)

    def _compute_calibrated_bonus(self) -> float:
        """Compute ``q99(score_pre_bonus)`` over every triple in
        ``self.triples`` (i.e. the ODE system's own triples).

        Pre-bonus score per triple is exactly the formula used in
        ``compute_triple_anomaly_score`` minus the constraint contribution:

            score_pre_bonus = energy * 5 + max_instability * 100

        where ``energy = compute_triple_energy_contribution`` and
        ``max_instability = max(emb_instability(s), emb_instability(o))``.
        The bonus equals the 99th-percentile value of this distribution, so
        adding it to a violator's score lifts the triple to the deep-tail
        zone of the dataset's empirical pre-bonus signal.

        Note: walks ``self.triples`` (NOT ``self.full_triples``). In the
        ablation flow ``full_triples`` is the entire perturbed graph
        (~310k on FB15k-237) while the ODE system was set up only on the
        sampled eval subset; ``compute_embedding_instability`` calls
        ``entity_list.index`` which raises ValueError for entities outside
        the ODE system's vocabulary. The calibration only meaningfully
        applies to triples the ODE has actually modelled, so iterating
        ``self.triples`` is the correct scope.

        Single pass, with per-entity instability cache for
        O(|ode_entities| + |ode_triples|) cost.
        """
        # Per-entity instability cache (same shape as the cache used in
        # detect_all_anomalies's instability_cache argument).
        instability_cache: Dict[str, float] = {}
        seen: set = set()
        for triple in self.triples:
            for ent in (triple.subject, triple.object):
                if ent not in seen:
                    instability_cache[ent] = self.compute_embedding_instability(ent)
                    seen.add(ent)

        scores_pre_bonus = []
        for triple in self.triples:
            energy = self.compute_triple_energy_contribution(triple)
            si = instability_cache.get(triple.subject, 0.0)
            oi = instability_cache.get(triple.object, 0.0)
            score = energy * 5.0 + max(si, oi) * 100.0
            scores_pre_bonus.append(score)

        arr = np.asarray(scores_pre_bonus, dtype=float)
        q50 = float(np.quantile(arr, 0.50))
        q99 = float(np.quantile(arr, 0.99))
        bonus = q99
        # Defend against degenerate distributions (e.g. all-zero energies on
        # a tiny graph) — fall back to the historical +500 only when q99
        # itself is non-finite or non-positive.
        if not np.isfinite(bonus) or bonus <= 0.0:
            print(f"    [calibrated-bonus] degenerate distribution "
                  f"(q99={q99:.4f}); falling back to 500.0")
            bonus = 500.0
        print(f"    [calibrated-bonus] pre-bonus score q50={q50:.4f}, "
              f"q99={q99:.4f}, calibrated bonus = q99 = {bonus:.4f}")
        # Cache the percentiles too so visualization scripts and audits can
        # read them without recomputing.
        self._calibrated_q50 = q50
        self._calibrated_q99 = q99
        return bonus
    
    def compute_embedding_instability(self, entity: str) -> float:
        """
        Measure how much an entity's embedding fluctuates
        
        High instability suggests the entity is involved in problematic triples
        
        Returns:
            instability_score: Variance of embedding trajectory
        """
        entity_idx = self.ode_system.entity_list.index(entity)
        dim = self.ode_system.embedding_dim
        
        start = entity_idx * dim
        end = start + dim
        
        # Get trajectory for this entity
        trajectory = self.solution.y[start:end, :]  # shape: (dim, time)
        
        # Compute variance across time for each dimension
        variances = np.var(trajectory, axis=1)
        
        # Total instability
        instability = np.sum(variances)
        
        return instability
    
    def compute_triple_energy_contribution(self, triple: Triple) -> float:
        """
        Measure how much a triple contributes to system energy
        
        High contribution suggests the triple is inconsistent
        
        Returns:
            energy_contribution: Energy without triple - Energy with triple
        """
        # Get final embeddings
        final_embeddings = self.ode_system.get_final_embeddings(self.solution)
        
        # Get embeddings for this triple
        h = final_embeddings.get(triple.subject)
        r = final_embeddings.get(triple.relation, np.zeros(self.ode_system.embedding_dim))
        t = final_embeddings.get(triple.object)
        
        if h is None or t is None:
            return 0.0
        
        # TransE-style energy: ||h + r - t||²
        residual = h + r - t
        energy = np.linalg.norm(residual) ** 2
        
        if np.isnan(energy) or np.isinf(energy):
            return 1000.0
            
        return energy
    
    def compute_constraint_violation_score(self, triple: Triple) -> Dict[str, int]:
        """
        Check which constraints this triple violates
        
        Returns:
            Dict mapping constraint name to violation count
        """
        violations = {}
        
        if not self.constraint_checker:
            return violations
            
        if not hasattr(self, '_cached_violations'):
            self._cached_violations = {}
            for constraint in self.constraint_checker.constraints:
                violating = constraint.get_violating_triples(self.full_triples)
                self._cached_violations[constraint.name] = set(violating)
                
        for constraint in self.constraint_checker.constraints:
            if triple in self._cached_violations[constraint.name]:
                violations[constraint.name] = 1
        
        return violations
    
    def compute_triple_anomaly_score(self, triple: Triple, instability_cache: Dict[str, float] = None) -> Tuple[float, List[str]]:
        """
        Compute overall anomaly score for a triple
        
        Args:
            instability_cache: Optional pre-computed instability scores
        
        Returns:
            (anomaly_score, anomaly_types)
        """
        anomaly_types = []
        score = 0.0
        
        # 1. Energy contribution (add raw value, no hard threshold gating)
        energy = self.compute_triple_energy_contribution(triple)
        if energy > 0.05:  # Lower threshold for reporting type
            anomaly_types.append("high_energy")
        score += energy * 5.0  # Scale up energy to be comparable
        
        # 2. Embedding instability (use cache if available)
        if instability_cache is not None:
            subject_instability = instability_cache.get(triple.subject, 0.0)
            object_instability = instability_cache.get(triple.object, 0.0)
        else:
            subject_instability = self.compute_embedding_instability(triple.subject)
            object_instability = self.compute_embedding_instability(triple.object)
        
        max_instability = max(subject_instability, object_instability)
        if max_instability > 0.005:  # Lower threshold (0.005)
            anomaly_types.append("unstable_embeddings")
        score += max_instability * 100.0  # Scale up instability (it's usually small variance)
        
        # 3. Constraint violations
        violations = self.compute_constraint_violation_score(triple)
        if violations:
            score += len(violations) * self.symbolic_bonus
            anomaly_types.extend(f"violates_{v}" for v in violations.keys())
        
        return score, anomaly_types
    
    def detect_all_anomalies(self, 
                            top_k: int = 20,
                            min_score: float = 0.5) -> List[AnomalyReport]:
        """
        Detect top-k most anomalous triples
        
        Args:
            top_k: Number of anomalies to return
            min_score: Minimum anomaly score threshold
            
        Returns:
            List of AnomalyReports sorted by anomaly score
        """
        print("\n" + "="*60)
        print("DETECTING ANOMALIES")
        print("="*60)
        
        # OPTIMIZATION: Pre-compute instability for all entities ONCE
        print("Pre-computing entity instabilities...")
        instability_cache = {}
        all_entities = set()
        for triple in self.triples:
            all_entities.add(triple.subject)
            all_entities.add(triple.object)
        
        for entity in tqdm(all_entities, desc="Computing instabilities", unit="entity"):
            instability_cache[entity] = self.compute_embedding_instability(entity)
        
        reports = []
        
        print(f"\nAnalyzing {len(self.triples)} triples...")
        for triple in tqdm(self.triples, desc="Detecting anomalies", unit="triple"):
            score, anomaly_types = self.compute_triple_anomaly_score(triple, instability_cache)
            
            if score >= min_score:
                # Generate basic explanation (no LLM - fast)
                explanation = self._generate_explanation(triple, score, anomaly_types)
                
                # Determine suggested action
                action = self._determine_action(anomaly_types, score)
                
                # Confidence based on score magnitude
                confidence = min(score / 10.0, 1.0)
                
                # Calculate plausibility without LLM
                energy = self.compute_triple_energy_contribution(triple)
                has_violation = "violates_" in str(anomaly_types)
                
                if has_violation and energy > 1.0:
                    plausibility_score = 0.9
                    risk_assessment = "confirmed_error"
                elif has_violation and energy <= 1.0:
                    plausibility_score = 0.3
                    risk_assessment = "false_positive_risk"
                elif not has_violation and energy > 1.0:
                    plausibility_score = 0.6
                    risk_assessment = "false_negative_risk"
                else:
                    plausibility_score = 0.5
                    risk_assessment = "uncertain"
                
                report = AnomalyReport(
                    triple=triple,
                    anomaly_score=score,
                    anomaly_types=anomaly_types,
                    explanation=explanation,
                    confidence=confidence,
                    suggested_action=action,
                    cot_analysis=None,  # Deferred to post-processing
                    plausibility_score=plausibility_score,
                    risk_assessment=risk_assessment
                )
                reports.append(report)
        
        # Sort by anomaly score
        reports.sort(key=lambda r: r.anomaly_score, reverse=True)
        
        # Return top-k
        top_reports = reports[:top_k]
        
        print(f"\nDetected {len(reports)} anomalies (score >= {min_score})")
        print(f"Returning top {len(top_reports)} anomalies")
        
        return top_reports
    
    def _generate_explanation(self, triple: Triple, score: float, 
                            anomaly_types: List[str]) -> str:
        """Generate human-readable explanation for anomaly"""
        parts = [f"Triple ({triple.subject}, {triple.relation}, {triple.object})"]
        parts.append(f"is anomalous (score: {score:.2f}) because:")
        
        for atype in anomaly_types:
            if atype == "high_energy":
                parts.append("  - It has high energy (poor fit with embeddings)")
            elif atype == "unstable_embeddings":
                parts.append("  - Its entities have unstable embedding trajectories")
            elif atype.startswith("violates_"):
                constraint_name = atype.replace("violates_", "")
                parts.append(f"  - It violates constraint: {constraint_name}")
        
        return "\n".join(parts)
    
    def _determine_action(self, anomaly_types: List[str], score: float) -> str:
        """Determine recommended action based on anomaly type"""
        if any("violates_" in t for t in anomaly_types):
            return "explain"  # Changed from "correct" to "explain" - prioritize understanding
        elif score > 5.0:
            return "verify"  # High score, needs human verification
        else:
            return "explain"  # Moderate score, default to explanation
    
    def print_summary(self, reports: List[AnomalyReport]):
        """Print human-readable summary of anomaly detection"""
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        
        if not reports:
            print("No anomalies detected!")
            return
        
        # Group by action
        by_action = defaultdict(list)
        for report in reports:
            by_action[report.suggested_action].append(report)
        
        print(f"\nTotal anomalies detected: {len(reports)}")
        print(f"  - To REMOVE: {len(by_action['remove'])}")
        print(f"  - To VERIFY: {len(by_action['verify'])}")
        print(f"  - To EXPLAIN: {len(by_action['explain'])}")
        print(f"  - To CORRECT: {len(by_action['correct'])}")
        
        print("\n" + "-"*60)
        print("TOP 10 ANOMALIES:")
        print("-"*60)
        
        for i, report in enumerate(reports[:10], 1):
            print(f"\n{i}. Score: {report.anomaly_score:.2f} | "
                  f"Confidence: {report.confidence:.2f} | "
                  f"Action: {report.suggested_action.upper()}")
            print(f"   Triple: ({report.triple.subject}, {report.triple.relation}, "
                  f"{report.triple.object})")
            print(f"   Types: {', '.join(report.anomaly_types)}")
            if report.cot_analysis:
                print(f"   [CoT]: {report.cot_analysis.get('explanation', '')[:100]}...")
        
        print("\n" + "="*60)
    
    def export_anomalies_to_csv(self, reports: List[AnomalyReport], filepath: str):
        """Export anomaly reports to CSV file"""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Subject', 'Relation', 'Object', 'Score', 
                           'Confidence', 'Action', 'Anomaly Types'])
            
            for report in reports:
                writer.writerow([
                    report.triple.subject,
                    report.triple.relation,
                    report.triple.object,
                    f"{report.anomaly_score:.4f}",
                    f"{report.confidence:.4f}",
                    report.suggested_action,
                    '; '.join(report.anomaly_types)
                ])
        
        print(f"Exported {len(reports)} anomalies to {filepath}")
    
    def visualize_anomaly_distribution(self, reports: List[AnomalyReport], 
                                      save_path: str = None):
        """Visualize distribution of anomaly scores and types"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Score distribution
            scores = [r.anomaly_score for r in reports]
            axes[0].hist(scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0].set_xlabel('Anomaly Score', fontsize=12)
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_title('Distribution of Anomaly Scores', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            # Anomaly type counts
            type_counts = defaultdict(int)
            for report in reports:
                for atype in report.anomaly_types:
                    type_counts[atype] += 1
            
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            axes[1].barh(types, counts, color='coral', edgecolor='black')
            axes[1].set_xlabel('Count', fontsize=12)
            axes[1].set_title('Anomaly Types', fontsize=14)
            axes[1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved anomaly visualization to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("matplotlib not available for plotting")


class AnomalyCorrector:
    """
    Suggests corrections for detected anomalies
    """
    
    def __init__(self, ode_system, constraint_checker: ConstraintChecker):
        self.ode_system = ode_system
        self.constraint_checker = constraint_checker
        self.explainer = ChainOfThoughtExplainer()
    
    def suggest_corrections(self, report: AnomalyReport) -> List[str]:
        """
        Suggest how to correct an anomaly
        
        Returns:
            List of suggested correction strategies
        """
        suggestions = []
        
        if report.suggested_action == "remove":
            suggestions.append(f"REMOVE triple: ({report.triple.subject}, "
                             f"{report.triple.relation}, {report.triple.object})")
        
        elif report.suggested_action == "verify":
            suggestions.append(f"VERIFY with domain expert: Is ({report.triple.subject}, "
                             f"{report.triple.relation}, {report.triple.object}) correct?")
        
        elif report.suggested_action == "correct":
            # 1. Embedding-based suggestions
            suggestions.append("Possible corrections (Embedding-based):")
            embedding_suggestions = self._suggest_alternative_objects(report.triple)
            suggestions.extend(embedding_suggestions)
            
            # 2. CoT-based suggestions (LLM)
            # Extract candidates from embedding suggestions for context
            candidates = [s.split("'")[1] for s in embedding_suggestions if "'" in s]
            
            cot_suggestions = self.explainer.suggest_corrections(
                report.triple, 
                violation_type=", ".join(report.anomaly_types),
                context_candidates=candidates
            )
            
            if cot_suggestions and 'suggestions' in cot_suggestions:
                suggestions.append("\nPossible corrections (CoT Logic-based):")
                for sug in cot_suggestions['suggestions']:
                    if isinstance(sug, dict):
                        suggestions.append(f"  - {sug.get('action')}: {sug.get('value')} ({sug.get('reason')})")
                    else:
                        suggestions.append(f"  - {sug}")
        
        return suggestions
    
    def _suggest_alternative_objects(self, triple: Triple, top_k: int = 3) -> List[str]:
        """Suggest alternative objects that might fit better"""
        final_embeddings = self.ode_system.get_final_embeddings(self.ode_system.solution)
        
        if triple.subject not in final_embeddings:
            return ["No suggestions available"]
        
        h = final_embeddings[triple.subject]
        r = final_embeddings.get(triple.relation, np.zeros_like(h))
        
        # Expected tail: h + r
        expected = h + r
        
        # Find closest entities
        distances = []
        for entity, emb in final_embeddings.items():
            if entity == triple.subject:  # Skip subject itself
                continue
            dist = np.linalg.norm(expected - emb)
            distances.append((entity, dist))
        
        distances.sort(key=lambda x: x[1])
        
        suggestions = []
        for entity, dist in distances[:top_k]:
            suggestions.append(f"  - Replace object with '{entity}' (distance: {dist:.4f})")
        
        return suggestions
