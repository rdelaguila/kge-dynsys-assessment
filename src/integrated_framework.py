"""
Integrated KG Debugging Framework
==================================
Combines ODE and PyKEEN approaches with full comparison capabilities
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# Import detection modules
# Import detection modules
from pykeen_detector import PyKEENDetector, PyKEENAnomalyReport
from utils.stats import StatisticalEvaluator
from data import DataLoader, Triple

# For ODE detection, you need to copy your existing files:
# - ode_system.py
# - constraints.py  
# - anomaly_detector.py


@dataclass
class IntegratedReport:
    """Unified report combining all detection methods"""
    triple: Triple
    ode_score: float = 0.0
    pykeen_score: float = 0.0
    constraint_score: float = 0.0
    consensus_score: float = 0.0
    detection_methods: List[str] = None
    explanations: Dict[str, str] = None
    action: str = "verify"
    
    def __post_init__(self):
        if self.detection_methods is None:
            self.detection_methods = []
        if self.explanations is None:
            self.explanations = {}
    
    def compute_consensus(self):
        """Compute consensus score across all methods"""
        scores = []
        if self.ode_score > 0:
            scores.append(self.ode_score)
        if self.pykeen_score > 0:
            scores.append(self.pykeen_score)
        if self.constraint_score > 0:
            scores.append(self.constraint_score)
        
        if scores:
            self.consensus_score = np.mean(scores)
            
        # Determine action
        if self.consensus_score > 0.8:
            self.action = "remove"
        elif self.consensus_score > 0.5:
            self.action = "verify"
        else:
            self.action = "correct"


class IntegratedKGDebugger:
    """
    Main integrated debugging system that runs both ODE and PyKEEN methods
    for comprehensive comparison
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize integrated debugger
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        self.ode_detector = None
        self.pykeen_detector = None
        self.constraint_checker = None
        self.statistical_evaluator = StatisticalEvaluator()
        
        # Results storage
        self.ode_results = []
        self.pykeen_results = []
        self.integrated_results = []
        
        print("="*80)
        print("INTEGRATED KG DEBUGGING FRAMEWORK INITIALIZED")
        print("="*80)
        print(f"ODE Detection: {config['use_ode']}")
        print(f"PyKEEN Detection: {config['use_pykeen']}")
        print(f"Constraint Checking: {config['use_constraints']}")
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration"""
        return {
            'use_ode': True,
            'use_pykeen': True,
            'use_constraints': True,
            'ode_config': {
                'embedding_dim': 50,
                't_span': (0.0, 5.0), # Reduced from 10.0 for faster testing
                'lambda_data': 1.0,
                'lambda_logic': 1.0,
                'lambda_reg': 0.1
            },
            'pykeen_config': {
                'model': 'TransE',
                'embedding_dim': 50,
                'epochs': 100,
                'batch_size': 32
            },
            'constraint_config': {
                'transitivity_relations': ['locatedIn', 'partOf'],
                'symmetry_relations': ['marriedTo', 'siblingOf'],
                'cardinality_constraints': [
                    {'relation': 'hasCapital', 'max': 1}
                ]
            }
        }
    
    def setup_ode_detector(self, triples: List[Triple]):
        """Setup ODE-based detector"""
        if not self.config['use_ode']:
            return
        
        print("\nSetting up ODE detector...")
        
        try:
            # Import ODE modules
            from ode_system import KGODESystem, ODEConfig
            from constraints import (ConstraintChecker, TransitivityConstraint,
                                     SymmetryConstraint, AntisymmetryConstraint)
            from anomaly_detector import AnomalyDetector
            
            # Setup constraints
            self.constraint_checker = ConstraintChecker()
            
            # Add transitivity constraints
            for rel in self.config['constraint_config']['transitivity_relations']:
                self.constraint_checker.add_constraint(TransitivityConstraint(rel))
            
            # Add symmetry constraints
            for rel in self.config['constraint_config']['symmetry_relations']:
                self.constraint_checker.add_constraint(SymmetryConstraint(rel))
            
            # Add antisymmetry (asymmetry) constraints — CRITICAL for detecting
            # inverse-triple violations (e.g. (B,_hypernym,A) where (A,_hypernym,B) ∈ G).
            # AntisymmetryConstraint now contributes to a continuous
            # gradient force scaled by lambda_asym (separate from lambda_logic
            # which governs transitivity + symmetry). The `margin_asym` and
            # the runtime `disable_asymmetry` toggle (used by the
            # logical_lambda_asym_zero sub-experiment in TASK 5) are propagated
            # through here.
            _margin_asym = float(self.config.get('ode_config', {}).get('margin_asym', 1.0))
            _disable_asym = bool(
                self.config['constraint_config'].get('disable_asymmetry', False)
            )
            for rel in self.config['constraint_config'].get('asymmetry_relations', []):
                self.constraint_checker.add_constraint(
                    AntisymmetryConstraint(rel, margin=_margin_asym,
                                           enabled=not _disable_asym))
            
            # Initialize embeddings for ENTITIES and RELATIONS.
            #
            # [perf-milestone-c, 2026-05-16] Prior to this fix the embedding dict
            # only contained entity rows. Constraint gradient code in
            # ``src/data.py`` reads the relation vector via
            # ``embeddings.get(self.relation, np.zeros_like(emb_h_np))``; with
            # no relation entries the fallback produced ``r = 0`` for every
            # constraint, which silently zeroes the antisymmetry hinge
            # gradient (since ``forward_sq - reverse_sq ≡ 0`` when ``r = 0``).
            # This made ``lambda_asym`` mechanically inert in every detection
            # result that went through ``IntegratedKGDebugger``, including the
            # §4.7 Wilcoxon panel of the vasimetry-submission.
            #
            # Initialising relations with the same ``N(0, 0.1)`` distribution
            # as entities is the minimal fix: the deterministic detection
            # pipeline now exercises ``lambda_asym`` honestly, and the §4.7
            # H_asym row can be re-evaluated with a non-trivial gradient
            # active.
            entities = set()
            relations = set()
            for t in triples:
                entities.add(t.subject)
                entities.add(t.object)
                relations.add(t.relation)

            embedding_dim = self.config['ode_config']['embedding_dim']
            embeddings = {
                e: np.random.randn(embedding_dim) * 0.1
                for e in entities
            }
            for r in relations:
                embeddings[r] = np.random.randn(embedding_dim) * 0.1
            
            # Create ODE config
            # lambda_asym + margin_asym carry the new asymmetry-as-
            # gradient knobs through to ODEConfig. lambda_asym defaults to
            # 1.0 to keep symmetry with the existing lambda_logic / lambda_data
            # defaults; setting it to 0.0 disables the asymmetry force at
            # inference (used by the logical_lambda_asym_zero sub-experiment).
            ode_config = ODEConfig(
                t_span=tuple(self.config['ode_config']['t_span']),
                lambda_data=self.config['ode_config'].get('lambda_data', 1.0),
                lambda_logic=self.config['ode_config'].get('lambda_logic', 1.0),
                lambda_reg=self.config['ode_config'].get('lambda_reg', 0.1),
                lambda_asym=self.config['ode_config'].get('lambda_asym', 1.0),
                margin_asym=self.config['ode_config'].get('margin_asym', 1.0),
                rtol=self.config['ode_config'].get('rtol', 1e-3),
                atol=self.config['ode_config'].get('atol', 1e-6),
                method=self.config['ode_config'].get('method', 'RK4'),
                device=self.config['ode_config'].get('device', 'mps' if torch.backends.mps.is_available() else 'cpu')
            )
            
            # Initialize ODE system
            self.ode_system = KGODESystem(
                embeddings=embeddings,
                triples=triples,
                constraint_checker=self.constraint_checker,
                config=ode_config
            )
            
            print("ODE detector ready")
            
        except ImportError:
            print("ODE modules not available, using mock detector")
            self.ode_system = None
    
    def setup_pykeen_detector(self, triples: List[Triple]):
        """Setup PyKEEN-based detector"""
        if not self.config['use_pykeen']:
            return
        
        print("\nSetting up PyKEEN detector...")
        
        # Setup constraints if not already done (for PyKEEN to use)
        if self.constraint_checker is None:
            self.setup_constraints(triples)
        
        self.pykeen_detector = PyKEENDetector(
            embedding_dim=self.config['pykeen_config']['embedding_dim']
        )
        
        # Train model
        metrics = self.pykeen_detector.train_model(
            triples,
            model_name=self.config['pykeen_config']['model'],
            epochs=self.config['pykeen_config']['epochs'],
            batch_size=self.config['pykeen_config']['batch_size'],
            learning_rate=self.config['pykeen_config'].get('learning_rate'),
            margin=self.config['pykeen_config'].get('margin'),
            negative_sampling=self.config['pykeen_config'].get('negative_sampling')
        )
        
        print(f"PyKEEN model trained: MRR = {metrics.get('mrr', 0.0):.4f}")
    
    def setup_constraints(self, triples: List[Triple]):
        """Setup constraint checker (used by both ODE and PyKEEN)"""
        if self.constraint_checker is not None:
            return  # Already setup
        
        try:
            from constraints import (ConstraintChecker, TransitivityConstraint,
                                     SymmetryConstraint, CardinalityConstraint,
                                     AntisymmetryConstraint)
            
            self.constraint_checker = ConstraintChecker()
            
            # Add transitivity constraints
            for rel in self.config['constraint_config']['transitivity_relations']:
                self.constraint_checker.add_constraint(TransitivityConstraint(rel))
            
            # Add symmetry constraints  
            for rel in self.config['constraint_config']['symmetry_relations']:
                self.constraint_checker.add_constraint(SymmetryConstraint(rel))
            
            # Add antisymmetry constraints (e.g. _hypernym in WN18RR)
            for rel in self.config['constraint_config'].get('asymmetry_relations', []):
                self.constraint_checker.add_constraint(AntisymmetryConstraint(rel))
            
            # Add cardinality constraints
            for constraint_def in self.config['constraint_config']['cardinality_constraints']:
                self.constraint_checker.add_constraint(
                    CardinalityConstraint(constraint_def['relation'], constraint_def['max'])
                )
            
            print(f"Setup {len(self.constraint_checker.constraints)} constraints")
            
        except ImportError:
            print("Constraint modules not available, using mock constraints")
            self.constraint_checker = None
    
    def run_ode_detection(self, triples: List[Triple], full_triples: List[Triple] = None) -> List[Dict]:
        """Run ODE-based anomaly detection"""
        if not self.config['use_ode'] or self.ode_system is None:
            return self._mock_ode_detection(triples)
        
        print("\n" + "="*60)
        print("RUNNING ODE-BASED DETECTION")
        print("="*60)
        
        try:
            from anomaly_detector import AnomalyDetector
            
            # Solve ODE
            solution = self.ode_system.solve()
            self.ode_system.solution = solution
            
            # Detect anomalies. symbolic_bonus=None means "auto-calibrate
            # using q99 - q50 of the per-triple pre-bonus score". A pinned
            # numeric value (e.g. 0.0 or the historical 500.0) overrides the
            # calibration. See AnomalyDetector.__init__ for the contract.
            detector = AnomalyDetector(
                self.ode_system,
                solution,
                self.constraint_checker,
                full_triples=full_triples,
                symbolic_bonus=self.config.get('symbolic_bonus', None),
            )

            n_triples = len(triples)

            # Use detector to evaluate ALL triples without truncating top_k
            reports = detector.detect_all_anomalies(top_k=None)
            
            # [phase3-step2-fix1] Recover the pre-bonus dynamical energy of each report
            # by subtracting the symbolic-constraint penalty added at
            # src/anomaly_detector.py:175 (`score += len(violations) * symbolic_bonus`).
            # The number of violations per report is recoverable from anomaly_types
            # (each violation contributes one `violates_<name>` tag at line 176).
            #
            # Equivalence with the previous `score % 500` hack:
            #   - For default symbolic_bonus=500 AND base score (energy*5 + instability*100)
            #     strictly less than 500, modulo and subtraction give the same result.
            #   - For base score >= 500, modulo wraps incorrectly (it strips an extra 500
            #     from a non-bonus contribution); subtraction is exact. This is a bug
            #     fix, not a behaviour change — see PROVENANCE.md and
            #     progress/diagnostic_phase3/step2_fixes.md.
            #   - For symbolic_bonus != 500, modulo subtracted the wrong amount; the new
            #     formula is the only correct one. The CLI flag --symbolic-bonus was
            #     added after the canonical run; the canonical run
            #     therefore always used bonus=500, so published numbers are unaffected.
            # Pull the *effective* bonus from the detector (handles both
            # the user-pinned and the auto-calibrated cases). The
            # detector lazily computes the calibrated value on first
            # access via `compute_triple_anomaly_score`, which by this
            # point in the flow has already run, so the value is cached.
            symbolic_bonus = float(detector.symbolic_bonus)
            raw_energies = []
            for r in reports:
                n_viol = sum(1 for t in r.anomaly_types if t.startswith('violates_'))
                e = r.anomaly_score - n_viol * symbolic_bonus
                raw_energies.append(e)
                
            import numpy as np
            from scipy.stats import skew, kurtosis
            try:
                import cv2  # noqa: F401  (used in BIMODAL branch below)
            except ImportError as e:
                # [phase3-cv2-fix] Surface this loud rather than letting the
                # outer except-block silently fall through to mock detection.
                # Previously, missing opencv-python caused run_ode_detection
                # to silently return mock anomalies (every 10th triple),
                # contaminating any logical-ablation evaluation. The package
                # is required by the BIMODAL Otsu branch; install via
                # `pip install opencv-python` (also in requirements.txt).
                raise ImportError(
                    "OpenCV (cv2) is required by IntegratedKGDebugger.run_ode_detection "
                    "for the bimodality/Otsu threshold path. Install with: "
                    "pip install opencv-python (also listed in requirements.txt). "
                    "Refusing to silently fall back to mock detection."
                ) from e

            x = np.array(raw_energies)
            n_val = len(x)
            if n_val > 3:
                # Remove precision loss warning
                with np.errstate(all='ignore'):
                    s = skew(x)
                    k = kurtosis(x, fisher=False)
                    bc = (s**2 + 1) / (k + 3 * ((n_val-1)**2) / ((n_val-2)*(n_val-3)))
            else:
                bc = 0.0
                
            data_normalized = ((x - x.min()) / (x.max() - x.min() + 1e-9) * 255).astype(np.uint8)
            
            if bc > 0.555:
                print(f"    ODE distribution is BIMODAL (BC={bc:.4f}). Applying Otsu threshold.")
                otsu_val, _ = cv2.threshold(data_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                real_threshold = x.min() + (otsu_val / 255.0) * (x.max() - x.min())
            else:
                # Knee detection on the sorted energy curve, matching the
                # configuration that produced the manuscript's published
                # numbers. No parameters modified: curve='convex',
                # direction='decreasing', interp_method='polynomial'.
                from kneed import KneeLocator
                n = len(x)
                sorted_e = np.sort(x)[::-1]  # descending
                xs = np.arange(n)
                # KneeLocator: concave=False (convex curve), direction='decreasing'
                knl = KneeLocator(xs, sorted_e, curve='convex', direction='decreasing', interp_method='polynomial')
                knee_idx = knl.knee
                if knee_idx is not None:
                    real_threshold = float(sorted_e[knee_idx])
                    knee_pct = knee_idx / n * 100
                    print(f"    ODE distribution is UNIMODAL (BC={bc:.4f}). Applying knee detection on sorted energy curve.")
                    print(f"    Knee idx={knee_idx} ({knee_pct:.1f}% of graph) energy={real_threshold:.4f}")
                else:
                    real_threshold = float(np.percentile(x, 95))
                    print(f"    ODE distribution is UNIMODAL (BC={bc:.4f}). Knee NOT DETECTED — falling back to p95 = {real_threshold:.4f}.")
                
            print(f"    Dynamic threshold: {real_threshold:.4f}")
            
            # Now filter the reports to those >= threshold
            final_reports = [r for r, e in zip(reports, raw_energies) if e >= real_threshold]
            reports = sorted(final_reports, key=lambda r: r.anomaly_score, reverse=True)
            print(f"    ODE flagged {len(reports)} anomalies ({len(reports)/n_triples*100:.1f}%)")
            
            # Convert to standard format
            results = []
            for report in reports:
                results.append({
                    'triple': report.triple,
                    'score': min(report.anomaly_score / 10.0, 1.0),  # Normalize
                    'types': report.anomaly_types,
                    'explanation': report.explanation,
                    'action': report.suggested_action
                })
            
            self.ode_results = results
            print(f"ODE detected {len(results)} anomalies")
            
            return results

        except Exception as e:
            # [phase3-cv2-fix] Re-raise instead of silently falling back to mock.
            # The previous broad mock-fallback masked real failures (missing cv2,
            # ODE solver crashes, malformed triples) and produced exactly
            # n_triples/10 fake anomalies — contaminating downstream metrics
            # without warning. If the caller actually wants mock data for
            # testing, they should import _mock_ode_detection explicitly.
            print(f"ODE detection failed (re-raising; no silent mock fallback): {e}")
            raise
    
    def _mock_ode_detection(self, triples: List[Triple]) -> List[Dict]:
        """Mock ODE detection for testing"""
        results = []
        for i, triple in enumerate(triples):
            # Simulate detection of specific patterns
            if "Asia" in str(triple.object) or "violat" in str(triple.relation) or i % 10 == 0:
                results.append({
                    'triple': triple,
                    'score': np.random.uniform(0.6, 1.0),
                    'types': ['mock_anomaly'],
                    'explanation': 'Mock ODE detection',
                    'action': 'verify'
                })
        return results
    
    def run_pykeen_detection(self, triples: List[Triple]) -> List[Dict]:
        """Run PyKEEN-based anomaly detection INCLUDING constraints"""
        if not self.config['use_pykeen'] or self.pykeen_detector is None:
            return []
        
        print("\n" + "="*60)
        print("RUNNING PYKEEN-BASED DETECTION")
        print("="*60)
        
        # Pass constraint_checker to PyKEEN (NEW!)
        checker = self.constraint_checker if self.config.get('use_constraints', True) else None
        reports = self.pykeen_detector.detect_all_anomalies(
            triples, 
            constraint_checker=checker
        )
        
        # Convert to standard format
        results = []
        for report in reports:
            results.append({
                'triple': report.triple,
                'score': report.anomaly_score,
                'types': report.anomaly_types,
                'explanation': report.explanation,
                'action': report.suggested_action
            })
        
        self.pykeen_results = results
        model_name = self.config['pykeen_config']['model']
        print(f"PyKEEN-{model_name} detected {len(results)} anomalies")
        
        return results, model_name

    
    def integrate_results(self, ode_results: List[Dict], 
                         pykeen_results: List[Dict],
                         pykeen_model_name: str = 'PyKEEN') -> List[IntegratedReport]:
        """Integrate results from all detection methods"""
        print("\n" + "="*60)
        print("INTEGRATING DETECTION RESULTS")
        print("="*60)
        
        # Combine all results
        integrated = {}
        
        # Process ODE results
        for result in ode_results:
            key = (result['triple'].subject, result['triple'].relation, result['triple'].object)
            if key not in integrated:
                integrated[key] = IntegratedReport(triple=result['triple'])
            
            integrated[key].ode_score = result['score']
            integrated[key].explanations['ode'] = result.get('explanation', '')
            if 'ODE' not in integrated[key].detection_methods:
                integrated[key].detection_methods.append('ODE')
        
        # Process PyKEEN results
        for result in pykeen_results:
            key = (result['triple'].subject, result['triple'].relation, result['triple'].object)
            if key not in integrated:
                integrated[key] = IntegratedReport(triple=result['triple'])
            
            integrated[key].pykeen_score = result['score']
            integrated[key].explanations['pykeen'] = result.get('explanation', '')
            method_label = f"PyKEEN-{pykeen_model_name}"
            if method_label not in integrated[key].detection_methods:
                integrated[key].detection_methods.append(method_label)
        
        # Compute consensus for all
        for report in integrated.values():
            report.compute_consensus()
        
        # Sort by consensus score
        final_reports = sorted(integrated.values(), 
                              key=lambda x: x.consensus_score,
                              reverse=True)
        
        self.integrated_results = final_reports
        
        print(f"Total unique anomalies: {len(final_reports)}")
        print(f"Detected by both methods: {sum(1 for r in final_reports if len(r.detection_methods) > 1)}")
        print(f"High consensus (>0.6): {sum(1 for r in final_reports if r.consensus_score > 0.6)}")
        
        return final_reports
    
    def compare_methods(self) -> Dict:
        """Statistically compare detection methods"""
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        
        # Prepare results for comparison
        method_results = {
            'ODE': [{'triple': r['triple'], 'score': r['score']} for r in self.ode_results],
            'PyKEEN': [{'triple': r['triple'], 'score': r['score']} for r in self.pykeen_results]
        }
        
        comparison = self.statistical_evaluator.compare_detection_methods(method_results)
        
        print(comparison['summary'])
        
        return comparison
    
    def run_complete_analysis(self, triples: List[Tuple[str, str, str]], full_triples: List[Triple] = None) -> Dict:
        """
        Run complete analysis pipeline
        
        Args:
            triples: List of (subject, relation, object) tuples
            full_triples: Optional complete graph for constraint validation
        Returns:
            Complete analysis results
        """
        # Convert to Triple objects if needed
        if triples and isinstance(triples[0], Triple):
            triple_objects = triples
        else:
            triple_objects = [Triple(s, r, o) for s, r, o in triples]
            
        if full_triples is None:
            full_triples = triple_objects
        elif not isinstance(full_triples[0], Triple):
            full_triples = [Triple(s, r, o) for s, r, o in full_triples]
        
        # Setup detectors
        if self.config['use_ode']:
            self.setup_ode_detector(triple_objects)
        
        if self.config['use_pykeen']:
            self.setup_pykeen_detector(triple_objects)
        
        # Run detections
        ode_results = self.run_ode_detection(triple_objects, full_triples) if self.config['use_ode'] else []
        
        pykeen_results = []
        pykeen_model_name = "None"
        if self.config['use_pykeen']:
            pykeen_results, pykeen_model_name = self.run_pykeen_detection(triple_objects)
        
        # Integrate results
        integrated_results = self.integrate_results(ode_results, pykeen_results, pykeen_model_name)
        
        # Compare methods
        comparison = self.compare_methods() if len(ode_results) > 0 and len(pykeen_results) > 0 else {}
        
        return {
            'ode_results': ode_results,
            'pykeen_results': pykeen_results,
            'pykeen_mrr': self.pykeen_detector.metrics.get('mrr', None) if self.pykeen_detector and hasattr(self.pykeen_detector, 'metrics') else None,
            'integrated_results': integrated_results,
            'comparison': comparison,
            'summary': self.generate_summary(integrated_results, pykeen_model_name)
        }
    
    def generate_summary(self, results: List[IntegratedReport], pykeen_model_name: str = "PyKEEN") -> str:
        """Generate analysis summary"""
        lines = ["="*80, "INTEGRATED ANALYSIS SUMMARY", "="*80]
        
        n_total = len(results)
        
        # 1. Experimental Context (Inferred from data)
        lines.append("\n" + "-"*40)
        lines.append("1. EXPERIMENTAL CONTEXT")
        lines.append("-" * 40)
        lines.append(f"Analysis performed on Knowledge Graph data.")
        lines.append(f"Objective: Compare ODE-based consistency dynamics vs PyKEEN ({pykeen_model_name}) embedding models.")
        
        # 2. Results Breakdown
        lines.append("\n" + "-"*40)
        lines.append("2. DETECTION RESULTS")
        lines.append("-" * 40)
        lines.append(f"Total anomalies detected: {n_total}")
        
        method_counts = {'ODE': 0, 'PyKEEN': 0, 'Both': 0}
        pykeen_label = f"PyKEEN-{pykeen_model_name}"
        
        for r in results:
            has_ode = 'ODE' in r.detection_methods
            has_pykeen = any(m.startswith('PyKEEN') for m in r.detection_methods)
            
            if has_ode and has_pykeen:
                method_counts['Both'] += 1
            elif has_ode:
                method_counts['ODE'] += 1
            elif has_pykeen:
                method_counts['PyKEEN'] += 1
        
        lines.append("\nMethod Performance:")
        lines.append(f"  - ODE only:       {method_counts['ODE']:4d} (Unique structural violations)")
        lines.append(f"  - {pykeen_label} only:    {method_counts['PyKEEN']:4d} (Statistical outliers)")
        lines.append(f"  - Both methods:   {method_counts['Both']:4d} (High confidence anomalies)")
        
        # 3. Conclusions
        lines.append("\n" + "-"*40)
        lines.append("3. CONCLUSIONS")
        lines.append("-" * 40)
        
        if method_counts['Both'] > 0:
            lines.append(f"- Agreement: Both methods agreed on {method_counts['Both']} cases, indicating strong reliability for these anomalies.")
        else:
            lines.append("- Agreement: Methods found disjoint sets of anomalies, suggesting they detect different types of errors.")
            
        if method_counts['ODE'] > method_counts['PyKEEN']:
            lines.append("- Performance: ODE detected more anomalies, indicating the dataset has significant structural/logical inconsistencies.")
        elif method_counts['PyKEEN'] > method_counts['ODE']:
            lines.append(f"- Performance: {pykeen_label} detected more anomalies, typical of random noise or distributional shifts.")
            
        # 4. Top Anomalies
        lines.append("\n" + "-"*40)
        lines.append("4. TOP DETECTED ANOMALIES (Sample)")
        lines.append("-" * 40)
        
        for i, r in enumerate(results[:10], 1):
            lines.append(f"\n[Anomaly #{i}]")
            lines.append(f"  Triple:    ({r.triple.subject}, {r.triple.relation}, {r.triple.object})")
            lines.append(f"  Consensus: {r.consensus_score:.3f} | Suggested Action: {r.action}")
            lines.append(f"  Detected by: {', '.join(r.detection_methods)}")
            
            if r.explanations.get('ode'):
                lines.append(f"  ODE Reason: {r.explanations['ode']}")
            if r.explanations.get('pykeen'):
                lines.append(f"  {pykeen_label} Reason: {r.explanations['pykeen']}")
        
        lines.append("\n" + "="*80)
        
        return "\n".join(lines)
    
    def export_results(self, results: Dict, output_dir: str = "results", dataset_name: str = ""):
        """Export all results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export integrated results to JSON
        integrated_data = []
        for r in results['integrated_results']:
            integrated_data.append({
                'triple': {
                    'subject': r.triple.subject,
                    'relation': r.triple.relation,
                    'object': r.triple.object
                },
                'scores': {
                    'ode': r.ode_score,
                    'pykeen': r.pykeen_score,
                    'consensus': r.consensus_score
                },
                'methods': r.detection_methods,
                'action': r.action
            })
        
        suffix = f"_{dataset_name}" if dataset_name else ""
        
        with open(output_path / f'integrated_results{suffix}.json', 'w') as f:
            json.dump(integrated_data, f, indent=2)
        
        # Export comparison if available
        if 'comparison' in results and results['comparison']:
            with open(output_path / f'method_comparison{suffix}.json', 'w') as f:
                # Convert numpy types to Python types
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                json.dump(convert_numpy(results['comparison']), f, indent=2)
        
        # Save summary
        with open(output_path / f'summary{suffix}.txt', 'w') as f:
            f.write(results['summary'])
        
        print(f"\nResults exported to {output_path}/")


# Main function
def run_integrated_debugging(dataset_name: str = 'small_sample',
                            inject_anomalies: bool = True,
                            config: Optional[Dict] = None) -> Dict:
    """
    Run complete integrated debugging pipeline
    
    Args:
        dataset_name: Name of dataset to load
        inject_anomalies: Whether to inject synthetic anomalies
        config: Configuration dictionary
        
    Returns:
        Complete results dictionary
    """
    print("\n" + "="*80)
    print("INTEGRATED KG DEBUGGING PIPELINE")
    print("="*80)
    
    # Load dataset
    loader = DataLoader()
    
    if dataset_name == 'small_sample':
        triples = loader.create_sample_dataset('small')
    elif dataset_name == 'medium_sample':
        triples = loader.create_sample_dataset('medium')
    else:
        triples = loader.load_dataset(dataset_name)
    
    print(f"\nLoaded {len(triples)} triples")
    
    # Inject anomalies if requested
    if inject_anomalies:
        triples = loader.inject_anomalies(triples, anomaly_rate=0.1)
    
    # Get statistics
    stats = loader.get_statistics(triples)
    print(f"\nDataset statistics:")
    print(f"  Entities: {stats['n_entities']}")
    print(f"  Relations: {stats['n_relations']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    
    # Initialize debugger
    debugger = IntegratedKGDebugger(config)
    
    # Run complete analysis
    results = debugger.run_complete_analysis(triples)
    
    # Export results
    debugger.export_results(results, dataset_name=dataset_name)
    
    return results

if __name__ == "__main__":
    # Run example
    results = run_integrated_debugging('small_sample', inject_anomalies=True)
