"""
Ablation Study with Perturbations
=================================
Introduces controlled perturbations to KGs and evaluates detection
quality with statistical significance tests.

Experiment Design:
1. Take original clean dataset
2. Introduce X% perturbations (varying X from 1% to 20%)
3. Run all detection methods
4. Measure Precision, Recall, F1 against known perturbations
5. Statistical tests: t-test, Wilcoxon for significance
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from scipy import stats
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data import Triple, DataLoader
from config import ABLATION_EPOCHS, ABLATION_BATCH


@dataclass
class PerturbationResult:
    """Results for a single perturbation level"""
    perturbation_pct: float
    n_perturbed: int
    n_detected: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    precision: float
    recall: float
    f1: float
    detection_scores: List[float]


# ─── Relation property maps for logical violation perturbations ───────────────

WN18RR_RELATION_PROPERTIES = {
    '_hypernym':                      {'asymmetric': True,  'transitive': True,  'functional': False},
    '_instance_hypernym':             {'asymmetric': True,  'transitive': False, 'functional': False},
    '_has_part':                      {'asymmetric': True,  'transitive': True,  'functional': False},
    '_member_meronym':                {'asymmetric': True,  'transitive': False, 'functional': False},
    '_synset_domain_topic_of':        {'asymmetric': True,  'transitive': False, 'functional': False},
    '_also_see':                      {'asymmetric': False, 'transitive': False, 'functional': False},
    '_derivationally_related_form':   {'asymmetric': False, 'transitive': False, 'functional': False},
    '_similar_to':                    {'asymmetric': False, 'transitive': False, 'functional': False},
    '_member_of_domain_usage':        {'asymmetric': True,  'transitive': False, 'functional': False},
    '_member_of_domain_region':       {'asymmetric': True,  'transitive': False, 'functional': False},
    '_verb_group':                    {'asymmetric': False, 'transitive': False, 'functional': False},
}

FB15K237_RELATION_PROPERTIES = {
    '/location/location/contains':              {'asymmetric': True,  'transitive': True,  'functional': False},
    # [2026-05-11] '/location/administrative_division/capital' replaced — the
    # original name does not exist in FB15k-237; the actual data-side relation
    # is '/location/country/capital' (functional, asymmetric).
    '/location/country/capital':                {'asymmetric': True,  'transitive': False, 'functional': True},
    '/people/person/nationality':               {'asymmetric': True,  'transitive': False, 'functional': True},
    '/people/person/gender':                    {'asymmetric': True,  'transitive': False, 'functional': True},
    '/organization/organization/headquarters./location/mailing_address/country':
                                                {'asymmetric': True,  'transitive': False, 'functional': True},
    # [2026-05-11] User-supplied additions, kept in sync with
    # get_constraint_config('FB15k-237') in run_all.py.
    '/music/artist/origin':                     {'asymmetric': True,  'transitive': False, 'functional': False},
    '/organization/organization/child./organization/organization_relationship/child':
                                                {'asymmetric': True,  'transitive': True,  'functional': False},
    '/award/award_winner/awards_won./award/award_honor/award_winner':
                                                {'asymmetric': True,  'transitive': False, 'functional': False},
    '/business/business_operation/industry':    {'asymmetric': True,  'transitive': False, 'functional': False},
}

CODEX_M_RELATION_PROPERTIES = {
    'P131': {'asymmetric': True, 'transitive': True, 'functional': False}, # located in the administrative territorial entity (part of)
    'P361': {'asymmetric': True, 'transitive': True, 'functional': False}, # part of
    'P17': {'asymmetric': True, 'transitive': False, 'functional': False}, # country 
    'P19': {'asymmetric': True, 'transitive': False, 'functional': False}, # place of birth
    'P20': {'asymmetric': True, 'transitive': False, 'functional': False}, # place of death
    'P27': {'asymmetric': True, 'transitive': False, 'functional': False}, # citizenship
    'P50': {'asymmetric': True, 'transitive': False, 'functional': False}, # author
    'P69': {'asymmetric': True, 'transitive': False, 'functional': False}, # educated at
    'P106': {'asymmetric': True, 'transitive': False, 'functional': False}, # occupation 
    'P108': {'asymmetric': True, 'transitive': False, 'functional': False}, # employer
    'P3373': {'asymmetric': False, 'transitive': True, 'functional': False}, # sibling
    'P26': {'asymmetric': False, 'transitive': False, 'functional': False}, # spouse
}


class KGPerturbator:
    """Introduces controlled perturbations to a Knowledge Graph"""
    
    PERTURBATION_TYPES = ['entity_swap', 'relation_swap', 'inverse']
    LOGICAL_VIOLATION_TYPES = ['asymmetry', 'transitivity', 'cardinality']

    def __init__(self, triples: List[Triple], seed: int = 42,
                 relation_properties: Dict = None):
        self.original_triples = triples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Build entity and relation sets
        self.entities = set()
        self.relations = set()
        self.triple_set = set()
        
        for t in triples:
            self.entities.add(t.subject)
            self.entities.add(t.object)
            self.relations.add(t.relation)
            self.triple_set.add((t.subject, t.relation, t.object))
        
        self.entities = list(self.entities)
        self.relations = list(self.relations)

        # Relation properties (asymmetric/transitive/functional)
        self.relation_properties = relation_properties or {}

        # Build auxiliary indices for logical perturbations
        self._build_indices()

    def _build_indices(self):
        """Build head→tails and (h,r,t)→position indices for logical ops."""
        # head → {relation → set(tails)}
        self.h_r_to_tails: Dict[str, Dict[str, Set[str]]] = {}
        # relation → list of (h, t) pairs
        self.rel_to_pairs: Dict[str, List[Tuple[str, str]]] = {}

        for t in self.original_triples:
            # h_r_to_tails
            if t.subject not in self.h_r_to_tails:
                self.h_r_to_tails[t.subject] = {}
            if t.relation not in self.h_r_to_tails[t.subject]:
                self.h_r_to_tails[t.subject][t.relation] = set()
            self.h_r_to_tails[t.subject][t.relation].add(t.object)

            # rel_to_pairs
            if t.relation not in self.rel_to_pairs:
                self.rel_to_pairs[t.relation] = []
            self.rel_to_pairs[t.relation].append((t.subject, t.object))

    # ── Standard random perturbations ─────────────────────────────────────────

    def perturb(self, percentage: float, 
                perturbation_types: List[str] = None) -> Tuple[List[Triple], Set[Tuple]]:
        """
        Perturb a percentage of triples with random corruption.
        
        Args:
            percentage: Percentage of triples to perturb (0-100)
            perturbation_types: Types of perturbations to apply
            
        Returns:
            Tuple of (perturbed_triples, set_of_perturbed_triples_as_tuples)
        """
        if perturbation_types is None:
            perturbation_types = self.PERTURBATION_TYPES
            
        n_perturb = int(len(self.original_triples) * percentage / 100)
        
        # Select triples to perturb
        indices_to_perturb = random.sample(range(len(self.original_triples)), n_perturb)
        perturbed_set = set()
        
        # Create perturbed copy
        new_triples = []
        
        for i, triple in enumerate(self.original_triples):
            if i in indices_to_perturb:
                perturbed_triple = self._perturb_single(triple, perturbation_types)
                new_triples.append(perturbed_triple)
                perturbed_set.add((perturbed_triple.subject, perturbed_triple.relation, perturbed_triple.object))
            else:
                new_triples.append(triple)
        
        return new_triples, perturbed_set
    
    def _perturb_single(self, triple: Triple, perturbation_types: List[str]) -> Triple:
        """Apply a random perturbation to a single triple"""
        ptype = random.choice(perturbation_types)
        
        if ptype == 'entity_swap':
            # Replace subject or object with random entity
            if random.random() < 0.5:
                new_subj = random.choice(self.entities)
                while new_subj == triple.subject:
                    new_subj = random.choice(self.entities)
                return Triple(new_subj, triple.relation, triple.object)
            else:
                new_obj = random.choice(self.entities)
                while new_obj == triple.object:
                    new_obj = random.choice(self.entities)
                return Triple(triple.subject, triple.relation, new_obj)
                
        elif ptype == 'relation_swap':
            # Replace relation with random relation
            new_rel = random.choice(self.relations)
            while new_rel == triple.relation:
                new_rel = random.choice(self.relations)
            return Triple(triple.subject, new_rel, triple.object)
            
        elif ptype == 'inverse':
            # Swap subject and object (creates invalid triple for non-symmetric relations)
            return Triple(triple.object, triple.relation, triple.subject)
        
        return triple

    # ── Logical violation perturbations ───────────────────────────────────────

    def perturb_logical(
        self,
        violation_type: str,
        n_violations: int,
        dataset_name: str = None
    ) -> Tuple[List[Triple], Set[Tuple[str, str, str]], Dict[str, int]]:
        """
        Inject n_violations triples that break a specific logical property.

        Args:
            violation_type: 'asymmetry' | 'transitivity' | 'cardinality' | 'all'
            n_violations: Number of violations to inject
            dataset_name: 'WN18RR' or 'FB15k-237' for relation property lookup

        Returns:
            (perturbed_triples, injected_violation_set, counts_by_type)
        """
        rel_props = self.relation_properties
        if not rel_props and dataset_name:
            if 'WN18RR' in (dataset_name or ''):
                rel_props = WN18RR_RELATION_PROPERTIES
            elif 'FB15k' in (dataset_name or '') or 'fb15k' in (dataset_name or ''):
                rel_props = FB15K237_RELATION_PROPERTIES
            elif 'codex-m' in (dataset_name or ''):
                rel_props = CODEX_M_RELATION_PROPERTIES

        injected: List[Triple] = []
        counts = {'asymmetry': 0, 'transitivity': 0, 'cardinality': 0}

        if violation_type == 'all':
            per_type = n_violations // 3
            remainders = [n_violations - per_type * 3, 0, 0]
            for i, vtype in enumerate(['asymmetry', 'transitivity', 'cardinality']):
                n = per_type + remainders[i]
                new_violations = self._inject_violations(vtype, n, rel_props)
                injected.extend(new_violations)
                counts[vtype] = len(new_violations)
        else:
            new_violations = self._inject_violations(violation_type, n_violations, rel_props)
            injected.extend(new_violations)
            counts[violation_type] = len(new_violations)

        # Build final triple list: original + injected violations
        final_triples = list(self.original_triples) + injected
        violation_set = {(t.subject, t.relation, t.object) for t in injected}

        print(f"  Injected logical violations: {counts}")
        print(f"  Total graph size: {len(self.original_triples)} + {len(injected)} = {len(final_triples)}")

        return final_triples, violation_set, counts

    def _inject_violations(
        self,
        violation_type: str,
        n: int,
        rel_props: Dict
    ) -> List[Triple]:
        """Generate n violations of a specific logical type."""
        if violation_type == 'asymmetry':
            return self._inject_asymmetry_violations(n, rel_props)
        elif violation_type == 'transitivity':
            return self._inject_transitivity_violations(n, rel_props)
        elif violation_type == 'cardinality':
            return self._inject_cardinality_violations(n, rel_props)
        return []

    def _inject_asymmetry_violations(self, n: int, rel_props: Dict) -> List[Triple]:
        """
        Asymmetry violation: for an asymmetric relation r and (A, r, B) ∈ G,
        inject (B, r, A) if it's not already in the graph.
        """
        asymmetric_rels = {
            r for r, props in rel_props.items()
            if props.get('asymmetric', False)
        } or self.relations  # fallback: treat all relations as asymmetric candidates

        candidates = []
        for triple in self.original_triples:
            if triple.relation in asymmetric_rels:
                inverse_key = (triple.object, triple.relation, triple.subject)
                if inverse_key not in self.triple_set:
                    candidates.append(Triple(triple.object, triple.relation, triple.subject))

        random.shuffle(candidates)
        return candidates[:n]

    def _inject_transitivity_violations(self, n: int, rel_props: Dict) -> List[Triple]:
        """
        Transitivity violation: for a transitive relation r,
        if (A, r, B) and (B, r, C) exist but (A, r, C) doesn't,
        inject a WRONG chain closer: (A, r, X) where X ≠ C and X is random.
        """
        transitive_rels = {
            r for r, props in rel_props.items()
            if props.get('transitive', False)
        }
        if not transitive_rels:
            # Fallback: use first relation as transitive candidate
            transitive_rels = {self.relations[0]} if self.relations else set()

        violations = []
        for rel in transitive_rels:
            if rel not in self.rel_to_pairs:
                continue
            pairs = self.rel_to_pairs[rel]
            pair_set = {(h, t) for h, t in pairs}

            for h, mid in pairs:
                if mid not in self.h_r_to_tails:
                    continue
                if rel not in self.h_r_to_tails[mid]:
                    continue
                # (h, rel, mid) and (mid, rel, t2) exist → transitivity expects (h, rel, t2)
                for t2 in self.h_r_to_tails[mid][rel]:
                    if (h, t2) not in pair_set:
                        # (h, rel, t2) should exist by transitivity but doesn't —
                        # Inject an inverted edge: (t2, rel, h) creating a cycle
                        # This guarantees TransitivityConstraint will flag the injected (t2, r, h)
                        # because (mid, r, t2) + (t2, r, h) lacks (mid, r, h)
                        if (t2, rel, h) not in self.triple_set:
                            violations.append(Triple(t2, rel, h))
                            if len(violations) >= n * 3:
                                break
                if len(violations) >= n * 3:
                    break

        random.shuffle(violations)
        return violations[:n]

    def _inject_cardinality_violations(self, n: int, rel_props: Dict) -> List[Triple]:
        """
        Cardinality violation: for a functional (1:1) relation r and (A, r, B),
        inject (A, r, C) where C ≠ B — creating two tails for the same head.
        """
        functional_rels = {
            r for r, props in rel_props.items()
            if props.get('functional', False)
        }
        if not functional_rels:
            # Fallback: infer functional relations as those with low avg tail count
            rel_tail_counts = {}
            for triple in self.original_triples:
                rel_tail_counts.setdefault(triple.relation, set()).add(
                    (triple.subject, triple.object)
                )
            # Relations where each head has on average ≤1.2 tails  
            functional_rels = {
                r for r, pairs in rel_tail_counts.items()
                if self._avg_tails_per_head(r) <= 1.2
            }

        violations = []
        for triple in self.original_triples:
            if triple.relation in functional_rels:
                # Add a second (different) tail for same head
                wrong_tail = random.choice(self.entities)
                attempts = 0
                while (wrong_tail == triple.object or
                       (triple.subject, triple.relation, wrong_tail) in self.triple_set):
                    wrong_tail = random.choice(self.entities)
                    attempts += 1
                    if attempts > 20:
                        break
                if attempts <= 20:
                    violations.append(Triple(triple.subject, triple.relation, wrong_tail))
                if len(violations) >= n * 2:
                    break

        random.shuffle(violations)
        return violations[:n]

    def _avg_tails_per_head(self, relation: str) -> float:
        """Compute average number of distinct tails per head for a relation."""
        if relation not in self.rel_to_pairs:
            return 0.0
        head_to_tails: Dict[str, Set[str]] = {}
        for h, t in self.rel_to_pairs[relation]:
            head_to_tails.setdefault(h, set()).add(t)
        if not head_to_tails:
            return 0.0
        return sum(len(tails) for tails in head_to_tails.values()) / len(head_to_tails)



class AblationStudy:
    """Runs ablation study across perturbation levels using pre-trained models"""
    
    def __init__(self, dataset_name: str, base_results_dir: str = "paper_results",
                 output_dir: str = None):
        self.dataset_name = dataset_name
        self.base_results_dir = Path(base_results_dir)
        
        if output_dir is None:
            self.output_dir = self.base_results_dir / dataset_name / 'ablation'
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perturbation levels for ablation (realistic anomaly rates)
        # 1-5% is typical for real-world KG error rates
        self.perturbation_levels = [1, 2, 5]
        
        # Results storage
        self.results_by_method: Dict[str, List[PerturbationResult]] = {}
        
        # Pre-trained parameters cache
        self.trained_params: Dict[str, Dict] = {}
        
    def run(self, triples: List[Triple], methods: List[str] = None):
        """
        Run complete ablation study using pre-trained model parameters.
        
        Args:
            triples: Original (clean) triples
            methods: Detection methods to evaluate (if None, uses all trained methods)
        """
        # Load pre-trained parameters first
        self._load_trained_params()
        
        # Use trained methods if not specified
        if methods is None:
            methods = list(self.trained_params.keys())
            if not methods:
                print("ERROR: No trained models found. Run run_experiments_v2.py first.")
                return
        
        # Filter to only methods that have been trained
        available_methods = [m for m in methods if m in self.trained_params]
        if not available_methods:
            print(f"ERROR: None of the requested methods {methods} have been trained.")
            print(f"Available: {list(self.trained_params.keys())}")
            return
        
        print("\n" + "="*80)
        print("ABLATION STUDY: PERTURBATION SENSITIVITY ANALYSIS")
        print("="*80)
        print(f"Dataset: {self.dataset_name}")
        print(f"Original triples: {len(triples)}")
        print(f"Perturbation levels: {self.perturbation_levels}%")
        print(f"Methods (pre-trained): {available_methods}")
        print("="*80)
        
        perturbator = KGPerturbator(triples)
        methods = available_methods  # Use only available methods
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method}")
            print(f"{'='*60}")
            
            method_results = []
            
            for pct in tqdm(self.perturbation_levels, desc=f"  {method} ablation"):
                # Perturb dataset
                perturbed_triples, perturbed_set = perturbator.perturb(pct)
                
                # Run detection
                detected_set, scores = self._run_detection(method, perturbed_triples)
                
                # Calculate metrics
                result = self._calculate_metrics(
                    pct, perturbed_set, detected_set, scores, len(triples)
                )
                method_results.append(result)
                
                print(f"    {pct}%: P={result.precision:.3f}, R={result.recall:.3f}, F1={result.f1:.3f}")
            
            self.results_by_method[method] = method_results
        
        # Run statistical tests
        self._run_statistical_tests()
        
        # Save results
        self._save_results()
        
        # Generate report
        self._generate_report()

    def run_logical(
        self,
        triples: List[Triple],
        violation_type: str = 'all',
        n_violations_per_level: List[int] = None,
        methods: List[str] = None,
    ):
        """
        Run ablation study using controlled logical violations instead of
        random entity/relation swaps.

        Unlike run(), perturbations are:
          - asymmetry   : inject inverse of asymmetric relation
          - transitivity: inject wrong transitive chain completion
          - cardinality : inject second tail for functional relation
          - all         : all three types, n_violations split equally

        Results are saved to ablation_logical/<violation_type>/ to keep them
        separate from the random-perturbation ablation.

        ─── Path A vs Path B (logical-violation evaluation) ──────────────────────
        This method is **Path B** of the two non-comparable evaluation paths in this
        codebase. The other is **Path A** in
        ``src/run_all.py::UnifiedExperimentRunner.run_ablation_stage``.

        - **Path B (this method)** — invoked by ``run_full_rerun.py`` and is
          canonical for **manuscript v4 Tables C.1–C.3 (random ablation only)**.
          Its KGE branch (``self._run_detection`` in this same file) explicitly
          sets ``config['use_constraints'] = False`` at line 678, so the
          symbolic constraint-checker is NOT folded into KGE outputs.
          ``run_full_rerun.py`` does NOT call ``run_logical`` (only ``run``);
          the canonical logical-ablation results are produced by Path A.

        - **Path A (``src/run_all.py::run_ablation_stage``)** — produced
          **manuscript v4 Tables 7/8 / D1–D3** (logical ablation, canonical
          commit ``879a39b``). Its KGE branch leaves the default
          ``use_constraints=True``, so the symbolic checker is folded into
          KGE detection sets. This is why the manuscript reports identical
          TransE / DistMult / MuRE F1 within each cell of those tables.

        The two paths produce **non-comparable** KGE columns when applied to
        logical violations. See ``PROVENANCE.md`` (repo root) for the full
        artefact split, ``progress/diagnostic_phase3/step1_provenance.md`` for
        the audit, and ``progress/diagnostic_phase3/step2_fixes.md`` for the
        documentation rationale of this docstring.

        Args:
            triples: Original (clean) triples.
            violation_type: 'asymmetry' | 'transitivity' | 'cardinality' | 'all'
            n_violations_per_level: Number of violations to inject at each level.
                Defaults to [50, 100, 200] (mimics 1%/2%/5% of a ~10k-triple graph).
            methods: Detection methods to evaluate (if None, uses all trained).
        """
        if n_violations_per_level is None:
            # Scale default levels to ~1%/2%/5% of the dataset
            n = len(triples)
            n_violations_per_level = [
                max(10, int(n * 0.01)),
                max(20, int(n * 0.02)),
                max(50, int(n * 0.05)),
            ]

        # Override output_dir for this mode
        logical_output_dir = (
            self.base_results_dir / self.dataset_name
            / 'ablation_logical' / violation_type
        )
        logical_output_dir.mkdir(parents=True, exist_ok=True)
        original_output_dir = self.output_dir
        self.output_dir = logical_output_dir

        # Load pre-trained parameters
        self._load_trained_params()

        if methods is None:
            methods = list(self.trained_params.keys())
            if not methods:
                print("ERROR: No trained models found. Run run_experiments_v2.py first.")
                self.output_dir = original_output_dir
                return

        available_methods = [m for m in methods if m in self.trained_params]
        if not available_methods:
            print(f"ERROR: None of {methods} have been trained.")
            self.output_dir = original_output_dir
            return

        print("\n" + "=" * 80)
        print("ABLATION STUDY (LOGICAL VIOLATIONS)")
        print("=" * 80)
        print(f"Dataset        : {self.dataset_name}")
        print(f"Original triples: {len(triples)}")
        print(f"Violation type : {violation_type}")
        print(f"Violation counts: {n_violations_per_level}")
        print(f"Methods        : {available_methods}")
        print(f"Output dir     : {logical_output_dir}")
        print("=" * 80)

        perturbator = KGPerturbator(triples)
        self.results_by_method = {}
        violation_counts_log = []

        for method in available_methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method}")
            print(f"{'='*60}")

            method_results = []

            for n_viol in tqdm(n_violations_per_level, desc=f"  {method} logical ablation"):
                # Inject logical violations
                perturbed_triples, violation_set, counts = perturbator.perturb_logical(
                    violation_type=violation_type,
                    n_violations=n_viol,
                    dataset_name=self.dataset_name,
                )
                violation_counts_log.append({
                    'requested': n_viol,
                    'injected': len(violation_set),
                    'counts': counts,
                })

                # Run detection
                detected_set, scores = self._run_detection(method, perturbed_triples)

                # Use n_viol as the "perturbation_pct" proxy for consistency
                pct_proxy = round(len(violation_set) / len(triples) * 100, 2)
                result = self._calculate_metrics(
                    pct_proxy, violation_set, detected_set, scores, len(triples)
                )
                method_results.append(result)

                print(
                    f"    n={len(violation_set)} ({pct_proxy:.1f}%): "
                    f"P={result.precision:.3f}, R={result.recall:.3f}, F1={result.f1:.3f}"
                )

            self.results_by_method[method] = method_results

        # Statistical tests
        self._run_statistical_tests()

        # Save (in logical output dir)
        self._save_results(extra_metadata={
            'ablation_mode': 'logical_violations',
            'violation_type': violation_type,
            'violation_counts_log': violation_counts_log,
        })

        # Report
        self._generate_report()

        # Restore original output dir
        self.output_dir = original_output_dir

    def _load_trained_params(self):
        """Load parameters from previously trained models in paper_results."""
        if getattr(self, 'trained_params', None) and len(self.trained_params) > 0:
            return True

        print("\nLoading pre-trained model parameters...")

        dataset_dir = self.base_results_dir / self.dataset_name

        # Check for ODE params
        ode_results = dataset_dir / 'ode' / 'results.json'
        if ode_results.exists():
            with open(ode_results) as f:
                data = json.load(f)
                self.trained_params['ode'] = data.get('parameters', {})
                print(f"  ✓ Loaded ODE params: {self.trained_params['ode']}")
        else:
            print(f"  ⚠ ODE results.json not found at {ode_results}. "
                  f"Run native stage first: python src/run_all.py --stages native")

        # Check for PyKEEN model params
        for method_dir in dataset_dir.iterdir():
            if method_dir.is_dir() and method_dir.name not in [
                'ode', 'ablation', 'ablation_logical', 'visualizations',
                'explanations', 'metrics',
            ]:
                results_file = method_dir / 'results.json'
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                        method_name = method_dir.name
                        self.trained_params[method_name] = data.get('parameters', {})
                        print(f"  ✓ Loaded {method_name} params: {self.trained_params[method_name]}")
                else:
                    print(f"  ⚠ No results.json for {method_dir.name}. "
                          f"Run native stage first.")

    
    def _run_detection(self, method: str, triples: List[Triple]) -> Tuple[Set[Tuple], List[float]]:
        """Run a detection method using pre-trained parameters"""
        from integrated_framework import IntegratedKGDebugger
        
        config = IntegratedKGDebugger.get_default_config()
        
        # Use pre-trained params if available
        params = self.trained_params.get(method, {})
        
        if method == 'ode':
            config['use_ode'] = True
            config['use_pykeen'] = False
            config['ode_config'] = {
                'embedding_dim': params.get('embedding_dim', 50),
                't_span': tuple(params.get('t_span', [0.0, 5.0])),
                'lambda_data': params.get('lambda_data', 1.0),
                'lambda_logic': params.get('lambda_logic', 1.0),
                'lambda_reg': params.get('lambda_reg', 0.1),
                'rtol': params.get('rtol', 0.005),
                'atol': params.get('atol', 1e-6),
                'device': 'mps'
            }
        else:
            config['use_ode'] = False
            config['use_pykeen'] = True
            config['use_constraints'] = False
            config['pykeen_config'] = {
                'model': method,
                'embedding_dim': params.get('embedding_dim', 50),
                'learning_rate': params.get('learning_rate', 0.001),
                'margin': params.get('margin', 1.0),
                'negative_sampling': params.get('negative_sampling', 5),
                # ABLATION_EPOCHS: re-train budget for perturbed subgraph.
                # Hyperparams are already optimised; 10 epochs is enough to
                # calibrate embedding geometry for detection.
                # See src/config.py for the authoritative value.
                'epochs': ABLATION_EPOCHS,
                'batch_size': ABLATION_BATCH,
            }
        
        debugger = IntegratedKGDebugger(config)
        results = debugger.run_complete_analysis(triples)
        
        # Extract detected triples
        detected_set = set()
        scores = []
        
        result_key = 'ode_results' if method == 'ode' else 'pykeen_results'
        for r in results.get(result_key, []):
            t = r['triple']
            detected_set.add((t.subject, t.relation, t.object))
            scores.append(r['score'])
        
        return detected_set, scores
    
    def _calculate_metrics(self, pct: float, perturbed_set: Set[Tuple], 
                          detected_set: Set[Tuple], scores: List[float],
                          total_triples: int) -> PerturbationResult:
        """Calculate precision, recall, F1"""
        
        true_positives = perturbed_set & detected_set
        false_positives = detected_set - perturbed_set
        false_negatives = perturbed_set - detected_set
        
        n_tp = len(true_positives)
        n_fp = len(false_positives)
        n_fn = len(false_negatives)
        
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return PerturbationResult(
            perturbation_pct=pct,
            n_perturbed=len(perturbed_set),
            n_detected=len(detected_set),
            n_true_positives=n_tp,
            n_false_positives=n_fp,
            n_false_negatives=n_fn,
            precision=precision,
            recall=recall,
            f1=f1,
            detection_scores=scores
        )
    
    def _run_statistical_tests(self):
        """Run statistical significance tests"""
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*60)
        
        self.statistical_results = {}
        
        methods = list(self.results_by_method.keys())
        
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                # Compare F1 scores across perturbation levels
                f1_m1 = [r.f1 for r in self.results_by_method[m1]]
                f1_m2 = [r.f1 for r in self.results_by_method[m2]]
                
                # Paired t-test (assumes normality)
                t_stat, t_pval = stats.ttest_rel(f1_m1, f1_m2)
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_pval = stats.wilcoxon(f1_m1, f1_m2)
                except ValueError:
                    w_stat, w_pval = np.nan, np.nan
                
                # Effect size (Cohen's d)
                diff = np.array(f1_m1) - np.array(f1_m2)
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                
                key = f"{m1}_vs_{m2}"
                self.statistical_results[key] = {
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_pvalue': w_pval,
                    'cohens_d': cohens_d,
                    'significant_t': t_pval < 0.05,
                    'significant_w': w_pval < 0.05 if not np.isnan(w_pval) else False
                }
                
                sig_marker = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else ""
                print(f"\n{m1} vs {m2}:")
                print(f"  t-test: t={t_stat:.3f}, p={t_pval:.4f} {sig_marker}")
                print(f"  Wilcoxon: W={w_stat:.3f}, p={w_pval:.4f}" if not np.isnan(w_stat) else "  Wilcoxon: N/A")
                print(f"  Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
        
        # Trend analysis: Does detection increase with perturbation?
        print("\n" + "-"*40)
        print("TREND ANALYSIS (Detection vs Perturbation)")
        print("-"*40)
        
        for method, results in self.results_by_method.items():
            pct_levels = [r.perturbation_pct for r in results]
            f1_values = [r.f1 for r in results]
            
            # Spearman correlation
            rho, pval = stats.spearmanr(pct_levels, f1_values)
            
            sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"{method}: ρ={rho:.3f}, p={pval:.4f} {sig_marker}")
            
            self.statistical_results[f"{method}_trend"] = {
                'spearman_rho': rho,
                'spearman_pvalue': pval,
                'significant': pval < 0.05
            }
    
    def _save_results(self, extra_metadata: Dict = None):
        """Save all results to JSON.
        
        Args:
            extra_metadata: Optional dict merged into the output JSON
                            (used by run_logical to store violation metadata).
        """
        output = {
            'dataset': self.dataset_name,
            'perturbation_levels': self.perturbation_levels,
            'methods': {}
        }
        
        if extra_metadata:
            output.update(extra_metadata)
        
        for method, results in self.results_by_method.items():
            output['methods'][method] = [
                {
                    'perturbation_pct': r.perturbation_pct,
                    'n_perturbed': r.n_perturbed,
                    'n_detected': r.n_detected,
                    'true_positives': r.n_true_positives,
                    'false_positives': r.n_false_positives,
                    'false_negatives': r.n_false_negatives,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1': r.f1
                }
                for r in results
            ]
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        output['statistical_tests'] = convert_to_native(self.statistical_results)

        # Use a mode-specific filename so logical and random runs don't overwrite each other
        ablation_mode = (extra_metadata or {}).get('ablation_mode', 'random')
        suffix = f'_ablation_{ablation_mode}' if ablation_mode != 'random' else '_ablation'
        fname = f'{self.dataset_name}{suffix}.json'
        
        with open(self.output_dir / fname, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {self.output_dir / fname}")
    
    def _generate_report(self):
        """Generate human-readable report"""
        lines = []
        lines.append("="*80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("="*80)
        lines.append(f"Dataset: {self.dataset_name}")
        lines.append("")
        
        # Results table
        lines.append("DETECTION PERFORMANCE BY PERTURBATION LEVEL")
        lines.append("-"*80)
        
        header = "Method      | " + " | ".join([f"{p}%" for p in self.perturbation_levels])
        lines.append(header)
        lines.append("-"*len(header))
        
        for method, results in self.results_by_method.items():
            f1_values = " | ".join([f"{r.f1:.2f}" for r in results])
            lines.append(f"{method:11s} | {f1_values}")
        
        lines.append("")
        
        # Statistical significance
        lines.append("STATISTICAL SIGNIFICANCE (p < 0.05)")
        lines.append("-"*80)
        
        for key, stats_result in self.statistical_results.items():
            if '_vs_' in key:
                if stats_result['significant_t']:
                    lines.append(f"✓ {key}: Significant difference (p={stats_result['t_pvalue']:.4f})")
                else:
                    lines.append(f"✗ {key}: No significant difference (p={stats_result['t_pvalue']:.4f})")
        
        lines.append("")
        
        # Trend significance
        lines.append("PERTURBATION SENSITIVITY (Recall increases with perturbation?)")
        lines.append("-"*80)
        
        for key, stats_result in self.statistical_results.items():
            if '_trend' in key:
                method = key.replace('_trend', '')
                if stats_result['significant']:
                    lines.append(f"✓ {method}: Significant trend (ρ={stats_result['spearman_rho']:.3f})")
                else:
                    lines.append(f"✗ {method}: No significant trend (ρ={stats_result['spearman_rho']:.3f})")
        
        report_text = "\n".join(lines)
        
        with open(self.output_dir / f'{self.dataset_name}_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to {self.output_dir / f'{self.dataset_name}_report.txt'}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate LaTeX tables
        self._generate_latex_tables()
    
    def _generate_visualizations(self):
        """Generate publication-quality visualizations"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("matplotlib not available, skipping visualizations")
            return
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 11,
            'figure.figsize': (10, 6),
            'figure.dpi': 150
        })
        
        methods = list(self.results_by_method.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        
        # 1. F1 Score vs Perturbation Level
        fig, ax = plt.subplots()
        for i, method in enumerate(methods):
            results = self.results_by_method[method]
            x = [r.perturbation_pct for r in results]
            y = [r.f1 for r in results]
            ax.plot(x, y, marker=markers[i % len(markers)], 
                   color=colors[i], label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Anomaly Detection Performance vs Perturbation ({self.dataset_name})')
        ax.legend(loc='best')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(viz_dir / 'f1_vs_perturbation.pdf', bbox_inches='tight')
        plt.savefig(viz_dir / 'f1_vs_perturbation.png', bbox_inches='tight')
        plt.close()
        
        # 2. Precision vs Recall curves
        fig, ax = plt.subplots()
        for i, method in enumerate(methods):
            results = self.results_by_method[method]
            precision = [r.precision for r in results]
            recall = [r.recall for r in results]
            ax.plot(recall, precision, marker=markers[i % len(markers)],
                   color=colors[i], label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Trade-off ({self.dataset_name})')
        ax.legend(loc='best')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(viz_dir / 'precision_recall.pdf', bbox_inches='tight')
        plt.savefig(viz_dir / 'precision_recall.png', bbox_inches='tight')
        plt.close()
        
        # 3. Grouped bar chart for F1 at different perturbation levels
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.perturbation_levels))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            results = self.results_by_method[method]
            f1_values = [r.f1 for r in results]
            ax.bar(x + i * width, f1_values, width, label=method, color=colors[i])
        
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'Method Comparison by Perturbation Level ({self.dataset_name})')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f'{p}%' for p in self.perturbation_levels])
        ax.legend(loc='upper left')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(viz_dir / 'f1_comparison_bars.pdf', bbox_inches='tight')
        plt.savefig(viz_dir / 'f1_comparison_bars.png', bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap of statistical significance
        n_methods = len(methods)
        sig_matrix = np.zeros((n_methods, n_methods))
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    sig_matrix[i, j] = 1.0
                else:
                    key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in self.statistical_results else f"{m2}_vs_{m1}"
                    if key in self.statistical_results:
                        pval = self.statistical_results[key]['t_pvalue']
                        sig_matrix[i, j] = 1 - pval  # Higher = more significant
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sig_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(methods)
        ax.set_title('Statistical Significance Matrix (1-pvalue)')
        plt.colorbar(im, ax=ax, label='1 - p-value')
        plt.tight_layout()
        plt.savefig(viz_dir / 'significance_heatmap.pdf', bbox_inches='tight')
        plt.savefig(viz_dir / 'significance_heatmap.png', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {viz_dir}")
    
    def _generate_latex_tables(self):
        """Generate LaTeX tables for the paper"""
        tables_dir = self.output_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)
        
        methods = list(self.results_by_method.keys())
        
        # Table 1: F1 Scores by Perturbation Level
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{F1 Scores by Perturbation Level ({self.dataset_name})}}")
        lines.append("\\label{tab:ablation_f1}")
        
        col_spec = "l" + "c" * len(self.perturbation_levels)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        header = "Method & " + " & ".join([f"{p}\\%" for p in self.perturbation_levels]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        for method in methods:
            results = self.results_by_method[method]
            f1_values = " & ".join([f"{r.f1:.3f}" for r in results])
            lines.append(f"{method} & {f1_values} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(tables_dir / 'ablation_f1.tex', 'w') as f:
            f.write('\n'.join(lines))
        
        # Table 2: Statistical Significance
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Statistical Significance Tests ({self.dataset_name})}}")
        lines.append("\\label{tab:ablation_significance}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Comparison & t-statistic & p-value & Cohen's d & Significant \\\\ ")
        lines.append("\\midrule")
        
        for key, stats_result in self.statistical_results.items():
            if '_vs_' in key:
                sig = "$\\checkmark$" if stats_result['significant_t'] else "$\\times$"
                lines.append(f"{key.replace('_', ' ')} & {stats_result['t_statistic']:.3f} & {stats_result['t_pvalue']:.4f} & {stats_result['cohens_d']:.3f} & {sig} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(tables_dir / 'ablation_significance.tex', 'w') as f:
            f.write('\n'.join(lines))
        
        # Table 3: Trend Analysis
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Perturbation Sensitivity Analysis ({self.dataset_name})}}")
        lines.append("\\label{tab:ablation_trend}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Method & Spearman $\\rho$ & p-value & Significant Trend \\\\ ")
        lines.append("\\midrule")
        
        for key, stats_result in self.statistical_results.items():
            if '_trend' in key:
                method = key.replace('_trend', '')
                sig = "$\\checkmark$" if stats_result['significant'] else "$\\times$"
                lines.append(f"{method} & {stats_result['spearman_rho']:.3f} & {stats_result['spearman_pvalue']:.4f} & {sig} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(tables_dir / 'ablation_trend.tex', 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ LaTeX tables saved to {tables_dir}")


def run_ablation_study(dataset_name: str = None):
    """Main entry point for ablation study"""
    
    # Load dataset
    loader = DataLoader()
    
    if dataset_name:
        datasets = [dataset_name]
    else:
        datasets = loader.list_datasets()
        print("Available datasets:", datasets)
        dataset_name = input("Select dataset: ").strip()
        datasets = [dataset_name]
    
    for ds in datasets:
        print(f"\n{'='*80}")
        print(f"Running ablation study on: {ds}")
        print(f"{'='*80}")
        
        triples = loader.load_dataset(ds)
        
        study = AblationStudy(ds)
        study.run(triples, methods=['ode', 'TransE', 'DistMult'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation study with perturbations")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=['ode', 'TransE', 'DistMult'],
                        help="Methods to evaluate")
    args = parser.parse_args()
    
    run_ablation_study(args.dataset)
