from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from typing import Dict, List
import joblib
import hashlib
from pathlib import Path
from scipy import sparse

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_triples_hash(triples: List['Triple']) -> str:
    """Generate a hash for a list of triples to identify dataset version"""
    # Heuristic: hash length + first 10 + last 10 triples
    if not triples:
        return "empty"
    
    sample = str(len(triples)) + str(triples[:10]) + str(triples[-10:])
    return hashlib.md5(sample.encode()).hexdigest()



@dataclass
class Triple:
    subject: str
    relation: str
    object: str


@dataclass
class IntegratedAnomalyReport:
    triple: Triple
    method_scores: Dict[str, float] = field(default_factory=dict)
    explanations: Dict[str, str] = field(default_factory=dict)
    consensus_score: float = 0.0
    confidence: float = 0.0
    action: str = "verify"

    def update_consensus(self):
        if self.method_scores:
            self.consensus_score = np.mean(list(self.method_scores.values()))
            self.confidence = len(self.method_scores) / 4.0  # Assuming 4 methods max

            if self.consensus_score > 0.8:
                self.action = "remove"
            elif self.consensus_score > 0.5:
                self.action = "verify"
            else:
                self.action = "correct"


"""
Constraint Module for KG-Debug-ODE
Defines logical constraints that KGs should satisfy
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Triple:
    """Represents a KG triple (subject, relation, object)"""
    subject: str
    relation: str
    object: str

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.relation == other.relation and
                self.object == other.object)


class Constraint(ABC):
    """Abstract base class for logical constraints"""

    def __init__(self, weight: float = 1.0, uses_embeddings: bool = False):
        self.weight = weight
        self.name = self.__class__.__name__
        self.uses_embeddings = uses_embeddings

    @abstractmethod
    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        """
        Compute constraint violation score

        Args:
            embeddings: Dict mapping entity/relation names to embedding vectors
            triples: List of triples in the KG

        Returns:
            violation_score: Higher means more violation (0 = satisfied)
        """
        pass

    @abstractmethod
    def get_violating_triples(self,
                              triples: List[Triple]) -> List[Triple]:
        """
        Identify specific triples that violate this constraint
        
        Returns:
            List of violating triples
        """
        pass

    def compute_gradient(self,
                         embeddings: Dict[str, np.ndarray],
                         triples: List[Triple]) -> Dict[str, np.ndarray]:
        """
        Compute gradient of violation w.r.t embeddings
        Default implementation returns empty dict (gradient 0)
        """
        return {}


class TransitivityConstraint(Constraint):
    """
    Enforces transitivity: (A, r, B) ∧ (B, r, C) → (A, r, C)

    Example: locatedIn relation
    - (Paris, locatedIn, France) ∧ (France, locatedIn, Europe)
      → Should have (Paris, locatedIn, Europe)
    """

    def __init__(self, relation: str, weight: float = 1.0):
        super().__init__(weight, uses_embeddings=True)
        self.relation = relation
        self.name = f"Transitivity({relation})"
        self.adj = None  # Cache for adjacency list

    def precompute_structure(self, triples: List[Triple]):
        """Precompute adjacency matrix (Sparse) with caching"""
        dataset_hash = get_triples_hash(triples)
        safe_rel = self.relation.replace('/', '_')
        cache_key = f"transitivity_{safe_rel}_{dataset_hash}"
        cache_path = CACHE_DIR / f"{cache_key}.joblib"
        
        if cache_path.exists():
            print(f"Loading cached structure for {self.name}")
            data = joblib.load(cache_path)
            self.adj_matrix = data['adj_matrix']
            self.entity_to_idx = data['entity_to_idx']
            self.idx_to_entity = data['idx_to_entity']
            
            # Reconstruct adjacency list from sparse matrix for compute_gradient
            # This is necessary because compute_gradient currently iterates over the graph
            self.adj = {}
            cx = self.adj_matrix.tocoo()
            for i, j in zip(cx.row, cx.col):
                subj = self.idx_to_entity[i]
                obj = self.idx_to_entity[j]
                if subj not in self.adj: self.adj[subj] = set()
                self.adj[subj].add(obj)
            return

        print(f"Computing structure for {self.name}...")
        # Build entity map
        entities = set()
        for t in triples:
            entities.add(t.subject)
            entities.add(t.object)
        
        self.entity_list = sorted(entities)
        self.entity_to_idx = {e: i for i, e in enumerate(self.entity_list)}
        self.idx_to_entity = {i: e for i, e in enumerate(self.entity_list)}
        n = len(entities)
        
        # Build sparse adjacency matrix
        rows = []
        cols = []
        data = []
        
        for t in triples:
            if t.relation == self.relation:
                if t.subject in self.entity_to_idx and t.object in self.entity_to_idx:
                    rows.append(self.entity_to_idx[t.subject])
                    cols.append(self.entity_to_idx[t.object])
                    data.append(1)
        
        self.adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        
        # Also populate self.adj for compute_gradient
        self.adj = {}
        for t in triples:
            if t.relation == self.relation:
                if t.subject not in self.adj: self.adj[t.subject] = set()
                self.adj[t.subject].add(t.object)
        
        # Save to cache
        joblib.dump({
            'adj_matrix': self.adj_matrix,
            'entity_to_idx': self.entity_to_idx,
            'idx_to_entity': self.idx_to_entity
        }, cache_path)
        print(f"Cached structure to {cache_path}")

    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        """
        Compute violation by checking if transitive closure is satisfied
        Optimized using sparse matrix multiplication
        """
        violation = 0.0

        if not hasattr(self, 'adj_matrix'):
            self.precompute_structure(triples)
        
        # A^2 gives paths of length 2
        # If (A^2)[i,j] > 0, there is a path i->k->j
        # We check if A[i,j] == 0 (missing direct link)
        
        A = self.adj_matrix
        A2 = A.dot(A)
        
        # Find pairs (i, j) where A2[i,j] > 0 but A[i,j] == 0
        # We can subtract A from A2 (treating >0 as 1) but we need to be careful with weights
        # Let's just iterate over non-zero elements of A2
        
        cx = A2.tocoo()
        
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i == j: continue # Ignore self-loops for now
            
            # Check if direct link exists
            # A is CSR, efficient access? Not O(1), but fast enough
            # Or convert A to set of pairs for O(1) check
            # Since we are iterating, let's just check
            
            # If (i, j) is NOT in A
            if A[i, j] == 0:
                # Violation found: i->k->j exists but i->j does not
                subj = self.idx_to_entity[i]
                obj = self.idx_to_entity[j]
                
                if subj in embeddings and obj in embeddings:
                    emb_a = embeddings[subj]
                    emb_c = embeddings[obj]
                    emb_r = embeddings.get(self.relation, np.zeros_like(emb_a))

                    # TransE-style: ||h + r - t||
                    dist = np.linalg.norm(emb_a + emb_r - emb_c)
                    violation += dist

        return violation * self.weight

    def get_violating_triples(self, triples: List[Triple]) -> List[Triple]:
        """
        Identify triples involved in transitivity violations.
        Returns pairs of triples (a,r,b) and (b,r,c) where (a,r,c) is missing.
        """
        if not hasattr(self, 'adj_matrix'):
            self.precompute_structure(triples)
            
        # Map triples for fast lookup
        triple_map = {(t.subject, t.relation, t.object): t for t in triples if t.relation == self.relation}
        
        A = self.adj_matrix
        # We need to find k for i->k->j
        # This is expensive with pure sparse matrices without keeping track of paths.
        # Fallback to iterating for detailed reporting (slower but necessary for this method)
        
        violations = []
        
        # Reconstruct adjacency list for traversal
        adj = {}
        for t in triples:
            if t.relation == self.relation:
                if t.subject not in adj: adj[t.subject] = []
                adj[t.subject].append(t.object)
                
        for a in adj:
            for b in adj[a]:
                for c in adj.get(b, []):
                    if c not in adj.get(a, []):
                        # Found violation: a->b->c but no a->c
                        # Add evidence triples
                        t1 = triple_map.get((a, self.relation, b))
                        t2 = triple_map.get((b, self.relation, c))
                        if t1: violations.append(t1)
                        if t2: violations.append(t2)
                        
        return list(set(violations)) # Remove duplicates

        return missing

    def compute_gradient(self,
                         embeddings: Dict[str, np.ndarray],
                         triples: List[Triple]) -> Dict[str, np.ndarray]:
        """Compute analytical gradient for transitivity"""
        grads = {}

        # Use cached adjacency
        if self.adj is None:
            self.precompute_structure(triples)
        
        adj = self.adj

        # Iterate violating triplets
        for a in adj:
            for b in adj.get(a, []):
                for c in adj.get(b, []):
                    # Should have (a, r, c)
                    if c not in adj.get(a, []):
                        if a in embeddings and c in embeddings:
                            emb_a = embeddings[a]
                            emb_c = embeddings[c]
                            emb_r = embeddings.get(self.relation, np.zeros_like(emb_a))

                            # dist = ||a + r - c||
                            residual = emb_a + emb_r - emb_c
                            dist = np.linalg.norm(residual)
                            
                            if dist > 1e-6:
                                # grad dist / grad a = residual / dist
                                g = residual / dist
                                
                                # Access/Init gradients
                                if a not in grads: grads[a] = np.zeros_like(emb_a)
                                if c not in grads: grads[c] = np.zeros_like(emb_c)
                                
                                # Accumulate
                                grads[a] += g * self.weight
                                grads[c] -= g * self.weight # deriv of -c is -1

        return grads


class SymmetryConstraint(Constraint):
    """
    Enforces symmetry: (A, r, B) → (B, r, A)

    Example: marriedTo, siblingOf, collaboratesWith
    """

    def __init__(self, relation: str, weight: float = 1.0):
        super().__init__(weight, uses_embeddings=True)
        self.relation = relation
        self.name = f"Symmetry({relation})"
        self.relevant_triples = None  # Cache

    def precompute_structure(self, triples: List[Triple]):
        """Cache triples relevant to this relation with persistence"""
        dataset_hash = get_triples_hash(triples)
        safe_rel = self.relation.replace('/', '_')
        cache_key = f"symmetry_{safe_rel}_{dataset_hash}"
        cache_path = CACHE_DIR / f"{cache_key}.joblib"
        
        if cache_path.exists():
            # print(f"Loading cached structure for {self.name}")
            self.relevant_triples = joblib.load(cache_path)
            return

        self.relevant_triples = [t for t in triples if t.relation == self.relation]
        
        joblib.dump(self.relevant_triples, cache_path)
        print(f"Cached structure to {cache_path}")

    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        violation = 0.0
        # Set lookup for fast existence check (global)
        existing = set(triples)
        
        if self.relevant_triples is None:
            self.precompute_structure(triples)

        for t in self.relevant_triples:
            # Check if reverse exists
                # Check if reverse exists
                reverse = Triple(t.object, t.relation, t.subject)
                if reverse not in existing:
                    # Measure asymmetry in embedding space
                    if t.subject in embeddings and t.object in embeddings:
                        emb_s = embeddings[t.subject]
                        emb_o = embeddings[t.object]
                        emb_r = embeddings.get(self.relation, np.zeros_like(emb_s))

                        # Should have symmetric distances
                        dist_forward = np.linalg.norm(emb_s + emb_r - emb_o)
                        dist_backward = np.linalg.norm(emb_o + emb_r - emb_s)
                        violation += abs(dist_forward - dist_backward)

        return violation * self.weight

    def get_violating_triples(self, triples: List[Triple]) -> List[Triple]:
        """
        Identify triples that violate symmetry (exist without their inverse).
        """
        if self.relevant_triples is None:
            self.precompute_structure(triples)
            
        existing = set(triples)
        violations = []
        
        for t in self.relevant_triples:
            reverse = Triple(t.object, t.relation, t.subject)
            if reverse not in existing:
                violations.append(t)
                
        return violations

        return missing

    def compute_gradient(self,
                         embeddings: Dict[str, np.ndarray],
                         triples: List[Triple]) -> Dict[str, np.ndarray]:
        """Compute analytical gradient for symmetry"""
        grads = {}
        existing = set(triples)
        
        if self.relevant_triples is None:
            self.precompute_structure(triples)

        for t in self.relevant_triples:
            if t.relation == self.relation:
                reverse = Triple(t.object, t.relation, t.subject)
                if reverse not in existing:
                    if t.subject in embeddings and t.object in embeddings:
                        emb_s = embeddings[t.subject]
                        emb_o = embeddings[t.object]
                        emb_r = embeddings.get(self.relation, np.zeros_like(emb_s))

                        # dist_fwd = ||s + r - o||
                        res_fwd = emb_s + emb_r - emb_o
                        dist_fwd = np.linalg.norm(res_fwd)
                        
                        # dist_bwd = ||o + r - s||
                        res_bwd = emb_o + emb_r - emb_s
                        dist_bwd = np.linalg.norm(res_bwd)
                        
                        # Loss = |dist_fwd - dist_bwd|
                        # Let delta = dist_fwd - dist_bwd
                        delta = dist_fwd - dist_bwd
                        sign = 1.0 if delta > 0 else -1.0
                        
                        if dist_fwd > 1e-6 and dist_bwd > 1e-6:
                            # dL/ds = sign * (d_dfwd/ds - d_dbwd/ds)
                            # d_dfwd/ds = res_fwd / dist_fwd
                            # d_dbwd/ds = -res_bwd / dist_bwd
                            
                            g_s_fwd = res_fwd / dist_fwd
                            g_s_bwd = -res_bwd / dist_bwd
                            
                            g_o_fwd = -res_fwd / dist_fwd
                            g_o_bwd = res_bwd / dist_bwd
                            
                            grad_s = sign * (g_s_fwd - g_s_bwd)
                            grad_o = sign * (g_o_fwd - g_o_bwd)
                            
                            if t.subject not in grads: grads[t.subject] = np.zeros_like(emb_s)
                            if t.object not in grads: grads[t.object] = np.zeros_like(emb_o)
                            
                            grads[t.subject] += grad_s * self.weight
                            grads[t.object] += grad_o * self.weight

        return grads


class AntisymmetryConstraint(Constraint):
    """
    Enforces antisymmetry: (A, r, B) → ¬(B, r, A)

    Example: parentOf, precedes, causes

    Two operating modes (block7 milestone — asymmetry-as-gradient):

    - ``treats_as_gradient=True`` (DEFAULT, post-block7): the constraint
      contributes a continuous force to the Neural ODE through
      ``compute_gradient`` (autograd-derived from the L_asym hinge loss).
      ``compute_violation`` returns 0 and ``get_violating_triples`` returns
      ``[]`` so the constraint does NOT also contribute to the symbolic
      bonus path in ``AnomalyDetector`` (avoiding double penalisation:
      gradient + bonus).
    - ``treats_as_gradient=False`` (legacy, pre-block7): symbolic-only.
      ``compute_violation`` counts violations as before; the symbolic
      bonus path captures them. ``compute_gradient`` returns ``{}``.

    L_asym formula (when in gradient mode, only fired on triples where
    the reverse ALSO exists on an antisymmetric relation):

        L_asym(h, r, t) = max(0, margin - (||h+r-t||² - ||t+r-h||²))

    IMPORTANT — `uses_embeddings` is ALWAYS False because the asymmetry
    gradient flows through a SEPARATE path (`compute_asymmetry_gradient`
    on the checker, scaled by `lambda_asym` in the ODE dynamics) rather
    than the existing `compute_total_gradient` aggregator (which is
    multiplied by `lambda_logic` and reserved for transitivity +
    symmetry, per user spec D6 of the block7 audit).

    The ``enabled`` flag is a runtime toggle reserved for the
    ``logical_lambda_asym_zero`` sub-experiment in
    ``run_bootstrap_validation_stage`` (TASK 5 of block7). When False,
    ``compute_gradient`` short-circuits to ``{}``, equivalent to
    ``lambda_asym=0`` for THIS constraint alone — without disturbing
    other AntisymmetryConstraint instances on different relations.
    """

    def __init__(self, relation: str, weight: float = 1.0,
                 treats_as_gradient: bool = True,
                 margin: float = 1.0,
                 enabled: bool = True):
        # uses_embeddings stays False so compute_total_gradient (under
        # lambda_logic) skips this constraint. The asym gradient is
        # invoked via the dedicated path on ConstraintChecker, scaled
        # by lambda_asym.
        super().__init__(weight, uses_embeddings=False)
        self.relation = relation
        self.name = f"Antisymmetry({relation})"
        self.treats_as_gradient = bool(treats_as_gradient)
        self.margin = float(margin)
        self.enabled = bool(enabled)

    def _get_violating_triples_legacy(self, triples: List[Triple]) -> List[Triple]:
        """Diagnostic helper: list all (h,r,t) with both directions present
        on this relation. Independent of operating mode — always returns
        the same set."""
        existing = set(triples)
        violations = []
        for t in triples:
            if t.relation == self.relation:
                reverse = Triple(t.object, t.relation, t.subject)
                if reverse in existing:
                    violations.append(t)
        return violations

    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        # Gradient mode: don't double-count via symbolic bonus.
        if self.treats_as_gradient or not self.enabled:
            return 0.0
        violation = 0.0
        existing = set(triples)
        for t in triples:
            if t.relation == self.relation:
                reverse = Triple(t.object, t.relation, t.subject)
                if reverse in existing:
                    violation += 1.0
        return violation * self.weight

    def get_violating_triples(self, triples: List[Triple]) -> List[Triple]:
        # Used by AnomalyDetector to drive the symbolic-bonus path. In
        # gradient mode (or when disabled), return empty so the bonus is
        # not added on these triples.
        if self.treats_as_gradient or not self.enabled:
            return []
        return self._get_violating_triples_legacy(triples)

    def compute_gradient(self,
                         embeddings: Dict[str, np.ndarray],
                         triples: List[Triple]) -> Dict[str, np.ndarray]:
        """L_asym hinge gradient via PyTorch autograd.

        Returns ``{}`` if either ``treats_as_gradient=False`` (legacy mode)
        or ``enabled=False`` (asym_zero sub-experiment override).
        Otherwise fires on triples where (t, r, h) also exists for the
        same antisymmetric ``self.relation``.

        Note: this method is NOT picked up by ``compute_total_gradient``
        because ``uses_embeddings=False``; instead the dedicated path
        ``ConstraintChecker.compute_asymmetry_gradient`` aggregates these
        contributions and the ODE dynamics scale them by ``lambda_asym``.
        """
        if not self.treats_as_gradient or not self.enabled:
            return {}

        import torch

        existing = set((tt.subject, tt.relation, tt.object) for tt in triples)
        seen = set()
        violating_pairs = []
        for tt in triples:
            if tt.relation != self.relation:
                continue
            h, t = tt.subject, tt.object
            key = frozenset({h, t})
            if key in seen:
                continue
            if (t, self.relation, h) in existing:
                violating_pairs.append((h, t))
                seen.add(key)

        if not violating_pairs:
            return {}

        grads: Dict[str, np.ndarray] = {}
        for h, t in violating_pairs:
            if h not in embeddings or t not in embeddings:
                continue
            emb_h_np = np.asarray(embeddings[h], dtype=np.float64)
            emb_t_np = np.asarray(embeddings[t], dtype=np.float64)
            emb_r_np = np.asarray(embeddings.get(
                self.relation, np.zeros_like(emb_h_np)), dtype=np.float64)

            h_t = torch.tensor(emb_h_np, requires_grad=True)
            t_t = torch.tensor(emb_t_np, requires_grad=True)
            r_t = torch.tensor(emb_r_np, requires_grad=False)

            forward_sq = torch.sum((h_t + r_t - t_t) ** 2)
            reverse_sq = torch.sum((t_t + r_t - h_t) ** 2)
            inner = float(self.margin) - (forward_sq - reverse_sq)
            L = torch.clamp(inner, min=0.0)

            if float(L.detach()) <= 0.0:
                continue

            L.backward()

            if h not in grads:
                grads[h] = np.zeros_like(emb_h_np)
            if t not in grads:
                grads[t] = np.zeros_like(emb_t_np)
            grads[h] += h_t.grad.detach().numpy() * self.weight
            grads[t] += t_t.grad.detach().numpy() * self.weight

        # Cast back to float32 for downstream stability with the ODE state
        for k in list(grads.keys()):
            grads[k] = grads[k].astype(np.float32)

        return grads


class CardinalityConstraint(Constraint):
    """
    Enforces cardinality: Entity can have at most N objects for relation r

    Example: hasCapital (max=1), bornIn (max=1), hasChild (unbounded)
    """

    def __init__(self, relation: str, max_cardinality: int, weight: float = 1.0):
        super().__init__(weight, uses_embeddings=False)
        self.relation = relation
        self.max_cardinality = max_cardinality
        self.name = f"Cardinality({relation}, max={max_cardinality})"

    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        violation = 0.0

        # Count objects per subject
        subject_counts = {}
        for t in triples:
            if t.relation == self.relation:
                if t.subject not in subject_counts:
                    subject_counts[t.subject] = 0
                subject_counts[t.subject] += 1

        # Check violations
        for subject, count in subject_counts.items():
            if count > self.max_cardinality:
                excess = count - self.max_cardinality
                violation += excess

        return violation * self.weight

    def get_violating_triples(self, triples: List[Triple]) -> List[Triple]:
        # Group by subject
        subject_groups = {}
        for t in triples:
            if t.relation == self.relation:
                if t.subject not in subject_groups:
                    subject_groups[t.subject] = []
                subject_groups[t.subject].append(t)

        # Return excess triples (those beyond max_cardinality)
        violations = []
        for subject, group in subject_groups.items():
            if len(group) > self.max_cardinality:
                # Return all but the first max_cardinality
                violations.extend(group[self.max_cardinality:])

        return violations


class DisjointnessConstraint(Constraint):
    """
    Enforces disjointness: If (A, r1, B) then not (A, r2, B)

    Example: (Person, worksAt, Company) and (Person, studiesAt, Company)
    can't both be true simultaneously
    """

    def __init__(self, relation1: str, relation2: str, weight: float = 1.0):
        super().__init__(weight, uses_embeddings=False)
        self.relation1 = relation1
        self.relation2 = relation2
        self.name = f"Disjointness({relation1}, {relation2})"

    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        violation = 0.0

        # Build sets of (subject, object) pairs for each relation
        r1_pairs = {(t.subject, t.object) for t in triples if t.relation == self.relation1}
        r2_pairs = {(t.subject, t.object) for t in triples if t.relation == self.relation2}

        # Count overlaps
        overlap = r1_pairs & r2_pairs
        violation = len(overlap)

        return violation * self.weight

    def get_violating_triples(self, triples: List[Triple]) -> List[Triple]:
        r1_pairs = {(t.subject, t.object): t for t in triples if t.relation == self.relation1}
        r2_pairs = {(t.subject, t.object): t for t in triples if t.relation == self.relation2}

        violations = []
        for pair in r1_pairs:
            if pair in r2_pairs:
                violations.append(r1_pairs[pair])
                violations.append(r2_pairs[pair])

        return violations


class DomainRangeConstraint(Constraint):
    """
    Enforces domain/range types: (A, r, B) → A ∈ Domain(r) ∧ B ∈ Range(r)

    Example: (Person, worksAt, Company) - domain is Person, range is Company
    """

    def __init__(self, relation: str, domain_type: str, range_type: str,
                 type_mapping: Dict[str, str], weight: float = 1.0):
        super().__init__(weight, uses_embeddings=False)
        self.relation = relation
        self.domain_type = domain_type
        self.range_type = range_type
        self.type_mapping = type_mapping  # entity -> type
        self.name = f"DomainRange({relation}: {domain_type}->{range_type})"

    def compute_violation(self,
                          embeddings: Dict[str, np.ndarray],
                          triples: List[Triple]) -> float:
        violation = 0.0

        for t in triples:
            if t.relation == self.relation:
                # Check domain
                subject_type = self.type_mapping.get(t.subject, "Unknown")
                if subject_type != self.domain_type:
                    violation += 1.0

                # Check range
                object_type = self.type_mapping.get(t.object, "Unknown")
                if object_type != self.range_type:
                    violation += 1.0

        return violation * self.weight

    def get_violating_triples(self, triples: List[Triple]) -> List[Triple]:
        violations = []

        for t in triples:
            if t.relation == self.relation:
                subject_type = self.type_mapping.get(t.subject, "Unknown")
                object_type = self.type_mapping.get(t.object, "Unknown")

                if (subject_type != self.domain_type or
                        object_type != self.range_type):
                    violations.append(t)

        return violations


class ConstraintChecker:
    """
    Manager class for multiple constraints
    """

    def __init__(self):
        self.constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint):
        """Add a constraint to check"""
        self.constraints.append(constraint)
        print(f"Added constraint: {constraint.name}")

    def compute_total_violation(self,
                                embeddings: Dict[str, np.ndarray],
                                triples: List[Triple],
                                only_embedding_sensitive: bool = False) -> Dict[str, float]:
        """
        Compute violation for all constraints

        Args:
            embeddings: Current embeddings
            triples: Data triples
            only_embedding_sensitive: If True, only check constraints that use embeddings

        Returns:
            Dict mapping constraint name to violation score
        """
        violations = {}
        total = 0.0

        for constraint in self.constraints:
            if only_embedding_sensitive and not constraint.uses_embeddings:
                continue
                
            score = constraint.compute_violation(embeddings, triples)
            violations[constraint.name] = score
            total += score

        violations['total'] = total
        return violations

    def get_all_violating_triples(self,
                                  triples: List[Triple]) -> Dict[str, List[Triple]]:
        """
        Get violating triples for each constraint

        Returns:
            Dict mapping constraint name to list of violating triples
        """
        result = {}

        for constraint in self.constraints:
            violating = constraint.get_violating_triples(triples)
            if violating:
                result[constraint.name] = violating

        return result

    def compute_total_gradient(self,
                              embeddings: Dict[str, np.ndarray],
                              triples: List[Triple]) -> Dict[str, np.ndarray]:
        """Aggregated gradient from all constraints with uses_embeddings=True
        (transitivity + symmetry). Antisymmetry is NOT included here — it
        flows through ``compute_asymmetry_gradient`` and is scaled by a
        separate ``lambda_asym`` in the ODE dynamics."""
        total_grads = {}

        for constraint in self.constraints:
            if not constraint.uses_embeddings:
                continue

            c_grads = constraint.compute_gradient(embeddings, triples)

            for entity, grad in c_grads.items():
                if entity not in total_grads:
                    total_grads[entity] = np.zeros_like(grad)
                total_grads[entity] += grad

        return total_grads

    def compute_asymmetry_gradient(self,
                                   embeddings: Dict[str, np.ndarray],
                                   triples: List[Triple]) -> Dict[str, np.ndarray]:
        """Aggregated gradient from AntisymmetryConstraint instances
        ONLY. This is a separate path from ``compute_total_gradient`` so
        that asymmetry can be scaled by its own ``lambda_asym`` in the
        ODE dynamics, independent of ``lambda_logic`` which governs
        transitivity + symmetry.

        AntisymmetryConstraint declares ``uses_embeddings=False`` so the
        existing ``compute_total_gradient`` aggregator skips it; this
        method picks up its ``compute_gradient`` output explicitly.
        """
        total_grads = {}
        for constraint in self.constraints:
            if not isinstance(constraint, AntisymmetryConstraint):
                continue
            c_grads = constraint.compute_gradient(embeddings, triples)
            for entity, grad in c_grads.items():
                if entity not in total_grads:
                    total_grads[entity] = np.zeros_like(grad)
                total_grads[entity] += grad
        return total_grads

    def summary(self, embeddings: Dict[str, np.ndarray],
                triples: List[Triple]) -> str:
        """Generate a human-readable summary"""
        violations = self.compute_total_violation(embeddings, triples)

        lines = ["=" * 60]
        lines.append("CONSTRAINT VIOLATION SUMMARY")
        lines.append("=" * 60)

        for name, score in violations.items():
            if name != 'total':
                lines.append(f"{name:40s}: {score:8.2f}")

        lines.append("-" * 60)
        lines.append(f"{'TOTAL':40s}: {violations['total']:8.2f}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ===================================================================
# DATA LOADING
# ===================================================================

import os
import random
import json
from pathlib import Path
from pykeen.triples import TriplesFactory

class DataLoader:
    """
    Handles loading of datasets and anomaly injection.
    Compatible with run_experiments.py and IntegratedKGDebugger.
    """
    
    def __init__(self, base_dir: str = "data"):
        # Resolve 'data' relative to the project root (parent of 'src')
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        self.base_dir = project_root / base_dir
        
        # Fallback: if absolute path provided or exists in cwd
        if not self.base_dir.exists():
            if Path(base_dir).exists():
               self.base_dir = Path(base_dir)
            
        self.base_dir.mkdir(exist_ok=True)

    def get_available_datasets(self) -> List[str]:
        """Returns a list of available datasets in the base directory."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]


    # -----------------------------------------------------------
    # PUBLIC DATASET LOADING
    # -----------------------------------------------------------
    def load_dataset(self, name: str) -> List[Triple]:
        """
        Loads a benchmark dataset. Expected folder structure:
            datasets/<name>/train.txt
            datasets/<name>/valid.txt
            datasets/<name>/test.txt
        Each file contains: h<TAB>r<TAB>t
        """
        folder = self.base_dir / name
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found at {folder}. "
                "Please download it or generate it first."
            )

        triples = []

        for split in ["train.txt", "valid.txt", "test.txt"]:
            path = folder / split
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        h, r, t = parts[:3]
                        triples.append(Triple(h, r, t))

        return triples

    # -----------------------------------------------------------
    # SAMPLE DATASETS ("small" and "medium")
    # -----------------------------------------------------------
    def create_sample_dataset(self, size: str = "small") -> List[Triple]:
        triples = []
        if size == "small":
            n_entities = 20
            n_relations = 5
            n_triples = 150

        elif size == "medium":
            n_entities = 80
            n_relations = 10
            n_triples = 600

        else:
            raise ValueError("Use 'small' or 'medium'")

        entities = [f"E{i}" for i in range(n_entities)]
        relations = [f"R{i}" for i in range(n_relations)]

        for _ in range(n_triples):
            h = random.choice(entities)
            r = random.choice(relations)
            t = random.choice(entities)
            triples.append(Triple(h, r, t))

        return triples

    # -----------------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------------
    def get_statistics(self, triples: List[Triple]) -> Dict:
        ents = set()
        rels = set()

        for tr in triples:
            ents.add(tr.subject)
            ents.add(tr.object)
            rels.add(tr.relation)

        return {
            "n_triples": len(triples),
            "n_entities": len(ents),
            "n_relations": len(rels),
            "avg_degree": (2 * len(triples)) / len(ents) if len(ents) > 0 else 0.0
        }

    # -----------------------------------------------------------
    # ANOMALY INJECTION
    # -----------------------------------------------------------
    def inject_anomalies(self, triples: List[Triple], anomaly_rate: float = 0.1):
        n_anomalies = int(len(triples) * anomaly_rate)
        if n_anomalies == 0:
            return triples

        entities = list({tr.subject for tr in triples} | {tr.object for tr in triples})
        relations = list({tr.relation for tr in triples})

        anomalies = []
        for _ in range(n_anomalies):
            h = random.choice(entities)
            t = random.choice(entities)
            r = random.choice(relations)
            anomalies.append(Triple(h, r, t))

        return triples + anomalies

    # -----------------------------------------------------------
    # PYKEEN COMPATIBILITY
    # -----------------------------------------------------------
    def prepare_experiment_data(self, triples: List[Triple]):
        """
        Converts a list of Triple objects to a PyKEEN TriplesFactory
        """
        arr = np.array(
            [[tr.subject, tr.relation, tr.object] for tr in triples],
            dtype=str
        )

        tf = TriplesFactory.from_labeled_triples(arr)

        return {
            "triples_factory": tf,
            "labeled_triples": arr
        }
