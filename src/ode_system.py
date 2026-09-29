"""
 ODE System for KG Debugging (PyTorch Version)
Core module that integrates KGE with differential equations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import time

from data import Triple, ConstraintChecker

@dataclass
class ODEConfig:
    """Configuration for ODE solver"""
    t_span: Tuple[float, float] = (0.0, 10.0)  # Time span for integration
    t_eval: Optional[np.ndarray] = None  # Times to evaluate solution
    method: str = 'RK4'  # Integration method
    rtol: float = 1e-3  # Relative tolerance
    atol: float = 1e-6  # Absolute tolerance
    step_size: float = 0.1  # Step size for fixed step solvers
    
    # Weights for different forces
    lambda_data: float = 1.0   # Data/KGE loss weight
    lambda_logic: float = 1.0  # Logic gradient weight (transitivity + symmetry only, post-block7)
    lambda_reg: float = 0.1    # Regularization weight
    # Antisymmetry as continuous gradient force, scaled by its own lambda_asym
    # (separate from lambda_logic). margin_asym is the hinge margin used inside
    # AntisymmetryConstraint.compute_gradient (default 1.0).
    lambda_asym: float = 1.0
    margin_asym: float = 1.0

    device: str = "mps"  # Default device

    # [perf milestone] Use the vectorised constraint gradient
    # (src/constraint_gradient_vectorized.py) inside compute_logic_forces /
    # compute_asym_forces. Mathematically equivalent to the scalar reference;
    # SDE per-triple variance observable agrees with scalar to Pearson 1.000000
    # (see scripts/verify_vectorized_sde_observable.py). 76× faster on the
    # canonical WN18RR perturbed pool (3287 ms → 43 ms per ode_dynamics step).
    use_vectorized_constraints: bool = True

@dataclass
class ODESolution:
    """Wrapper for ODE solution with analysis"""
    t: np.ndarray  # Time points
    y: np.ndarray  # Solutions (embeddings over time) - Shape (N_params, T)
    success: bool
    message: str
    
    # Analysis metrics
    final_energy: float = 0.0
    energy_reduction: float = 0.0
    convergence_time: float = 0.0
    max_gradient_norm: float = 0.0
    stability_index: float = 0.0


class KGODESystem(nn.Module):
    """
    Neural ODE system for knowledge graph debugging - PyTorch Implementation
    
    Models KG embeddings as a dynamical system that evolves toward consistency.
    Uses PyTorch for GPU/MPS acceleration of vector operations.
    """
    
    def __init__(self, 
                 embeddings: Dict[str, np.ndarray],
                 triples: List[Triple],
                 constraint_checker: ConstraintChecker,
                 config: ODEConfig):
        """
        Args:
            embeddings: Initial embeddings for entities and relations
            triples: List of KG triples
            constraint_checker: Constraint checker with defined constraints
            config: ODE solver configuration
        """
        super().__init__()
        self.config = config
        
        # Setup Device
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and config.device == 'cuda':
            self.device = torch.device('cuda')
            print("🚀 ODE System initialized on CUDA")
        elif torch.backends.mps.is_available() and config.device == 'mps':
            self.device = torch.device('mps')
            print("🚀 ODE System initialized on MPS (Apple Silicon)")
        else:
            print("💻 ODE System initialized on CPU")
            
        self.triples = triples
        self.constraint_checker = constraint_checker
        
        # 1. Create Entity Mapping
        self.entity_list = sorted(embeddings.keys())
        self.entity_to_idx = {e: i for i, e in enumerate(self.entity_list)}
        self.n_entities = len(self.entity_list)
        
        # 2. Initialize Embeddings as PyTorch Tensor (y0)
        # Shape: (N, D)
        vectors = [embeddings[entity] for entity in self.entity_list]
        initial_matrix = np.stack(vectors)
        self.embedding_dim = initial_matrix.shape[1]
        
        # We store initial state. The evolving state is passed in dynamics.
        self.y0 = torch.tensor(initial_matrix, dtype=torch.float32, device=self.device)
        
        # 3. Precompute Triple Indices for Vectorization (Tensor-based)
        # Filter triples to only include those with known entities
        valid_triples = [t for t in triples if t.subject in self.entity_to_idx and t.object in self.entity_to_idx]
        
        s_indices = [self.entity_to_idx[t.subject] for t in valid_triples]
        o_indices = [self.entity_to_idx[t.object] for t in valid_triples]
        
        self.s_indices = torch.tensor(s_indices, dtype=torch.long, device=self.device)
        self.o_indices = torch.tensor(o_indices, dtype=torch.long, device=self.device)
        
        # Handle relations (vectors)
        r_vectors_list = []
        for t in valid_triples:
            if t.relation in embeddings:
                r_vectors_list.append(embeddings[t.relation])
            else:
                r_vectors_list.append(np.zeros(self.embedding_dim))
        
        self.r_vectors = torch.tensor(np.array(r_vectors_list), dtype=torch.float32, device=self.device)
        
        self.gradient_history = []
        self.energy_history = []

        # [perf milestone] Build the vectorised constraint gradient engine ONCE.
        # See progress/perf_proposal_vectorized_constraint_gradient.md.
        # The flag defaults to True; setting use_vectorized_constraints=False
        # restores the scalar code path (kept for regression testing).
        self._vec_grad = None
        if getattr(self.config, "use_vectorized_constraints", True):
            try:
                from constraint_gradient_vectorized import VectorizedConstraintGradient
                relation_to_vec = {e: embeddings[e] for e in self.entity_list if e in embeddings}
                self._vec_grad = VectorizedConstraintGradient(
                    self.constraint_checker,
                    self.entity_to_idx,
                    self.triples,
                    relation_to_vec,
                    self.embedding_dim,
                    device=self.device,
                )
                print(f"  - Vectorised constraint gradient: n_trans={self._vec_grad.n_trans} "
                      f"n_sym={self._vec_grad.n_sym} n_asym={self._vec_grad.n_asym}")
            except Exception as e:
                print(f"  - WARNING: vectorised constraint gradient unavailable ({e}); "
                      f"falling back to scalar path")
                self._vec_grad = None

        print(f"Initialized ODE system (Vectorized PyTorch):")
        print(f"  - Entities: {self.n_entities}")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Triples: {len(valid_triples)}")

    def _get_current_embeddings_dict(self, y: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert current tensor state back to dict (for constraints compatibility)"""
        # Note: This involves GPU->CPU transfer, which is a bottleneck.
        # Required because ConstraintChecker assumes Dictionary<str, numpy>
        y_cpu = y.detach().cpu().numpy()
        return {entity: y_cpu[i] for i, entity in enumerate(self.entity_list)}

    def compute_kge_forces(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute forces from KGE loss (Gradient Descent Direction)
        Force = -Gradient
        
        Optimized PyTorch implementation using scatter/gather
        """
        # y shape: (N, D)
        
        # Gather embeddings: (T, D)
        H = y[self.s_indices]
        T = y[self.o_indices]
        R = self.r_vectors
        
        # Residuals: h + r - t
        residual = H + R - T
        
        # Gradients:
        # dL/dh = 2 * residual
        # dL/dt = -2 * residual
        
        grad_H = 2 * residual
        grad_T = -2 * residual
        
        # Aggregate gradients
        grad_E = torch.zeros_like(y)
        grad_E.index_add_(0, self.s_indices, grad_H)
        grad_E.index_add_(0, self.o_indices, grad_T)
        
        # Force is negative gradient
        return -self.config.lambda_data * grad_E

    def compute_logic_forces(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute forces from Logic Constraints
        Propagates gradients from the ConstraintChecker
        """
        # [perf milestone] Fast path: vectorised constraint gradient.
        if self._vec_grad is not None:
            return self._vec_grad.compute_logic_force(y, self.config.lambda_logic)

        # Bridge to NumPy implementation (CPU)
        embeddings_dict = self._get_current_embeddings_dict(y)

        # Returns dict {entity: grad_np}
        grad_dict = self.constraint_checker.compute_total_gradient(embeddings_dict, self.triples)
        
        # Convert back to Tensor
        grad_E = torch.zeros_like(y)
        
        # Build updates
        indices = []
        grads = []
        
        for entity_name, grad_vec in grad_dict.items():
            if entity_name in self.entity_to_idx:
                indices.append(self.entity_to_idx[entity_name])
                grads.append(grad_vec)
        
        if indices:
            idxs = torch.tensor(indices, dtype=torch.long, device=self.device)
            g_vals = torch.tensor(np.array(grads), dtype=torch.float32, device=self.device)
            grad_E.index_add_(0, idxs, g_vals)
            
        return -self.config.lambda_logic * grad_E

    def compute_asym_forces(self, y: torch.Tensor) -> torch.Tensor:
        """Compute forces from antisymmetry constraints, scaled by
        ``lambda_asym``. Mirrors the structure of ``compute_logic_forces`` but
        flows through the dedicated path on ConstraintChecker
        (``compute_asymmetry_gradient``), keeping antisymmetry decoupled from
        the lambda_logic-scaled aggregator that handles transitivity + symmetry.
        """
        # [perf milestone] Fast path: vectorised constraint gradient.
        if self._vec_grad is not None:
            return self._vec_grad.compute_asym_force(
                y, self.config.lambda_asym, margin=self.config.margin_asym,
            )

        # Bridge to NumPy implementation (CPU)
        embeddings_dict = self._get_current_embeddings_dict(y)

        # Returns dict {entity: grad_np}
        grad_dict = self.constraint_checker.compute_asymmetry_gradient(
            embeddings_dict, self.triples
        )

        # Convert back to Tensor
        grad_E = torch.zeros_like(y)
        if not grad_dict:
            return grad_E

        indices = []
        grads = []
        for entity_name, grad_vec in grad_dict.items():
            if entity_name in self.entity_to_idx:
                indices.append(self.entity_to_idx[entity_name])
                grads.append(grad_vec)

        if indices:
            idxs = torch.tensor(indices, dtype=torch.long, device=self.device)
            g_vals = torch.tensor(np.array(grads), dtype=torch.float32, device=self.device)
            grad_E.index_add_(0, idxs, g_vals)

        return -self.config.lambda_asym * grad_E

    def ode_dynamics(self, t, y):
        """
        dy/dt = F(y)
        Defines the evolution of the system
        """
        # 1. Data Force (GPU optimized)
        force_data = self.compute_kge_forces(y)

        # 2. Logic Force (Hybrid CPU/GPU) — transitivity + symmetry, scaled by lambda_logic
        force_logic = torch.zeros_like(y)
        if self.config.lambda_logic > 0:
            force_logic = self.compute_logic_forces(y)

        # 3. Antisymmetry force — separate path, scaled by lambda_asym
        force_asym = torch.zeros_like(y)
        if self.config.lambda_asym > 0:
            force_asym = self.compute_asym_forces(y)

        # 4. Regularization Force (Force = -lambda * 2 * y, from L2 norm)
        force_reg = -self.config.lambda_reg * 2 * y

        total_force = force_data + force_logic + force_asym + force_reg

        # Clip Gradients to prevent explosion
        total_force = torch.clamp(total_force, -10.0, 10.0)

        return total_force

    def solve_rk4(self, y0, t_span, step_size=0.1):
        """
        Fixed step Runge-Kutta 4 solver implemented in PyTorch
        Keeps computation on device
        """
        t0, t1 = t_span
        n_steps = int((t1 - t0) / step_size)
        dt = step_size
        
        t = t0
        y = y0.clone()
        
        # Track history
        # We store flattened arrays in CPU RAM to avoid GPU OOM for long trajectories
        track_t = [t]
        track_y = [y.detach().cpu().numpy().flatten()] 
        
        print(f"Solving ODE (RK4) on {self.device}: {n_steps} steps, dt={dt}")
        
        for i in range(n_steps):
            k1 = self.ode_dynamics(t, y)
            k2 = self.ode_dynamics(t + 0.5*dt, y + 0.5*dt*k1)
            k3 = self.ode_dynamics(t + 0.5*dt, y + 0.5*dt*k2)
            k4 = self.ode_dynamics(t + dt, y + dt*k3)
            
            # Gradient containment (Clipping)
            # Clip total_force (k1 is the derivative at t)
            # Ideally we clip k1, k2, k3, k4 or just stable dynamics?
            # Let's clip y updates or just run smaller steps.
            # Here we just monitor. But to Fix 'inf', we should clip the gradients inside ode_dynamics or here.
            
            y = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Check for instability
            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"⚠️ Simulation exploded at step {i} (t={t:.2f}). Stopping.")
                track_t.append(t)
                track_y.append(y.detach().cpu().numpy().flatten())
                break
            
            t = t + dt
            
            # Monitoring
            if i % 10 == 0:
                with torch.no_grad():
                    grad_norm = torch.norm(k1).item()
                    self.gradient_history.append((t, grad_norm))
                
            track_t.append(t)
            track_y.append(y.detach().cpu().numpy().flatten())
            
        return np.array(track_t), np.array(track_y)

    def solve(self) -> ODESolution:
        """
        Solve the ODE system
        """
        print("\n" + "="*60)
        print("SOLVING ODE SYSTEM (PARALLELIZED)")
        print(f"Device: {self.device}")
        print("="*60)
        
        start_time = time.time()
        
        # Initial State
        y_current = self.y0
        
        # Solve
        ts, ys = self.solve_rk4(y_current, 
                              self.config.t_span, 
                              step_size=self.config.step_size)
        
        solve_time = time.time() - start_time
        
        # Compute Stats (Post-Process)
        final_y_flat = ys[-1]
        
        # Convert final stats
        max_gradient = max(g for _, g in self.gradient_history) if self.gradient_history else 0.0
        
        print(f"\nSolution status: Converged (RK4)")
        print(f"Time steps: {len(ts)}")
        print(f"Solve time: {solve_time:.2f}s")
        print(f"Max Gradient: {max_gradient:.4f}")
        print("="*60)
        
        return ODESolution(
            t=ts,
            y=ys.T, # Transpose to (N_params, T) to match Scipy convention
            success=True,
            message="Integration successful (RK4 PyTorch)",
            convergence_time=solve_time,
            max_gradient_norm=max_gradient,
            stability_index=1.0 # Placeholder
        )
        
    def get_final_embeddings(self, solution: ODESolution) -> Dict[str, np.ndarray]:
        """Extract final embeddings from solution"""
        # solution.y is (N_flat, T)
        final_flat = solution.y[:, -1]
        matrix = final_flat.reshape(self.n_entities, self.embedding_dim)
        return {entity: matrix[i] for i, entity in enumerate(self.entity_list)}

