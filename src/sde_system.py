""" SDE wrapper around the existing ODE — block-sde milestone.

Extends ``KGODESystem`` (which carries the trained drift function as fixed
dynamics parameterised by the cached lambdas) with a stochastic diffusion
term so that the embeddings evolve as

    dE(t) = f(E(t)) dt + sigma * dW(t)

where ``f`` is the **same** ``ode_dynamics`` of the deterministic  ODE
(reused as-is — *no retraining*) and the diffusion is additive scalar
sigma *

The Fokker-Planck
analysis points out that the SDE's stationary density depends explicitly on
the potential gradient direction, so an observable derived from the
*variance* of the stochastic trajectories should be sensitive to
lambda_asym where the deterministic energy-knee observable was not.

Usage
-----

.. code-block:: python

    from ode_system import KGODESystem
    from sde_system import KGSDESystem, compute_per_triple_variance

    kgode = KGODESystem(embeddings, triples, constraint_checker, config)
    sde = KGSDESystem(kgode, sigma=0.05)

    # Per-triple variance observable over k stochastic trajectories
    variances = compute_per_triple_variance(
        sde,
        triples_to_evaluate,
        k_trajectories=20,
        t_span=(0.0, 5.0),
        n_steps=50,
        seed_base=42,
    )

Reproducibility note
--------------------

The stochastic seed is supplied via ``torchsde.BrownianInterval(entropy=...)``.
Two integrations with the same ``entropy`` produce bit-identical trajectories;
two integrations with different ``entropy`` produce independent trajectories.
``torch.manual_seed`` alone is **not** sufficient — torchsde uses its own
Brownian path generator.
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import torch
import torchsde

if TYPE_CHECKING:
    from ode_system import KGODESystem


class KGSDESystem(torchsde.SDEIto):
    """SDE wrapper that delegates the drift to an existing ``KGODESystem``.

    The diffusion is additive scalar ``sigma * I`` (``noise_type="diagonal"``
    in torchsde, with each embedding dimension perturbed independently). The
    Itô convention is the standard choice for Brownian noise without
    multiplicative state dependence, and matches the Fokker-Planck reasoning
    in the milestone prompt.
    """

    sde_type = "ito"

    def __init__(self, ode_system: "KGODESystem", sigma: float = 0.05):
        """
        Args:
            ode_system: an already-initialised ``KGODESystem``. The drift
                function ``ode_dynamics`` is reused without modification.
            sigma: scalar diffusion coefficient. Must be > 0 for the SDE
                to be non-degenerate. The variance of the terminal state
                under zero drift scales as ``sigma**2 * (t1 - t0)``.
        """
        super().__init__(noise_type="diagonal")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.ode = ode_system
        self.sigma = float(sigma)

    def f(self, t, y):
        """Drift function: delegate to the existing ODE dynamics.

        ``KGODESystem.ode_dynamics`` returns the negated gradient sum
        (data + logic + asym + reg) clamped to ±10. We do not modify it.
        """
        return self.ode.ode_dynamics(t, y)

    def g(self, t, y):
        """Diffusion function: additive scalar sigma on every dimension.

        Returns a tensor of shape ``y.shape`` with all entries equal to
        ``self.sigma`` — the diagonal of the diffusion matrix.
        """
        if self.sigma == 0.0:
            return torch.zeros_like(y)
        return self.sigma * torch.ones_like(y)


def integrate_one_trajectory(
    sde: KGSDESystem,
    y0: torch.Tensor,
    t_span: Tuple[float, float] = (0.0, 5.0),
    n_steps: int = 50,
    entropy: int = 42,
    method: str = "euler",
) -> torch.Tensor:
    """Integrate the SDE once with a fixed Brownian seed.

    Returns the full trajectory tensor of shape ``(n_steps + 1, *y0.shape)``.
    Use ``[-1]`` for the terminal state. ``method="euler"`` is the
    Euler-Maruyama scheme — the canonical first-order method for Itô SDEs.

    The Brownian motion is constructed from ``entropy``. Two calls with the
    same ``entropy`` produce bit-identical trajectories.

    *Implementation note*
    ---------------------

    ``torchsde.sdeint`` has a large one-time setup cost (~280 s on the canonical
    WN18RR perturbed pool of size 40,943×50, measured 2026-05-16 during the
    perf milestone). The integration loop is plain Euler-Maruyama, so we
    inline it here using ``BrownianInterval`` for the noise. This bypasses
    the wrapper overhead while keeping the same noise realization, dt, and
    drift/diffusion semantics as a ``torchsde.sdeint(sde, ..., method="euler",
    bm=bm)`` call.
    """
    if method != "euler":
        # Fall back to torchsde for non-Euler methods (kept for completeness;
        # the canonical run uses Euler-Maruyama).
        ts = torch.linspace(t_span[0], t_span[1], n_steps + 1, device=y0.device)
        bm = torchsde.BrownianInterval(
            t0=float(t_span[0]), t1=float(t_span[1]), size=y0.shape,
            entropy=int(entropy), device=y0.device, dtype=y0.dtype,
        )
        return torchsde.sdeint(sde, y0, ts, method=method, bm=bm)

    # Manual Euler-Maruyama loop (the fast path).
    t0_v = float(t_span[0])
    t1_v = float(t_span[1])
    dt = (t1_v - t0_v) / n_steps
    bm = torchsde.BrownianInterval(
        t0=t0_v, t1=t1_v, size=y0.shape, entropy=int(entropy),
        device=y0.device, dtype=y0.dtype,
    )
    ys = torch.empty((n_steps + 1, *y0.shape), dtype=y0.dtype, device=y0.device)
    ys[0] = y0
    y = y0.clone()
    t = t0_v
    with torch.no_grad():
        for i in range(n_steps):
            t_next = t + dt
            drift = sde.f(t, y)
            diffusion = sde.g(t, y)
            dW = bm(t, t_next)
            y = y + drift * dt + diffusion * dW
            ys[i + 1] = y
            t = t_next
    return ys


def compute_per_triple_variance(
    sde: KGSDESystem,
    triples,
    k_trajectories: int = 20,
    t_span: Tuple[float, float] = (0.0, 5.0),
    n_steps: int = 50,
    seed_base: int = 42,
    method: str = "euler",
    verbose: bool = False,
) -> torch.Tensor:
    """Per-triple variance observable over ``k`` stochastic trajectories.

    For each triple ``(h, r, t)`` in ``triples`` and for each of ``k``
    trajectories, integrate the SDE forward and measure the *terminal*
    state of the entities involved (``h`` and ``t``). Then compute the
    cross-trajectory variance of the per-triple energy::

        E_i(h, r, t) = || h_i(T) + r - t_i(T) ||^2     (the same data energy
                                                       used by the ODE)
        V(h, r, t)  = Var_i [ E_i(h, r, t) ]            (variance over k)

    The hypothesis under Fokker-Planck: ``V`` should depend on ``lambda_asym``
    where the deterministic detection-F1 observable did not.

    Implementation note
    -------------------

    We integrate **once** from the shared ``y0`` of the SDE (the initial
    embedding matrix) per trajectory ``k``, then index into the terminal state
    by ``(h_idx, t_idx)`` per triple. This is O(k * n_steps) ODE evaluations
    total, not O(n_triples * k * n_steps). Triples share trajectories — each
    trajectory is a *single integration of the whole system* under one
    Brownian realisation.

    Args:
        sde: instantiated ``KGSDESystem`` wrapping a ``KGODESystem``.
        triples: iterable of ``Triple`` namedtuples. We use ``triple.subject``
            and ``triple.object`` to index into the SDE's ``ode.entity_to_idx``
            map; triples whose endpoints are unknown to the ODE are skipped
            (with a ``NaN`` in the corresponding output position).
        k_trajectories: number of independent SDE realisations.
        t_span: integration window in pseudo-time.
        n_steps: number of solver steps (Euler-Maruyama).
        seed_base: base entropy for ``BrownianInterval``. Trajectory ``k``
            uses ``entropy = seed_base * 10_000 + k``.
        method: torchsde solver method. ``"euler"`` recommended for Itô
            SDEs with additive diagonal noise.

    Returns:
        Tensor of shape ``(n_triples,)`` with the per-triple energy variance
        across the ``k`` trajectories. ``NaN`` for triples with unknown
        endpoints.
    """
    if k_trajectories < 2:
        raise ValueError(f"k_trajectories must be >= 2 (need variance), got {k_trajectories}")
    if t_span[1] <= t_span[0]:
        raise ValueError(f"t_span end must be > start, got {t_span}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    import time as _time
    # Collect the terminal state of every entity for each of the k trajectories.
    # Shape after stacking: (k, N_entities, D).
    terminals = []
    t_start = _time.time()
    for k_idx in range(k_trajectories):
        ent = int(seed_base) * 10_000 + int(k_idx)
        if verbose:
            print(f"    [sde] integrating trajectory {k_idx + 1}/{k_trajectories} "
                  f"(entropy={ent}, elapsed_total={_time.time() - t_start:.1f}s)",
                  flush=True)
        t_k = _time.time()
        ys = integrate_one_trajectory(
            sde, sde.ode.y0, t_span=t_span, n_steps=n_steps,
            entropy=ent, method=method,
        )
        if verbose:
            print(f"    [sde] trajectory {k_idx + 1} done in {_time.time() - t_k:.1f}s",
                  flush=True)
        terminals.append(ys[-1])  # (N_entities, D)
    terminals = torch.stack(terminals, dim=0)  # (k, N_entities, D)

    # Map triples to entity indices using the ODE's pre-built map.
    entity_to_idx = sde.ode.entity_to_idx
    r_vec_all = sde.ode.r_vectors  # (T_pool, D) — relation embeddings per pool triple

    # The KGODESystem keeps its training-pool triple list at self.triples
    # so we can rebuild a (s, r, o) → pool_index map to recover relation
    # embeddings for the eval set. Eval triples that aren't in the pool
    # return NaN (caller decides how to handle).
    eval_triples_known = getattr(sde.ode, "triples", None)
    if eval_triples_known is None:
        # Can't recover relations. Caller must pass a relation dict.
        raise RuntimeError(
            "KGSDESystem requires the wrapped KGODESystem to expose "
            "its triples list (self.triples). Cannot recover relation "
            "embeddings for arbitrary evaluation triples."
        )

    # Build (s, r, o) → index map for the ODE's known triples
    known_index = {}
    for i, t in enumerate(eval_triples_known):
        key = (t.subject, t.relation, t.object)
        known_index[key] = i

    variances = []
    for triple in triples:
        key = (triple.subject, triple.relation, triple.object)
        i = known_index.get(key)
        if i is None or triple.subject not in entity_to_idx or triple.object not in entity_to_idx:
            variances.append(float("nan"))
            continue
        s_idx = entity_to_idx[triple.subject]
        o_idx = entity_to_idx[triple.object]
        r_vec = r_vec_all[i]  # (D,)
        # Per-trajectory energy: || h(T) + r - t(T) ||^2
        h_term = terminals[:, s_idx, :]  # (k, D)
        t_term = terminals[:, o_idx, :]  # (k, D)
        residuals = h_term + r_vec.unsqueeze(0) - t_term  # (k, D)
        energies = (residuals ** 2).sum(dim=-1)  # (k,)
        var_k = energies.var(unbiased=False).item()
        variances.append(var_k)

    return torch.tensor(variances, dtype=torch.float32, device="cpu")
