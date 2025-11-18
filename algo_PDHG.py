from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from lp_precondition import PreconditionerData
from mps_process import LPData
from operations import proj_box, proj_dual


@dataclass
class PDHGResult:
    """Container for PDHG outputs."""

    x_scaled: np.ndarray
    y_scaled: np.ndarray
    x: np.ndarray
    y: np.ndarray
    objective: float
    primal_residual: float
    dual_residual: float
    iterations: int
    converged: bool
    k_multiplications: int
    objective_history: List[float] = field(default_factory=list)
    primal_residual_history: List[float] = field(default_factory=list)
    dual_residual_history: List[float] = field(default_factory=list)
    objective_distance_trace: List[Tuple[int, float]] = field(default_factory=list)


def pdhg(
    lp: LPData,
    *,
    max_K_multi: int,
    tol: float,
    omega: float = 1.0,
    eta_scale: float = 0.9,
    check_every: int = 100,
    objective_stride: int = 0,
    reference_objective: Optional[float] = None,
    precond: Optional[PreconditionerData] = None,
    power_iterations: int = 20,
) -> PDHGResult:
    """Run the Primal-Dual Hybrid Gradient algorithm on the provided LP."""
    n = lp.n_vars
    m_total = lp.m_ineq + lp.m_eq
    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    if eta_scale <= 0.0:
        raise ValueError("eta_scale must be positive.")
    if objective_stride < 0:
        raise ValueError("objective_stride must be non-negative.")
    if objective_stride > 0 and reference_objective is None:
        raise ValueError("reference_objective is required when objective_stride > 0.")

    if lp.m_ineq and lp.m_eq:
        K = np.vstack([lp.G, lp.A])
    elif lp.m_ineq:
        K = lp.G.copy()
    elif lp.m_eq:
        K = lp.A.copy()
    else:
        K = np.zeros((0, n), dtype=float)

    q = np.concatenate([lp.h, lp.b])

    tracker = _KMultiplicationTracker(
        lp=lp,
        reference_objective=reference_objective,
        stride=objective_stride,
    )

    # Use precomputed norm_2 from LPData for reproducibility
    # Norms are computed during problem loading with fixed seed
    # Note: power_iterations parameter is now ignored (fixed to 20 iterations during loading)
    norm_K = lp.norm_2
    denom = max(norm_K, 1e-8)
    eta = eta_scale / denom
    tau = eta / omega
    sigma = eta * omega

    x = np.clip(np.zeros(n, dtype=float), lp.lower, lp.upper)
    y = np.zeros(m_total, dtype=float)

    def apply_K(vec: np.ndarray, *, sample_primal: Optional[np.ndarray] = None) -> np.ndarray:
        result = K @ vec
        tracker.bump(sample_primal)
        return result

    def apply_K_transpose(
        vec: np.ndarray, *, sample_primal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        result = K.T @ vec
        tracker.bump(sample_primal)
        return result

    objective_history: List[float] = []
    primal_residual_history: List[float] = []
    dual_residual_history: List[float] = []

    converged = False
    iterations = 0

    k = 0
    while tracker.count < max_K_multi:
        k += 1
        x_prev = x.copy()
        grad = lp.c - apply_K_transpose(y, sample_primal=None) if m_total else lp.c
        x = proj_box(x_prev - tau * grad, lp.lower, lp.upper)
        x_bar = 2.0 * x - x_prev

        if m_total:
            y = proj_dual(
                y + sigma * (q - apply_K(x_bar, sample_primal=x)),
                lp.m_ineq,
            )

        # Check if we've exceeded max_K_multi after the optimization step
        if tracker.count >= max_K_multi:
            iterations = k
            break

        if k % check_every == 0:
            # Disable counting for residual computation (statistics)
            tracker.disable_counting()
            primal_residual, dual_residual = _compute_residuals(
                lp,
                x,
                y,
                lambda vec: apply_K_transpose(vec, sample_primal=x),
            )
            tracker.enable_counting()
            primal_residual_history.append(primal_residual)
            dual_residual_history.append(dual_residual)
            objective = float(lp.c @ x + lp.obj_offset)
            objective_history.append(objective)
            if max(primal_residual, dual_residual) <= tol:
                converged = True
                iterations = k
                break

    if not objective_history:
        objective = float(lp.c @ x + lp.obj_offset)
        objective_history.append(objective)
        # Disable counting for final residual computation
        tracker.disable_counting()
        primal_residual, dual_residual = _compute_residuals(
            lp,
            x,
            y,
            lambda vec: apply_K_transpose(vec, sample_primal=x),
        )
        tracker.enable_counting()
        primal_residual_history.append(primal_residual)
        dual_residual_history.append(dual_residual)
    else:
        objective = objective_history[-1]
        primal_residual = primal_residual_history[-1]
        dual_residual = dual_residual_history[-1]

    x_scaled = x.copy()
    y_scaled = y.copy()
    if precond is not None:
        x = precond.to_primal(x_scaled)
        y = precond.to_dual(y_scaled)

    return PDHGResult(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        x=x,
        y=y,
        objective=objective,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        iterations=iterations,
        converged=converged,
        k_multiplications=tracker.count,
        objective_history=objective_history,
        primal_residual_history=primal_residual_history,
        dual_residual_history=dual_residual_history,
        objective_distance_trace=tracker.samples,
    )


def _compute_residuals(
    lp: LPData,
    x: np.ndarray,
    y: np.ndarray,
    apply_k_transpose: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float]:
    if lp.m_ineq:
        ineq_violation = np.minimum(lp.G @ x - lp.h, 0.0)
        primal_ineq = float(np.linalg.norm(ineq_violation, ord=np.inf))
    else:
        primal_ineq = 0.0

    if lp.m_eq:
        eq_violation = lp.A @ x - lp.b
        primal_eq = float(np.linalg.norm(eq_violation, ord=np.inf))
    else:
        primal_eq = 0.0

    primal_residual = max(primal_ineq, primal_eq)

    if y.size:
        dual_residual = float(np.linalg.norm(lp.c - apply_k_transpose(y), ord=np.inf))
    else:
        dual_residual = float(np.linalg.norm(lp.c, ord=np.inf))

    return primal_residual, dual_residual


class _KMultiplicationTracker:
    """Track K-matrix multiplications and objective distance samples."""

    def __init__(
        self,
        *,
        lp: LPData,
        reference_objective: Optional[float],
        stride: int,
    ) -> None:
        self.lp = lp
        self.reference_objective = reference_objective
        self.stride = stride
        self.count = 0
        self._count_enabled = True  # Flag to enable/disable counting
        if stride > 0 and reference_objective is not None:
            self._next_checkpoint: Optional[int] = stride
        else:
            self._next_checkpoint = None
        self.samples: List[Tuple[int, float]] = []

    def bump(self, primal: Optional[np.ndarray] = None, increments: int = 1) -> None:
        if self._count_enabled:
            self.count += increments
        if self._next_checkpoint is None or primal is None or self.reference_objective is None:
            return
        while self.count >= self._next_checkpoint:
            objective = float(self.lp.c @ primal + self.lp.obj_offset)
            distance = abs(objective - self.reference_objective)
            self.samples.append((self._next_checkpoint, distance))
            self._next_checkpoint += self.stride

    def disable_counting(self) -> None:
        """Disable counting of K-multiplications (for statistics/convergence checks)."""
        self._count_enabled = False

    def enable_counting(self) -> None:
        """Enable counting of K-multiplications (for optimization steps)."""
        self._count_enabled = True


__all__ = ["PDHGResult", "pdhg"]
