from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from lp_precondition import PreconditionerData
from mps_process import LPData
from operations import (
    compute_duality_gap,
    estimate_spectral_norm,
    proj_box,
    proj_dual,
    project_lambda,
    weighted_norm,
    weighted_norm_sq,
)


@dataclass
class PDLPResult:
    """Outputs collected from running PDLP."""

    x_scaled: np.ndarray
    y_scaled: np.ndarray
    x: np.ndarray
    y: np.ndarray
    objective: float
    dual_objective: float
    duality_gap: float
    primal_residual: float
    dual_residual: float
    iterations: int
    outer_iterations: int
    converged: bool
    k_multiplications: int
    objective_history: List[float] = field(default_factory=list)
    dual_objective_history: List[float] = field(default_factory=list)
    duality_gap_history: List[float] = field(default_factory=list)
    primal_residual_history: List[float] = field(default_factory=list)
    dual_residual_history: List[float] = field(default_factory=list)
    objective_distance_trace: List[Tuple[int, float]] = field(default_factory=list)


def pdlp(
    lp: LPData,
    *,
    max_outer: int = 50,
    max_inner: int = 2000,
    tol: float = 1e-6,
    beta_params: Optional[Dict[str, float]] = None,
    eta0_scale: float = 1.0,
    theta_weight: float = 0.5,
    eps_zero: float = 1e-12,
    check_every: int = 50,
    objective_stride: int = 0,
    reference_objective: Optional[float] = None,
    precond: Optional[PreconditionerData] = None,
) -> PDLPResult:
    """Run PDLP with adaptive modules."""
    if max_outer <= 0 or max_inner <= 0:
        raise ValueError("max_outer and max_inner must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if objective_stride < 0:
        raise ValueError("objective_stride must be non-negative.")
    if objective_stride > 0 and reference_objective is None:
        raise ValueError("reference_objective is required when objective_stride > 0.")

    n = lp.n_vars
    m_total = lp.m_ineq + lp.m_eq

    if lp.m_ineq and lp.m_eq:
        K = np.vstack([lp.G, lp.A])
    elif lp.m_ineq:
        K = lp.G.copy()
    elif lp.m_eq:
        K = lp.A.copy()
    else:
        K = np.zeros((0, n), dtype=float)

    q = np.concatenate([lp.h, lp.b]) if m_total else np.zeros((0,), dtype=float)

    tracker = _KMultiplicationTracker(
        lp=lp,
        reference_objective=reference_objective,
        stride=objective_stride,
    )

    # Step size initialization
    if K.size:
        spectral_norm = estimate_spectral_norm(K, matvec_callback=lambda: tracker.bump())
        norm_inf = np.linalg.norm(K, ord=np.inf)
        norm_bound = max(spectral_norm, norm_inf, 1e-6)
        eta_max = 1.0 / max(spectral_norm, 1e-6)
    else:
        norm_bound = 1.0
        eta_max = 1.0
    eta_hat = min(eta0_scale / norm_bound, eta_max)

    # Primal weight initialization
    omega = _initialize_primal_weight(lp.c, q, eps_zero)

    x = np.clip(np.zeros(n, dtype=float), lp.lower, lp.upper)
    y = np.zeros(m_total, dtype=float)

    def apply_K(vec: np.ndarray, *, sample_primal: Optional[np.ndarray] = None) -> np.ndarray:
        if not m_total:
            return np.zeros((0,), dtype=float)
        result = K @ vec
        tracker.bump(sample_primal)
        return result

    def apply_K_transpose(
        vec: np.ndarray, *, sample_primal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not m_total:
            return np.zeros(n, dtype=float)
        result = K.T @ vec
        tracker.bump(sample_primal)
        return result

    z_curr = (x.copy(), y.copy())
    z_prev_start = (x.copy(), y.copy())

    beta_values = _default_beta_params() if beta_params is None else _validate_beta_params(beta_params)

    objective_history: List[float] = []
    dual_objective_history: List[float] = []
    duality_gap_history: List[float] = []
    primal_residual_history: List[float] = []
    dual_residual_history: List[float] = []

    converged = False
    iterations = 0
    outer_iterations = 0

    for epoch in range(max_outer):
        outer_iterations = epoch + 1
        t = 0
        accumulated_eta = 0.0
        z_start = (z_curr[0].copy(), z_curr[1].copy())
        z_avg = (z_curr[0].copy(), z_curr[1].copy())
        z_candidate_prev = z_start
        mu_candidate_prev = np.inf

        mu_prev_epoch = (
            _normalized_gap(
                lp,
                z_start,
                z_prev_start,
                omega,
                eps_zero,
                tracker=tracker,
                K=K,
                q=q,
            )
            if epoch > 0
            else np.inf
        )

        while t < max_inner:
            iterations += 1
            z_next, eta_accepted, eta_hat = _adaptive_step_of_pdhg(
                lp=lp,
                q=q,
                omega=omega,
                eta_hat=eta_hat,
                eta_max=eta_max,
                k=iterations - 1,
                z=z_curr,
                apply_K=apply_K,
                apply_K_transpose=apply_K_transpose,
            )

            accumulated_eta += eta_accepted
            if accumulated_eta > 0.0:
                weight = eta_accepted / accumulated_eta
                z_avg = (
                    (1.0 - weight) * z_avg[0] + weight * z_next[0],
                    (1.0 - weight) * z_avg[1] + weight * z_next[1],
                )
            else:
                z_avg = (z_next[0].copy(), z_next[1].copy())

            z_candidate = _get_restart_candidate(
                lp=lp,
                z_new=z_next,
                z_avg=z_avg,
                z_start=z_start,
                omega=omega,
                eps_zero=eps_zero,
                tracker=tracker,
                K=K,
                q=q,
            )
            mu_candidate = _normalized_gap(
                lp,
                z_candidate,
                z_start,
                omega,
                eps_zero,
                tracker=tracker,
                K=K,
                q=q,
            )

            if iterations % check_every == 0:
                metrics = _evaluate_metrics(
                    lp,
                    z_candidate,
                    tol,
                    eps_zero,
                    apply_K=apply_K,
                    apply_K_transpose=apply_K_transpose,
                    tracker=tracker,
                    K=K,
                    q=q,
                )
                objective_history.append(metrics.objective)
                dual_objective_history.append(metrics.dual_objective)
                duality_gap_history.append(metrics.duality_gap)
                primal_residual_history.append(metrics.primal_residual)
                dual_residual_history.append(metrics.dual_residual)
                if metrics.converged:
                    converged = True
                    z_curr = z_candidate
                    break

            restart = _should_restart(
                mu_candidate=mu_candidate,
                mu_prev_epoch=mu_prev_epoch,
                mu_candidate_prev=mu_candidate_prev,
                t=t + 1,
                k=iterations,
                beta_params=beta_values,
            )

            z_curr = z_next
            t += 1

            if restart:
                z_candidate_prev = z_candidate
                mu_candidate_prev = mu_candidate
                break

            z_candidate_prev = z_candidate
            mu_candidate_prev = mu_candidate

        if converged:
            break

        z_prev_start = z_start
        z_curr = z_candidate_prev
        omega = _primal_weight_update(
            z_curr=z_curr,
            z_prev=z_prev_start,
            omega_prev=omega,
            theta=theta_weight,
            eps_zero=eps_zero,
        )

    final_metrics = _evaluate_metrics(
        lp,
        z_curr,
        tol,
        eps_zero,
        apply_K=apply_K,
        apply_K_transpose=apply_K_transpose,
        tracker=tracker,
        K=K,
        q=q,
    )

    if not objective_history:
        objective_history.append(final_metrics.objective)
        dual_objective_history.append(final_metrics.dual_objective)
        duality_gap_history.append(final_metrics.duality_gap)
        primal_residual_history.append(final_metrics.primal_residual)
        dual_residual_history.append(final_metrics.dual_residual)

    x_scaled = z_curr[0]
    y_scaled = z_curr[1]
    if precond is not None:
        x = precond.to_primal(x_scaled)
        y = precond.to_dual(y_scaled)
    else:
        x = x_scaled.copy()
        y = y_scaled.copy()

    return PDLPResult(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        x=x,
        y=y,
        objective=final_metrics.objective,
        dual_objective=final_metrics.dual_objective,
        duality_gap=final_metrics.duality_gap,
        primal_residual=final_metrics.primal_residual,
        dual_residual=final_metrics.dual_residual,
        iterations=iterations,
        outer_iterations=outer_iterations,
        converged=final_metrics.converged,
        k_multiplications=tracker.count,
        objective_history=objective_history,
        dual_objective_history=dual_objective_history,
        duality_gap_history=duality_gap_history,
        primal_residual_history=primal_residual_history,
        dual_residual_history=dual_residual_history,
        objective_distance_trace=tracker.samples,
    )


def _adaptive_step_of_pdhg(
    *,
    lp: LPData,
    q: np.ndarray,
    omega: float,
    eta_hat: float,
    eta_max: float,
    k: int,
    z: Tuple[np.ndarray, np.ndarray],
    apply_K: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    apply_K_transpose: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
    """Module 1: safeguarded adaptive PDHG step."""
    x, y = z
    eta = min(eta_hat, eta_max)
    m_total = y.size

    while True:
        grad = lp.c - apply_K_transpose(y, sample_primal=None) if m_total else lp.c
        x_next = proj_box(x - (eta / omega) * grad, lp.lower, lp.upper)
        x_bar = 2.0 * x_next - x

        if m_total:
            y_next = proj_dual(
                y + (eta * omega) * (q - apply_K(x_bar, sample_primal=x_next)),
                lp.m_ineq,
            )
        else:
            y_next = y.copy()

        delta_x = x_next - x
        delta_y = y_next - y
        numerator = weighted_norm_sq(delta_x, delta_y, omega)
        if numerator <= 0.0:
            # No movement; accept current iterate with unchanged step size.
            return (x_next, y_next), eta, eta

        denom = float(
            delta_y @ apply_K(delta_x, sample_primal=None)
        ) if m_total else 0.0
        if denom <= 0.0:
            bar_eta = np.inf
        else:
            bar_eta = numerator / (2.0 * denom)

        grow_candidate = min((1.0 + (k + 1) ** (-0.6)) * eta, eta_max)
        if np.isfinite(bar_eta) and bar_eta > 0.0:
            shrink_candidate = (1.0 - (k + 1) ** (-0.3)) * bar_eta
            shrink_candidate = max(shrink_candidate, 1e-12)
            eta_next = min(shrink_candidate, grow_candidate)
        else:
            eta_next = grow_candidate

        if not np.isfinite(bar_eta) or bar_eta > 0.0 and eta <= bar_eta + 1e-12:
            return (x_next, y_next), eta, min(eta_next, eta_max)

        eta = max(min(eta_next, eta_max), 1e-12)


def _get_restart_candidate(
    *,
    lp: LPData,
    z_new: Tuple[np.ndarray, np.ndarray],
    z_avg: Tuple[np.ndarray, np.ndarray],
    z_start: Tuple[np.ndarray, np.ndarray],
    omega: float,
    eps_zero: float,
    tracker: "_KMultiplicationTracker",
    K: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Module 2: choose restart candidate."""
    mu_new = _normalized_gap(
        lp,
        z_new,
        z_start,
        omega,
        eps_zero,
        tracker=tracker,
        K=K,
        q=q,
    )
    mu_avg = _normalized_gap(
        lp,
        z_avg,
        z_start,
        omega,
        eps_zero,
        tracker=tracker,
        K=K,
        q=q,
    )
    if mu_new < mu_avg:
        return z_new[0].copy(), z_new[1].copy()
    return z_avg[0].copy(), z_avg[1].copy()


def _primal_weight_update(
    *,
    z_curr: Tuple[np.ndarray, np.ndarray],
    z_prev: Tuple[np.ndarray, np.ndarray],
    omega_prev: float,
    theta: float,
    eps_zero: float,
) -> float:
    """Module 3: update primal weight."""
    delta_x = np.linalg.norm(z_curr[0] - z_prev[0])
    delta_y = np.linalg.norm(z_curr[1] - z_prev[1])
    if delta_x <= eps_zero or delta_y <= eps_zero:
        return omega_prev

    ratio = delta_y / max(delta_x, eps_zero)
    return float(np.exp(theta * np.log(ratio) + (1.0 - theta) * np.log(max(omega_prev, eps_zero))))


def _normalized_gap(
    lp: LPData,
    z: Tuple[np.ndarray, np.ndarray],
    z_ref: Tuple[np.ndarray, np.ndarray],
    omega: float,
    eps_zero: float,
    *,
    tracker: "_KMultiplicationTracker",
    K: np.ndarray,
    q: np.ndarray,
) -> float:
    """Normalized duality gap heuristic."""
    _, _, gap = compute_duality_gap(
        lp,
        z[0],
        z[1],
        omega=1.0,
        K=K,
        q=q,
        matvec_callback=lambda: tracker.bump(z[0]),
    )
    radius = max(weighted_norm(z[0] - z_ref[0], z[1] - z_ref[1], omega), eps_zero)
    return abs(gap) / radius


def _should_restart(
    *,
    mu_candidate: float,
    mu_prev_epoch: float,
    mu_candidate_prev: float,
    t: int,
    k: int,
    beta_params: Dict[str, float],
) -> bool:
    """Evaluate restart triggers."""
    sufficient_decay = mu_candidate <= beta_params["sufficient"] * mu_prev_epoch
    necessary_decay = (
        mu_candidate <= beta_params["necessary"] * mu_prev_epoch and mu_candidate > mu_candidate_prev
    )
    long_inner = t >= beta_params["artificial"] * k
    return sufficient_decay or necessary_decay or long_inner


def _evaluate_metrics(
    lp: LPData,
    z: Tuple[np.ndarray, np.ndarray],
    tol: float,
    eps_zero: float,
    *,
    apply_K: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    apply_K_transpose: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    tracker: "_KMultiplicationTracker",
    K: np.ndarray,
    q: np.ndarray,
):
    x, y = z
    if lp.m_ineq:
        ineq_violation = np.minimum(lp.G @ x - lp.h, 0.0)
    else:
        ineq_violation = np.zeros(0, dtype=float)

    if lp.m_eq:
        eq_violation = lp.A @ x - lp.b
    else:
        eq_violation = np.zeros(0, dtype=float)

    residual_vec = np.concatenate([eq_violation, ineq_violation]) if eq_violation.size + ineq_violation.size else np.zeros(0)
    primal_residual = float(np.linalg.norm(residual_vec, ord=2)) if residual_vec.size else 0.0

    if K.size:
        reduced_cost = lp.c - apply_K_transpose(y, sample_primal=x)
    else:
        reduced_cost = lp.c.copy()
    lam = project_lambda(reduced_cost, lp.lower, lp.upper)
    dual_residual_vec = lp.c - (apply_K_transpose(y, sample_primal=x) if K.size else 0.0) - lam
    if isinstance(dual_residual_vec, np.ndarray):
        dual_residual = float(np.linalg.norm(dual_residual_vec, ord=2))
    else:
        dual_residual = float(np.linalg.norm(lp.c - lam, ord=2))

    primal_obj, dual_obj, gap = compute_duality_gap(
        lp,
        x,
        y,
        omega=1.0,
        K=K,
        q=q,
        matvec_callback=lambda: tracker.bump(x),
    )
    tol_scale = tol * (1.0 + abs(primal_obj) + abs(dual_obj))
    converged = abs(gap) <= tol_scale and primal_residual <= tol_scale and dual_residual <= tol_scale

    return _Metrics(
        objective=primal_obj,
        dual_objective=dual_obj,
        duality_gap=gap,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        converged=converged,
    )


@dataclass
class _Metrics:
    objective: float
    dual_objective: float
    duality_gap: float
    primal_residual: float
    dual_residual: float
    converged: bool


class _KMultiplicationTracker:
    """Track K-matrix multiplications and objective distance samples for PDLP."""

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
        if stride > 0 and reference_objective is not None:
            self._next_checkpoint: Optional[int] = stride
        else:
            self._next_checkpoint = None
        self.samples: List[Tuple[int, float]] = []

    def bump(self, primal: Optional[np.ndarray] = None, increments: int = 1) -> None:
        self.count += increments
        if self._next_checkpoint is None or primal is None or self.reference_objective is None:
            return
        while self.count >= self._next_checkpoint:
            objective = float(self.lp.c @ primal + self.lp.obj_offset)
            distance = abs(objective - self.reference_objective)
            self.samples.append((self._next_checkpoint, distance))
            self._next_checkpoint += self.stride


def _initialize_primal_weight(c: np.ndarray, q: np.ndarray, eps_zero: float) -> float:
    norm_q = np.linalg.norm(q)
    if norm_q > eps_zero:
        return float(np.linalg.norm(c) / max(norm_q, eps_zero))
    return 1.0


def _default_beta_params() -> Dict[str, float]:
    return {"sufficient": 0.9, "necessary": 0.1, "artificial": 0.5}


def _validate_beta_params(beta_params: Dict[str, float]) -> Dict[str, float]:
    keys = {"sufficient", "necessary", "artificial"}
    missing = keys - beta_params.keys()
    if missing:
        raise ValueError(f"Missing beta parameters: {missing}")
    return {key: float(beta_params[key]) for key in keys}


__all__ = ["PDLPResult", "pdlp"]

