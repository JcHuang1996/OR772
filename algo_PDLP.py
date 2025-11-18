from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Callable, Dict, List, Optional, Tuple

import cvxpy as cp
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
    max_K_multi: int = 100000,
    max_outer: int = 1000000,  # Large default to avoid bounding outer loops
    max_inner: int = 1000000,  # Large default (effectively unbounded)
    tol: float = 1e-6,
    beta_params: Optional[Dict[str, float]] = None,
    kkt_params: Optional[Dict[str, float]] = None,
    eta0_scale: float = 1.0,
    theta_weight: float = 0.5,
    eps_zero: float = 1e-12,
    check_every: int = 100,
    objective_stride: int = 0,
    reference_objective: Optional[float] = None,
    precond: Optional[PreconditionerData] = None,
    restart_mode: str = "normalized_gap",  # "normalized_gap" or "kkt"
    enable_adaptive_step: bool = True,
    enable_primal_weight: bool = True,
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
    if restart_mode not in ("normalized_gap", "kkt"):
        raise ValueError('restart_mode must be "normalized_gap" or "kkt".')

    # initialize necessary matrix, vector, etc.
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

    q = np.concatenate([lp.h, lp.b])  # m_total is always > 0 by problem structure

    # initialize tracker for the number of K-matrix multiplications
    tracker = _KMultiplicationTracker(
        lp=lp,
        reference_objective=reference_objective,
        stride=objective_stride,
    )

    # Step size initialization - disable counting for spectral norm estimation
    # Cache spectral norm to avoid recomputing it multiple times
    tracker.disable_counting()
    spectral_norm = estimate_spectral_norm(K, matvec_callback=lambda: tracker.bump())
    norm_inf = np.linalg.norm(K, ord=np.inf)
    norm_bound = max(spectral_norm, norm_inf, 1e-6)
    eta_hat = eta0_scale / norm_bound # eta0_scale = 1 in the original implementation

    # Compute fixed step size for non-adaptive mode (0.9 / 2-norm of K)
    # Reuse cached spectral_norm instead of recomputing
    norm_K_2 = spectral_norm
    eta_fixed = 0.9 / max(norm_K_2, 1e-8)
    tracker.enable_counting()

    # Primal weight initialization
    if enable_primal_weight:
        omega = _initialize_primal_weight(lp.c, q, eps_zero)
    else:
        omega = 1.0

    x = np.clip(np.zeros(n, dtype=float), lp.lower, lp.upper)
    y = np.zeros(m_total, dtype=float)

    # define the K-matrix multiplication and its transpose. Sample_primal is only for tracking progress, not regarding the actual computation.
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

    # initialize parameters, variables, and trackers
    z_curr = (x.copy(), y.copy())
    z_prev_start = (x.copy(), y.copy())

    beta_values = _default_beta_params() if beta_params is None else _validate_beta_params(beta_params)
    kkt_values = _default_kkt_params() if kkt_params is None else _validate_kkt_params(kkt_params)

    objective_history: List[float] = []
    dual_objective_history: List[float] = []
    duality_gap_history: List[float] = []
    primal_residual_history: List[float] = []
    dual_residual_history: List[float] = []

    # initialize convergence flags and counters
    converged = False
    iterations = 0
    outer_iterations = 0

    # compute the backup step size for the inner iteration if the denominator is < 0 or the numerator is 0
    # Reuse cached spectral_norm instead of recomputing
    norm_K = spectral_norm  # Reuse cached value
    denom_backup = max(norm_K, 1e-8)
    eta_backup = 0.9 / denom_backup


    for epoch in range(max_outer):

        # start of the outer iteration
        outer_iterations = epoch + 1
        t = 0
        accumulated_eta = 0.0
        z_start = (z_curr[0].copy(), z_curr[1].copy())
        z_avg = (z_curr[0].copy(), z_curr[1].copy())
        z_candidate_prev = z_start
        mu_candidate_prev = np.inf
        K_candidate_prev = np.inf

        # computing the \mu_n(z^{n,0}, z^{n-1,0}) in the paper.
        # i.e. the normalized gap between the current and previous starting points of the outer iteration
        # Disable counting for restart metric computation (statistics)
        tracker.disable_counting()
        if restart_mode == "normalized_gap":
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
                else np.inf         # by the definition, if epoch = 0 ,then r = 0, so mu_prev_epoch = inf
            )
            K0 = np.inf
        else:
            # KKT-based restart: compute K0 for this epoch
            K0 = _kkt_error(lp, z_start, omega, K=K, q=q)
            mu_prev_epoch = np.inf
        tracker.enable_counting()

        while t < max_inner:    # typically, the max_inner should not be reached due to the long inner loop restart criteria. Just in case.
            # Check max_K_multi before optimization step (each step does 2 K-mults: K and K.T)
            if tracker.count >= max_K_multi:
                break
            
            # start of the inner iteration
            iterations += 1
            if enable_adaptive_step:
                z_next, eta_accepted, eta_hat = _adaptive_step_of_pdhg(
                    lp=lp,
                    q=q,
                    omega=omega,
                    eta_hat=eta_hat,
                    k=iterations,
                    z=z_curr,
                    apply_K=apply_K,
                    apply_K_transpose=apply_K_transpose,
                    eta_backup=eta_backup
                )
            else:
                z_next, eta_accepted, eta_hat = _fixed_step_of_pdhg(
                    lp=lp,
                    q=q,
                    omega=omega,
                    eta=eta_fixed,
                    z=z_curr,
                    apply_K=apply_K,
                    apply_K_transpose=apply_K_transpose,
                )
            
            # Check max_K_multi right after optimization step (in case we went slightly over)
            if tracker.count >= max_K_multi:
                z_curr = z_next
                converged = False  # Mark as not converged since we hit the limit
                break
            
            # Minimal tracking hook for accepted steps (enable by setting env PDLP_DEBUG_STEP=1)
            if os.environ.get("PDLP_DEBUG_STEP", ""):
                try:
                    dx_norm = float(np.linalg.norm(z_next[0] - z_curr[0]))
                    dy_norm = float(np.linalg.norm(z_next[1] - z_curr[1]))
                    x_norm = float(np.linalg.norm(z_next[0]))
                    y_norm = float(np.linalg.norm(z_next[1]))
                    print(
                        f"[PDLP step] k={iterations-1} eta={eta_accepted:.3e} "
                        f"eta_hat_next={eta_hat:.3e} ||dx||={dx_norm:.3e} ||dy||={dy_norm:.3e} "
                        f"||x||={x_norm:.3e} ||y||={y_norm:.3e}"
                    )
                except Exception:
                    # Do not interrupt the algorithm due to debugging
                    pass

            # update the average z. accumulated_eta updated here for computing the eta_accepted faster.
            accumulated_eta += eta_accepted
            if accumulated_eta > 0.0:
                weight = eta_accepted / accumulated_eta
                z_avg = (
                    (1.0 - weight) * z_avg[0] + weight * z_next[0],
                    (1.0 - weight) * z_avg[1] + weight * z_next[1],
                )
            else:
                z_avg = (z_next[0].copy(), z_next[1].copy())

            # get the restart candidate
            # Disable counting for restart candidate computation (statistics)
            tracker.disable_counting()
            if restart_mode == "normalized_gap":
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
            else:
                z_candidate = _get_restart_candidate_kkt(
                    lp=lp,
                    z_new=z_next,
                    z_avg=z_avg,
                    omega=omega,
                    K=K,
                    q=q,
                )
            tracker.enable_counting()

            # compute the restart metric for decision
            # Disable counting for restart metric computation (statistics)
            tracker.disable_counting()
            if restart_mode == "normalized_gap":
                # mu_candidate is \mu_n(z_{c}^{n,t+1}, z^{n,0}) in the paper
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
                K_c = np.inf
            else:
                # KKT-based metric
                K_c = _kkt_error(lp, z_candidate, omega, K=K, q=q)
                mu_candidate = np.inf
            tracker.enable_counting()

            if iterations % check_every == 0:
                # Disable counting for metrics evaluation (statistics/convergence)
                tracker.disable_counting()
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
                tracker.enable_counting()
                objective_history.append(metrics.objective)
                dual_objective_history.append(metrics.dual_objective)
                duality_gap_history.append(metrics.duality_gap)
                primal_residual_history.append(metrics.primal_residual)
                dual_residual_history.append(metrics.dual_residual)
                if metrics.converged:
                    converged = True
                    z_curr = z_candidate
                    break

            if restart_mode == "normalized_gap":
                restart = _should_restart(
                    mu_candidate=mu_candidate,
                    mu_prev_epoch=mu_prev_epoch,
                    mu_candidate_prev=mu_candidate_prev,
                    t=t + 1,
                    k=iterations,
                    beta_params=beta_values,
                )
            else:
                restart = _should_restart_kkt(
                    K0=K0,
                    K_c=K_c,
                    K_last=K_candidate_prev,
                    t=t + 1,
                    k=iterations,
                    theta_params=kkt_values,
                )

            z_curr = z_next
            t += 1

            if restart:
                z_candidate_prev = z_candidate
                if restart_mode == "normalized_gap":
                    mu_candidate_prev = mu_candidate
                else:
                    K_candidate_prev = K_c
                break

            z_candidate_prev = z_candidate
            if restart_mode == "normalized_gap":
                mu_candidate_prev = mu_candidate
            else:
                K_candidate_prev = K_c
        
        # Check max_K_multi after inner loop (in case we exited due to restart)
        if tracker.count >= max_K_multi:
            break

        if converged:
            break

        z_prev_start = z_start
        z_curr = z_candidate_prev
        if enable_primal_weight:
            omega = _primal_weight_update(
                z_curr=z_curr,
                z_prev=z_prev_start,
                omega_prev=omega,
                theta=theta_weight,
                eps_zero=eps_zero,
            )

    # Disable counting for final metrics evaluation (statistics)
    tracker.disable_counting()
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
    tracker.enable_counting()

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
    k: int,
    z: Tuple[np.ndarray, np.ndarray],
    apply_K: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    apply_K_transpose: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    eta_backup: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
    """Module 1: safeguarded adaptive PDHG step."""
    x, y = z
    eta = eta_hat

    internal_i = 0
    while internal_i < 100000: # prevent infinite loop. The limit should not be reached when the algo works normally.
        internal_i += 1
        grad = lp.c - apply_K_transpose(y, sample_primal=None)
        x_next = proj_box(x - (eta / omega) * grad, lp.lower, lp.upper)
        x_bar = 2.0 * x_next - x

        y_next = proj_dual(
            y + (eta * omega) * (q - apply_K(x_bar, sample_primal=None)),
            lp.m_ineq,
        )

        delta_x = x_next - x
        delta_y = y_next - y
        numerator = weighted_norm_sq(delta_x, delta_y, omega)

        # In the Julia implementation, "movement" is 0.5 * weighted_norm_sq
        # and "interaction" is the curvature term. For our LP case (no Q),
        # interaction = delta_y' * K * delta_x. When interaction <= 0,
        # they set step_size_limit = Inf and always accept the step.
        interaction = abs(float(delta_y @ apply_K(delta_x, sample_primal=None)))
        if interaction <= 0.0 or numerator <= 0.0:
            # Corresponds to interaction <= 0 in Julia: treat as
            # step_size_limit = Inf, i.e., bar_eta = Inf.
            bar_eta = np.inf
        else:
            # movement = 0.5 * numerator, interaction = denom
            # so bar_eta = movement / interaction = numerator / (2 * denom)
            bar_eta = numerator / (2.0 * interaction)

        grow_candidate = (1.0 + (k + 1) ** (-0.6)) * eta
        if np.isfinite(bar_eta):
            shrink_candidate = (1.0 - (k + 1) ** (-0.3)) * bar_eta
            eta_next = min(shrink_candidate, grow_candidate)
        else:
            eta_next = grow_candidate

        if not np.isfinite(bar_eta) or (eta <= bar_eta):
            return (x_next, y_next), eta, eta_next

        eta = float(eta_next)


def _fixed_step_of_pdhg(
    *,
    lp: LPData,
    q: np.ndarray,
    omega: float,
    eta: float,
    z: Tuple[np.ndarray, np.ndarray],
    apply_K: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    apply_K_transpose: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
    """Fixed step size PDHG step (used when adaptive step is disabled)."""
    x, y = z
    grad = lp.c - apply_K_transpose(y, sample_primal=None)
    x_next = proj_box(x - (eta / omega) * grad, lp.lower, lp.upper)
    x_bar = 2.0 * x_next - x
    y_next = proj_dual(
        y + (eta * omega) * (q - apply_K(x_bar, sample_primal=None)),
        lp.m_ineq,
    )
    # For fixed step, eta_accepted and eta_hat are both the fixed eta
    return (x_next, y_next), eta, eta


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


def _get_restart_candidate_kkt(
    *,
    lp: LPData,
    z_new: Tuple[np.ndarray, np.ndarray],
    z_avg: Tuple[np.ndarray, np.ndarray],
    omega: float,
    K: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """KKT-based: choose restart candidate by comparing KKT errors at current weight."""
    K_new = _kkt_error(lp, z_new, omega, K=K, q=q)
    K_avg = _kkt_error(lp, z_avg, omega, K=K, q=q)
    if K_new < K_avg:
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
    omega_new = float(np.exp(theta * np.log(ratio) + (1.0 - theta) * np.log(max(omega_prev, eps_zero))))
    # Clamp omega to avoid numerical issues
    return max(1e-8, min(omega_new, 1e8))


def _trust_region_linear_weighted(
    z: Tuple[np.ndarray, np.ndarray],
    g: Tuple[np.ndarray, np.ndarray],
    lower: Tuple[np.ndarray, np.ndarray],
    upper: Tuple[np.ndarray, np.ndarray],
    r: float,
    omega: float,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Solve:  minimize g^T z_hat
           s.t. ||z_hat - z||_omega <= r,   lower <= z_hat <= upper  (componentwise)
    
    Uses weighted norm: ||(x,y)||_omega^2 = omega*||x||^2 + (1/omega)*||y||^2
    """
    x, y = z
    g_x, g_y = g
    l_x, l_y = lower
    u_x, u_y = upper
    
    if r <= 0 or not np.isfinite(r):
        return z, float(g_x @ x + g_y @ y)
    
    # Clamp omega to avoid numerical issues
    omega = max(1e-8, min(omega, 1e8))
    sqrt_omega = np.sqrt(omega)
    sqrt_omega_inv = 1.0 / sqrt_omega
    radius = max(0.0, float(r))
    solver_iter_scale = max(1, int(np.ceil(max_iter)))
    
    x_var = cp.Variable(shape=x.shape)
    y_var = cp.Variable(shape=y.shape)
    objective = cp.Minimize(g_x @ x_var + g_y @ y_var)
    
    diff = cp.hstack(
        [
            sqrt_omega * (x_var - x),
            sqrt_omega_inv * (y_var - y),
        ]
    )
    constraints = [
        x_var >= l_x,
        x_var <= u_x,
        y_var >= l_y,
        y_var <= u_y,
        cp.norm(diff, 2) <= radius + max(tol, 0.0),
    ]
    
    problem = cp.Problem(objective, constraints)
    solved = False
    ecos_opts = {
        "abstol": max(tol, 1e-9),
        "reltol": max(tol, 1e-9),
        "max_iters": max(100, solver_iter_scale * 10),
    }
    scs_opts = {
        "eps": max(tol, 1e-6),
        "max_iters": max(500, solver_iter_scale * 20),
    }
    for solver, opts in ((cp.ECOS, ecos_opts), (cp.SCS, scs_opts)):
        try:
            problem.solve(solver=solver, **opts)
        except (cp.SolverError, ValueError):
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            solved = True
            break
    
    if not solved or x_var.value is None or y_var.value is None:
        return z, float(g_x @ x + g_y @ y)
    
    x_hat = np.asarray(x_var.value).reshape(x.shape)
    y_hat = np.asarray(y_var.value).reshape(y.shape)
    x_hat = np.nan_to_num(x_hat, nan=x, posinf=u_x, neginf=l_x)
    y_hat = np.nan_to_num(y_hat, nan=y, posinf=u_y, neginf=l_y)
    T = float(g_x @ x_hat + g_y @ y_hat)
    if not np.isfinite(T):
        return z, float(g_x @ x + g_y @ y)
    
    return (x_hat, y_hat), T


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
    """Normalized duality gap: ρ_r(z) = (1/r) max_{z_hat: ||z_hat - z||_ω <= r} {L(x, ŷ) - L(x̂, y)}."""
    x, y = z
    x_ref, y_ref = z_ref
    
    # Compute radius
    r_val = max(weighted_norm(x - x_ref, y - y_ref, omega), eps_zero)
    
    if r_val == 0.0:
        _, _, gap = compute_duality_gap(
            lp, x, y, omega=1.0, K=K, q=q, matvec_callback=lambda: tracker.bump(x)
        )
        return abs(gap)
    
    # Build gradient in the same way as Julia's bound_optimal_objective:
    # primal_gradient = ∂L/∂x = c - Kᵀ y
    # dual_gradient   = ∂L/∂y = q - K x
    # and z-gradient is [primal_gradient; -dual_gradient].
    if K.size:
        primal_grad = lp.c - K.T @ y
        dual_grad = q - K @ x
        g_x = primal_grad
        g_y = -dual_grad          # = K @ x - q
        tracker.bump(x)
        tracker.bump(x)
    else:
        # No constraints: gradients reduce to c and q
        primal_grad = lp.c.copy()
        dual_grad = q.copy()
        g_x = primal_grad
        g_y = -dual_grad
    
    # Lower bounds: l_x = lower, l_y[i] = 0 for inequality constraints, -inf for equality
    l_x = lp.lower.copy()
    l_y = np.full_like(y, -np.inf)
    if lp.m_ineq > 0:
        l_y[:lp.m_ineq] = 0.0
    
    # Upper bounds: u_x = upper, u_y = +inf (dual variables are unbounded above)
    u_x = lp.upper.copy()
    u_y = np.full_like(y, np.inf)
    
    # Solve trust-region subproblem
    _, T = _trust_region_linear_weighted(
        z=(x, y),
        g=(g_x, g_y),
        lower=(l_x, l_y),
        upper=(u_x, u_y),
        r=r_val,
        omega=omega,
    )
    
    # Constant term chosen so that (C - T) equals the localized duality gap
    C = float(primal_grad @ x - dual_grad @ y)
    
    # Handle numerical issues
    if not np.isfinite(C) or not np.isfinite(T):
        # Fallback: use simple duality gap
        _, _, gap = compute_duality_gap(
            lp, x, y, omega=1.0, K=K, q=q, matvec_callback=lambda: tracker.bump(x)
        )
        return abs(gap) / max(r_val, eps_zero)
    
    # Normalized duality gap: ρ_r(z) = (C - T) / r, always non-negative
    rho = max(0.0, (C - T) / r_val)
    if not np.isfinite(rho):
        # Fallback: use simple duality gap
        _, _, gap = compute_duality_gap(
            lp, x, y, omega=1.0, K=K, q=q, matvec_callback=lambda: tracker.bump(x)
        )
        rho = abs(gap) / max(r_val, eps_zero)
    return rho


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


def _kkt_error(
    lp: LPData,
    z: Tuple[np.ndarray, np.ndarray],
    omega: float,
    *,
    K: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Compute KKT error for current outer weight omega:
    KKT(z) = sqrt( omega^2 * ||r_p||^2 + omega^-2 * ||r_d||^2 + gap^2 ).
    Here we take ε (lambda) as the projection of reduced cost (standard for LP).
    """
    x, y = z
    # Primal residual: [Ax - b; (h - Gx)_+]
    if lp.m_eq:
        Ax_b = lp.A @ x - lp.b
    else:
        Ax_b = np.zeros(0, dtype=float)
    if lp.m_ineq:
        h_minus_Gx_pos = np.maximum(lp.h - (lp.G @ x), 0.0)
    else:
        h_minus_Gx_pos = np.zeros(0, dtype=float)
    r_p = np.concatenate([Ax_b, h_minus_Gx_pos]) if Ax_b.size + h_minus_Gx_pos.size else np.zeros(0)
    norm_rp = float(np.linalg.norm(r_p, ord=2)) if r_p.size else 0.0

    # Dual residual: c - K^T y - ε, with ε = proj_lambda(c - K^T y)
    if K.size:
        reduced_cost = lp.c - K.T @ y
    else:
        reduced_cost = lp.c.copy()
    eps_vec = project_lambda(reduced_cost, lp.lower, lp.upper)
    r_d = reduced_cost - eps_vec
    norm_rd = float(np.linalg.norm(r_d, ord=2))

    # Gap term: q^T y + l^T ε^+ - u^T ε^- - c^T x
    # Careful: l or u may contain +/-inf; exclude non-finite contributions.
    eps_pos = np.maximum(eps_vec, 0.0)
    eps_neg = np.minimum(eps_vec, 0.0)
    lower_mask = np.isfinite(lp.lower)
    upper_mask = np.isfinite(lp.upper)
    l_dot = float((lp.lower[lower_mask] @ eps_pos[lower_mask]) if lower_mask.any() else 0.0)
    u_dot = float((lp.upper[upper_mask] @ eps_neg[upper_mask]) if upper_mask.any() else 0.0)
    gap_val = float(q @ y + l_dot - u_dot - lp.c @ x)

    # Weighted combination
    omega_clamped = max(1e-8, min(omega, 1e8))
    val = (omega_clamped ** 2) * (norm_rp ** 2) + (omega_clamped ** -2) * (norm_rd ** 2) + (gap_val ** 2)
    if not np.isfinite(val):
        return np.inf
    return float(np.sqrt(val))


def _should_restart_kkt(
    *,
    K0: float,
    K_c: float,
    K_last: float,
    t: int,
    k: int,
    theta_params: Dict[str, float],
) -> bool:
    """
    KKT-based restart criteria:
      1) K_c <= theta_sufficient * K0
      2) K_c <= theta_necessary * K0 and K_c > K_last
      3) t >= theta_artificial * k
    """
    sufficient_decay = K_c <= theta_params["sufficient"] * K0
    necessary_decay = (K_c <= theta_params["necessary"] * K0) and (K_c > K_last)
    long_inner = t >= theta_params["artificial"] * k
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
    # Reuse reduced_cost instead of recomputing apply_K_transpose
    dual_residual_vec = reduced_cost - lam
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


def _default_kkt_params() -> Dict[str, float]:
    # Suggested example in docs: (0.2, 0.8, 0.36)
    return {"sufficient": 0.2, "necessary": 0.8, "artificial": 0.36}


def _validate_kkt_params(kkt_params: Dict[str, float]) -> Dict[str, float]:
    keys = {"sufficient", "necessary", "artificial"}
    missing = keys - kkt_params.keys()
    if missing:
        raise ValueError(f"Missing KKT parameters: {missing}")
    return {key: float(kkt_params[key]) for key in keys}


__all__ = ["PDLPResult", "pdlp"]
