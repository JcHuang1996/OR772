from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from mps_process import LPData


def proj_box(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Projection onto the box {x | lower <= x <= upper} with element-wise bounds."""
    return np.minimum(np.maximum(x, lower), upper)


def proj_dual(y: np.ndarray, m_ineq: int) -> np.ndarray:
    """Projection of dual variables so that inequality multipliers remain non-negative."""
    if m_ineq == 0:
        return y
    projected = y.copy()
    projected[:m_ineq] = np.maximum(projected[:m_ineq], 0.0)
    return projected


def estimate_spectral_norm(
    K: np.ndarray,
    *,
    iters: int = 20,
    seed: Optional[int] = None,
    matvec_callback: Optional[Callable[[], None]] = None,
) -> float:
    """Estimate ||K||_2 using power iteration."""
    if K.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    n = K.shape[1]
    v = rng.standard_normal(n)
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        return 0.0
    v /= v_norm
    for _ in range(max(iters, 1)):
        w = K @ v
        if matvec_callback is not None:
            matvec_callback()
        w_norm = np.linalg.norm(w)
        if w_norm == 0.0:
            return 0.0
        v = K.T @ (w / w_norm)
        if matvec_callback is not None:
            matvec_callback()
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            return 0.0
        v /= v_norm
    return float(np.linalg.norm(K @ v))


def weighted_norm_sq(x: np.ndarray, y: np.ndarray, omega: float) -> float:
    """Squared weighted norm ||(x, y)||_omega^2."""
    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    return float(omega * np.dot(x, x) + np.dot(y, y) / omega)


def weighted_norm(x: np.ndarray, y: np.ndarray, omega: float) -> float:
    """Weighted norm ||(x, y)||_omega."""
    return float(np.sqrt(weighted_norm_sq(x, y, omega)))


def project_lambda(v: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project reduced costs onto the dual domain Lambda induced by bounds."""
    # Vectorized implementation for better performance
    lam = v.copy()
    
    # Case 1: both bounds are infinite -> set to 0
    both_infinite = ~np.isfinite(lower) & ~np.isfinite(upper)
    lam[both_infinite] = 0.0
    
    # Case 2: lower infinite, upper finite -> min(value, 0)
    lower_inf_upper_finite = ~np.isfinite(lower) & np.isfinite(upper)
    lam[lower_inf_upper_finite] = np.minimum(lam[lower_inf_upper_finite], 0.0)
    
    # Case 3: lower finite, upper infinite -> max(value, 0)
    lower_finite_upper_inf = np.isfinite(lower) & ~np.isfinite(upper)
    lam[lower_finite_upper_inf] = np.maximum(lam[lower_finite_upper_inf], 0.0)
    
    # Case 4: both finite -> keep value (already copied from v)
    # No action needed as lam already equals v for these indices
    
    return lam


def compute_duality_gap(
    lp: "LPData",
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    *,
    K: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    matvec_callback: Optional[Callable[[], None]] = None,
) -> Tuple[float, float, float]:
    """Compute primal objective, dual candidate, and their gap."""
    if K is None:
        if lp.m_ineq and lp.m_eq:
            K_local = np.vstack([lp.G, lp.A])
            q_local = np.concatenate([lp.h, lp.b])
        elif lp.m_ineq:
            K_local = lp.G
            q_local = lp.h
        elif lp.m_eq:
            K_local = lp.A
            q_local = lp.b
        else:
            K_local = np.zeros((0, lp.n_vars), dtype=float)
            q_local = np.zeros((0,), dtype=float)
    else:
        K_local = K
        if q is not None:
            q_local = q
        elif lp.m_ineq and lp.m_eq:
            q_local = np.concatenate([lp.h, lp.b])
        elif lp.m_ineq:
            q_local = lp.h
        elif lp.m_eq:
            q_local = lp.b
        else:
            q_local = np.zeros((0,), dtype=float)

    if K_local.size:
        reduced_cost = lp.c - K_local.T @ y
        if matvec_callback is not None:
            matvec_callback()
    else:
        reduced_cost = lp.c.copy()
    lam = project_lambda(reduced_cost, lp.lower, lp.upper)

    lam_pos = np.maximum(lam, 0.0)
    lam_neg = np.minimum(lam, 0.0)

    finite_lower = np.where(np.isfinite(lp.lower), lp.lower, 0.0)
    finite_upper = np.where(np.isfinite(lp.upper), lp.upper, 0.0)

    primal_obj = float(lp.c @ x + lp.obj_offset)
    dual_term = float(q_local @ y) if y.size else 0.0
    bound_term = float(finite_lower @ lam_pos + finite_upper @ lam_neg)
    dual_obj = dual_term + bound_term + lp.obj_offset

    gap = primal_obj - dual_obj
    return primal_obj, dual_obj, gap


__all__ = [
    "proj_box",
    "proj_dual",
    "estimate_spectral_norm",
    "weighted_norm",
    "weighted_norm_sq",
    "project_lambda",
    "compute_duality_gap",
]