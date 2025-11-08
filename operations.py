import numpy as np


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


def estimate_spectral_norm(K: np.ndarray, *, iters: int = 20, seed: int | None = None) -> float:
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
        w_norm = np.linalg.norm(w)
        if w_norm == 0.0:
            return 0.0
        v = K.T @ (w / w_norm)
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            return 0.0
        v /= v_norm
    return float(np.linalg.norm(K @ v))


__all__ = ["proj_box", "proj_dual", "estimate_spectral_norm"]