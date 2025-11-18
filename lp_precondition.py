from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from mps_process import LPData
from operations import estimate_spectral_norm

PrecondMethod = Literal["none", "pc", "ruiz", "ruiz_pc"]


@dataclass
class PreconditionerData:
    """Diagonal scalings produced by the preconditioner."""

    D1: np.ndarray  # length m_ineq + m_eq
    D2: np.ndarray  # length n
    method: PrecondMethod
    m_ineq: int
    m_eq: int

    def to_primal(self, x_scaled: np.ndarray) -> np.ndarray:
        return x_scaled * self.D2

    def to_dual(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.D1


def _apply_ruiz(K: np.ndarray, max_iters: int, tol: float, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = K.shape
    D1 = np.ones(m, dtype=float)
    D2 = np.ones(n, dtype=float)
    K_scaled = K.copy()
    for _ in range(max_iters):
        if K_scaled.size == 0:
            break
        row_norms = np.maximum(np.max(np.abs(K_scaled), axis=1), eps)
        col_norms = np.maximum(np.max(np.abs(K_scaled), axis=0), eps)
        S1 = 1.0 / np.sqrt(row_norms)
        S2 = 1.0 / np.sqrt(col_norms)
        K_scaled = (S1[:, None]) * K_scaled * (S2[None, :])
        D1 *= S1
        D2 *= S2
        rows_close = np.all(np.abs(np.max(np.abs(K_scaled), axis=1) - 1.0) <= tol)
        cols_close = np.all(np.abs(np.max(np.abs(K_scaled), axis=0) - 1.0) <= tol)
        if rows_close and cols_close:
            break
    return K_scaled, D1, D2


def _apply_pc(K: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = K.shape
    if K.size == 0:
        return K.copy(), np.ones(m, dtype=float), np.ones(n, dtype=float)
    row_norms = np.maximum(np.linalg.norm(K, axis=1), eps)
    col_norms = np.maximum(np.linalg.norm(K, axis=0), eps)
    S1 = row_norms ** (-0.5)
    S2 = col_norms ** (0.5)
    K_scaled = (S1[:, None]) * K * (S2[None, :])
    return K_scaled, S1, S2


def apply_preconditioner(
    lp: LPData,
    method: PrecondMethod = "ruiz_pc",
    *,
    tol: float = 1e-2,
    max_iters: int = 10,
    eps: float = 1e-8,
) -> Tuple[LPData, PreconditionerData]:
    m_total = lp.m_ineq + lp.m_eq
    if method == "none" or m_total == 0:
        method_name = method
        identity = PreconditionerData(
            D1=np.ones(m_total, dtype=float),
            D2=np.ones(lp.n_vars, dtype=float),
            method=method_name,
            m_ineq=lp.m_ineq,
            m_eq=lp.m_eq,
        )
        return lp, identity

    if lp.m_ineq and lp.m_eq:
        K = np.vstack([lp.G, lp.A])
    elif lp.m_ineq:
        K = lp.G.copy()
    else:
        K = lp.A.copy()

    accum_D1 = np.ones(m_total, dtype=float)
    accum_D2 = np.ones(lp.n_vars, dtype=float)
    K_work = K.copy()

    if method in ("ruiz", "ruiz_pc"):
        K_work, D1_ruiz, D2_ruiz = _apply_ruiz(K_work, max_iters=max_iters, tol=tol, eps=eps)
        accum_D1 *= D1_ruiz
        accum_D2 *= D2_ruiz

    if method in ("pc", "ruiz_pc"):
        K_work, D1_pc, D2_pc = _apply_pc(K_work, eps=eps)
        accum_D1 *= D1_pc
        accum_D2 *= D2_pc

    D1_ineq = accum_D1[: lp.m_ineq] if lp.m_ineq else np.ones(0, dtype=float)
    D1_eq = accum_D1[lp.m_ineq :] if lp.m_eq else np.ones(0, dtype=float)

    G_scaled = lp.G.copy()
    if lp.m_ineq:
        G_scaled = D1_ineq[:, None] * G_scaled * accum_D2[None, :]

    A_scaled = lp.A.copy()
    if lp.m_eq:
        A_scaled = D1_eq[:, None] * A_scaled * accum_D2[None, :]

    c_scaled = accum_D2 * lp.c
    h_scaled = D1_ineq * lp.h if lp.m_ineq else lp.h.copy()
    b_scaled = D1_eq * lp.b if lp.m_eq else lp.b.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        lower_scaled = lp.lower / accum_D2
        upper_scaled = lp.upper / accum_D2

    # Recompute norms for the preconditioned matrix K_scaled
    # Build K_scaled matrix: K_scaled = [G_scaled; A_scaled]
    if G_scaled.shape[0] > 0 and A_scaled.shape[0] > 0:
        K_scaled = np.vstack([G_scaled, A_scaled])
    elif G_scaled.shape[0] > 0:
        K_scaled = G_scaled
    elif A_scaled.shape[0] > 0:
        K_scaled = A_scaled
    else:
        K_scaled = np.zeros((0, lp.n_vars), dtype=float)
    
    # Compute norms with fixed seed for reproducibility
    if K_scaled.size > 0:
        norm_2_scaled = estimate_spectral_norm(K_scaled, iters=20, seed=42)
        norm_inf_scaled = float(np.linalg.norm(K_scaled, ord=np.inf))
    else:
        norm_2_scaled = 0.0
        norm_inf_scaled = 0.0

    scaled_lp = LPData(
        c=c_scaled,
        A=A_scaled,
        b=b_scaled,
        G=G_scaled,
        h=h_scaled,
        lower=lower_scaled,
        upper=upper_scaled,
        obj_offset=lp.obj_offset,
        m_eq=lp.m_eq,
        m_ineq=lp.m_ineq,
        n_vars=lp.n_vars,
        norm_2=norm_2_scaled,
        norm_inf=norm_inf_scaled,
    )

    precond = PreconditionerData(
        D1=accum_D1,
        D2=accum_D2,
        method=method,
        m_ineq=lp.m_ineq,
        m_eq=lp.m_eq,
    )
    return scaled_lp, precond


__all__ = ["PrecondMethod", "PreconditionerData", "apply_preconditioner"]
