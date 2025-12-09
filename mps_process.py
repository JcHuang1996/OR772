import bz2
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from highspy import Highs, HighsStatus, ObjSense, HighsVarType  # type: ignore[import]

from operations import estimate_spectral_norm


@dataclass
class LPData:
    """Dense canonical LP representation."""

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    G: np.ndarray
    h: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    obj_offset: float
    m_eq: int
    m_ineq: int
    n_vars: int
    norm_2: float  # Spectral norm (2-norm) of constraint matrix K
    norm_inf: float  # Infinity norm of constraint matrix K


@contextmanager
def _normalized_model_path(path: str) -> Iterator[str]:
    path_obj = Path(path)
    if path_obj.suffix == ".bz2":
        suffix = path_obj.with_suffix("").suffix or ".mps"
        with bz2.open(path_obj, "rb") as src, tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(src.read())
            tmp.flush()
            tmp_path = tmp.name
        try:
            yield tmp_path
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        yield str(path_obj)


def _load_highs_model(path: str) -> Tuple[Highs, LPData]:
    # Initialize HiGHS solver and load MPS model file
    highs = Highs()
    highs.setOptionValue("output_flag", False)
    with _normalized_model_path(path) as actual_path:
        status = highs.readModel(actual_path)
    if status != HighsStatus.kOk:
        highs.clear()
        raise RuntimeError(f"HiGHS failed to read model: status {status}.")

    lp = highs.getLp()

    # Get problem dimensions
    # n: number of variables (dimension of x in PDHG notation)
    # m: number of constraints (rows in the constraint matrix)
    n = lp.num_col_
    m = lp.num_row_

    # Relax any integrality constraints to make it a continuous LP
    if hasattr(lp, "integrality_") and len(lp.integrality_) != 0:
        indices = np.arange(n, dtype=np.int32)
        types = np.full(n, HighsVarType.kContinuous.value, dtype=np.uint8)
        status = highs.changeColsIntegrality(int(n), indices, types)
        if status != HighsStatus.kOk:
            highs.clear()
            raise RuntimeError("Failed to relax integrality constraints in model.")
        lp = highs.getLp()

    # Read objective vector c (for primal problem: min c^T x in PDHG notation)
    c = np.asarray(lp.col_cost_, dtype=float)
    # Read variable bounds: l (lower) and u (upper) for X = {x | l ≤ x ≤ u} in PDHG notation
    lb = np.asarray(lp.col_lower_, dtype=float)  # l: lower bounds
    ub = np.asarray(lp.col_upper_, dtype=float)  # u: upper bounds
    # Read row constraint bounds (will be used to separate equality and inequality constraints)
    row_lower = np.asarray(lp.row_lower_, dtype=float)
    row_upper = np.asarray(lp.row_upper_, dtype=float)
    obj_offset = float(lp.offset_) if hasattr(lp, "offset_") else 0.0

    # Handle maximization: convert max to min by negating objective
    # Ensures standard form: min c^T x (as in PDHG primal formulation)
    _, sense = getattr(highs, "getObjectiveSense")()
    if sense == ObjSense.kMaximize:
        c = -c
        obj_offset = -obj_offset

    # Extract sparse constraint matrix from HiGHS (column-compressed format)
    start = lp.a_matrix_.start_
    index = lp.a_matrix_.index_
    value = lp.a_matrix_.value_

    # Convert sparse matrix to dense format (m x n constraint matrix)
    # This matrix contains all constraints before separating into A (equality) and G (inequality)
    matrix = np.zeros((m, n), dtype=float)
    for j in range(n):
        col_start = start[j]
        col_end = start[j + 1]
        for kk in range(col_start, col_end):
            i = index[kk]
            matrix[i, j] = value[kk]

    # Separate constraints into equality (Ax = b) and inequality (Gx ≥ h) forms
    # Following PDHG notation:
    # - A: equality constraint matrix (for Ax = b)
    # - b: equality constraint RHS (for Ax = b)
    # - G: inequality constraint matrix (for Gx ≥ h)
    # - h: inequality constraint RHS (for Gx ≥ h)
    eq_rows = []
    b_eq = []
    G_rows = []
    h_vals = []
    tol = 1e-9
    for i in range(m):
        lower = row_lower[i]
        upper = row_upper[i]
        row = matrix[i]
        # Equality constraint: lower == upper (within tolerance)
        # Contributes to A (equality matrix) and b (equality RHS)
        if np.isfinite(lower) and np.isfinite(upper) and abs(lower - upper) <= tol:
            eq_rows.append(row)
            b_eq.append(lower)
        else:
            # Inequality constraint: lower ≤ row·x or row·x ≤ upper
            # Convert to standard form Gx ≥ h:
            # - If lower bound exists: row·x ≥ lower → add row to G, lower to h
            # - If upper bound exists: row·x ≤ upper → add -row to G, -upper to h
            # (Note: converts "≤" to "≥" by multiplying by -1)
            if np.isfinite(lower):
                G_rows.append(row)
                h_vals.append(lower)
            if np.isfinite(upper):
                G_rows.append(-row)
                h_vals.append(-upper)

    # Finalize constraint matrices and RHS vectors in PDHG notation:
    # A: equality constraint matrix (m_eq × n) for Ax = b
    # b: equality constraint RHS (m_eq × 1) for Ax = b
    # G: inequality constraint matrix (m_ineq × n) for Gx ≥ h
    # h: inequality constraint RHS (m_ineq × 1) for Gx ≥ h
    # Note: Later, K = [G; A] (vertical stacking) and q = [h; b] (vertical stacking)
    # such that K^T = [G^T, A^T] and q^T = [h^T, b^T] as in PDHG notation
    A_eq = np.asarray(eq_rows, dtype=float) if eq_rows else np.zeros((0, n), dtype=float)
    b_eq = np.asarray(b_eq, dtype=float) if b_eq else np.zeros((0,), dtype=float)
    G = np.asarray(G_rows, dtype=float) if G_rows else np.zeros((0, n), dtype=float)
    h = np.asarray(h_vals, dtype=float) if h_vals else np.zeros((0,), dtype=float)

    # Compute matrix norms for reproducibility (using fixed seed)
    # Build K matrix: K = [G; A]
    if G.shape[0] > 0 and A_eq.shape[0] > 0:
        K = np.vstack([G, A_eq])
    elif G.shape[0] > 0:
        K = G
    elif A_eq.shape[0] > 0:
        K = A_eq
    else:
        K = np.zeros((0, n), dtype=float)
    
    # Compute norms with fixed seed for reproducibility
    # Use seed=42 for deterministic results across runs
    if K.size > 0:
        norm_2 = estimate_spectral_norm(K, iters=20, seed=42)
        norm_inf = float(np.linalg.norm(K, ord=np.inf))
    else:
        norm_2 = 0.0
        norm_inf = 0.0

    lp_data = LPData(
        c=c,
        A=A_eq,
        b=b_eq,
        G=G,
        h=h,
        lower=lb,
        upper=ub,
        obj_offset=obj_offset,
        m_eq=A_eq.shape[0],
        m_ineq=G.shape[0],
        n_vars=n,
        norm_2=norm_2,
        norm_inf=norm_inf,
    )
    return highs, lp_data


REFERENCE_CSV = Path(__file__).parent / "lp_cases" / "mps_compare_output.csv"


def _instance_name_from_path(path: str) -> str:
    name = Path(path).name
    for suffix in (".bz2", ".gz", ".zip"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    if name.lower().endswith(".mps"):
        name = name[: -len(".mps")]
    return name


@lru_cache(maxsize=1)
def _reference_objective_series() -> pd.Series:
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(f"Reference CSV not found at {REFERENCE_CSV}")
    df = pd.read_csv(REFERENCE_CSV)
    if "Name" not in df.columns or "Gurobi Optimal Value" not in df.columns:
        raise ValueError(
            "CSV must contain 'Name' and 'Gurobi Optimal Value' columns."
        )
    df["Name"] = df["Name"].astype(str).str.strip()
    df = df[df["Name"] != ""]
    df = df.dropna(subset=["Gurobi Optimal Value"])
    if df.empty:
        raise ValueError(f"No reference objectives parsed from {REFERENCE_CSV}")
    return df.set_index("Name")["Gurobi Optimal Value"].astype(float)


def get_reference_objective(path: str) -> float:
    """Return the reference objective value for the instance referenced by path."""
    instance = _instance_name_from_path(path)
    series = _reference_objective_series()
    if instance not in series:
        raise KeyError(f"No reference objective found for instance '{instance}'.")
    return float(series[instance])


def read_mps(path: str) -> LPData:
    """Load an LP from an MPS file and convert it to dense canonical form."""
    highs, lp_data = _load_highs_model(path)
    highs.clear()
    return lp_data


__all__ = [
    "LPData",
    "read_mps",
]