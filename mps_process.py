import bz2
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from highspy import Highs, HighsStatus, ObjSense, HighsVarType  # type: ignore[import]
import gurobipy as gp
from gurobipy import GRB


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


@dataclass
class HighsResult:
    """Optimal solution reported by HiGHS."""

    x: np.ndarray
    objective: float


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
    highs = Highs()
    highs.setOptionValue("output_flag", False)
    with _normalized_model_path(path) as actual_path:
        status = highs.readModel(actual_path)
    if status != HighsStatus.kOk:
        highs.clear()
        raise RuntimeError(f"HiGHS failed to read model: status {status}.")

    lp = highs.getLp()

    n = lp.num_col_
    m = lp.num_row_

    if hasattr(lp, "integrality_") and len(lp.integrality_) != 0:
        indices = np.arange(n, dtype=np.int32)
        types = np.full(n, HighsVarType.kContinuous.value, dtype=np.uint8)
        status = highs.changeColsIntegrality(int(n), indices, types)
        if status != HighsStatus.kOk:
            highs.clear()
            raise RuntimeError("Failed to relax integrality constraints in model.")
        lp = highs.getLp()

    c = np.asarray(lp.col_cost_, dtype=float)
    lb = np.asarray(lp.col_lower_, dtype=float)
    ub = np.asarray(lp.col_upper_, dtype=float)
    row_lower = np.asarray(lp.row_lower_, dtype=float)
    row_upper = np.asarray(lp.row_upper_, dtype=float)
    obj_offset = float(lp.offset_) if hasattr(lp, "offset_") else 0.0

    _, sense = getattr(highs, "getObjectiveSense")()
    if sense == ObjSense.kMaximize:
        c = -c
        obj_offset = -obj_offset

    start = lp.a_matrix_.start_
    index = lp.a_matrix_.index_
    value = lp.a_matrix_.value_

    matrix = np.zeros((m, n), dtype=float)
    for j in range(n):
        col_start = start[j]
        col_end = start[j + 1]
        for kk in range(col_start, col_end):
            i = index[kk]
            matrix[i, j] = value[kk]

    eq_rows = []
    b_eq = []
    G_rows = []
    h_vals = []
    tol = 1e-9
    for i in range(m):
        lower = row_lower[i]
        upper = row_upper[i]
        row = matrix[i]
        if np.isfinite(lower) and np.isfinite(upper) and abs(lower - upper) <= tol:
            eq_rows.append(row)
            b_eq.append(lower)
        else:
            if np.isfinite(lower):
                G_rows.append(row)
                h_vals.append(lower)
            if np.isfinite(upper):
                G_rows.append(-row)
                h_vals.append(-upper)

    A_eq = np.asarray(eq_rows, dtype=float) if eq_rows else np.zeros((0, n), dtype=float)
    b_eq = np.asarray(b_eq, dtype=float) if b_eq else np.zeros((0,), dtype=float)
    G = np.asarray(G_rows, dtype=float) if G_rows else np.zeros((0, n), dtype=float)
    h = np.asarray(h_vals, dtype=float) if h_vals else np.zeros((0,), dtype=float)

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
    )
    return highs, lp_data


def read_mps(path: str) -> LPData:
    """Load an LP from an MPS file and convert it to dense canonical form."""
    highs, lp_data = _load_highs_model(path)
    highs.clear()
    return lp_data


def solve_with_highs(
    path: str,
    *,
    log_to_console: bool = False,
    log_file: Optional[str] = None,
) -> HighsResult:
    """Solve the LP in the provided MPS file with HiGHS."""
    highs, lp_data = _load_highs_model(path)
    highs.setOptionValue("output_flag", bool(log_to_console))
    if log_file is not None:
        highs.setOptionValue("log_file", str(log_file))
    run_status = highs.run()
    if run_status != HighsStatus.kOk:
        highs.clear()
        raise RuntimeError(f"HiGHS failed to solve model: status {run_status}.")

    solution = highs.getSolution()
    x = np.asarray(solution.col_value, dtype=float)
    objective = float(lp_data.c @ x + lp_data.obj_offset)
    highs.clear()
    return HighsResult(x=x, objective=objective)

def solve_with_gurobi(
    path: str,
    *,
    log_to_console: bool = False,
    log_file: Optional[str] = None,
) -> HighsResult:
    """Solve the LP in the provided MPS file with Gurobi."""

    # Load model from MPS file
    model = gp.read(path)
    if not log_to_console:
        model.Params.OutputFlag = 0
    if log_file is not None:
        model.Params.LogFile = str(log_file)
    model.optimize()
    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        model.dispose()
        raise RuntimeError(f"Gurobi failed to solve model: status {model.Status}.")

    # Map results to dense x (with bounds) via LPData
    _, lp_data = _load_highs_model(path)  # To get mapping/order and objective coeffs
    x_vars = []
    for var in model.getVars():
        x_vars.append(var.X)
    x = np.asarray(x_vars, dtype=float)
    objective = float(lp_data.c @ x + lp_data.obj_offset)
    model.dispose()
    return HighsResult(x=x, objective=objective)

def solve_lp_reference(
    path: str,
    *,
    solver: str = "highs",
    log_to_console: bool = False,
    log_file: Optional[str] = None,
) -> HighsResult:
    """Solve the LP in the provided MPS file with the selected solver (HiGHS or Gurobi)."""
    if solver.lower() == "highs":
        return solve_with_highs(path, log_to_console=log_to_console, log_file=log_file)
    elif solver.lower() == "gurobi":
        return solve_with_gurobi(path, log_to_console=log_to_console, log_file=log_file)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Use 'highs' or 'gurobi'.")


__all__ = [
    "LPData",
    "HighsResult",
    "read_mps",
    "solve_with_highs",
    "solve_with_gurobi",
    "solve_lp_reference",
]