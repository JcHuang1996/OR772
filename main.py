from pathlib import Path

from algo_PDHG import PDHGResult, pdhg
from algo_PDLP import PDLPResult, pdlp
from lp_precondition import PrecondMethod, PreconditionerData, apply_preconditioner
from mps_process import LPData, get_reference_objective, read_mps
from typing import Optional, Dict


def run_pdhg(
    case_path: str,
    *,
    max_iters: int = 100_000,
    tol: float = 1e-5,
    omega: float = 1.0,
    eta_scale: float = 0.9,
    check_every: int = 25,
    objective_stride: int = 0,
    precond: PrecondMethod = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
) -> tuple[PDHGResult, PreconditionerData, LPData, float]:
    path = Path(case_path).expanduser().resolve()
    reference_objective = get_reference_objective(str(path))
    lp = read_mps(str(path))
    lp_scaled, precond_data = apply_preconditioner(
        lp,
        method=precond,
        tol=precond_tol,
        max_iters=precond_iters,
    )

    result = pdhg(
        lp_scaled,
        max_iters=max_iters,
        tol=tol,
        omega=omega,
        eta_scale=eta_scale,
        check_every=check_every,
        objective_stride=objective_stride,
        reference_objective=reference_objective,
        precond=precond_data if precond != "none" else None,
    )

    return result, precond_data, lp, reference_objective


def run_pdlp(
    case_path: str,
    *,
    max_outer: int = 50,
    max_inner: int = 2000,
    tol: float = 1e-6,
    check_every: int = 50,
    objective_stride: int = 0,
    precond: PrecondMethod = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
    beta_params: Optional[Dict[str, float]] = None,
    restart_mode: str = "normalized_gap",
    kkt_params: Optional[Dict[str, float]] = None,
) -> tuple[PDLPResult, PreconditionerData, LPData, float]:
    path = Path(case_path).expanduser().resolve()
    reference_objective = get_reference_objective(str(path))
    lp = read_mps(str(path))
    lp_scaled, precond_data = apply_preconditioner(
        lp,
        method=precond,
        tol=precond_tol,
        max_iters=precond_iters,
    )

    result = pdlp(
        lp_scaled,
        max_outer=max_outer,
        max_inner=max_inner,
        tol=tol,
        check_every=check_every,
        objective_stride=objective_stride,
        reference_objective=reference_objective,
        precond=precond_data if precond != "none" else None,
        beta_params=beta_params,
        restart_mode=restart_mode,
        kkt_params=kkt_params,
    )

    return result, precond_data, lp, reference_objective


def resolve_case_paths(target: Path) -> list[Path]:
    """Return a sorted list of case files from a file or directory input."""
    path = target.expanduser().resolve()
    if path.is_file():
        if _is_supported_case(path):
            return [path]
        raise ValueError(f"Unsupported case file extension: {path.name}")
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")

    case_paths = sorted(p for p in path.iterdir() if _is_supported_case(p))
    if not case_paths:
        raise ValueError(f"No .mps or .mps.bz2 files found under {path}")
    return case_paths


def _is_supported_case(path: Path) -> bool:
    """Check whether the path points to a supported case file."""
    if not path.is_file():
        return False
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return False
    if suffixes[-1] == ".mps":
        return True
    return len(suffixes) >= 2 and suffixes[-2:] == [".mps", ".bz2"]


__all__ = ["run_pdhg", "run_pdlp", "resolve_case_paths"]