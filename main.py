from pathlib import Path

from algo_PDHG import PDHGResult, pdhg
from lp_precondition import PrecondMethod, PreconditionerData, apply_preconditioner
from mps_process import HighsResult, LPData, read_mps, solve_with_highs


def run_pdhg(
    case_path: str,
    *,
    max_iters: int = 100_000,
    tol: float = 1e-5,
    omega: float = 1.0,
    eta_scale: float = 0.9,
    check_every: int = 25,
    precond: PrecondMethod = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
    verbose: bool = False,
) -> tuple[PDHGResult, PreconditionerData, LPData, HighsResult]:
    path = Path(case_path).expanduser().resolve()
    reference_result = solve_with_highs(str(path), log_to_console=verbose)
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
        precond=precond_data if precond != "none" else None,
    )

    if verbose:
        print(f"Reference objective: {reference_result.objective:.6f}")
        print(f"PDHG objective     : {result.objective:.6f}")
        print(f"Objective gap      : {result.objective - reference_result.objective:.6e}")
        print(f"Primal residual    : {result.primal_residual:.3e}")
        print(f"Dual residual      : {result.dual_residual:.3e}")
        print(f"Iterations         : {result.iterations}")
        print(f"Converged          : {result.converged}")

    return result, precond_data, lp, reference_result


__all__ = ["run_pdhg"]