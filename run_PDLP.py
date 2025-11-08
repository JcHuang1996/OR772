from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, cast

from algo_PDLP import pdlp
from lp_precondition import PrecondMethod, apply_preconditioner
from mps_process import read_mps, solve_lp_reference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PDLP solver on an MPS model.")
    parser.add_argument("model", type=Path, help="Path to the MPS model file.")
    parser.add_argument(
        "--max-outer",
        type=int,
        default=50,
        help="Maximum number of outer restart epochs (default: 50).",
    )
    parser.add_argument(
        "--max-inner",
        type=int,
        default=2000,
        help="Maximum number of inner iterations per epoch (default: 2000).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Termination tolerance for residuals and duality gap (default: 1e-6).",
    )
    parser.add_argument(
        "--precond",
        type=str,
        default="ruiz_pc",
        choices=["none", "pc", "ruiz", "ruiz_pc"],
        help="Diagonal preconditioner to apply (default: ruiz_pc).",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=50,
        help="Frequency of convergence checks (default: 50 iterations).",
    )
    parser.add_argument(
        "--ref-solver",
        choices=["highs", "gurobi"],
        default="highs",
        help="Reference solver for the objective comparison (default: highs).",
    )
    parser.add_argument(
        "--ref-log",
        "--highs-log",
        dest="ref_log",
        type=Path,
        default=None,
        help="Optional log file for the reference solver (alias: --highs-log).",
    )
    parser.add_argument(
        "--ref-verbose",
        "--highs-verbose",
        dest="ref_verbose",
        action="store_true",
        help="Stream reference solver output to the console (alias: --highs-verbose).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    model_path = args.model.expanduser().resolve()
    log_file = str(args.ref_log) if args.ref_log is not None else None
    reference = solve_lp_reference(
        str(model_path),
        solver=args.ref_solver,
        log_to_console=args.ref_verbose,
        log_file=log_file,
    )
    lp = read_mps(str(model_path))
    lp_scaled, precond = apply_preconditioner(lp, method=cast(PrecondMethod, args.precond))

    result = pdlp(
        lp_scaled,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        tol=args.tol,
        check_every=args.check_every,
        precond=precond,
    )

    print("PDLP run summary")
    print("----------------")
    print(f"Model:           {model_path}")
    print(f"Reference solver:{args.ref_solver}")
    print(f"Preconditioner:  {precond.method}")
    print(f"Outer epochs:    {result.outer_iterations}")
    print(f"Iterations:      {result.iterations}")
    print(f"Reference obj:   {reference.objective:.6f}")
    print(f"Objective:       {result.objective:.6f}")
    print(f"Dual objective:  {result.dual_objective:.6f}")
    print(f"Objective gap:   {result.objective - reference.objective:.3e}")
    print(f"Duality gap:     {result.duality_gap:.3e}")
    print(f"Primal residual: {result.primal_residual:.3e}")
    print(f"Dual residual:   {result.dual_residual:.3e}")
    print(f"Converged:       {result.converged}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

