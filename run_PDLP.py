import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from algo_PDLP import PDLPResult
from main import ReferenceSolver, resolve_case_paths, run_pdlp
from mps_process import HighsResult


def _print_summary(case_path: Path, reference: HighsResult, result: PDLPResult) -> None:
    print(f"=== PDLP Summary: {case_path.name} ===")
    print(f"Reference objective : {reference.objective:.6f}")
    print(f"PDLP objective      : {result.objective:.6f}")
    print(f"Dual objective      : {result.dual_objective:.6f}")
    print(f"Objective gap       : {result.objective - reference.objective:.3e}")
    print(f"Duality gap         : {result.duality_gap:.3e}")
    print(f"Primal residual     : {result.primal_residual:.3e}")
    print(f"Dual residual       : {result.dual_residual:.3e}")
    print(f"Outer epochs        : {result.outer_iterations}")
    print(f"Iterations          : {result.iterations}")
    print(f"Converged           : {result.converged}")
    print(f"K multiplications   : {result.k_multiplications}")
    # if result.objective_distance_trace:
    #     print("Objective distance trace (K, |obj-ref|):")
    #     for k_mult, distance in result.objective_distance_trace:
    #         print(f"  {k_mult:>10d}  {distance:.3e}")
    # else:
    #     print("Objective distance trace: (not recorded)")
    print()


def _collect_case_results(
    cases: List[Path],
    *,
    max_outer: int,
    max_inner: int,
    tol: float,
    check_every: int,
    objective_stride: int,
    precond: str,
    precond_tol: float,
    precond_iters: int,
    ref_solver: ReferenceSolver,
    ref_verbose: bool,
) -> List[Tuple[Path, PDLPResult, HighsResult]]:
    case_results: List[Tuple[Path, PDLPResult, HighsResult]] = []
    for case_path in cases:
        result, _, _, reference = run_pdlp(
            str(case_path),
            max_outer=max_outer,
            max_inner=max_inner,
            tol=tol,
            check_every=check_every,
            objective_stride=objective_stride,
            precond=precond,  # type: ignore[arg-type]
            precond_tol=precond_tol,
            precond_iters=precond_iters,
            ref_solver=ref_solver,
            ref_verbose=ref_verbose,
        )
        case_results.append((case_path, result, reference))
    return case_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PDLP on LP cases in MPS format.")
    parser.add_argument(
        "target",
        help="Path to a .mps/.mps.bz2 file or a directory containing such files.",
    )
    parser.add_argument("--max-outer", type=int, default=50)
    parser.add_argument("--max-inner", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument(
        "--objective-stride",
        type=int,
        default=0,
        help="Record objective distance every N K-multiplications (0 disables).",
    )
    parser.add_argument(
        "--precond",
        type=str,
        default="ruiz_pc",
        choices=["none", "pc", "ruiz", "ruiz_pc"],
    )
    parser.add_argument("--precond-tol", type=float, default=1e-2)
    parser.add_argument("--precond-iters", type=int, default=10)
    parser.add_argument(
        "--ref-solver",
        type=str,
        default="highs",
        choices=["highs", "gurobi"],
        help="Reference solver for objective comparison (default: highs).",
    )
    parser.add_argument(
        "--ref-verbose",
        action="store_true",
        help="Stream reference solver logs to stdout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Directory where timestamped outputs will be created (default: output/).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case PDLP summaries.",
    )
    args = parser.parse_args()

    if args.objective_stride < 0:
        parser.error("--objective-stride must be non-negative.")

    target_path = Path(args.target)
    case_paths = resolve_case_paths(target_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDLP run started at {timestamp}")
    print(f"Cases discovered: {len(case_paths)}")
    print(f"Output directory: {output_dir}")
    print()

    case_results = _collect_case_results(
        case_paths,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        tol=args.tol,
        check_every=args.check_every,
        objective_stride=args.objective_stride,
        precond=args.precond,
        precond_tol=args.precond_tol,
        precond_iters=args.precond_iters,
        ref_solver=args.ref_solver,
        ref_verbose=args.ref_verbose,
    )

    for case_path, result, _ in case_results:
        if result.objective_distance_trace:
            trace_filename = (
                f"{timestamp}_{case_path.name.replace('.', '_')}_pdlp_trace.csv"
            )
            trace_path = output_dir / trace_filename
            with trace_path.open("w", newline="") as trace_file:
                trace_writer = csv.writer(trace_file)
                trace_writer.writerow(["k_multiplications", "objective_distance"])
                trace_writer.writerows(result.objective_distance_trace)
            print(f"Objective trace CSV written to {trace_path}")

    if len(case_results) > 1:
        records = [
            {
                "case": str(case_path),
                "reference_objective": reference.objective,
                "objective": result.objective,
                "k_multiplications": result.k_multiplications,
                "outer_iterations": result.outer_iterations,
                "iterations": result.iterations,
                "converged": result.converged,
            }
            for case_path, result, reference in case_results
        ]
        csv_path = output_dir / f"{timestamp}_pdlp_summary.csv"
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "case",
                    "reference_objective",
                    "objective",
                    "k_multiplications",
                    "outer_iterations",
                    "iterations",
                    "converged",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
        print(f"Summary CSV written to {csv_path}")

    for case_path, result, reference in case_results:
        print(
            f"{case_path.name}: ref={reference.objective:.6f}, "
            f"obj={result.objective:.6f}, gap={result.objective - reference.objective:.3e}, "
            f"k-mults={result.k_multiplications}, "
            f"outer={result.outer_iterations}, iters={result.iterations}, converged={result.converged}"
        )
    print()

    if args.verbose:
        for case_path, result, reference in case_results:
            _print_summary(case_path, reference, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())