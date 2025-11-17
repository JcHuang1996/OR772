import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from algo_PDLP import PDLPResult
from main import resolve_case_paths, run_pdlp


def _print_summary(case_path: Path, reference: float, result: PDLPResult) -> None:
    print(f"=== PDLP Summary: {case_path.name} ===")
    print(f"Reference objective : {reference:.6f}")
    print(f"PDLP objective      : {result.objective:.6f}")
    print(f"Dual objective      : {result.dual_objective:.6f}")
    print(f"Objective gap       : {result.objective - reference:.3e}")
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
    max_K_multi: int,
    tol: float,
    check_every: int,
    objective_stride: int,
    precond: str,
    precond_tol: float,
    precond_iters: int,
    beta_params: Optional[Dict[str, float]] = None,
    restart_mode: str = "normalized_gap",
    kkt_params: Optional[Dict[str, float]] = None,
) -> List[Tuple[Path, PDLPResult, float]]:
    case_results: List[Tuple[Path, PDLPResult, float]] = []
    for case_path in cases:
        result, _, _, reference = run_pdlp(
            str(case_path),
            max_K_multi=max_K_multi,
            tol=tol,
            check_every=check_every,
            objective_stride=objective_stride,
            precond=precond,  # type: ignore[arg-type]
            precond_tol=precond_tol,
            precond_iters=precond_iters,
            beta_params=beta_params,
            restart_mode=restart_mode,
            kkt_params=kkt_params,
        )
        case_results.append((case_path, result, reference))
    return case_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PDLP on LP cases in MPS format.")
    parser.add_argument(
        "target",
        help="Path to a .mps/.mps.bz2 file or a directory containing such files.",
    )
    parser.add_argument("--max-K-multi", type=int, default=100000, dest="max_K_multi")
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
    parser.add_argument(
        "--beta-sufficient",
        type=float,
        default=None,
        help="Beta parameter for sufficient decay restart (default: 0.9).",
    )
    parser.add_argument(
        "--beta-necessary",
        type=float,
        default=None,
        help="Beta parameter for necessary decay restart (default: 0.1).",
    )
    parser.add_argument(
        "--beta-artificial",
        type=float,
        default=None,
        help="Beta parameter for artificial restart (default: 0.5).",
    )
    parser.add_argument(
        "--restart-mode",
        type=str,
        default="normalized_gap",
        choices=["normalized_gap", "kkt"],
        help='Restart module to use: "normalized_gap" (default) or "kkt".',
    )
    parser.add_argument(
        "--kkt-sufficient",
        type=float,
        default=None,
        help="KKT sufficient decay threshold (default: 0.2).",
    )
    parser.add_argument(
        "--kkt-necessary",
        type=float,
        default=None,
        help="KKT necessary decay threshold (default: 0.8).",
    )
    parser.add_argument(
        "--kkt-artificial",
        type=float,
        default=None,
        help="KKT artificial long-inner threshold (default: 0.36).",
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
    
    # Build beta_params dict if any beta parameter is provided
    beta_params = None
    if args.beta_sufficient is not None or args.beta_necessary is not None or args.beta_artificial is not None:
        beta_params = {}
        if args.beta_sufficient is not None:
            beta_params["sufficient"] = args.beta_sufficient
        if args.beta_necessary is not None:
            beta_params["necessary"] = args.beta_necessary
        if args.beta_artificial is not None:
            beta_params["artificial"] = args.beta_artificial
        print(f"Beta parameters: {beta_params}")
    # Build kkt_params dict if any kkt parameter is provided
    kkt_params = None
    if args.kkt_sufficient is not None or args.kkt_necessary is not None or args.kkt_artificial is not None:
        kkt_params = {}
        if args.kkt_sufficient is not None:
            kkt_params["sufficient"] = args.kkt_sufficient
        if args.kkt_necessary is not None:
            kkt_params["necessary"] = args.kkt_necessary
        if args.kkt_artificial is not None:
            kkt_params["artificial"] = args.kkt_artificial
        print(f"KKT parameters: {kkt_params}")
    print()

    case_results = _collect_case_results(
        case_paths,
        max_K_multi=args.max_K_multi,
        tol=args.tol,
        check_every=args.check_every,
        objective_stride=args.objective_stride,
        precond=args.precond,
        precond_tol=args.precond_tol,
        precond_iters=args.precond_iters,
        beta_params=beta_params,
        restart_mode=args.restart_mode,
        kkt_params=kkt_params,
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
                "reference_objective": reference,
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
            f"{case_path.name}: ref={reference:.6f}, "
            f"obj={result.objective:.6f}, gap={result.objective - reference:.3e}, "
            f"k-mults={result.k_multiplications}, "
            f"outer={result.outer_iterations}, iters={result.iterations}, converged={result.converged}"
        )
    print()

    # Show verbose output if requested or if only one case
    if args.verbose or len(case_results) == 1:
        for case_path, result, reference in case_results:
            _print_summary(case_path, reference, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())