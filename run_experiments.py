"""Unified command-line entry point for all PDHG and PDLP experiments."""

import argparse
from datetime import datetime
from pathlib import Path

from typing import Dict, List, Optional

from experiments import (
    run_pdhg_power_iterations_experiment,
    run_pdhg_primal_weight_experiment,
    run_pdhg_step_size_scale_experiment,
    save_pdhg_experiment_csv,
    plot_pdhg_sgm10,
    save_experiment_data,
    run_pdlp_module_experiment,
    run_pdlp_grid_search_experiment,
    save_pdlp_experiment_csv,
    plot_pdlp_fraction_by_parameter,
    plot_objective_gap_trace_single_case,
    get_case_names_from_folder,
    compute_solved_fractions,
    normalize_objective_gap,
)
from main import resolve_case_paths


# Define all experiment types
PDHG_EXPERIMENTS = ["power_iterations", "step_size_scale", "primal_weight"]
PDLP_EXPERIMENTS = [
    "restart_primal_weight",
    "restart_adaptive_step",
    "restart_all_modules",
    "grid_search_artificial",
    "grid_search_necessary",
    "grid_search_sufficient",
]


def generate_merged_module_plots(
    all_module_results: Dict[str, List],
    cases_folder: Path,
    output_base: Path,
    tolerances: List[float],
    selected_cases: Optional[List[str]] = None,
) -> None:
    """Generate merged plots for PDLP module experiments."""
    from experiments import PDLPExperimentResult
    
    merged_plots_dir = output_base / "merged_module_plots"
    merged_plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating merged plots for module experiments")
    print(f"{'='*60}")
    print(f"Output directory: {merged_plots_dir}")
    
    # Get all case names
    if selected_cases:
        case_names = selected_cases
    else:
        all_cases = get_case_names_from_folder(cases_folder)
        case_names = all_cases[:4] if len(all_cases) >= 4 else all_cases
    
    # Generate fraction plots: one per tolerance, with curves for each experiment type
    for tol in tolerances:
        fraction_plot_path = merged_plots_dir / f"plot_fraction_tol{tol:.0e}.png"
        
        # Group results by experiment type for this tolerance
        import matplotlib.pyplot as plt
        k_limits_all = set()
        for exp_type, results in all_module_results.items():
            for r in results:
                if r.tolerance == tol:
                    k_limits_all.add(r.k_multiplications)
        
        max_k = max(k_limits_all) if k_limits_all else 100000
        k_limits_dense = list(range(0, max_k + 1, max(1, max_k // 200)))
        
        plt.figure(figsize=(10, 6))
        for exp_type, results in all_module_results.items():
            filtered_results = [r for r in results if r.tolerance == tol]
            if filtered_results:
                fractions = compute_solved_fractions(filtered_results, tol, k_limits_dense)
                plt.plot(k_limits_dense, fractions, label=exp_type, linewidth=2, marker="o", markersize=3)
        
        plt.xlabel("K-multiplication number")
        plt.ylabel("Fraction of problem solved")
        tol_label = f"Tolerance {tol:.0e}" if tol < 0.01 else f"Tolerance {tol}"
        plt.title(f"PDLP Solved Fraction Curve ({tol_label})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fraction_plot_path, dpi=150)
        plt.close()
        print(f"Merged fraction plot saved to {fraction_plot_path}")
    
    # Generate objective gap trace plots: one per case per tolerance, with curves for each experiment type
    for case_name in case_names:
        for tol in tolerances:
            trace_plot_path = merged_plots_dir / f"plot_objective_gap_{case_name}_tol{tol:.0e}.png"
            
            # Collect all results for this case and tolerance, grouped by experiment type
            all_results_for_case_tol: List[PDLPExperimentResult] = []
            for exp_type, results in all_module_results.items():
                for r in results:
                    if r.tolerance == tol and r.case_name == case_name:
                        # Add experiment type to label by modifying parameter_config temporarily
                        # Create a copy to avoid modifying original
                        import copy
                        r_copy = copy.deepcopy(r)
                        r_copy.parameter_config["_experiment_type"] = exp_type
                        all_results_for_case_tol.append(r_copy)
            
            if all_results_for_case_tol:
                # Use a custom plotting approach to label by experiment type
                from main import get_reference_objective
                
                case_paths = {p.stem: p for p in resolve_case_paths(cases_folder)}
                if case_name not in case_paths:
                    print(f"Warning: Case {case_name} not found in folder")
                    continue
                
                case_path = case_paths[case_name]
                reference_obj = get_reference_objective(str(case_path))
                
                plt.figure(figsize=(10, 6))
                for result in all_results_for_case_tol:
                    trace = result.result.objective_distance_trace
                    if not trace:
                        continue
                    
                    try:
                        normalized_trace = normalize_objective_gap(trace, reference_obj)
                        k_vals = [t[0] for t in normalized_trace]
                        gap_vals = [t[1] for t in normalized_trace]
                        exp_type = result.parameter_config.get("_experiment_type", "unknown")
                        plt.plot(k_vals, gap_vals, label=exp_type, linewidth=2, marker="o", markersize=3)
                    except Exception as e:
                        print(f"Warning: Could not plot trace for {case_name}: {e}")
                        continue
                
                plt.xlabel("K-multiplication number")
                plt.ylabel("Normalized objective gap")
                tol_label = f"Tolerance {tol:.0e}" if tol < 0.01 else f"Tolerance {tol}"
                plt.title(f"PDLP Objective Gap Trace: {case_name} ({tol_label})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale("log")
                plt.tight_layout()
                plt.savefig(trace_plot_path, dpi=150)
                plt.close()
                print(f"Merged objective gap trace plot saved to {trace_plot_path}")


def run_single_experiment(
    algorithm: str,
    experiment_type: str,
    cases_folder: Path,
    output_base: Path,
    args,
    selected_cases: Optional[List[str]] = None,
) -> int:
    """Run a single experiment and save results."""
    experiment_dir = output_base / experiment_type
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {algorithm.upper()} experiment: {experiment_type}")
    print(f"{'='*60}")
    print(f"Cases folder: {cases_folder}")
    print(f"Output directory: {experiment_dir}")
    print()

    # Run PDHG experiments
    if algorithm == "pdhg":
        if experiment_type == "power_iterations":
            results = run_pdhg_power_iterations_experiment(
                cases_folder,
                args.power_iterations,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
            )
        elif experiment_type == "step_size_scale":
            results = run_pdhg_step_size_scale_experiment(
                cases_folder,
                args.eta_scales,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
            )
        elif experiment_type == "primal_weight":
            results = run_pdhg_primal_weight_experiment(
                cases_folder,
                args.omega_values,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
            )
        else:
            print(f"Error: Unknown PDHG experiment type: {experiment_type}")
            return 1

        # Save CSV
        csv_path = experiment_dir / "summary.csv"
        save_pdhg_experiment_csv(results, csv_path)
        print(f"CSV saved to {csv_path}")

        # Generate combined plot for all tolerances
        plot_path = experiment_dir / "plot_sgm10.png"
        plot_pdhg_sgm10(results, plot_path, args.tolerances)
        print(f"Plot saved to {plot_path}")

        # Save pickle data
        pickle_path = experiment_dir / "data.pkl"
        save_experiment_data(results, pickle_path)
        print(f"Pickle data saved to {pickle_path}")

    # Run PDLP experiments
    elif algorithm == "pdlp":
        if experiment_type == "restart_primal_weight":
            results = run_pdlp_module_experiment(
                cases_folder,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
                restart_mode="kkt",
                enable_adaptive_step=False,
                enable_primal_weight=True,
            )
        elif experiment_type == "restart_adaptive_step":
            results = run_pdlp_module_experiment(
                cases_folder,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
                restart_mode="kkt",
                enable_adaptive_step=True,
                enable_primal_weight=False,
            )
        elif experiment_type == "restart_all_modules":
            results = run_pdlp_module_experiment(
                cases_folder,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
                restart_mode="kkt",
                enable_adaptive_step=True,
                enable_primal_weight=True,
            )
        elif experiment_type == "grid_search_artificial":
            fixed_params = {"necessary": 0.8, "sufficient": 0.2}
            results = run_pdlp_grid_search_experiment(
                cases_folder,
                "artificial",
                args.kkt_artificial,
                fixed_params,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
            )
        elif experiment_type == "grid_search_necessary":
            fixed_params = {"sufficient": 0.2, "artificial": 0.36}
            results = run_pdlp_grid_search_experiment(
                cases_folder,
                "necessary",
                args.kkt_necessary,
                fixed_params,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
            )
        elif experiment_type == "grid_search_sufficient":
            fixed_params = {"necessary": 0.8, "artificial": 0.36}
            results = run_pdlp_grid_search_experiment(
                cases_folder,
                "sufficient",
                args.kkt_sufficient,
                fixed_params,
                args.tolerances,
                max_K_multi=args.max_K_multi,
                check_every=args.check_every,
            )
        else:
            print(f"Error: Unknown PDLP experiment type: {experiment_type}")
            return 1

        # Save CSV
        csv_path = experiment_dir / "summary.csv"
        save_pdlp_experiment_csv(results, csv_path)
        print(f"CSV saved to {csv_path}")

        # Handle module experiments differently from grid search
        if experiment_type in ["restart_primal_weight", "restart_adaptive_step", "restart_all_modules"]:
            # Module experiments: don't generate plots here, will be merged later
            print("Note: Plots for module experiments will be generated in merged folder")
        else:
            # Grid search experiments: generate plots with parameter values
            # Determine parameter name from experiment type
            if experiment_type == "grid_search_artificial":
                param_name = "artificial"
            elif experiment_type == "grid_search_necessary":
                param_name = "necessary"
            elif experiment_type == "grid_search_sufficient":
                param_name = "sufficient"
            else:
                param_name = None

            # Generate solved fraction plots for each tolerance
            for tol in args.tolerances:
                fraction_plot_path = experiment_dir / f"plot_fraction_tol{tol:.0e}.png"
                plot_pdlp_fraction_by_parameter(results, fraction_plot_path, tol, param_name)
                print(f"Solved fraction plot saved to {fraction_plot_path}")

            # Generate objective gap trace plots for each case-tolerance combination
            if selected_cases:
                case_names = selected_cases
            else:
                # Use all cases for grid search
                all_cases = get_case_names_from_folder(cases_folder)
                case_names = all_cases

            for case_name in case_names:
                for tol in args.tolerances:
                    trace_plot_path = experiment_dir / f"plot_objective_gap_{case_name}_tol{tol:.0e}.png"
                    plot_objective_gap_trace_single_case(
                        results, case_name, trace_plot_path, tol, cases_folder, param_name
                    )
                    print(f"Objective gap trace plot saved to {trace_plot_path}")

        # Save pickle data
        pickle_path = experiment_dir / "data.pkl"
        save_experiment_data(results, pickle_path)
        print(f"Pickle data saved to {pickle_path}")

    print(f"✓ Experiment '{experiment_type}' completed successfully!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run numerical experiments for PDHG or PDLP algorithms."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["pdhg", "pdlp", "all"],
        help="Algorithm to run experiments for: pdhg, pdlp, or all",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        help="Experiment type to run (required unless --run-all is used)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments for the specified algorithm (or both if algorithm='all')",
    )
    parser.add_argument(
        "--cases-folder",
        type=Path,
        required=True,
        help="Folder containing MPS case files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output/)",
    )
    parser.add_argument(
        "--selected-cases",
        type=str,
        nargs="*",
        default=None,
        help="List of case names for objective gap trace plots (for PDLP)",
    )

    # PDHG-specific arguments
    parser.add_argument(
        "--power-iterations",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25],
        help="Power iteration values for power_iterations experiment (default: 10 15 20 25)",
    )
    parser.add_argument(
        "--eta-scales",
        type=float,
        nargs="+",
        default=[0.8, 0.85, 0.9, 0.95],
        help="Step size scales for step_size_scale experiment (default: 0.8 0.85 0.9 0.95)",
    )
    parser.add_argument(
        "--omega-values",
        type=float,
        nargs="+",
        default=[0.9, 0.95, 1.0, 1.05, 1.1],
        help="Primal weight values for primal_weight experiment (default: 0.9 0.95 1.0 1.05 1.1)",
    )

    # PDLP-specific arguments
    parser.add_argument(
        "--kkt-artificial",
        type=float,
        nargs="+",
        default=[0.16, 0.26, 0.36, 0.46, 0.56],
        help="KKT artificial parameter values for grid search (default: 0.16 0.26 0.36 0.46 0.56)",
    )
    parser.add_argument(
        "--kkt-necessary",
        type=float,
        nargs="+",
        default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="KKT necessary parameter values for grid search (default: 0.7 0.75 0.8 0.85 0.9 0.95)",
    )
    parser.add_argument(
        "--kkt-sufficient",
        type=float,
        nargs="+",
        default=[0.1, 0.15, 0.2, 0.25, 0.3],
        help="KKT sufficient parameter values for grid search (default: 0.1 0.15 0.2 0.25 0.3)",
    )

    # Common arguments
    parser.add_argument("--max-K-multi", type=int, default=100000, dest="max_K_multi", help="Max K-multiplications for both PDHG and PDLP")
    parser.add_argument(
        "--check-every",
        type=int,
        default=100,
        dest="check_every",
        help="Check convergence every N iterations (default: 100)",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[1e-4, 1e-8],
        help="Tolerance values to test (default: 1e-4 1e-8)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.run_all and (not args.algorithm or not args.experiment_type):
        parser.error(
            "Either --run-all must be specified, or both --algorithm and --experiment-type must be provided."
        )

    # Prepare output directory
    cases_folder = args.cases_folder.expanduser().resolve()
    if not cases_folder.exists():
        print(f"Error: Cases folder does not exist: {cases_folder}")
        return 1

    case_folder_name = cases_folder.name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Determine which experiments to run
    if args.run_all:
        if args.algorithm == "all" or args.algorithm is None:
            algorithms_to_run = ["pdhg", "pdlp"]
        else:
            algorithms_to_run = [args.algorithm]
        
        experiments_to_run = []
        for alg in algorithms_to_run:
            if alg == "pdhg":
                experiments_to_run.extend([("pdhg", exp) for exp in PDHG_EXPERIMENTS])
            elif alg == "pdlp":
                experiments_to_run.extend([("pdlp", exp) for exp in PDLP_EXPERIMENTS])
    else:
        # Single experiment
        experiments_to_run = [(args.algorithm, args.experiment_type)]

    # Run experiments
    total_experiments = len(experiments_to_run)
    print(f"\n{'='*70}")
    print(f"Starting batch run: {total_experiments} experiment(s)")
    print(f"Cases folder: {cases_folder}")
    print(f"{'='*70}\n")

    failed_experiments = []
    # Store module experiment results for merged plotting
    module_results: Dict[str, List] = {}
    
    for idx, (algorithm, experiment_type) in enumerate(experiments_to_run, 1):
        algorithm_prefix = algorithm.upper()
        output_base = args.output_root.expanduser().resolve() / f"{algorithm_prefix}-{case_folder_name}-{timestamp}"
        
        print(f"\n[{idx}/{total_experiments}] Processing: {algorithm.upper()} - {experiment_type}")
        
        result = run_single_experiment(
            algorithm=algorithm,
            experiment_type=experiment_type,
            cases_folder=cases_folder,
            output_base=output_base,
            args=args,
            selected_cases=args.selected_cases,
        )
        
        if result != 0:
            failed_experiments.append((algorithm, experiment_type))
        else:
            # Store module experiment results for later merging
            if algorithm == "pdlp" and experiment_type in ["restart_primal_weight", "restart_adaptive_step", "restart_all_modules"]:
                # Load results from pickle file
                experiment_dir = output_base / experiment_type
                pickle_path = experiment_dir / "data.pkl"
                if pickle_path.exists():
                    import pickle
                    with pickle_path.open("rb") as f:
                        module_results[experiment_type] = pickle.load(f)
    
    # Generate merged plots for module experiments if any were run
    if module_results:
        algorithm_prefix = "PDLP"
        output_base = args.output_root.expanduser().resolve() / f"{algorithm_prefix}-{case_folder_name}-{timestamp}"
        generate_merged_module_plots(
            module_results,
            cases_folder,
            output_base,
            args.tolerances,
            args.selected_cases,
        )

    # Final summary
    print(f"\n{'='*70}")
    print("Batch run completed!")
    print(f"{'='*70}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {total_experiments - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for alg, exp in failed_experiments:
            print(f"  - {alg.upper()}: {exp}")
        return 1
    
    print("\n✓ All experiments completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

