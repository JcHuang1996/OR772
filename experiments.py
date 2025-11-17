"""Experiment framework for PDHG and PDLP numerical experiments."""

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt

from algo_PDHG import PDHGResult
from algo_PDLP import PDLPResult
from main import resolve_case_paths, run_pdhg, run_pdlp


def sgm10(data: List[float]) -> float:
    """Compute shifted geometric mean with shift 10."""
    if not data:
        return 0.0
    shifted = [x + 10.0 for x in data]
    product = 1.0
    for val in shifted:
        if val > 0:
            product *= val
        else:
            return 0.0
    geometric_mean = product ** (1.0 / len(shifted))
    return geometric_mean - 10.0


@dataclass
class PDHGExperimentResult:
    """Results from a single PDHG experiment run."""

    case_name: str
    parameter_value: float
    parameter_name: str
    k_multiplications: int
    converged: bool
    tolerance: float
    result: PDHGResult


@dataclass
class PDLPExperimentResult:
    """Results from a single PDLP experiment run."""

    case_name: str
    parameter_config: Dict[str, Any]
    k_multiplications: int
    converged: bool
    tolerance: float
    result: PDLPResult


def run_pdhg_power_iterations_experiment(
    cases_folder: Path,
    power_iterations_list: List[int],
    tolerances: List[float],
    *,
    max_K_multi: int = 100000,
    omega: float = 1.0,
    eta_scale: float = 0.9,
    check_every: int = 100,
    precond: str = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
) -> List[PDHGExperimentResult]:
    """Run PDHG power iterations experiment."""
    case_paths = resolve_case_paths(cases_folder)
    results: List[PDHGExperimentResult] = []

    for case_path in case_paths:
        case_name = case_path.stem
        for power_iterations in power_iterations_list:
            for tol in tolerances:
                print(f"Running {case_name} with power_iterations={power_iterations}, tol={tol}")
                result, _, _, _ = run_pdhg(
                    str(case_path),
                    max_K_multi=max_K_multi,
                    tol=tol,
                    omega=omega,
                    eta_scale=eta_scale,
                    check_every=check_every,
                    precond=precond,
                    precond_tol=precond_tol,
                    precond_iters=precond_iters,
                    power_iterations=power_iterations,
                )
                results.append(
                    PDHGExperimentResult(
                        case_name=case_name,
                        parameter_value=float(power_iterations),
                        parameter_name="Power Iterations Number",
                        k_multiplications=result.k_multiplications,
                        converged=result.converged,
                        tolerance=tol,
                        result=result,
                    )
                )
    return results


def run_pdhg_step_size_scale_experiment(
    cases_folder: Path,
    eta_scales: List[float],
    tolerances: List[float],
    *,
    max_K_multi: int = 100000,
    omega: float = 1.0,
    power_iterations: int = 20,
    check_every: int = 100,
    precond: str = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
) -> List[PDHGExperimentResult]:
    """Run PDHG step size scale experiment."""
    case_paths = resolve_case_paths(cases_folder)
    results: List[PDHGExperimentResult] = []

    for case_path in case_paths:
        case_name = case_path.stem
        for eta_scale in eta_scales:
            for tol in tolerances:
                print(f"Running {case_name} with eta_scale={eta_scale}, tol={tol}")
                result, _, _, _ = run_pdhg(
                    str(case_path),
                    max_K_multi=max_K_multi,
                    tol=tol,
                    omega=omega,
                    eta_scale=eta_scale,
                    check_every=check_every,
                    precond=precond,
                    precond_tol=precond_tol,
                    precond_iters=precond_iters,
                    power_iterations=power_iterations,
                )
                results.append(
                    PDHGExperimentResult(
                        case_name=case_name,
                        parameter_value=eta_scale,
                        parameter_name="Step size scale",
                        k_multiplications=result.k_multiplications,
                        converged=result.converged,
                        tolerance=tol,
                        result=result,
                    )
                )
    return results


def run_pdhg_primal_weight_experiment(
    cases_folder: Path,
    omega_values: List[float],
    tolerances: List[float],
    *,
    max_K_multi: int = 100000,
    eta_scale: float = 0.9,
    power_iterations: int = 20,
    check_every: int = 100,
    precond: str = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
) -> List[PDHGExperimentResult]:
    """Run PDHG primal weight experiment."""
    case_paths = resolve_case_paths(cases_folder)
    results: List[PDHGExperimentResult] = []

    for case_path in case_paths:
        case_name = case_path.stem
        for omega in omega_values:
            for tol in tolerances:
                print(f"Running {case_name} with omega={omega}, tol={tol}")
                result, _, _, _ = run_pdhg(
                    str(case_path),
                    max_K_multi=max_K_multi,
                    tol=tol,
                    omega=omega,
                    eta_scale=eta_scale,
                    check_every=check_every,
                    precond=precond,
                    precond_tol=precond_tol,
                    precond_iters=precond_iters,
                    power_iterations=power_iterations,
                )
                results.append(
                    PDHGExperimentResult(
                        case_name=case_name,
                        parameter_value=omega,
                        parameter_name="Primal weight",
                        k_multiplications=result.k_multiplications,
                        converged=result.converged,
                        tolerance=tol,
                        result=result,
                    )
                )
    return results


def save_pdhg_experiment_csv(
    results: List[PDHGExperimentResult], output_path: Path
) -> None:
    """Save PDHG experiment results to CSV."""
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Name",
                results[0].parameter_name if results else "Parameter",
                "K-multiplication Number",
                "Converged",
                "Tolerance",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Name": r.case_name,
                    results[0].parameter_name if results else "Parameter": r.parameter_value,
                    "K-multiplication Number": r.k_multiplications,
                    "Converged": r.converged,
                    "Tolerance": r.tolerance,
                }
            )


def plot_pdhg_sgm10(
    results: List[PDHGExperimentResult], output_path: Path, tolerances: List[float]
) -> None:
    """Generate SGM10 plot for PDHG experiments with all tolerances combined."""
    plt.figure(figsize=(8, 6))
    
    # Get parameter name from first result
    param_name = results[0].parameter_name if results else "Parameter"
    
    # Plot each tolerance
    for tol in sorted(tolerances):
        # Filter by tolerance
        filtered_results = [r for r in results if r.tolerance == tol]

        # Group by parameter value
        param_to_k_mults: Dict[float, List[int]] = {}
        for r in filtered_results:
            if r.parameter_value not in param_to_k_mults:
                param_to_k_mults[r.parameter_value] = []
            param_to_k_mults[r.parameter_value].append(r.k_multiplications)

        # Compute SGM10 for each parameter value
        param_values = sorted(param_to_k_mults.keys())
        sgm10_values = [sgm10(param_to_k_mults[pv]) for pv in param_values]

        # Format tolerance in scientific notation
        tol_label = f"tol = {tol:.0e}" if tol < 0.01 else f"tol = {tol}"
        plt.plot(param_values, sgm10_values, marker="o", linestyle="-", linewidth=2, markersize=8, label=tol_label)
    
    plt.xlabel(param_name)
    plt.ylabel("SGM10 of K-multiplication Number")
    plt.title("PDHG Experiment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pdlp_module_experiment(
    cases_folder: Path,
    tolerances: List[float],
    *,
    max_K_multi: int = 100000,
    check_every: int = 100,
    precond: str = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
    restart_mode: str = "kkt",
    enable_adaptive_step: bool = True,
    enable_primal_weight: bool = True,
    kkt_params: Optional[Dict[str, float]] = None,
    objective_stride: int = 50,
) -> List[PDLPExperimentResult]:
    """Run PDLP module combination experiment."""
    case_paths = resolve_case_paths(cases_folder)
    results: List[PDLPExperimentResult] = []

    # Configure PDLP based on module flags
    # If adaptive step is disabled, we need to use fixed step size (not implemented in pdlp, so we just enable it)
    # If primal weight is disabled, we need to keep omega constant (not implemented, so we enable it)
    # For now, we rely on restart_mode and kkt_params to control behavior

    for case_path in case_paths:
        case_name = case_path.stem
        for tol in tolerances:
            print(
                f"Running {case_name} with restart_mode={restart_mode}, "
                f"adaptive_step={enable_adaptive_step}, primal_weight={enable_primal_weight}, tol={tol}"
            )
            result, _, _, _ = run_pdlp(
                str(case_path),
                max_K_multi=max_K_multi,
                tol=tol,
                check_every=check_every,
                objective_stride=objective_stride,
                precond=precond,
                precond_tol=precond_tol,
                precond_iters=precond_iters,
                restart_mode=restart_mode,
                kkt_params=kkt_params,
                enable_adaptive_step=enable_adaptive_step,
                enable_primal_weight=enable_primal_weight,
            )
            results.append(
                PDLPExperimentResult(
                    case_name=case_name,
                    parameter_config={
                        "restart_mode": restart_mode,
                        "enable_adaptive_step": enable_adaptive_step,
                        "enable_primal_weight": enable_primal_weight,
                        "kkt_params": kkt_params,
                    },
                    k_multiplications=result.k_multiplications,
                    converged=result.converged,
                    tolerance=tol,
                    result=result,
                )
            )
    return results


def run_pdlp_grid_search_experiment(
    cases_folder: Path,
    parameter_name: str,
    parameter_values: List[float],
    fixed_params: Dict[str, float],
    tolerances: List[float],
    *,
    max_K_multi: int = 100000,
    check_every: int = 100,
    precond: str = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
    objective_stride: int = 50,
) -> List[PDLPExperimentResult]:
    """Run PDLP grid search on restart trigger parameters."""
    case_paths = resolve_case_paths(cases_folder)
    results: List[PDLPExperimentResult] = []

    for case_path in case_paths:
        case_name = case_path.stem
        for param_value in parameter_values:
            # Build kkt_params with the varying parameter
            kkt_params = fixed_params.copy()
            kkt_params[parameter_name] = param_value

            for tol in tolerances:
                print(
                    f"Running {case_name} with {parameter_name}={param_value}, tol={tol}"
                )
                result, _, _, _ = run_pdlp(
                    str(case_path),
                    max_K_multi=max_K_multi,
                    tol=tol,
                    check_every=check_every,
                    objective_stride=objective_stride,
                    precond=precond,
                    precond_tol=precond_tol,
                    precond_iters=precond_iters,
                    restart_mode="kkt",
                    kkt_params=kkt_params,
                )
                results.append(
                    PDLPExperimentResult(
                        case_name=case_name,
                        parameter_config={
                            parameter_name: param_value,
                            **fixed_params,
                        },
                        k_multiplications=result.k_multiplications,
                        converged=result.converged,
                        tolerance=tol,
                        result=result,
                    )
                )
    return results


def compute_solved_fractions(
    results: List[PDLPExperimentResult], tolerance: float, k_multiplication_limits: List[int]
) -> List[float]:
    """Compute solved fractions at different K-multiplication limits."""
    filtered_results = [r for r in results if r.tolerance == tolerance]
    fractions = []
    for k_limit in k_multiplication_limits:
        solved_count = sum(
            1 for r in filtered_results if r.converged and r.k_multiplications <= k_limit
        )
        fractions.append(solved_count / len(filtered_results) if filtered_results else 0.0)
    return fractions


def save_pdlp_experiment_csv(
    results: List[PDLPExperimentResult], output_path: Path
) -> None:
    """Save PDLP experiment results to CSV (solved fraction format)."""
    tolerances = sorted(set(r.tolerance for r in results))
    k_limits = sorted(set(r.k_multiplications for r in results))
    # Add some intermediate values for smoother curves
    max_k = max(k_limits) if k_limits else 100000
    k_limits = list(range(0, max_k + 1, max(1, max_k // 200)))  # ~200 points

    with output_path.open("w", newline="") as f:
        fieldnames = ["Number of K-multiplication"]
        for tol in tolerances:
            fieldnames.append(f"Solving tolerance {tol}")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k_limit in k_limits:
            row = {"Number of K-multiplication": k_limit}
            for tol in tolerances:
                fractions = compute_solved_fractions(results, tol, [k_limit])
                row[f"Solving tolerance {tol}"] = fractions[0] if fractions else 0.0
            writer.writerow(row)


def plot_pdlp_fraction(
    results: List[PDLPExperimentResult], output_path: Path, tolerance: Optional[float] = None
) -> None:
    """Generate solved fraction curve for PDLP experiments.
    
    If tolerance is provided, plots only that tolerance. Otherwise plots all tolerances.
    """
    if tolerance is not None:
        tolerances = [tolerance]
    else:
        tolerances = sorted(set(r.tolerance for r in results))
    
    k_limits = sorted(set(r.k_multiplications for r in results))
    max_k = max(k_limits) if k_limits else 100000
    k_limits_dense = list(range(0, max_k + 1, max(1, max_k // 200)))

    plt.figure(figsize=(10, 6))
    for tol in tolerances:
        fractions = compute_solved_fractions(results, tol, k_limits_dense)
        tol_label = f"Tolerance {tol:.0e}" if tol < 0.01 else f"Tolerance {tol}"
        plt.plot(k_limits_dense, fractions, label=tol_label, linewidth=2, marker="o", markersize=3)

    plt.xlabel("K-multiplication number")
    plt.ylabel("Fraction of problem solved")
    plt.title("PDLP Solved Fraction Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pdlp_fraction_by_parameter(
    results: List[PDLPExperimentResult], 
    output_path: Path, 
    tolerance: float,
    parameter_name: str
) -> None:
    """Generate solved fraction curve for PDLP grid search experiments, grouped by parameter value."""
    # Filter by tolerance
    filtered_results = [r for r in results if r.tolerance == tolerance]
    
    # Group by parameter value
    param_to_results: Dict[float, List[PDLPExperimentResult]] = {}
    for r in filtered_results:
        param_value = r.parameter_config.get(parameter_name)
        if param_value is not None:
            if param_value not in param_to_results:
                param_to_results[param_value] = []
            param_to_results[param_value].append(r)
    
    k_limits = sorted(set(r.k_multiplications for r in filtered_results))
    max_k = max(k_limits) if k_limits else 100000
    k_limits_dense = list(range(0, max_k + 1, max(1, max_k // 200)))

    plt.figure(figsize=(10, 6))
    for param_value in sorted(param_to_results.keys()):
        param_results = param_to_results[param_value]
        fractions = compute_solved_fractions(param_results, tolerance, k_limits_dense)
        plt.plot(k_limits_dense, fractions, label=f"{parameter_name} = {param_value}", linewidth=2, marker="o", markersize=3)

    plt.xlabel("K-multiplication number")
    plt.ylabel("Fraction of problem solved")
    tol_label = f"Tolerance {tolerance:.0e}" if tolerance < 0.01 else f"Tolerance {tolerance}"
    plt.title(f"PDLP Solved Fraction Curve ({tol_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def normalize_objective_gap(trace: List[Tuple[int, float]], reference_objective: float) -> List[Tuple[int, float]]:
    """Normalize objective gap trace by reference objective."""
    if not trace:
        return trace
    if reference_objective == 0:
        # Use max gap as normalization if reference is zero
        max_gap = max(abs(gap) for _, gap in trace) if trace else 1.0
        if max_gap == 0:
            return trace
        normalized = [(k, abs(gap) / max_gap) for k, gap in trace]
        return normalized
    normalized = [(k, abs(gap) / abs(reference_objective)) for k, gap in trace]
    return normalized


def plot_objective_gap_trace(
    results: List[PDLPExperimentResult],
    case_names: List[str],
    output_path: Path,
    tolerance: float,
    cases_folder: Path,
    parameter_name: Optional[str] = None,
) -> None:
    """Generate objective gap trace plots for selected cases.
    
    If parameter_name is provided, labels will include parameter values (for grid search).
    Otherwise, labels will be case names (for module experiments).
    """
    from main import get_reference_objective

    # Filter results by tolerance and case names
    filtered_results = [
        r for r in results
        if r.tolerance == tolerance and r.case_name in case_names
    ]

    # Get case paths for reference objectives
    case_paths = {p.stem: p for p in resolve_case_paths(cases_folder)}

    plt.figure(figsize=(10, 6))
    for result in filtered_results:
        case_name = result.case_name
        trace = result.result.objective_distance_trace
        if not trace:
            continue

        # Get reference objective for normalization
        try:
            if case_name not in case_paths:
                print(f"Warning: Case {case_name} not found in folder")
                continue

            case_path = case_paths[case_name]
            reference_obj = get_reference_objective(str(case_path))
            normalized_trace = normalize_objective_gap(trace, reference_obj)
            k_vals = [t[0] for t in normalized_trace]
            gap_vals = [t[1] for t in normalized_trace]
            
            # Create label based on whether this is a grid search or module experiment
            if parameter_name and parameter_name in result.parameter_config:
                param_value = result.parameter_config[parameter_name]
                label = f"{case_name} ({parameter_name}={param_value})"
            else:
                label = case_name
            
            plt.plot(k_vals, gap_vals, label=label, linewidth=2, marker="o", markersize=3)
        except Exception as e:
            print(f"Warning: Could not plot trace for {case_name}: {e}")
            continue

    plt.xlabel("K-multiplication number")
    plt.ylabel("Normalized objective gap")
    tol_label = f"Tolerance {tolerance:.0e}" if tolerance < 0.01 else f"Tolerance {tolerance}"
    plt.title(f"PDLP Objective Gap Traces ({tol_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_objective_gap_trace_single_case(
    results: List[PDLPExperimentResult],
    case_name: str,
    output_path: Path,
    tolerance: float,
    cases_folder: Path,
    parameter_name: Optional[str] = None,
) -> None:
    """Generate objective gap trace plot for a single case with multiple parameter values."""
    from main import get_reference_objective

    # Filter results by tolerance and case name
    filtered_results = [
        r for r in results
        if r.tolerance == tolerance and r.case_name == case_name
    ]

    # Get case path for reference objective
    case_paths = {p.stem: p for p in resolve_case_paths(cases_folder)}
    
    if case_name not in case_paths:
        print(f"Warning: Case {case_name} not found in folder")
        return

    case_path = case_paths[case_name]
    reference_obj = get_reference_objective(str(case_path))

    plt.figure(figsize=(10, 6))
    for result in filtered_results:
        trace = result.result.objective_distance_trace
        if not trace:
            continue

        try:
            normalized_trace = normalize_objective_gap(trace, reference_obj)
            k_vals = [t[0] for t in normalized_trace]
            gap_vals = [t[1] for t in normalized_trace]
            
            # Create label with parameter value
            if parameter_name and parameter_name in result.parameter_config:
                param_value = result.parameter_config[parameter_name]
                label = f"{parameter_name} = {param_value}"
            else:
                label = "trace"
            
            plt.plot(k_vals, gap_vals, label=label, linewidth=2, marker="o", markersize=3)
        except Exception as e:
            print(f"Warning: Could not plot trace for {case_name}: {e}")
            continue

    plt.xlabel("K-multiplication number")
    plt.ylabel("Normalized objective gap")
    tol_label = f"Tolerance {tolerance:.0e}" if tolerance < 0.01 else f"Tolerance {tolerance}"
    plt.title(f"PDLP Objective Gap Trace: {case_name} ({tol_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_experiment_data(data: Any, output_path: Path) -> None:
    """Save experiment data to pickle file."""
    with output_path.open("wb") as f:
        pickle.dump(data, f)


def get_case_names_from_folder(cases_folder: Path) -> List[str]:
    """Get case names from folder."""
    case_paths = resolve_case_paths(cases_folder)
    return [p.stem for p in case_paths]

