from dataclasses import dataclass
from pathlib import Path

from algo_PDHG import PDHGResult, pdhg
from algo_PDLP import PDLPResult, pdlp
from lp_precondition import PrecondMethod, PreconditionerData, apply_preconditioner
from mps_process import LPData, get_reference_objective, read_mps
from typing import Optional, Dict, Union


@dataclass
class PreprocessedCase:
    """Preprocessed case data for experiments (loaded once, reused multiple times)."""
    
    case_path: Path
    case_name: str
    lp: LPData  # Original LP data
    lp_scaled: LPData  # Preconditioned LP data
    precond_data: PreconditionerData
    reference_objective: float


def preload_cases(
    cases_folder: Path,
    *,
    precond: PrecondMethod = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
) -> Dict[str, PreprocessedCase]:
    """
    Preload and preprocess all cases from a folder.
    
    This function reads all MPS files, applies preconditioning, and computes norms
    once at the beginning. The results can be reused across multiple algorithm runs
    with different parameters, improving efficiency and ensuring reproducibility.
    
    Args:
        cases_folder: Path to folder containing MPS files
        precond: Preconditioning method
        precond_tol: Preconditioning tolerance
        precond_iters: Maximum preconditioning iterations
        
    Returns:
        Dictionary mapping case names to PreprocessedCase objects
    """
    case_paths = resolve_case_paths(cases_folder)
    preprocessed: Dict[str, PreprocessedCase] = {}
    
    print(f"Preloading {len(case_paths)} cases...")
    for case_path in case_paths:
        case_name = case_path.stem
        print(f"  Loading {case_name}...")
        
        reference_objective = get_reference_objective(str(case_path))
        lp = read_mps(str(case_path))
        lp_scaled, precond_data = apply_preconditioner(
            lp,
            method=precond,
            tol=precond_tol,
            max_iters=precond_iters,
        )
        
        preprocessed[case_name] = PreprocessedCase(
            case_path=case_path,
            case_name=case_name,
            lp=lp,
            lp_scaled=lp_scaled,
            precond_data=precond_data,
            reference_objective=reference_objective,
        )
    
    print(f"Preloaded {len(preprocessed)} cases.")
    return preprocessed


def run_pdhg(
    case_path: str,
    *,
    max_K_multi: int = 100000,
    tol: float = 1e-5,
    omega: float = 1.0,
    eta_scale: float = 0.9,
    check_every: int = 25,
    objective_stride: int = 0,
    precond: PrecondMethod = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
    power_iterations: int = 20,
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
        max_K_multi=max_K_multi,
        tol=tol,
        omega=omega,
        eta_scale=eta_scale,
        check_every=check_every,
        objective_stride=objective_stride,
        reference_objective=reference_objective,
        precond=precond_data if precond != "none" else None,
        power_iterations=power_iterations,
    )

    return result, precond_data, lp, reference_objective


def run_pdlp(
    case_path_or_preprocessed: Union[str, PreprocessedCase],
    *,
    max_K_multi: int = 100000,
    tol: float = 1e-6,
    check_every: int = 50,
    objective_stride: int = 0,
    precond: PrecondMethod = "ruiz_pc",
    precond_tol: float = 1e-2,
    precond_iters: int = 10,
    beta_params: Optional[Dict[str, float]] = None,
    restart_mode: str = "normalized_gap",
    kkt_params: Optional[Dict[str, float]] = None,
    enable_adaptive_step: bool = True,
    enable_primal_weight: bool = True,
) -> tuple[PDLPResult, PreconditionerData, LPData, float]:
    """
    Run PDLP algorithm on a case.
    
    Args:
        case_path_or_preprocessed: Either a path string to an MPS file, or a PreprocessedCase
                                   object. If PreprocessedCase is provided, the preprocessed data
                                   is reused (more efficient for experiments).
        Other arguments: Algorithm parameters
        
    Returns:
        Tuple of (result, precond_data, lp, reference_objective)
    """
    # Handle both string path and PreprocessedCase
    if isinstance(case_path_or_preprocessed, PreprocessedCase):
        # Use preprocessed data
        preprocessed = case_path_or_preprocessed
        lp_scaled = preprocessed.lp_scaled
        precond_data = preprocessed.precond_data
        lp = preprocessed.lp
        reference_objective = preprocessed.reference_objective
    else:
        # Load from path (backward compatibility)
        path = Path(case_path_or_preprocessed).expanduser().resolve()
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
        max_K_multi=max_K_multi,
        tol=tol,
        check_every=check_every,
        objective_stride=objective_stride,
        reference_objective=reference_objective,
        precond=precond_data if precond != "none" else None,
        beta_params=beta_params,
        restart_mode=restart_mode,
        kkt_params=kkt_params,
        enable_adaptive_step=enable_adaptive_step,
        enable_primal_weight=enable_primal_weight,
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


__all__ = ["run_pdhg", "run_pdlp", "resolve_case_paths", "preload_cases", "PreprocessedCase"]