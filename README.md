# PD Solver Coursework Toolkit

This repository contains minimal Primal-Dual Hybrid Gradient (PDHG) and Primal-Dual Linear Programming (PDLP) solvers for LP instances provided in MPS or compressed `.mps.bz2` form. The shared workflow:

1. Load the LP with HiGHS into a dense canonical representation.
2. Optionally apply diagonal preconditioning (Ruiz, Pock–Chambolle, or both).
3. Run PDHG or PDLP while tracking convergence, matrix–vector products with `K`, and objective progress relative to a reference solver (HiGHS or Gurobi).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the solvers

Both entry points share the same structure:

```bash
# Single case
python run_PDHG.py lp_cases/testlp.mps --verbose
python run_PDLP.py lp_cases/testlp.mps

# Directory (all *.mps / *.mps.bz2 files)
python run_PDHG.py lp_cases/
python run_PDLP.py lp_cases/ --objective-stride 25
```

### Common options

- `--precond {none,pc,ruiz,ruiz_pc}`: choose diagonal scaling (default `ruiz_pc`).
- `--ref-solver {highs,gurobi}` / `--ref-verbose`: configure the reference objective solver.
- `--objective-stride N`: record the absolute objective gap every `N` matrix multiplications with `K` (`0` disables recording).
- `--output-root PATH`: base directory for timestamped outputs (defaults to `output/`).
- `--verbose`: print detailed per-case summaries (objective traces, residuals, etc.).

Algorithm-specific parameters are still available (`--max-iters`, `--omega`, `--check-every`, `--max-inner`, `--max-outer`, …).

## Running Experiments

The `run_experiments.py` script provides a framework for running systematic numerical experiments on PDHG and PDLP algorithms.

### PDLP Module Experiments

Run all three PDLP module experiments (restart + adaptive step size; restart + primal weight update; all modules) with merged output generation:

```bash
python run_experiments.py --algorithm pdlp --experiment-type restart_primal_weight --cases-folder lp_cases/netlib_exp && python run_experiments.py --algorithm pdlp --experiment-type restart_adaptive_step --cases-folder lp_cases/netlib_exp && python run_experiments.py --algorithm pdlp --experiment-type restart_all_modules --cases-folder lp_cases/netlib_exp
```

Or run them individually (merged plots are generated automatically when all three complete):

```bash
python run_experiments.py --algorithm pdlp --experiment-type restart_primal_weight --cases-folder lp_cases/netlib_exp
python run_experiments.py --algorithm pdlp --experiment-type restart_adaptive_step --cases-folder lp_cases/netlib_exp
python run_experiments.py --algorithm pdlp --experiment-type restart_all_modules --cases-folder lp_cases/netlib_exp
```

### PDLP Parameter Grid Search Experiments

Run a grid search experiment for PDLP restart parameters:

```bash
# Grid search over "necessary" parameter
python run_experiments.py --algorithm pdlp --experiment-type grid_search_necessary --cases-folder lp_cases/netlib_exp

# Grid search over "artificial" parameter
python run_experiments.py --algorithm pdlp --experiment-type grid_search_artificial --cases-folder lp_cases/netlib_exp

# Grid search over "sufficient" parameter
python run_experiments.py --algorithm pdlp --experiment-type grid_search_sufficient --cases-folder lp_cases/netlib_exp
```

**Customizing grid search parameters:**
- `--kkt-necessary`: necessary parameter values (default: `0.7 0.75 0.8 0.85 0.9 0.95`)
- `--kkt-artificial`: artificial parameter values (default: `0.16 0.26 0.36 0.46 0.56`)
- `--kkt-sufficient`: sufficient parameter values (default: `0.1 0.15 0.2 0.25 0.3`)
- `--tolerances`: tolerance values (default: `1e-4 1e-8`)
- `--max-K-multi`: maximum K-multiplications (default: `100000`)

All experiment outputs are saved in timestamped directories under the `output/` folder.

### Outputs

```text
output/
  └── 20250101-101530/
        ├── 20250101-101530_pdhg_summary.csv      # multi-case runs only
        ├── 20250101-101530_case_mps_pdhg_trace.csv
        └── …
```

Each run creates a timestamped subdirectory summarising:

- Console tables listing final objective, reference objective, absolute gap, total `K` multiplications, and convergence flag.
- A multi-case summary CSV (if more than one model is processed).
- Optional per-case trace CSVs (`*_trace.csv`) whenever `--objective-stride > 0`, capturing `(K multiplications, |objective − reference|)` samples.

## Files

- `mps_process.py`: MPS reader and HiGHS/Gurobi reference solver wrappers (stdout logging only).
- `lp_precondition.py`: Diagonal preconditioners (Ruiz and Pock–Chambolle).
- `operations.py`: Projection and norm utilities shared by both solvers.
- `algo_PDHG.py`: PDHG algorithm, instrumentation, and result container.
- `algo_PDLP.py`: PDLP algorithm with adaptive modules and instrumentation.
- `main.py`: Shared orchestration helpers (loading, preconditioning, reference solving).
- `run_PDHG.py`, `run_PDLP.py`: CLI entry points supporting single-case or batch execution.

