# PDHG Coursework Implementation

This project provides a minimal implementation of the Primal-Dual Hybrid Gradient (PDHG) method for linear programs given in MPS format. The workflow:

1. Read the LP instance with HiGHS and convert it to a dense canonical form.
2. Optionally apply diagonal preconditioning (Ruiz, Pock–Chambolle, or both).
3. Run PDHG with simple projections and record convergence statistics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running PDHG

```bash
python run_PDHG.py --case /Users/huangjiacheng/OR772/lp_cases/testlp.mps --verbose
python run_PDHG.py --case /Users/huangjiacheng/OR772/lp_cases/qap15.mps.bz2 --verbose
```

Use `--precond {none,pc,ruiz,ruiz_pc}` to choose the scaling and adjust PDHG parameters with the other CLI options.

## Files

- `mps_process.py`: MPS reader and HiGHS-based reference solver.
- `lp_precondition.py`: Diagonal preconditioners (Ruiz and Pock–Chambolle).
- `operations.py`: Common projection and norm utilities.
- `algo_PDHG.py`: PDHG algorithm and result container.
- `main.py`: PDHG orchestration helpers (reused by other run scripts).
- `run_PDHG.py`: Command-line entry point for PDHG.

