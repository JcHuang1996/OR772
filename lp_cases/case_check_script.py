# -*- coding: utf-8 -*-
# @Time     : 2025/11/14
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One-shot script:
1. Load reference CSV.
2. Read all .mps files from a given directory.
3. For each .mps file:
      - match instance name to CSV
      - solve LP with Gurobi
      - compare optimal values
4. Write one CSV summarizing all results,
   including mismatch indicator column `matched`.

Fill in the three paths below before running.
"""

import os
import pandas as pd


# ============================================================
# ======== USER SETTINGS (EDIT THESE THREE PATHS) ============
# ============================================================

# Folder containing *.mps files
MPS_DIR = "/Users/huangjiacheng/Downloads/lp-data-netlib-main/mps_files"

# Reference CSV (the one generated earlier)
REF_CSV = "/Users/huangjiacheng/OR772/lp_cases/netlib_reference.csv"

# Output CSV to write comparison result
OUT_CSV = "/Users/huangjiacheng/OR772/lp_cases/mps_compare_output.csv"

# Tolerance for comparing numerical optimal values
COMPARE_TOL = 1e-6

# ============================================================
# ====================== HELPER FUNC =========================
# ============================================================

def load_reference_csv(path: str) -> pd.DataFrame:
    """
    Load reference CSV and index by lowercase name.
    Expected columns: Name, Rows, Cols, Nonzeros, Optimal Value
    """
    df = pd.read_csv(path)

    if "Name" not in df.columns:
        raise ValueError("Missing 'Name' column in reference CSV.")

    df["Name"] = df["Name"].astype(str).str.lower()
    df = df.set_index("Name")
    return df


def list_mps_files(folder: str):
    """
    List all *.mps files (non-recursive).
    """
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".mps")
    )


def solve_mps_with_gurobi(mps_path: str):
    """
    Solve the MPS file as a linear program with Gurobi.
    Return optimal objective value or None if failed.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(
            "gurobipy is required. Please install Gurobi first."
        ) from e

    try:
        model = gp.read(mps_path)
    except gp.GurobiError as e:
        print(f"[Warning] Could not read MPS: {mps_path}: {e}")
        return None

    # Force LP relaxation
    for v in model.getVars():
        v.vtype = GRB.CONTINUOUS

    print(f"Solving: {mps_path}")
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return model.ObjVal
    else:
        print(f"[Warning] Non-optimal status ({model.Status}) for: {mps_path}")
        return None


# ============================================================
# ======================== MAIN LOGIC ========================
# ============================================================

def main():
    # 1. Load reference CSV
    ref_df = load_reference_csv(REF_CSV)

    # 2. List MPS files
    mps_files = list_mps_files(MPS_DIR)
    if not mps_files:
        print("No .mps files found. Exiting.")
        return

    results = []

    for mps_path in mps_files:
        name_stem = os.path.splitext(os.path.basename(mps_path))[0].lower()

        # Skip if instance not found in reference CSV
        if name_stem not in ref_df.index:
            print(f"[Info] {name_stem} not in reference CSV → skipped.")
            continue

        row = ref_df.loc[name_stem]
        given_opt_str = str(row["Optimal Value"])

        # Try numerical value for comparison
        try:
            given_val = float(given_opt_str)
        except (ValueError, TypeError):
            given_val = None

        # 3. Solve using Gurobi
        gurobi_val = solve_mps_with_gurobi(mps_path)

        # 4. Determine match (1) or mismatch (0)
        if given_val is not None and gurobi_val is not None:
            matched = 1 if abs(given_val - gurobi_val) <= COMPARE_TOL else 0
        else:
            matched = 0

        # 5. Collect result
        results.append({
            "Name": name_stem,
            "Rows": int(row["Rows"]),
            "Cols": int(row["Cols"]),
            "Nonzeros": int(row["Nonzeros"]),
            "Given Optimal Value": given_opt_str,
            "Gurobi Optimal Value": gurobi_val,
            "matched": matched,
        })

    # 6. Save output CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n=== Done! Result saved to: {OUT_CSV} ===")


if __name__ == "__main__":
    main()