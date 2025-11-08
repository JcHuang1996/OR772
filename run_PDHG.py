import argparse

from main import run_pdhg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDHG on an LP in MPS format.")
    parser.add_argument("--case", required=True, help="Path to the .mps or .mps.bz2 file.")
    parser.add_argument("--max-iters", type=int, default=50_000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--eta-scale", type=float, default=0.9)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument(
        "--precond",
        type=str,
        default="ruiz_pc",
        choices=["none", "pc", "ruiz", "ruiz_pc"],
    )
    parser.add_argument("--precond-tol", type=float, default=1e-2)
    parser.add_argument("--precond-iters", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_pdhg(
        args.case,
        max_iters=args.max_iters,
        tol=args.tol,
        omega=args.omega,
        eta_scale=args.eta_scale,
        check_every=args.check_every,
        precond=args.precond,  # type: ignore[arg-type]
        precond_tol=args.precond_tol,
        precond_iters=args.precond_iters,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()