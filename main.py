"""
main.py — convenience runner for local development (outside Docker).

Usage
-----
    python3 main.py --status        # check which pipeline outputs exist
    python3 main.py --fom           # run FOM parametric sweep
    python3 main.py --rom           # train POD-ROM + Kalman filters
    python3 main.py --bayes         # run Bayesian inference
    python3 main.py --reconstruct   # posterior field reconstruction + figures
    python3 main.py --dashboard     # launch Streamlit dashboard
    python3 main.py --all           # run everything in sequence
"""

import argparse
import sys
from pathlib import Path

SNAPSHOT_DIR  = Path("/workspace/data/snapshots")
ROM_DIR       = Path("/workspace/data/rom")
POSTERIOR_DIR = Path("/workspace/data/posterior")


def check_status():
    checks = [
        (SNAPSHOT_DIR / "metadata.json",        "FOM snapshots"),
        (ROM_DIR / "rom_metadata.json",          "POD-ROM + Kalman results"),
        (POSTERIOR_DIR / "gp_emulator.pkl",      "GP emulator"),
        (POSTERIOR_DIR / "trace.nc",             "MCMC trace (Bayesian)"),
        (POSTERIOR_DIR / "T_posterior_mean.npy", "Posterior field reconstruction"),
    ]
    print("\nPipeline status")
    print("=" * 50)
    for path, label in checks:
        status = "✓  present" if path.exists() else "✗  missing"
        print(f"  {status:<14} {label}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Thermo-mechanical digital twin — local pipeline runner"
    )
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--fom",         action="store_true")
    parser.add_argument("--rom",         action="store_true")
    parser.add_argument("--bayes",       action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--dashboard",   action="store_true")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.status:
        check_status()
        return

    if args.all or args.fom:
        print("\n── Step 1: FOM parametric sweep ──")
        from run_parametric import main as fom_main
        fom_main()

    if args.all or args.rom:
        print("\n── Step 2: POD-ROM + Kalman filters ──")
        from train_rom import main as rom_main
        rom_main()

    if args.all or args.bayes:
        print("\n── Step 3: Bayesian inference ──")
        from bayesian_inference import run_bayesian_inference
        run_bayesian_inference()

    if args.all or args.reconstruct:
        print("\n── Step 4: Posterior field reconstruction ──")
        from field_reconstruction import run_field_reconstruction
        from posterior_analysis import analyse_posterior
        run_field_reconstruction()
        analyse_posterior()

    if args.all or args.dashboard:
        print("\n── Step 5: Launching dashboard ──")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "data_driven/dashboard.py",
            "--server.port=8501", "--server.address=0.0.0.0",
        ], check=True)


if __name__ == "__main__":
    main()
