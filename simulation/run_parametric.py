"""
run_parametric.py
=================
Runs the FOM across all LHS parameter samples and saves snapshot matrices.
This script runs inside the FEniCSx Docker container.

Usage (inside container):
    python3 run_parametric.py

Outputs written to /workspace/data/snapshots/:
    T_snapshots_train.npy   (N_T_dofs, N_TRAIN * N_STEPS)
    u_snapshots_train.npy   (N_u_dofs, N_TRAIN * N_STEPS)
    T_snapshots_test.npy    (N_T_dofs, N_TEST  * N_STEPS)
    u_snapshots_test.npy    (N_u_dofs, N_TEST  * N_STEPS)
    params_train.npy        (N_TRAIN, 3)
    params_test.npy         (N_TEST,  3)
    dof_coords.npy          (N_T_dofs, 2)
    metadata.json
"""

import numpy as np
import json
import time
from pathlib import Path

from config import *
from fom_solver import run_simulation, get_dof_coordinates, plot_validation


def latin_hypercube_sample(n, ranges, seed=42):
    """
    Simple LHS over 3 parameters.
    ranges = [(lo, hi), (lo, hi), (lo, hi)]
    Returns array of shape (n, 3).
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, len(ranges)))
    for j, (lo, hi) in enumerate(ranges):
        perm = rng.permutation(n)
        u    = (perm + rng.random(n)) / n
        samples[:, j] = lo + u * (hi - lo)
    return samples


def main():
    Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)

    ranges = [K_RANGE, H_RANGE, Q0_RANGE]

    params_train = latin_hypercube_sample(
        N_SAMPLES_TRAIN, ranges, seed=42
    )
    params_test  = latin_hypercube_sample(
        N_SAMPLES_TEST,  ranges, seed=999
    )

    print("=" * 60)
    print(f"Thermo-Mechanical FOM Parametric Sweep")
    print(f"Training samples : {N_SAMPLES_TRAIN}")
    print(f"Test samples     : {N_SAMPLES_TEST}")
    print(f"Time steps/run   : {N_STEPS}")
    print(f"Grid             : {NX}x{NY} quads")
    print("=" * 60)

    # ── Training snapshots ────────────────────────────────────────────────
    T_train_list, u_train_list = [], []
    t0_total = time.time()

    for i, (k, h, Q0) in enumerate(params_train):
        print(f"\n[Train {i+1}/{N_SAMPLES_TRAIN}]  "
              f"k={k:.1f}  h={h:.1f}  Q0={Q0:.2e}")
        t0 = time.time()
        T_s, u_s, N_T, N_u = run_simulation(k, h, Q0, verbose=False)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  | "
              f"T_max={T_s.max():.1f}K")
        T_train_list.append(T_s)
        u_train_list.append(u_s)

    T_train = np.concatenate(T_train_list, axis=1)
    u_train = np.concatenate(u_train_list, axis=1)

    # ── Test snapshots ────────────────────────────────────────────────────
    T_test_list, u_test_list = [], []
    for i, (k, h, Q0) in enumerate(params_test):
        print(f"\n[Test {i+1}/{N_SAMPLES_TEST}]  "
              f"k={k:.1f}  h={h:.1f}  Q0={Q0:.2e}")
        T_s, u_s, N_T, N_u = run_simulation(k, h, Q0, verbose=False)
        T_test_list.append(T_s)
        u_test_list.append(u_s)

    T_test = np.concatenate(T_test_list, axis=1)
    u_test = np.concatenate(u_test_list, axis=1)

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(f"{SNAPSHOT_DIR}/T_snapshots_train.npy", T_train)
    np.save(f"{SNAPSHOT_DIR}/u_snapshots_train.npy", u_train)
    np.save(f"{SNAPSHOT_DIR}/T_snapshots_test.npy",  T_test)
    np.save(f"{SNAPSHOT_DIR}/u_snapshots_test.npy",  u_test)
    np.save(f"{SNAPSHOT_DIR}/params_train.npy",      params_train)
    np.save(f"{SNAPSHOT_DIR}/params_test.npy",       params_test)

    # DOF coordinates for dashboard plotting
    coords = get_dof_coordinates()
    np.save(f"{SNAPSHOT_DIR}/dof_coords.npy", coords)

    metadata = {
        "N_T_dofs":         int(N_T),
        "N_u_dofs":         int(N_u),
        "N_samples_train":  N_SAMPLES_TRAIN,
        "N_samples_test":   N_SAMPLES_TEST,
        "N_steps":          N_STEPS,
        "dt":               DT,
        "t_end":            T_END,
        "geometry":         {"L": L, "H": H, "NX": NX, "NY": NY},
        "param_ranges":     {
            "k":  list(K_RANGE),
            "h":  list(H_RANGE),
            "Q0": list(Q0_RANGE)
        },
        "total_wall_time_s": round(time.time() - t0_total, 2),
        "T_train_shape":    list(T_train.shape),
        "u_train_shape":    list(u_train.shape),
    }
    with open(f"{SNAPSHOT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"All snapshots saved to {SNAPSHOT_DIR}/")
    print(f"T_train : {T_train.shape}")
    print(f"u_train : {u_train.shape}")
    print(f"Total time: {time.time()-t0_total:.1f}s")
    print("=" * 60)

    # ── Validation plot for first test sample ─────────────────────────────
    print("\nGenerating validation plot...")
    k0, h0, Q0_0 = params_test[0]
    T_v, u_v, _, _ = run_simulation(k0, h0, Q0_0, verbose=True)
    plot_validation(T_v, u_v, k0, FIGURE_DIR)
    print("Done. Run data_driven pipeline next.")


if __name__ == "__main__":
    main()
