"""
train_rom.py
============
Orchestrator for the ML container.
Runs in sequence:
  1. Build POD-ROM from snapshots
  2. Run LKF assimilation on test trajectory
  3. Run EnKF with k-identification on test trajectory
  4. Save all results for the Streamlit dashboard

Run this once before launching the dashboard:
    python3 train_rom.py

All outputs go to /workspace/data/rom/
"""

import numpy as np
import json
import time
from pathlib import Path

from config import (
    SNAPSHOT_DIR, FIGURE_DIR, DT,
    N_STEPS, N_SAMPLES_TEST
)
from pod_rom import build_and_save_rom
from kalman_filter import (
    place_sensors, build_observation_matrix,
    LinearKalmanFilter, EnsembleKalmanFilter,
    run_lkf_loop, run_enkf_loop
)

ROM_DIR    = "/workspace/data/rom"
RESULT_DIR = "/workspace/data/rom"


def main():
    Path(ROM_DIR).mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Step 1: Build POD-ROM ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Building POD-ROM")
    print("=" * 60)
    Phi_T, Phi_u, A_r, b_r, K_cu = build_and_save_rom()

    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        meta = json.load(f)
    N_T = meta["N_T_dofs"]
    N_u = meta["N_u_dofs"]
    r_T = Phi_T.shape[1]

    # ── Step 2: Load test trajectory (first test sample) ──────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Loading test trajectory")
    print("=" * 60)
    T_test_all = np.load(f"{SNAPSHOT_DIR}/T_snapshots_test.npy")
    u_test_all = np.load(f"{SNAPSHOT_DIR}/u_snapshots_test.npy")
    params_test = np.load(f"{SNAPSHOT_DIR}/params_test.npy")

    # use first test sample
    T_true = T_test_all[:, :N_STEPS]
    u_true = u_test_all[:, :N_STEPS]
    k_true = float(params_test[0, 0])
    print(f"  Test sample: k={k_true:.1f}  "
          f"T range: [{T_true.min():.1f}, {T_true.max():.1f}] K")

    # ── Step 3: Build observation model ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Building observation model")
    print("=" * 60)
    T_nodes, u_nodes = place_sensors(N_T, N_u, seed=0)
    H, R = build_observation_matrix(Phi_T, Phi_u, K_cu, T_nodes, u_nodes)
    print(f"  Sensors: {len(T_nodes)} temperature + {len(u_nodes)} strain")
    print(f"  H shape: {H.shape}")

    # ── Step 4: Run LKF ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Running Linear Kalman Filter")
    print("=" * 60)
    lkf = LinearKalmanFilter(A_r, b_r, K_cu, H, R, Phi_T, Phi_u)
    q0  = Phi_T.T @ T_true[:, 0]
    lkf.initialise(q0)
    T_lkf, u_lkf, err_T_lkf, err_u_lkf, std_T_lkf = run_lkf_loop(
        lkf, T_true, u_true, T_nodes, u_nodes
    )
    print(f"  Max T error (LKF): {err_T_lkf.max()*100:.3f}%")
    print(f"  Max u error (LKF): {err_u_lkf.max()*100:.3f}%")

    # ── Step 5: Run EnKF with k identification ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Running Ensemble Kalman Filter (k identification)")
    print("=" * 60)
    k_nominal = 150.0

    # Physics-correct scaling:
    # A_r captures diffusion → scales with k/k_nominal
    # b_r captures heat source Q0 → does NOT scale with k
    # (Q0 is treated as known/nominal here; EnKF identifies k only)
    def A_r_func(k):
        return A_r * (k / k_nominal)

    def b_r_func(k):
        return b_r   # b_r is Q0-dependent, not k-dependent

    # Initialise EnKF ensemble from the true IC (same as LKF)
    # Starting from q_T ~ N(0, 0.1) causes immediate divergence since
    # the true reduced state at t=0 is q0 = Phi_T.T @ T_true[:, 0] ~ O(300)
    enkf = EnsembleKalmanFilter(
        A_r_func, b_r_func, K_cu, H, R, Phi_T, Phi_u,
        k_true=k_true, seed=42, q0_init=q0
    )
    T_enkf, u_enkf, err_T_enkf, err_u_enkf, k_hist = run_enkf_loop(
        enkf, T_true, u_true, T_nodes, u_nodes
    )
    print(f"  True k            : {k_true:.1f}")
    print(f"  Final k estimate  : {k_hist[-1]:.2f}")
    print(f"  k error           : {abs(k_hist[-1]-k_true)/k_true*100:.2f}%")
    print(f"  Max T error (EnKF): {err_T_enkf.max()*100:.3f}%")

    # ── Step 6: Save all results ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Saving results for dashboard")
    print("=" * 60)
    t_axis = np.arange(N_STEPS) * DT

    np.save(f"{RESULT_DIR}/T_true.npy",      T_true)
    np.save(f"{RESULT_DIR}/u_true.npy",      u_true)
    np.save(f"{RESULT_DIR}/T_lkf.npy",       T_lkf)
    np.save(f"{RESULT_DIR}/u_lkf.npy",       u_lkf)
    np.save(f"{RESULT_DIR}/T_enkf.npy",      T_enkf)
    np.save(f"{RESULT_DIR}/u_enkf.npy",      u_enkf)
    np.save(f"{RESULT_DIR}/err_T_lkf.npy",   err_T_lkf)
    np.save(f"{RESULT_DIR}/err_u_lkf.npy",   err_u_lkf)
    np.save(f"{RESULT_DIR}/err_T_enkf.npy",  err_T_enkf)
    np.save(f"{RESULT_DIR}/k_hist.npy",      k_hist)
    np.save(f"{RESULT_DIR}/t_axis.npy",      t_axis)
    np.save(f"{RESULT_DIR}/T_sensor_nodes.npy", T_nodes)
    np.save(f"{RESULT_DIR}/u_sensor_nodes.npy", u_nodes)

    results_meta = {
        "k_true":           k_true,
        "k_final_estimate": float(k_hist[-1]),
        "max_err_T_lkf_pct":  float(err_T_lkf.max() * 100),
        "max_err_u_lkf_pct":  float(err_u_lkf.max() * 100),
        "max_err_T_enkf_pct": float(err_T_enkf.max() * 100),
        "n_T_sensors": int(len(T_nodes)),
        "n_u_sensors": int(len(u_nodes)),
        "wall_time_s":  round(time.time() - t_start, 2),
    }
    with open(f"{RESULT_DIR}/results_meta.json", "w") as f:
        json.dump(results_meta, f, indent=2)

    print(f"\n  All results saved to {RESULT_DIR}/")
    print(f"  Total wall time: {time.time()-t_start:.1f}s")
    print("\nRun:  streamlit run dashboard.py  to launch the app.")


if __name__ == "__main__":
    main()
