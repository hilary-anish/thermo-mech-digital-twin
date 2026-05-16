"""
pod_rom.py
==========
Dual-field POD-ROM for the thermo-mechanical system.

Reads snapshot matrices produced by the FEniCSx container.
Produces reduced bases and operators saved to /workspace/data/rom/.

Steps
-----
1. Load T and u snapshot matrices
2. Compute truncated SVD on each field independently
3. Plot singular value decay (Figure 1)
4. Project thermal snapshots -> reduced coordinates q_T
5. Project mechanical snapshots -> reduced coordinates q_u
6. Identify reduced thermal operator A_r, b_r via least-squares (OpInf)
7. Identify coupling matrix K_cu via least-squares
8. Validate ROM on held-out test trajectories
9. Save all ROM matrices
"""

import numpy as np
import json
import time
import os
from pathlib import Path

from config import (
    SNAPSHOT_DIR, FIGURE_DIR, DT, N_STEPS,
    ENERGY_THRESHOLD, N_SAMPLES_TRAIN, N_SAMPLES_TEST
)

ROM_DIR = "/workspace/data/rom"


# ── SVD-based POD basis ────────────────────────────────────────────────────

def compute_pod_basis(S, field_name, threshold=ENERGY_THRESHOLD):
    """
    S : ndarray (N_dofs, n_snapshots)
    Returns Phi (N_dofs, r), sigma (all singular values), r
    """
    print(f"\nComputing POD basis: {field_name}")
    t0 = time.time()
    U, sigma, _ = np.linalg.svd(S, full_matrices=False)

    energy  = np.cumsum(sigma**2) / np.sum(sigma**2)
    r       = int(np.searchsorted(energy, threshold)) + 1
    r       = min(r, len(sigma))
    Phi     = U[:, :r]

    print(f"  DOFs           : {S.shape[0]}")
    print(f"  Snapshots      : {S.shape[1]}")
    print(f"  Rank r         : {r}  ({energy[r-1]*100:.4f}% energy)")
    print(f"  Compression    : {S.shape[1]}/{r} = {S.shape[1]/r:.1f}x")
    print(f"  SVD time       : {time.time()-t0:.2f}s")
    return Phi, sigma, r


def plot_svd_decay(sigma_T, sigma_u, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sigma, name, col in zip(
        axes, [sigma_T, sigma_u],
        ["Thermal field (T)", "Mechanical field (u)"],
        ["tab:red", "tab:blue"]
    ):
        r = np.arange(1, len(sigma) + 1)
        energy = np.cumsum(sigma**2) / np.sum(sigma**2)
        ax2 = ax.twinx()
        ax.semilogy(r, sigma / sigma[0], color=col, lw=2,
                    label="Normalised σᵢ/σ₁")
        ax2.plot(r, energy * 100, "k--", lw=1.5, alpha=0.6,
                 label="Cumulative energy [%]")
        ax.axhline(1e-4, color=col, ls=":", alpha=0.4)
        ax.set_xlabel("Mode index r")
        ax.set_ylabel("Normalised singular value", color=col)
        ax2.set_ylabel("Cumulative energy [%]", color="gray")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.suptitle("POD singular value decay — dual-field ROM", fontsize=12)
    plt.tight_layout()
    out = Path(save_dir) / "svd_decay.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Operator Inference ─────────────────────────────────────────────────────

def fit_thermal_operator(q_T_all, dt, params_train=None, k_nominal=150.0):
    """
    Identifies A_r, b_r from reduced thermal trajectory data.
    Solves: dq_T/dt = A_r * q_T + b_r  via least-squares.

    IMPORTANT (Peherstorfer & Willcox 2016): OpInf identifies operators at a
    FIXED parameter point. Fitting with mixed-k snapshots corrupts the linear
    k-scaling assumption used by the Kalman filter and jax_surrogate.
    We therefore filter to snapshots whose k is closest to k_nominal before
    fitting. If params_train is None, all snapshots are used (legacy behaviour).

    q_T_all    : (r_T, n_samples * N_STEPS)  — all training reduced coords
    params_train : (n_samples, 3)             — [k, h, Q0] for each run
    """
    print("\nFitting thermal operator (Operator Inference)...")

    if params_train is not None:
        # Select the 5 runs whose k is closest to k_nominal
        k_vals   = params_train[:, 0]
        n_select = min(5, len(k_vals))
        closest  = np.argsort(np.abs(k_vals - k_nominal))[:n_select]
        n_steps_per = q_T_all.shape[1] // len(k_vals)
        idx_cols = np.concatenate([
            np.arange(i * n_steps_per, (i + 1) * n_steps_per)
            for i in closest
        ])
        q_T = q_T_all[:, idx_cols]
        print(f"  Using {n_select} runs near k_nominal={k_nominal}  "
              f"(k values: {k_vals[closest].round(1).tolist()})")
    else:
        q_T = q_T_all
        print("  WARNING: fitting on all mixed-k snapshots (use params_train for correctness)")

    dq_dt = (q_T[:, 1:] - q_T[:, :-1]) / dt          # (r_T, n_cols-1)
    ones  = np.ones((1, q_T.shape[1] - 1))
    Psi   = np.vstack([q_T[:, :-1], ones])             # (r_T+1, n_cols-1)

    Theta, res, rank, sv = np.linalg.lstsq(
        Psi.T, dq_dt.T, rcond=None
    )
    Theta = Theta.T                                     # (r_T, r_T+1)
    A_r   = Theta[:, :-1]
    b_r   = Theta[:, -1]

    rho = max(abs(np.linalg.eigvals(A_r)))
    print(f"  A_r shape      : {A_r.shape}")
    print(f"  Spectral radius: {rho:.4f}  "
          f"({'stable' if rho < 1 else 'UNSTABLE — check DT'})")
    return A_r, b_r


def fit_coupling_operator(q_T, q_u, params_train=None, k_nominal=150.0):
    """
    Identifies K_cu such that q_u ≈ K_cu @ q_T.
    Uses k-nominal snapshots only for consistency with A_r identification.
    """
    print("\nFitting coupling operator K_cu...")
    if params_train is not None:
        k_vals   = params_train[:, 0]
        n_select = min(5, len(k_vals))
        closest  = np.argsort(np.abs(k_vals - k_nominal))[:n_select]
        n_steps_per = q_T.shape[1] // len(k_vals)
        idx_cols = np.concatenate([
            np.arange(i * n_steps_per, (i + 1) * n_steps_per)
            for i in closest
        ])
        q_T_fit = q_T[:, idx_cols]
        q_u_fit = q_u[:, idx_cols]
        print(f"  Using {n_select} runs near k_nominal for K_cu")
    else:
        q_T_fit = q_T
        q_u_fit = q_u
    K_cu, _, _, _ = np.linalg.lstsq(q_T_fit.T, q_u_fit.T, rcond=None)
    K_cu = K_cu.T
    print(f"  K_cu shape     : {K_cu.shape}")
    return K_cu


# ── ROM prediction ─────────────────────────────────────────────────────────

def rom_predict(q0, n_steps, dt, A_r, b_r, K_cu, Phi_T, Phi_u):
    """
    Rolls out the ROM from initial condition q0.
    Uses explicit Euler for the thermal ODE.

    Returns
    -------
    T_rom : (N_T_dofs, n_steps+1)
    u_rom : (N_u_dofs, n_steps+1)
    """
    r_T = len(b_r)
    q_T = np.zeros((r_T, n_steps + 1))
    q_T[:, 0] = q0

    for n in range(n_steps):
        q_T[:, n + 1] = q_T[:, n] + dt * (A_r @ q_T[:, n] + b_r)

    q_u  = K_cu @ q_T
    T_rom = Phi_T @ q_T
    u_rom = Phi_u @ q_u
    return T_rom, u_rom, q_T


# ── Validation ────────────────────────────────────────────────────────────

def validate_rom(Phi_T, Phi_u, A_r, b_r, K_cu,
                 T_test, u_test, dt, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_steps = T_test.shape[1] - 1
    q0      = Phi_T.T @ T_test[:, 0]
    T_rom, u_rom, _ = rom_predict(
        q0, n_steps, dt, A_r, b_r, K_cu, Phi_T, Phi_u
    )

    # relative L2 error at each step
    def rel_l2(approx, truth):
        nrm = np.linalg.norm(truth, axis=0)
        nrm = np.where(nrm < 1e-12, 1e-12, nrm)
        return np.linalg.norm(approx - truth, axis=0) / nrm

    err_T = rel_l2(T_rom[:, 1:], T_test[:, 1:])
    err_u = rel_l2(u_rom[:, 1:], u_test[:, 1:])

    t_ax = np.arange(1, n_steps + 1) * dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(t_ax, err_T * 100, "tab:red",  lw=2)
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("Rel. L2 error [%]")
    axes[0].set_title("Thermal field: ROM vs FOM"); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_ax, err_u * 100, "tab:blue", lw=2)
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("Rel. L2 error [%]")
    axes[1].set_title("Displacement field: ROM vs FOM"); axes[1].grid(True, alpha=0.3)

    max_T = err_T.max() * 100
    max_u = err_u.max() * 100
    plt.suptitle(
        f"ROM validation — max T error: {max_T:.3f}%  "
        f"max u error: {max_u:.3f}%", fontsize=11
    )
    plt.tight_layout()
    out = Path(save_dir) / "rom_validation.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Max T error : {max_T:.4f}%")
    print(f"  Max u error : {max_u:.4f}%")
    print(f"  Saved: {out}")
    return err_T, err_u


# ── Main ───────────────────────────────────────────────────────────────────

def build_and_save_rom():
    Path(ROM_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load snapshots ────────────────────────────────────────────────────
    print("Loading snapshots...")
    T_train = np.load(f"{SNAPSHOT_DIR}/T_snapshots_train.npy")
    u_train = np.load(f"{SNAPSHOT_DIR}/u_snapshots_train.npy")
    T_test  = np.load(f"{SNAPSHOT_DIR}/T_snapshots_test.npy")
    u_test  = np.load(f"{SNAPSHOT_DIR}/u_snapshots_test.npy")

    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        meta = json.load(f)

    print(f"T_train : {T_train.shape}")
    print(f"u_train : {u_train.shape}")

    # ── POD bases ─────────────────────────────────────────────────────────
    Phi_T, sigma_T, r_T = compute_pod_basis(T_train, "thermal")
    Phi_u, sigma_u, r_u = compute_pod_basis(u_train, "mechanical")
    plot_svd_decay(sigma_T, sigma_u, FIGURE_DIR)

    # ── Project to reduced coords ─────────────────────────────────────────
    q_T_train = Phi_T.T @ T_train
    q_u_train = Phi_u.T @ u_train

    # ── Identify operators ────────────────────────────────────────────────
    # Pass params_train so OpInf uses k≈k_nominal snapshots only
    # (Peherstorfer & Willcox 2016: OpInf valid at fixed parameter point)
    params_train_data = np.load(f"{SNAPSHOT_DIR}/params_train.npy")
    A_r, b_r = fit_thermal_operator(q_T_train, DT,
                                    params_train=params_train_data,
                                    k_nominal=150.0)
    K_cu     = fit_coupling_operator(q_T_train, q_u_train,
                                    params_train=params_train_data,
                                    k_nominal=150.0)

    # ── Validate on first test trajectory ─────────────────────────────────
    print("\nValidating on held-out test trajectory...")
    n_test_steps = N_STEPS
    T_test_traj  = T_test[:, :n_test_steps + 1]
    u_test_traj  = u_test[:, :n_test_steps + 1]
    validate_rom(Phi_T, Phi_u, A_r, b_r, K_cu,
                 T_test_traj, u_test_traj, DT, FIGURE_DIR)

    # ── Save ROM ──────────────────────────────────────────────────────────
    np.save(f"{ROM_DIR}/Phi_T.npy", Phi_T)
    np.save(f"{ROM_DIR}/Phi_u.npy", Phi_u)
    np.save(f"{ROM_DIR}/A_r.npy",   A_r)
    np.save(f"{ROM_DIR}/b_r.npy",   b_r)
    np.save(f"{ROM_DIR}/K_cu.npy",  K_cu)
    np.save(f"{ROM_DIR}/sigma_T.npy", sigma_T)
    np.save(f"{ROM_DIR}/sigma_u.npy", sigma_u)

    rom_meta = {
        "r_T": int(r_T), "r_u": int(r_u),
        "N_T_dofs": int(Phi_T.shape[0]),
        "N_u_dofs": int(Phi_u.shape[0]),
        "dt": DT,
        "energy_threshold": ENERGY_THRESHOLD,
    }
    with open(f"{ROM_DIR}/rom_metadata.json", "w") as f:
        json.dump(rom_meta, f, indent=2)

    print(f"\nROM saved to {ROM_DIR}/")
    print(f"  Phi_T  : {Phi_T.shape}")
    print(f"  Phi_u  : {Phi_u.shape}")
    print(f"  A_r    : {A_r.shape}")
    print(f"  K_cu   : {K_cu.shape}")
    return Phi_T, Phi_u, A_r, b_r, K_cu


if __name__ == "__main__":
    build_and_save_rom()
