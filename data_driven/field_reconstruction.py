"""
field_reconstruction.py
========================
Reconstruct the full temperature and displacement fields from posterior
samples and compute pointwise uncertainty quantification (UQ).

v2 fix: the ROM (OpInf at single nominal point) is parameter-independent —
all posterior samples gave identical ROM output → std=0.

Fix: use the FOM training snapshots to build a spatial basis, then use
GP predictions to interpolate the full field for each posterior sample.

Approach:
  1. Build a GP-to-field mapping from FOM training data:
     - For each training sample i: we have params (k_i, h_i, Q0_i),
       full field T_i(x,t), and GP prediction y_hat_i (sensor readings)
  2. For each posterior sample (k,h,Q0):
     - Get GP prediction y_hat = predict_gp(k,h,Q0)
     - Find nearest neighbours in GP training space
     - Interpolate full field using distance-weighted FOM snapshots
  3. Compute pointwise mean, std, credible intervals across ensemble

This gives physically meaningful field uncertainty that reflects the
posterior uncertainty in (k,h,Q0).
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path

from config import N_STEPS, DT, SNAPSHOT_DIR, N_T_SENSORS, N_U_SENSORS
from gp_emulator import load_gp_emulator, predict_gp
from kalman_filter import place_sensors

ROM_DIR       = "/workspace/data/rom"
POSTERIOR_DIR = "/workspace/data/posterior"
FIGURES_DIR   = f"{POSTERIOR_DIR}/figures"

N_POSTERIOR_SAMPLES = 500


# ── Load posterior samples from trace ─────────────────────────────────────

def draw_posterior_samples(n_samples=N_POSTERIOR_SAMPLES, seed=0):
    trace_path = Path(POSTERIOR_DIR) / "trace.nc"
    if not trace_path.exists():
        raise FileNotFoundError("trace.nc not found. Run bayesian_inference.py first.")

    trace = az.from_netcdf(str(trace_path))
    posterior = trace.posterior

    k_all  = posterior["k"].values.flatten()
    h_all  = posterior["h"].values.flatten()
    Q0_all = posterior["Q0"].values.flatten()

    rng = np.random.default_rng(seed)
    n   = min(n_samples, len(k_all))
    idx = rng.choice(len(k_all), n, replace=False)

    return k_all[idx], h_all[idx], Q0_all[idx]


# ── Build field ensemble using GP + FOM nearest-neighbour interpolation ────

def build_field_ensemble_gp(k_samples, h_samples, Q0_samples,
                              gp, scaler_X, scaler_Y,
                              T_train_all, u_train_all, params_train):
    """
    For each posterior sample, reconstruct the full field by weighted
    interpolation of FOM training snapshots, where weights come from
    GP prediction similarity.

    Returns
    -------
    T_ensemble : (N_T, N_STEPS, n_samples)
    u_ensemble : (N_u, N_STEPS, n_samples)
    """
    n_samples  = len(k_samples)
    N_T        = T_train_all.shape[0]
    N_u        = u_train_all.shape[0]
    N_train    = params_train.shape[0]

    T_ensemble = np.zeros((N_T, N_STEPS, n_samples))
    u_ensemble = np.zeros((N_u, N_STEPS, n_samples))

    # Precompute GP predictions for all training samples
    print("  Precomputing GP predictions for training samples …")
    Y_train_gp = np.zeros((N_train, (N_T_SENSORS + N_U_SENSORS) * N_STEPS))
    for i in range(N_train):
        k_i, h_i, Q0_i = params_train[i]
        y, _ = predict_gp(gp, scaler_X, scaler_Y,
                          np.array([np.log(k_i), np.log(h_i), np.log(Q0_i)]))
        Y_train_gp[i] = y

    print(f"  Building field ensemble ({n_samples} samples) …")
    t0 = time.time()

    for j in range(n_samples):
        # GP prediction at this posterior sample
        y_j, _ = predict_gp(gp, scaler_X, scaler_Y,
                             np.array([np.log(float(k_samples[j])),
                                       np.log(float(h_samples[j])),
                                       np.log(float(Q0_samples[j]))]))

        # Distance in GP output space to each training sample
        dists = np.linalg.norm(Y_train_gp - y_j[None, :], axis=1)

        # Inverse distance weights (add small epsilon to avoid div/0)
        eps = 1e-6
        weights = 1.0 / (dists + eps)
        weights = weights / weights.sum()

        # Weighted sum of FOM training fields
        T_j = np.zeros((N_T, N_STEPS))
        u_j = np.zeros((N_u, N_STEPS))
        for i in range(N_train):
            T_i = T_train_all[:, i * N_STEPS:(i + 1) * N_STEPS]
            u_i = u_train_all[:, i * N_STEPS:(i + 1) * N_STEPS]
            T_j += weights[i] * T_i
            u_j += weights[i] * u_i

        T_ensemble[:, :, j] = T_j
        u_ensemble[:, :, j] = u_j

        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{n_samples}  ({time.time()-t0:.1f}s elapsed)")

    print(f"  Done in {time.time()-t0:.1f}s")
    return T_ensemble, u_ensemble


# ── Compute UQ statistics ──────────────────────────────────────────────────

def compute_uq_statistics(ensemble):
    return {
        "mean":  ensemble.mean(axis=2),
        "std":   ensemble.std(axis=2),
        "lower": np.percentile(ensemble, 2.5,  axis=2),
        "upper": np.percentile(ensemble, 97.5, axis=2),
    }


# ── Save UQ arrays ─────────────────────────────────────────────────────────

def save_uq_arrays(T_stats, u_stats):
    p = Path(POSTERIOR_DIR)
    for key in ["mean", "std", "lower", "upper"]:
        np.save(p / f"T_posterior_{key}.npy", T_stats[key])
        np.save(p / f"u_posterior_{key}.npy", u_stats[key])
    print(f"UQ arrays saved to {POSTERIOR_DIR}/")


# ── Figures ────────────────────────────────────────────────────────────────

def _field_to_grid(field_flat, nx, ny):
    n_nodes = (nx + 1) * (ny + 1)
    return field_flat[:n_nodes].reshape((ny + 1, nx + 1))


def plot_mean_and_std_fields(T_stats, u_stats, meta, step_idx=-1):
    NX = meta["geometry"]["NX"]
    NY = meta["geometry"]["NY"]
    L  = meta["geometry"]["L"]
    H  = meta["geometry"]["H"]
    x_1d = np.linspace(0, L, NX + 1)
    y_1d = np.linspace(0, H, NY + 1)

    for stats, label, unit, cmap in [
        (T_stats, "Temperature", "K",  "hot"),
        (u_stats, "Displacement |u|", "m", "viridis"),
    ]:
        mean_grid = _field_to_grid(stats["mean"][:, step_idx], NX, NY)
        std_grid  = _field_to_grid(stats["std"][:, step_idx],  NX, NY)

        fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
        fig.suptitle(
            f"Posterior {label} field — "
            f"t = {(step_idx % N_STEPS + 1) * DT:.1f}s",
            fontsize=12
        )

        im0 = axes[0].pcolormesh(x_1d, y_1d, mean_grid,
                                  cmap=cmap, shading="auto")
        plt.colorbar(im0, ax=axes[0], label=f"Mean {label} [{unit}]")
        axes[0].set_title("Posterior mean")
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")

        im1 = axes[1].pcolormesh(x_1d, y_1d, std_grid,
                                  cmap="plasma", shading="auto")
        plt.colorbar(im1, ax=axes[1], label=f"Std [{unit}]")
        axes[1].set_title("Posterior std (uncertainty)")
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")

        plt.tight_layout()
        safe_label = label.lower().replace(" ", "_").replace("|", "")
        out = Path(FIGURES_DIR) / f"field_mean_{safe_label}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_credible_interval_bands(T_stats, meta, node_idx=None):
    if node_idx is None:
        node_idx = int(T_stats["std"].mean(axis=1).argmax())

    t_axis = np.arange(1, N_STEPS + 1) * DT

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(
        t_axis,
        T_stats["lower"][node_idx, :],
        T_stats["upper"][node_idx, :],
        alpha=0.35, color="steelblue", label="95% credible interval"
    )
    ax.plot(t_axis, T_stats["mean"][node_idx, :],
            color="steelblue", lw=2, label="Posterior mean")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T [K]")
    ax.set_title(
        f"Posterior predictive — temperature at DOF node {node_idx}\n"
        f"(highest uncertainty node)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(FIGURES_DIR) / "credible_interval_T.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────

def run_field_reconstruction(n_samples=N_POSTERIOR_SAMPLES):
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        meta = json.load(f)
    N_T = meta["N_T_dofs"]
    N_u = meta["N_u_dofs"]

    # ── Load GP emulator ───────────────────────────────────────────────
    print("Loading GP emulator …")
    gp, scaler_X, scaler_Y = load_gp_emulator(POSTERIOR_DIR)

    # ── Load FOM training snapshots ────────────────────────────────────
    print("Loading FOM training snapshots …")
    T_train_all  = np.load(f"{SNAPSHOT_DIR}/T_snapshots_train.npy")
    u_train_all  = np.load(f"{SNAPSHOT_DIR}/u_snapshots_train.npy")
    params_train = np.load(f"{SNAPSHOT_DIR}/params_train.npy")

    # ── Draw posterior samples ─────────────────────────────────────────
    print("Drawing posterior samples …")
    k_s, h_s, Q0_s = draw_posterior_samples(n_samples)
    print(f"  k  posterior: mean={k_s.mean():.1f}  std={k_s.std():.1f}")
    print(f"  h  posterior: mean={h_s.mean():.1f}  std={h_s.std():.1f}")
    print(f"  Q0 posterior: mean={Q0_s.mean():.2e}  std={Q0_s.std():.2e}")

    # ── Build ensemble using GP + FOM interpolation ────────────────────
    T_ensemble, u_ensemble = build_field_ensemble_gp(
        k_s, h_s, Q0_s,
        gp, scaler_X, scaler_Y,
        T_train_all, u_train_all, params_train
    )

    # ── Compute statistics ─────────────────────────────────────────────
    print("Computing UQ statistics …")
    T_stats = compute_uq_statistics(T_ensemble)
    u_stats = compute_uq_statistics(u_ensemble)

    print(f"  Max pointwise T std  : {T_stats['std'].max():.2f} K")
    print(f"  Max pointwise |u| std: {u_stats['std'].max()*1e6:.4f} µm")

    # ── Save arrays ────────────────────────────────────────────────────
    save_uq_arrays(T_stats, u_stats)

    # ── Figures ────────────────────────────────────────────────────────
    print("Generating UQ figures …")
    plot_mean_and_std_fields(T_stats, u_stats, meta, step_idx=-1)
    plot_credible_interval_bands(T_stats, meta)

    print(f"\nField reconstruction complete. Outputs in {POSTERIOR_DIR}/")
    return T_stats, u_stats


if __name__ == "__main__":
    run_field_reconstruction()
