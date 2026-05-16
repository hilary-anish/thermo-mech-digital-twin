"""
bayesian_inference.py
=====================
Defines and runs the Bayesian inverse problem using PyMC.

Pipeline
--------
1. Load the FOM test trajectory and generate noisy synthetic observations y_obs.
2. Build or load a GP emulator trained on FOM training snapshots.
3. Define a PyMC probabilistic model:
       priors  : log_k, log_h, log_Q0 ~ Normal (log-space)
       forward : GP emulator predicts sensor readings
       likelihood : y_obs ~ Normal(y_hat, sigma_obs)  via pm.Potential
4. Sample the posterior with SMC (Sequential Monte Carlo).
5. Save the ArviZ InferenceData trace to /workspace/data/posterior/trace.nc
6. Print a convergence summary (r_hat, ESS).

Why FOM-GP and not ROM-GP?
--------------------------
The ROM (OpInf at a single nominal point) has no parameter sensitivity —
it produces identical output for any (k, h, Q0). The FOM snapshots contain
real physics at 50 different parameter points and are directly usable as
GP training data. See gp_emulator.py for details.

Why SMC and not NUTS?
---------------------
The likelihood uses pm.Potential (black-box, no gradient). SMC handles
this naturally via particle tempering. NUTS requires gradients.
PyMC issue #7078: as_op cannot be pickled for parallel chains → chains=1.
"""

import numpy as np
import json
import time
from pathlib import Path

import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pytensor.compile.ops import as_op

from config import (
    SNAPSHOT_DIR, N_STEPS,
    K_RANGE, H_RANGE, Q0_RANGE,
    N_T_SENSORS, N_U_SENSORS
)
from kalman_filter import place_sensors
from observations import generate_synthetic_observations, sensor_sigma_vector
from gp_emulator import build_gp_emulator, load_gp_emulator, predict_gp

ROM_DIR       = "/workspace/data/rom"
POSTERIOR_DIR = "/workspace/data/posterior"

# ── Prior hyperparameters (log-space Normal) ───────────────────────────────
# Centers at geometric midpoints of physical ranges.
# Sigmas so that 2-sigma interval covers the full range.
PRIOR = {
    "log_k":  {"mu": (np.log(K_RANGE[1])  + np.log(K_RANGE[0]))  / 2,
               "sigma": (np.log(K_RANGE[1])  - np.log(K_RANGE[0]))  / 4},
    "log_h":  {"mu": (np.log(H_RANGE[1])  + np.log(H_RANGE[0]))  / 2,
               "sigma": (np.log(H_RANGE[1])  - np.log(H_RANGE[0]))  / 4},
    "log_Q0": {"mu": (np.log(Q0_RANGE[1]) + np.log(Q0_RANGE[0])) / 2,
               "sigma": (np.log(Q0_RANGE[1]) - np.log(Q0_RANGE[0])) / 4},
}


# ── PyMC model builder ─────────────────────────────────────────────────────

def build_pymc_model(y_obs: np.ndarray,
                     gp,
                     scaler_X,
                     scaler_Y,
                     sigma_obs: np.ndarray):
    """
    Construct the PyMC probabilistic model.

    Parameters
    ----------
    y_obs      : (n_obs * N_STEPS,)  flattened observed sensor vector
    gp, scaler_X, scaler_Y : trained GP emulator components
    sigma_obs  : (n_obs,) per-sensor noise std, tiled to match y_obs

    Returns
    -------
    model : pm.Model
    """
    n_obs_flat = len(y_obs)
    # Tile sigma to match y_obs length
    n_t = n_obs_flat // (N_T_SENSORS + N_U_SENSORS)
    sigma_tiled = np.repeat(sigma_obs, n_t)
    log_sigma_sum = np.sum(np.log(sigma_tiled))

    with pm.Model() as model:

        # ── Priors ──────────────────────────────────────────────────────
        log_k  = pm.Normal("log_k",  mu=PRIOR["log_k"]["mu"],
                            sigma=PRIOR["log_k"]["sigma"])
        log_h  = pm.Normal("log_h",  mu=PRIOR["log_h"]["mu"],
                            sigma=PRIOR["log_h"]["sigma"])
        log_Q0 = pm.Normal("log_Q0", mu=PRIOR["log_Q0"]["mu"],
                            sigma=PRIOR["log_Q0"]["sigma"])

        # ── Back-transformed physical parameters ─────────────────────
        k  = pm.Deterministic("k",  pm.math.exp(log_k))
        h  = pm.Deterministic("h",  pm.math.exp(log_h))
        Q0 = pm.Deterministic("Q0", pm.math.exp(log_Q0))

        # ── Log-likelihood via pm.Potential (black-box) ──────────────
        @as_op(itypes=[pt.dscalar, pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
        def loglike_op(lk, lh, lQ0):
            log_params = np.array([float(lk), float(lh), float(lQ0)])
            y_hat, _ = predict_gp(gp, scaler_X, scaler_Y, log_params)
            # y_hat is (n_obs * N_STEPS,) — same shape as y_obs
            residuals = (y_obs - y_hat) / sigma_tiled
            loglike = (-0.5 * np.sum(residuals**2)
                       - log_sigma_sum
                       - 0.5 * n_obs_flat * np.log(2 * np.pi))
            return np.array(loglike, dtype=np.float64)

        pm.Potential("likelihood", loglike_op(log_k, log_h, log_Q0))

    return model


# ── Main inference run ─────────────────────────────────────────────────────

def run_bayesian_inference(
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        force_retrain_gp: bool = False
):
    """
    Full Bayesian inference pipeline.

    Parameters
    ----------
    draws            : SMC particles
    tune             : unused (kept for CLI compatibility)
    chains           : unused (SMC uses chains=1 due to PyMC issue #7078)
    force_retrain_gp : if True, retrain GP even if gp_emulator.pkl exists
    """
    Path(POSTERIOR_DIR).mkdir(parents=True, exist_ok=True)
    t_wall = time.time()

    # ── Load test snapshot (first test sample = inference target) ──────
    print("=" * 60)
    print("Bayesian Inverse Problem — Loading data")
    print("=" * 60)

    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        meta = json.load(f)
    N_T = meta["N_T_dofs"]
    N_u = meta["N_u_dofs"]

    T_true_all  = np.load(f"{SNAPSHOT_DIR}/T_snapshots_test.npy")
    u_true_all  = np.load(f"{SNAPSHOT_DIR}/u_snapshots_test.npy")
    params_test = np.load(f"{SNAPSHOT_DIR}/params_test.npy")

    T_true = T_true_all[:, :N_STEPS]
    u_true = u_true_all[:, :N_STEPS]
    k_true, h_true, Q0_true = (float(params_test[0, 0]),
                                float(params_test[0, 1]),
                                float(params_test[0, 2]))
    print(f"  True parameters: k={k_true:.1f}  h={h_true:.1f}  Q0={Q0_true:.2e}")

    # ── Place sensors ──────────────────────────────────────────────────
    T_nodes, u_nodes = place_sensors(N_T, N_u, seed=0)
    print(f"  T sensor nodes: {T_nodes}")

    # ── Generate synthetic observations from FOM ───────────────────────
    y_obs_2d = generate_synthetic_observations(T_true, u_true,
                                                T_nodes, u_nodes, seed=42)
    y_obs = y_obs_2d.flatten().astype(np.float64)
    print(f"  y_obs shape (flat): {y_obs.shape}")
    print(f"  y_obs range: {y_obs.min():.1f} – {y_obs.max():.1f} K")

    # ── Build or load GP emulator ──────────────────────────────────────
    gp_path = Path(POSTERIOR_DIR) / "gp_emulator.pkl"
    if gp_path.exists() and not force_retrain_gp:
        print("\nLoading existing GP emulator …")
        gp, scaler_X, scaler_Y = load_gp_emulator(POSTERIOR_DIR)
    else:
        print("\nTraining GP emulator on FOM snapshots …")
        gp, scaler_X, scaler_Y = build_gp_emulator(
            N_T, N_u, T_nodes, u_nodes,
            posterior_dir=POSTERIOR_DIR
        )

    # ── Verify GP is informative ───────────────────────────────────────
    print("\nGP sanity check (T sensor mean at varying k):")
    for k_test in [60, 100, 150, 194, 200]:
        y_t, _ = predict_gp(gp, scaler_X, scaler_Y,
                             np.array([np.log(k_test), np.log(h_true),
                                       np.log(Q0_true)]))
        print(f"  k={k_test:3d}: T_mean={y_t[:N_T_SENSORS].mean():.1f} K")

    # ── Build PyMC model ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building PyMC model …")
    print("=" * 60)

    # Model discrepancy inflation (Kennedy & O'Hagan 2001)
    sigma_obs_raw = sensor_sigma_vector()
    sigma_model_T = 5.0    # K — reduced now that GP is trained on FOM directly
    sigma_model_u = 1.0e-5 # m
    sigma_T_inf = np.sqrt(sigma_obs_raw[:N_T_SENSORS]**2 + sigma_model_T**2)
    sigma_u_inf = np.sqrt(sigma_obs_raw[N_T_SENSORS:]**2 + sigma_model_u**2)
    sigma_obs   = np.concatenate([sigma_T_inf, sigma_u_inf])
    print(f"  sigma_T (inflated): {sigma_obs[:N_T_SENSORS].mean():.2f} K")
    print(f"  sigma_u (inflated): {sigma_obs[N_T_SENSORS:].mean():.2e}")

    model = build_pymc_model(y_obs, gp, scaler_X, scaler_Y, sigma_obs)

    # ── Sample with SMC ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Running SMC  (particles={draws}, chains=1)")
    print("=" * 60)

    import time as _time
    t0 = _time.time()

    with model:
        trace = pm.sample_smc(
            draws=draws,
            chains=1,
            correlation_threshold=0.9,
            cores=0,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42,
        )

    elapsed = _time.time() - t0
    print(f"  SMC done — {int(elapsed//60)}m {int(elapsed%60)}s")

    # ── Save trace ─────────────────────────────────────────────────────
    trace_path = Path(POSTERIOR_DIR) / "trace.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Trace saved to {trace_path}")

    # ── Posterior summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Posterior summary  (k, h, Q0)")
    print("=" * 60)
    summary = az.summary(trace, var_names=["k", "h", "Q0"])
    print(summary)

    r_hat_vals = summary["r_hat"].values
    ess_vals   = summary["ess_bulk"].values
    converged  = all(r_hat_vals < 1.05) and all(ess_vals > 200)
    print(f"\n  r_hat max : {r_hat_vals.max():.4f}  "
          f"({'OK' if r_hat_vals.max() < 1.05 else 'WARN'})")
    print(f"  ESS min   : {ess_vals.min():.0f}  "
          f"({'OK' if ess_vals.min() > 200 else 'WARN'})")

    # ── Save summary JSON ──────────────────────────────────────────────
    posterior_summary = {
        "k_true":    k_true,   "h_true":  h_true,  "Q0_true":  Q0_true,
        "k_mean":    float(summary.loc["k",  "mean"]),
        "k_sd":      float(summary.loc["k",  "sd"]),
        "k_hdi_3%":  float(summary.loc["k",  "hdi_3%"]),
        "k_hdi_97%": float(summary.loc["k",  "hdi_97%"]),
        "h_mean":    float(summary.loc["h",  "mean"]),
        "h_sd":      float(summary.loc["h",  "sd"]),
        "Q0_mean":   float(summary.loc["Q0", "mean"]),
        "Q0_sd":     float(summary.loc["Q0", "sd"]),
        "r_hat_max": float(r_hat_vals.max()),
        "ess_min":   float(ess_vals.min()),
        "converged": bool(converged),
        "n_draws":   draws,
        "wall_time_s": round(time.time() - t_wall, 1),
    }
    summary_path = Path(POSTERIOR_DIR) / "posterior_summary.json"
    with open(summary_path, "w") as f:
        json.dump(posterior_summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    print(f"Total wall time: {time.time()-t_wall:.1f}s")

    return trace


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bayesian inverse problem — thermo-mechanical digital twin"
    )
    parser.add_argument("--draws",      type=int,  default=2000)
    parser.add_argument("--tune",       type=int,  default=1000)
    parser.add_argument("--chains",     type=int,  default=4)
    parser.add_argument("--retrain-gp", action="store_true", dest="retrain_gp")
    args = parser.parse_args()

    run_bayesian_inference(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        force_retrain_gp=args.retrain_gp,
    )
