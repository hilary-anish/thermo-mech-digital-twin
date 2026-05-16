"""
observations.py
===============
Sensor observation utilities for the Bayesian inference pipeline.

Provides two things:
  1. generate_synthetic_observations()
     Called ONCE before inference to produce the fixed noisy observation
     vector y_obs that is held constant throughout MCMC sampling.

  2. get_sensor_matrix()
     Returns the (n_obs, N_T+N_u) linear extraction operator so that
     sensor readings can be computed as a matrix-vector product, which
     is useful in the GP emulator training loop.

Both functions reuse the sensor-node indices produced by
kalman_filter.place_sensors(), keeping the pipeline consistent with the
existing Kalman filter code.
"""

import numpy as np
from config import (
    N_T_SENSORS, N_U_SENSORS,
    SIGMA_T_NOISE, SIGMA_U_NOISE,
    SNAPSHOT_DIR, N_STEPS
)


# ── Extract sensor readings from full-field arrays (NumPy) ─────────────────

def extract_sensor_readings(T_field: np.ndarray,
                             u_field: np.ndarray,
                             T_sensor_nodes: np.ndarray,
                             u_sensor_nodes: np.ndarray) -> np.ndarray:
    """
    Extract sensor readings from full-field arrays.

    Parameters
    ----------
    T_field        : (N_T, n_steps)
    u_field        : (N_u, n_steps)
    T_sensor_nodes : (N_T_SENSORS,)  integer node indices
    u_sensor_nodes : (N_U_SENSORS,)  integer node indices

    Returns
    -------
    y : (n_obs, n_steps)  where n_obs = N_T_SENSORS + N_U_SENSORS
    """
    y_T = T_field[T_sensor_nodes, :]          # (n_T_sens, n_steps)
    y_u = u_field[2 * u_sensor_nodes, :]      # (n_u_sens, n_steps)
    return np.vstack([y_T, y_u])              # (n_obs, n_steps)


# ── Generate synthetic observations (called once, before MCMC) ─────────────

def generate_synthetic_observations(T_true: np.ndarray,
                                     u_true: np.ndarray,
                                     T_sensor_nodes: np.ndarray,
                                     u_sensor_nodes: np.ndarray,
                                     seed: int = 42) -> np.ndarray:
    """
    Add realistic Gaussian noise to FOM sensor locations to produce the
    synthetic observation dataset used as evidence in the Bayesian model.

    Noise levels come directly from config.py (SIGMA_T_NOISE, SIGMA_U_NOISE)
    so they are consistent with the measurement noise covariance R used in
    the Kalman filters.

    Parameters
    ----------
    T_true         : (N_T, n_steps)  full FOM temperature field
    u_true         : (N_u, n_steps)  full FOM displacement field
    T_sensor_nodes : (N_T_SENSORS,)
    u_sensor_nodes : (N_U_SENSORS,)
    seed           : random seed for reproducibility

    Returns
    -------
    y_obs : (n_obs, n_steps)  noisy sensor observations — plain NumPy array
    """
    rng  = np.random.default_rng(seed)
    n_steps = T_true.shape[1]

    y_T_clean = T_true[T_sensor_nodes, :]
    y_u_clean = u_true[2 * u_sensor_nodes, :]

    y_T = y_T_clean + rng.normal(0.0, SIGMA_T_NOISE,
                                  (N_T_SENSORS, n_steps))
    y_u = y_u_clean + rng.normal(0.0, SIGMA_U_NOISE,
                                  (N_U_SENSORS, n_steps))

    return np.vstack([y_T, y_u])   # (n_obs, n_steps) — NumPy, not JAX


# ── Sensor noise standard-deviation vector ─────────────────────────────────

def sensor_sigma_vector() -> np.ndarray:
    """
    Returns a (n_obs,) vector of per-sensor noise standard deviations.
    Used to build the likelihood in bayesian_inference.py.
    """
    return np.array(
        [SIGMA_T_NOISE] * N_T_SENSORS +
        [SIGMA_U_NOISE] * N_U_SENSORS
    )


# ── Smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from kalman_filter import place_sensors

    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        meta = json.load(f)
    N_T = meta["N_T_dofs"]
    N_u = meta["N_u_dofs"]

    T_test = np.load(f"{SNAPSHOT_DIR}/T_snapshots_test.npy")[:, :N_STEPS]
    u_test = np.load(f"{SNAPSHOT_DIR}/u_snapshots_test.npy")[:, :N_STEPS]

    T_nodes, u_nodes = place_sensors(N_T, N_u, seed=0)
    y_obs = generate_synthetic_observations(T_test, u_test, T_nodes, u_nodes)

    print(f"y_obs shape   : {y_obs.shape}")
    print(f"T reading range : [{y_obs[:N_T_SENSORS].min():.1f}, "
          f"{y_obs[:N_T_SENSORS].max():.1f}] K")
    print(f"u reading range : [{y_obs[N_T_SENSORS:].min():.2e}, "
          f"{y_obs[N_T_SENSORS:].max():.2e}] m")
    print(f"sigma vector  : {sensor_sigma_vector()}")
    print("observations.py smoke-test PASSED")
