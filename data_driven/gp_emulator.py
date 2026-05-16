"""
gp_emulator.py
==============
GP emulator trained directly on FOM snapshots.

Key change from v1:
  v1 used the ROM (rom_predict_scaled) to generate 500 synthetic training
  points. The ROM has no parameter sensitivity (OpInf at a single nominal
  point), so the GP learned a flat function → useless for inference.

  v2 uses the 50 real FOM training snapshots directly.
  Input  : log(k), log(h), log(Q0)  — 50 points from LHS
  Output : sensor readings flattened — 50 × (n_obs * N_STEPS)
  This gives real physics-based parameter sensitivity.

The GP interpolates between the 50 FOM evaluations. At inference time,
predict_gp() is called for arbitrary (k,h,Q0) and returns the predicted
sensor readings which are compared to the observed y_obs in the likelihood.
"""

import numpy as np
import pickle
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

from config import (
    SNAPSHOT_DIR, N_STEPS,
    K_RANGE, H_RANGE, Q0_RANGE,
    N_T_SENSORS, N_U_SENSORS
)
from kalman_filter import place_sensors
from observations import extract_sensor_readings


# ── Build GP emulator from FOM snapshots ───────────────────────────────────

def build_gp_emulator(n_T_dofs: int,
                      n_u_dofs: int,
                      T_sensor_nodes: np.ndarray,
                      u_sensor_nodes: np.ndarray,
                      snapshot_dir: str = SNAPSHOT_DIR,
                      posterior_dir: str = "/workspace/data/posterior"):
    """
    Train a GP emulator directly on FOM training snapshots.

    Parameters
    ----------
    n_T_dofs        : number of thermal DOFs (from metadata)
    n_u_dofs        : number of displacement DOFs (from metadata)
    T_sensor_nodes  : (N_T_SENSORS,) temperature sensor node indices
    u_sensor_nodes  : (N_U_SENSORS,) displacement sensor node indices
    snapshot_dir    : path to FOM snapshots
    posterior_dir   : path to save gp_emulator.pkl

    Returns
    -------
    gp        : fitted GaussianProcessRegressor
    scaler_X  : StandardScaler for inputs
    scaler_Y  : StandardScaler for outputs
    """
    # ── Load FOM training snapshots ────────────────────────────────────
    T_train_all = np.load(f"{snapshot_dir}/T_snapshots_train.npy")
    u_train_all = np.load(f"{snapshot_dir}/u_snapshots_train.npy")
    params_train = np.load(f"{snapshot_dir}/params_train.npy")  # (N_train, 3)

    N_train = params_train.shape[0]
    print(f"  FOM training samples: {N_train}")
    print(f"  T_train_all shape: {T_train_all.shape}")

    # ── Extract sensor readings for each training sample ───────────────
    # Snapshots are concatenated: T_train_all[:, i*N_STEPS:(i+1)*N_STEPS]
    # is the i-th training simulation
    X = np.zeros((N_train, 3))   # log(k), log(h), log(Q0)
    Y = []

    for i in range(N_train):
        k_i, h_i, Q0_i = params_train[i]
        T_i = T_train_all[:, i * N_STEPS:(i + 1) * N_STEPS]
        u_i = u_train_all[:, i * N_STEPS:(i + 1) * N_STEPS]

        X[i] = [np.log(k_i), np.log(h_i), np.log(Q0_i)]

        y_sens = extract_sensor_readings(T_i, u_i,
                                          T_sensor_nodes, u_sensor_nodes)
        Y.append(y_sens.flatten())   # (n_obs * N_STEPS,)

    Y = np.array(Y)   # (N_train, n_obs * N_STEPS)
    print(f"  GP input  X shape: {X.shape}")
    print(f"  GP output Y shape: {Y.shape}")
    print(f"  Y mean: {Y.mean():.2f}  std: {Y.std():.2f}  "
          f"range: {Y.min():.1f} – {Y.max():.1f}")

    # ── Fit GP ─────────────────────────────────────────────────────────
    scaler_X = StandardScaler().fit(X)
    scaler_Y = StandardScaler().fit(Y)
    X_s = scaler_X.transform(X)
    Y_s = scaler_Y.transform(Y)

    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
             Matern(length_scale=np.ones(3),
                    length_scale_bounds=(1e-2, 1000.0),
                    nu=2.5)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,              # slightly larger nugget for FOM noise
        n_restarts_optimizer=10,
        normalize_y=False
    )
    print("Fitting GP on FOM data …")
    gp.fit(X_s, Y_s)
    print(f"GP fitted. Kernel: {gp.kernel_}")

    # ── Held-out validation on test snapshots ─────────────────────────
    try:
        T_test_all  = np.load(f"{snapshot_dir}/T_snapshots_test.npy")
        u_test_all  = np.load(f"{snapshot_dir}/u_snapshots_test.npy")
        params_test = np.load(f"{snapshot_dir}/params_test.npy")
        N_test = min(params_test.shape[0], 5)   # use up to 5 test samples

        X_val, Y_val = [], []
        for i in range(N_test):
            k_i, h_i, Q0_i = params_test[i]
            T_i = T_test_all[:, i * N_STEPS:(i + 1) * N_STEPS]
            u_i = u_test_all[:, i * N_STEPS:(i + 1) * N_STEPS]
            X_val.append([np.log(k_i), np.log(h_i), np.log(Q0_i)])
            y_sens = extract_sensor_readings(T_i, u_i,
                                              T_sensor_nodes, u_sensor_nodes)
            Y_val.append(y_sens.flatten())

        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        Y_pred_s, _ = gp.predict(scaler_X.transform(X_val), return_std=True)
        Y_pred = scaler_Y.inverse_transform(Y_pred_s)
        rel_err = np.linalg.norm(Y_pred - Y_val, axis=1) / \
                  (np.linalg.norm(Y_val, axis=1) + 1e-12)
        print(f"  GP validation relative error: "
              f"mean={rel_err.mean()*100:.1f}%  max={rel_err.max()*100:.1f}%")
    except Exception as e:
        print(f"  GP validation skipped: {e}")

    # ── Save ───────────────────────────────────────────────────────────
    Path(posterior_dir).mkdir(parents=True, exist_ok=True)
    out = Path(posterior_dir) / "gp_emulator.pkl"
    with open(out, "wb") as f:
        pickle.dump({"gp": gp, "scaler_X": scaler_X, "scaler_Y": scaler_Y}, f)
    print(f"  GP emulator saved to {out}")

    return gp, scaler_X, scaler_Y


# ── Load GP emulator ───────────────────────────────────────────────────────

def load_gp_emulator(posterior_dir: str = "/workspace/data/posterior"):
    path = Path(posterior_dir) / "gp_emulator.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["gp"], d["scaler_X"], d["scaler_Y"]


# ── Predict ────────────────────────────────────────────────────────────────

def predict_gp(gp, scaler_X, scaler_Y, log_params: np.ndarray):
    """
    Predict sensor readings for a single (log_k, log_h, log_Q0) triple.

    Parameters
    ----------
    log_params : (3,) array — [log_k, log_h, log_Q0]

    Returns
    -------
    y_mean : (n_obs * N_STEPS,)  predicted sensor readings
    y_std  : (n_obs * N_STEPS,)  predictive std
    """
    X = scaler_X.transform(log_params.reshape(1, -1))
    Y_s, std_s = gp.predict(X, return_std=True)
    y_mean = scaler_Y.inverse_transform(Y_s)[0]
    # approximate std back-transform (assumes diagonal scaler)
    y_std  = std_s[0] * scaler_Y.scale_
    return y_mean, y_std


# ── Smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        meta = json.load(f)

    T_nodes, u_nodes = place_sensors(meta["N_T_dofs"], meta["N_u_dofs"], seed=0)
    print(f"Sensor nodes — T: {T_nodes}  u: {u_nodes}")

    gp, sX, sY = build_gp_emulator(
        meta["N_T_dofs"], meta["N_u_dofs"],
        T_nodes, u_nodes
    )

    # predict at true test params
    params_test = np.load(f"{SNAPSHOT_DIR}/params_test.npy")
    k0, h0, Q0_0 = params_test[0]
    y, ys = predict_gp(gp, sX, sY,
                       np.array([np.log(k0), np.log(h0), np.log(Q0_0)]))
    print(f"\nPrediction at true params (k={k0:.1f}, h={h0:.1f}, Q0={Q0_0:.0f}):")
    print(f"  y[:5]  = {y[:5].round(2)}")
    print(f"  ys[:5] = {ys[:5].round(4)}")
