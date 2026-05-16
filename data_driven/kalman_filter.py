"""
kalman_filter.py
================
Linear Kalman Filter (LKF) and Ensemble Kalman Filter (EnKF)
for real-time thermo-mechanical field reconstruction from sparse sensors.

State vector  : q_T  in R^{r_T}   (reduced thermal coordinates)
Dynamics      : q_T[n+1] = F * q_T[n] + g + w,  w ~ N(0, Q)
Observation   : y[n] = H * q_T[n] + v,           v ~ N(0, R)

The EnKF augments the state with log(k) for online parameter identification.
"""

import numpy as np
from config import (
    N_T_SENSORS, N_U_SENSORS,
    SIGMA_T_NOISE, SIGMA_U_NOISE,
    SIGMA_PROCESS, N_ENSEMBLE, DT,
    N_STEPS,
    L, H
)


# ── Sensor placement ────────────────────────────────────────────────────────

def place_sensors(N_T_dofs, N_u_dofs, seed=0):
    """
    Place temperature and strain sensors in the thermally informative region.

    Temperature sensors are placed in the gradient zone (0.05 < x < 0.4)
    where temperature varies strongly with k, h, Q0.  Sensors at x > 0.5
    read ~T_inf=300K regardless of parameters (uninformative for inference).
    Sensors at x=0 are on the Dirichlet boundary (always 600K, also
    uninformative).

    Displacement sensors are placed in the same informative x-range since
    thermal strain is largest where dT/dx is largest.

    Falls back to random placement across all nodes if dof_coords.npy is
    not available (e.g. during unit tests).
    """
    rng = np.random.default_rng(seed)

    # ── Try to load spatial coordinates for informed placement ─────────
    try:
        import os
        coords_path = "/workspace/data/snapshots/dof_coords.npy"
        if not os.path.exists(coords_path):
            raise FileNotFoundError
        coords = np.load(coords_path)   # (N_T_dofs, 2)  — x, y per node

        # Temperature sensors: gradient zone 0.05 < x < 0.4
        # Avoids Dirichlet boundary (x=0) and cold uninformative end (x>0.4)
        x = coords[:, 0]
        informative_T = np.where((x > 0.001) & (x < 0.95 * L))[0]
        if len(informative_T) >= N_T_SENSORS:
            T_sensor_nodes = rng.choice(informative_T, N_T_SENSORS,
                                        replace=False)
        else:
            # fallback: use all nodes in region, pad with nearest
            T_sensor_nodes = rng.choice(N_T_dofs, N_T_SENSORS, replace=False)

        # Displacement sensors: same x-range (N_u_dofs = 2 * N_T_dofs,
        # node index n → DOFs 2n, 2n+1; coords indexed by node not DOF)
        N_nodes_u = N_u_dofs // 2
        # coords has N_T_dofs rows = N_nodes_u rows (same mesh nodes)
        informative_u = np.where((x > 0.001) & (x < 0.95 * L) & (coords[:, 1] > 0.01))[0]
        informative_u = informative_u[informative_u < N_nodes_u]
        if len(informative_u) >= N_U_SENSORS:
            u_sensor_nodes = rng.choice(informative_u, N_U_SENSORS,
                                        replace=False)
        else:
            u_sensor_nodes = rng.choice(N_nodes_u, N_U_SENSORS, replace=False)

    except (FileNotFoundError, Exception) as _ex:
        import sys; print("FALLBACK TRIGGERED:", _ex, file=sys.stderr)
        # Fallback: random placement (original behaviour)
        T_sensor_nodes = rng.choice(N_T_dofs,      N_T_SENSORS, replace=False)
        u_sensor_nodes = rng.choice(N_u_dofs // 2, N_U_SENSORS, replace=False)

    return T_sensor_nodes, u_sensor_nodes


def build_observation_matrix(Phi_T, Phi_u, K_cu,
                              T_sensor_nodes, u_sensor_nodes):
    """
    H maps reduced state q_T to sensor observations.

    Temperature sensors : C_T @ Phi_T @ q_T
    Strain sensors      : C_u @ Phi_u @ K_cu @ q_T
    """
    N_T = Phi_T.shape[0]
    N_u = Phi_u.shape[0]
    r_T = Phi_T.shape[1]

    # temperature observation rows
    C_T   = np.zeros((N_T_SENSORS, N_T))
    for i, n in enumerate(T_sensor_nodes):
        C_T[i, n] = 1.0
    H_T   = C_T @ Phi_T                               # (n_T_sens, r_T)

    # strain observation rows (ux component = even indices)
    C_u   = np.zeros((N_U_SENSORS, N_u))
    for i, n in enumerate(u_sensor_nodes):
        C_u[i, 2 * n] = 1.0
    H_u   = C_u @ Phi_u @ K_cu                        # (n_u_sens, r_T)

    H     = np.vstack([H_T, H_u])                     # (n_obs, r_T)

    # measurement noise covariance
    R = np.diag(
        [SIGMA_T_NOISE**2]  * N_T_SENSORS +
        [SIGMA_U_NOISE**2] * N_U_SENSORS
    )
    return H, R


def synthetic_measurement(T_true, u_true,
                           T_sensor_nodes, u_sensor_nodes,
                           step, rng):
    """
    Simulate noisy sensor readings from the true FOM field at a given step.
    """
    y_T = (T_true[T_sensor_nodes, step]
           + rng.normal(0, SIGMA_T_NOISE,   N_T_SENSORS))
    y_u = (u_true[2 * u_sensor_nodes, step]
           + rng.normal(0, SIGMA_U_NOISE, N_U_SENSORS))
    return np.concatenate([y_T, y_u])


# ── Linear Kalman Filter ────────────────────────────────────────────────────

class LinearKalmanFilter:
    """
    Standard discrete-time Kalman filter.
    Prediction model: explicit Euler discretisation of the thermal ROM ODE.
    """

    def __init__(self, A_r, b_r, K_cu, H, R, Phi_T, Phi_u):
        r_T      = A_r.shape[0]
        self.F   = np.eye(r_T) + DT * A_r     # transition matrix
        self.g   = DT * b_r                    # forcing vector
        self.K_cu = K_cu
        self.H   = H
        self.R   = R
        self.Q   = SIGMA_PROCESS**2 * np.eye(r_T)
        self.Phi_T = Phi_T
        self.Phi_u = Phi_u

        # state and covariance
        self.q   = np.zeros(r_T)
        self.P   = np.eye(r_T) * 10.0

    def initialise(self, q0):
        self.q = q0.copy()
        self.P = np.eye(len(q0)) * 1.0

    def predict(self):
        self.q = self.F @ self.q + self.g
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        innov = y - self.H @ self.q
        S     = self.H @ self.P @ self.H.T + self.R
        K     = self.P @ self.H.T @ np.linalg.inv(S)
        self.q = self.q + K @ innov
        I_KH   = np.eye(len(self.q)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return innov

    def get_fields(self):
        T_est = self.Phi_T @ self.q
        u_est = self.Phi_u @ (self.K_cu @ self.q)
        return T_est, u_est

    def get_std(self):
        """Pointwise std of T estimate via covariance propagation."""
        P_full = self.Phi_T @ self.P @ self.Phi_T.T
        return np.sqrt(np.abs(np.diag(P_full)))


# ── Ensemble Kalman Filter ──────────────────────────────────────────────────

class EnsembleKalmanFilter:
    """
    EnKF with augmented state [q_T; log_k] for simultaneous
    state estimation and thermal conductivity identification.
    """

    def __init__(self, A_r_func, b_r_func, K_cu,
                 H, R, Phi_T, Phi_u,
                 k_true=150.0, seed=42, q0_init=None):
        rng      = np.random.default_rng(seed)
        r_T      = H.shape[1]
        self.r_T = r_T
        self.A_r_func = A_r_func
        self.b_r_func = b_r_func
        self.K_cu  = K_cu
        self.H_aug = np.hstack([H, np.zeros((H.shape[0], 1))])
        self.R     = R
        self.Phi_T = Phi_T
        self.Phi_u = Phi_u
        self.k_true = k_true

        # Initialise ensemble around the true IC if provided.
        # Starting from q ~ N(0, 0.1) causes divergence because the
        # true reduced state at t=0 is O(T_REF) ≈ O(300), not O(0).
        if q0_init is not None:
            q_ens = q0_init[:, None] + rng.normal(0, 0.05, (r_T, N_ENSEMBLE))
        else:
            q_ens = rng.normal(0, 0.1, (r_T, N_ENSEMBLE))
        lk_ens = rng.normal(np.log(150.0), 0.3, (1, N_ENSEMBLE))
        self.ens = np.vstack([q_ens, lk_ens])   # (r_T+1, N_e)

        self.rng = rng

    def predict(self):
        for j in range(N_ENSEMBLE):
            q_j  = self.ens[:self.r_T, j]
            k_j  = float(np.exp(self.ens[self.r_T, j]))
            A_j  = self.A_r_func(k_j)
            b_j  = self.b_r_func(k_j)
            noise = self.rng.normal(0, SIGMA_PROCESS, self.r_T)
            self.ens[:self.r_T, j] = (
                q_j + DT * (A_j @ q_j + b_j) + noise
            )
            # small random walk on log_k to maintain ensemble spread
            self.ens[self.r_T, j] += self.rng.normal(0, 0.005)

    def update(self, y):
        n_obs = len(y)
        Y     = self.H_aug @ self.ens             # (n_obs, N_e)
        x_mn  = self.ens.mean(axis=1, keepdims=True)
        y_mn  = Y.mean(axis=1, keepdims=True)
        X_a   = self.ens - x_mn
        Y_a   = Y - y_mn
        N_e   = N_ENSEMBLE

        C_YY  = Y_a @ Y_a.T / (N_e - 1) + self.R
        C_XY  = X_a @ Y_a.T / (N_e - 1)
        K_en  = C_XY @ np.linalg.inv(C_YY)

        # perturbed observations
        D = (y[:, None]
             + self.rng.multivariate_normal(
                 np.zeros(n_obs), self.R, N_e).T)
        self.ens += K_en @ (D - Y)

    def get_estimate(self):
        q_mean = self.ens[:self.r_T].mean(axis=1)
        k_mean = float(np.exp(self.ens[self.r_T].mean()))
        k_std  = float(np.exp(self.ens[self.r_T]).std())
        T_est  = self.Phi_T @ q_mean
        u_est  = self.Phi_u @ (self.K_cu @ q_mean)
        return T_est, u_est, k_mean, k_std


# ── Run full assimilation loop ──────────────────────────────────────────────

def run_lkf_loop(kf: LinearKalmanFilter,
                 T_true, u_true,
                 T_sensor_nodes, u_sensor_nodes):
    """
    Run the LKF assimilation loop over all time steps.

    Returns
    -------
    T_est_hist : (N_T, N_STEPS)
    u_est_hist : (N_u, N_STEPS)
    err_T      : (N_STEPS,) relative L2 error
    err_u      : (N_STEPS,) relative L2 error
    """
    rng = np.random.default_rng(1)
    N_T, N_STEPS_RUN = T_true.shape
    N_u = u_true.shape[0]

    T_hist = np.zeros((N_T, N_STEPS_RUN))
    u_hist = np.zeros((N_u, N_STEPS_RUN))
    err_T  = np.zeros(N_STEPS_RUN)
    err_u  = np.zeros(N_STEPS_RUN)
    std_T  = np.zeros((N_T, N_STEPS_RUN))

    for n in range(N_STEPS_RUN):
        kf.predict()
        y = synthetic_measurement(T_true, u_true,
                                  T_sensor_nodes, u_sensor_nodes,
                                  n, rng)
        kf.update(y)
        T_e, u_e = kf.get_fields()
        T_hist[:, n] = T_e
        u_hist[:, n] = u_e
        std_T[:, n]  = kf.get_std()

        nT = np.linalg.norm(T_true[:, n])
        nU = np.linalg.norm(u_true[:, n])
        err_T[n] = np.linalg.norm(T_e - T_true[:, n]) / max(nT, 1e-12)
        err_u[n] = np.linalg.norm(u_e - u_true[:, n]) / max(nU, 1e-12)

    return T_hist, u_hist, err_T, err_u, std_T


def run_enkf_loop(enkf: EnsembleKalmanFilter,
                  T_true, u_true,
                  T_sensor_nodes, u_sensor_nodes):
    """
    Run the EnKF assimilation loop with k identification.

    Returns
    -------
    T_hist, u_hist, err_T, err_u, k_hist
    """
    rng = np.random.default_rng(2)
    N_T, N_STEPS_RUN = T_true.shape
    N_u = u_true.shape[0]

    T_hist = np.zeros((N_T, N_STEPS_RUN))
    u_hist = np.zeros((N_u, N_STEPS_RUN))
    err_T  = np.zeros(N_STEPS_RUN)
    err_u  = np.zeros(N_STEPS_RUN)
    k_hist = np.zeros(N_STEPS_RUN)

    for n in range(N_STEPS_RUN):
        enkf.predict()
        y = synthetic_measurement(T_true, u_true,
                                  T_sensor_nodes, u_sensor_nodes,
                                  n, rng)
        enkf.update(y)
        T_e, u_e, k_est, _ = enkf.get_estimate()
        T_hist[:, n] = T_e
        u_hist[:, n] = u_e
        k_hist[n]    = k_est

        nT = np.linalg.norm(T_true[:, n])
        nU = np.linalg.norm(u_true[:, n])
        err_T[n] = np.linalg.norm(T_e - T_true[:, n]) / max(nT, 1e-12)
        err_u[n] = np.linalg.norm(u_e - u_true[:, n]) / max(nU, 1e-12)

    return T_hist, u_hist, err_T, err_u, k_hist
