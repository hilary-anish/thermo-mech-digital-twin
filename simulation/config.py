"""
config.py — shared parameters for both simulation and ML containers.
Both containers mount /workspace/data as a shared Docker volume.
Simulation writes to /workspace/data/snapshots/
data-driven pipeline reads from /workspace/data/snapshots/

Geometry change (v2):
  L reduced from 1.0m → 0.1m so that thermal diffusion penetrates the full
  fin within the simulation window.
  Diffusion length scale: sqrt(alpha * T_END) = sqrt(8e-5 * 100) ≈ 0.09m
  T_END increased from 10s → 100s, DT from 0.5s → 5s (N_STEPS=20 unchanged).
"""

import numpy as np

# ── Geometry ──────────────────────────────────────────────────────────────
L = 0.1          # fin length [m]  (was 1.0m — reduced for thermal penetration)
H = 0.1          # fin height [m]
NX = 40          # elements in x
NY = 4           # elements in y

# ── Fixed material properties ─────────────────────────────────────────────
RHO   = 2700.0   # density [kg/m3]
CP    = 900.0    # specific heat [J/kg/K]
E_MOD = 70e9     # Young's modulus [Pa]
NU    = 0.33     # Poisson ratio
ALPHA = 23e-6    # thermal expansion [1/K]
T_REF = 300.0    # stress-free reference temperature [K]
T_HOT = 600.0    # left Dirichlet temperature [K]
T_INF = 300.0    # ambient temperature [K]

# ── Time stepping ─────────────────────────────────────────────────────────
T_END = 100.0    # simulation end time [s]  (was 10s)
DT    = 5.0      # time step [s]            (was 0.5s)
N_STEPS = int(T_END / DT)   # = 20  (unchanged)

# ── Parametric ranges (LHS sweep) ─────────────────────────────────────────
K_RANGE    = (50.0,  200.0)   # thermal conductivity [W/mK]
H_RANGE    = (5.0,   80.0)    # convection coefficient [W/m2K]
Q0_RANGE   = (1e4,   5e5)     # heat source amplitude [W/m3]

# ── Snapshot collection ───────────────────────────────────────────────────
N_SAMPLES_TRAIN = 50         # training parameter samples
N_SAMPLES_TEST  = 10         # test parameter samples (held out)

# ── ROM ───────────────────────────────────────────────────────────────────
ENERGY_THRESHOLD = 0.9999     # POD truncation criterion

# ── Kalman Filter ─────────────────────────────────────────────────────────
N_T_SENSORS    = 5            # temperature sensor count
N_U_SENSORS    = 3            # strain sensor count
SIGMA_T_NOISE  = 1.0          # temperature measurement noise [K]
SIGMA_U_NOISE  = 10e-6        # strain measurement noise [-]
SIGMA_PROCESS  = 0.01         # process noise standard deviation
N_ENSEMBLE     = 200          # EnKF ensemble size

# ── Paths (inside containers — mounted volume) ────────────────────────────
DATA_DIR       = "/workspace/data"
SNAPSHOT_DIR   = "/workspace/data/snapshots"
FIGURE_DIR     = "/workspace/data/figures"
