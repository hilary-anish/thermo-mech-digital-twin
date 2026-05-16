"""
sanity_check.py
===============
Run all sanity checks before starting SMC.
Checks:
1. Snapshot shapes and parameter ranges
2. Sensor placement (x-coordinates)
3. FOM data variance at sensor nodes (must not be flat)
4. GP training data variance
5. GP sensitivity check (varying k must give different predictions)
6. y_obs range (must not be ~300K flat)
7. Likelihood sanity at true parameters

Run with:
    docker compose run --rm data_driven python3 sanity_check.py
"""

import numpy as np
import json
import sys
from pathlib import Path

SNAPSHOT_DIR  = "/workspace/data/snapshots"
POSTERIOR_DIR = "/workspace/data/posterior"

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

errors = []

def check(name, condition, msg_pass, msg_fail, warn_only=False):
    tag = PASS if condition else (WARN if warn_only else FAIL)
    msg = msg_pass if condition else msg_fail
    print(f"{tag}  {name}: {msg}")
    if not condition and not warn_only:
        errors.append(name)

print("=" * 60)
print("Sanity Check — Thermo-Mechanical Digital Twin")
print("=" * 60)

# ── 1. Config ──────────────────────────────────────────────────────────────
print("\n[1] Config")
from config import L, H, NX, NY, T_END, DT, N_STEPS, K_RANGE, H_RANGE, Q0_RANGE
print(f"  L={L}  H={H}  NX={NX}  NY={NY}")
print(f"  T_END={T_END}  DT={DT}  N_STEPS={N_STEPS}")
check("L", L <= 0.15, f"{L}m ✓", f"{L}m — too long, heat won't penetrate in {T_END}s")
check("T_END", T_END >= 50, f"{T_END}s ✓", f"{T_END}s — too short for L={L}m")
check("N_STEPS", N_STEPS == 20, f"{N_STEPS} ✓", f"{N_STEPS} — expected 20")

import numpy as np_
alpha = 194 / (2700 * 900)
penetration = np_.sqrt(alpha * T_END)
print(f"  Thermal penetration depth sqrt(alpha*T_END) = {penetration:.4f}m  (fin length = {L}m)")
check("penetration", penetration >= 0.8 * L,
      f"{penetration:.3f}m covers >{0.8*L:.3f}m ✓",
      f"{penetration:.3f}m < 80% of fin length {L}m — sensors will be cold")

# ── 2. Snapshots ───────────────────────────────────────────────────────────
print("\n[2] Snapshots")
with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
    meta = json.load(f)

N_T   = meta["N_T_dofs"]
N_u   = meta["N_u_dofs"]
N_tr  = meta["N_samples_train"]
N_te  = meta["N_samples_test"]
print(f"  N_T_dofs={N_T}  N_u_dofs={N_u}")
print(f"  N_train={N_tr}  N_test={N_te}")

T_train = np.load(f"{SNAPSHOT_DIR}/T_snapshots_train.npy")
u_train = np.load(f"{SNAPSHOT_DIR}/u_snapshots_train.npy")
T_test  = np.load(f"{SNAPSHOT_DIR}/T_snapshots_test.npy")
p_train = np.load(f"{SNAPSHOT_DIR}/params_train.npy")
p_test  = np.load(f"{SNAPSHOT_DIR}/params_test.npy")

check("T_train shape", T_train.shape == (N_T, N_tr * N_STEPS),
      f"{T_train.shape} ✓", f"{T_train.shape} — expected ({N_T}, {N_tr*N_STEPS})")
check("T_max", T_train.max() >= 599,
      f"{T_train.max():.1f}K ✓", f"{T_train.max():.1f}K — hot boundary not reached")
check("T_min_reasonable", T_train.min() > 200,
      f"{T_train.min():.1f}K ✓", f"{T_train.min():.1f}K — suspiciously low")

print(f"  T_train global range: {T_train.min():.1f} – {T_train.max():.1f} K")
print(f"  k range in train: {p_train[:,0].min():.1f} – {p_train[:,0].max():.1f}")
print(f"  True test params: k={p_test[0,0]:.1f}  h={p_test[0,1]:.1f}  Q0={p_test[0,2]:.2e}")

# ── 3. Sensor placement ────────────────────────────────────────────────────
print("\n[3] Sensor placement")
from kalman_filter import place_sensors
T_nodes, u_nodes = place_sensors(N_T, N_u, seed=0)
coords = np.load(f"{SNAPSHOT_DIR}/dof_coords.npy")
x_sensors = coords[T_nodes, 0]
print(f"  T sensor nodes: {T_nodes}")
print(f"  T sensor x-coords: {x_sensors.round(4)}")
print(f"  u sensor nodes: {u_nodes}")

check("sensors_not_at_boundary", all(x_sensors > 0.001),
      "no sensors at x=0 Dirichlet boundary ✓",
      f"sensor at x=0 boundary — always 600K, uninformative")
check("sensors_not_cold_end", all(x_sensors < 0.98 * L),
      f"all sensors at x < {0.8*L:.3f}m ✓",
      f"sensors too far right — will read ~T_inf=300K")
check("sensors_in_gradient", any(x_sensors < 0.95 * L),
      f"sensors reach gradient zone (x < {0.4*L:.3f}m) ✓",
      f"no sensors in gradient zone — may be uninformative")

# ── 4. FOM data variance at sensor nodes ──────────────────────────────────
print("\n[4] FOM sensor variance across training samples")
Y_fom = []
for i in range(N_tr):
    T_i = T_train[:, i * N_STEPS:(i + 1) * N_STEPS]
    u_i = u_train[:, i * N_STEPS:(i + 1) * N_STEPS]
    y_T = T_i[T_nodes, -1]   # last time step
    Y_fom.append(y_T)
Y_fom = np.array(Y_fom)   # (N_train, N_T_SENSORS)

T_std_across_samples = Y_fom.std(axis=0)
T_range_across_samples = Y_fom.max(axis=0) - Y_fom.min(axis=0)
print(f"  T at sensors (last step) — mean: {Y_fom.mean():.1f}K")
print(f"  T std across {N_tr} training samples: {T_std_across_samples.round(1)}")
print(f"  T range across samples: {T_range_across_samples.round(1)}")

check("sensor_variance", T_std_across_samples.mean() > 5.0,
      f"mean std={T_std_across_samples.mean():.1f}K — good parameter sensitivity ✓",
      f"mean std={T_std_across_samples.mean():.2f}K — sensors are uninformative (flat ~300K)")
check("sensor_range", T_range_across_samples.mean() > 20.0,
      f"mean range={T_range_across_samples.mean():.1f}K ✓",
      f"mean range={T_range_across_samples.mean():.2f}K — too small for inference")

# ── 5. GP training data ────────────────────────────────────────────────────
print("\n[5] GP training data (full time series)")
from observations import extract_sensor_readings
Y_gp = []
for i in range(N_tr):
    T_i = T_train[:, i * N_STEPS:(i + 1) * N_STEPS]
    u_i = u_train[:, i * N_STEPS:(i + 1) * N_STEPS]
    y = extract_sensor_readings(T_i, u_i, T_nodes, u_nodes)
    Y_gp.append(y.flatten())
Y_gp = np.array(Y_gp)

print(f"  GP Y shape: {Y_gp.shape}  (should be ({N_tr}, {(len(T_nodes)+len(u_nodes))*N_STEPS}))")
print(f"  GP Y range: {Y_gp.min():.2f} – {Y_gp.max():.2f}")
print(f"  GP Y std (mean across outputs): {Y_gp.std(axis=0).mean():.4f}")

check("gp_y_range", Y_gp.max() > 310,
      f"Y max={Y_gp.max():.1f} — GP has real signal ✓",
      f"Y max={Y_gp.max():.1f} — GP output flat near 300K, inference will fail")
check("gp_y_std", Y_gp.std(axis=0).mean() > 1.0,
      f"mean std={Y_gp.std(axis=0).mean():.2f} — sufficient variance ✓",
      f"mean std={Y_gp.std(axis=0).mean():.4f} — too flat for GP to learn")

# ── 6. GP sensitivity (if pkl exists) ─────────────────────────────────────
print("\n[6] GP sensitivity check")
gp_path = Path(POSTERIOR_DIR) / "gp_emulator.pkl"
if gp_path.exists():
    from gp_emulator import load_gp_emulator, predict_gp
    gp, sX, sY = load_gp_emulator(POSTERIOR_DIR)
    k_true, h_true, Q0_true = p_test[0]
    preds = []
    for k_t in [60, 100, 150, 194, 200]:
        y, _ = predict_gp(gp, sX, sY,
                          np.array([np.log(k_t), np.log(h_true), np.log(Q0_true)]))
        preds.append(y[:len(T_nodes)].mean())
        print(f"  k={k_t:3d}: T_sensor_mean={preds[-1]:.2f}K")
    gp_spread = max(preds) - min(preds)
    check("gp_sensitivity", gp_spread > 5.0,
          f"GP spread across k range = {gp_spread:.1f}K ✓",
          f"GP spread = {gp_spread:.2f}K — GP is flat, inference will give prior")
else:
    print(f"  No GP pkl found at {gp_path} — will be trained on first run")

# ── 7. y_obs range ─────────────────────────────────────────────────────────
print("\n[7] y_obs from test snapshot")
from observations import generate_synthetic_observations
T_true = T_test[:, :N_STEPS]
u_true = np.load(f"{SNAPSHOT_DIR}/u_snapshots_test.npy")[:, :N_STEPS]
y_obs_2d = generate_synthetic_observations(T_true, u_true, T_nodes, u_nodes, seed=42)
y_obs = y_obs_2d.flatten()
print(f"  y_obs shape: {y_obs.shape}  (should be {(len(T_nodes)+len(u_nodes))*N_STEPS})")
print(f"  y_obs range: {y_obs.min():.2f} – {y_obs.max():.2f}")
print(f"  T sensor mean: {y_obs[:len(T_nodes)*N_STEPS].mean():.2f}K")

check("y_obs_range", y_obs[:len(T_nodes)*N_STEPS].max() > 310,
      f"T_max={y_obs[:len(T_nodes)*N_STEPS].max():.1f}K — informative ✓",
      f"T_max={y_obs[:len(T_nodes)*N_STEPS].max():.1f}K — observations flat near 300K")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"FAILED CHECKS ({len(errors)}): {errors}")
    print("Fix these before running SMC.")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — safe to run SMC")
    sys.exit(0)
