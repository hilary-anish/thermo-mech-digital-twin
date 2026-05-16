# Thermo-Mechanical Digital Twin

> POD-ROM · Ensemble Kalman Filter · Bayesian Inversion · FEniCSx · PyMC · Streamlit

A containerised end-to-end digital twin for a 2D coupled thermo-mechanical aluminium fin. Combines finite element simulation, reduced-order modelling, real-time data assimilation, and Bayesian parameter identification with full uncertainty quantification.

---

## Overview

The system identifies unknown material properties (thermal conductivity k, convection coefficient h, heat source Q₀) from sparse sensor measurements, reconstructs the full temperature and displacement fields in real time, and quantifies uncertainty in all estimates.

```
FEniCSx FOM  →  POD-ROM  →  Kalman Filter  →  GP Emulator  →  Bayesian SMC  →  Dashboard
  (truth)       (3 modes)    (real-time)       (surrogate)     (posteriors)     (Streamlit)
```

---

## Results

| Metric | Value |
|---|---|
| POD compression ratio | 333× (99.999% energy, 3 modes) |
| LKF temperature reconstruction error | 16.5% |
| EnKF temperature reconstruction error | **4.3%** |
| Bayesian k identification error | **1.8%** (198.3 vs true 194.2 W/mK) |
| Posterior k uncertainty | ±2.4 W/mK |
| SMC runtime (500 particles) | ~37 seconds |

---

## Physics

A 2D aluminium fin (0.1m × 0.1m, 40×4 quad mesh) with:

- **Left edge** — Dirichlet boundary: T = 600K (heat source)
- **All other edges** — Robin boundary: convective cooling h(T − T∞), T∞ = 300K
- **Bottom edge** — mechanically clamped: u = 0
- **Volumetric heat source** Q₀ throughout the domain

**Thermal PDE (heat equation):**

$$\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + Q_0$$

**Mechanical PDE (linear thermoelasticity):**

$$\nabla \cdot \sigma = 0, \quad \sigma = \lambda \text{tr}(\varepsilon)\mathbf{I} + 2\mu\varepsilon - (3\lambda+2\mu)\alpha(T-T_{ref})\mathbf{I}$$

One-way coupling — temperature drives thermal expansion, mechanics does not feed back into thermal.

**Material properties (Aluminium):**

| Property | Value |
|---|---|
| Density ρ | 2700 kg/m³ |
| Specific heat cp | 900 J/kgK |
| Young's modulus E | 70 GPa |
| Poisson's ratio ν | 0.33 |
| Thermal expansion α | 23×10⁻⁶ /K |

---

## Pipeline

### 1. FOM Parametric Sweep (`simulation/`)

FEniCSx solves the coupled PDE system for 60 parameter combinations sampled via Latin Hypercube Sampling:

- k ∈ [50, 200] W/mK
- h ∈ [5, 80] W/m²K  
- Q₀ ∈ [10⁴, 5×10⁵] W/m³

Produces snapshot matrices:
```
T_snapshots_train.npy  (205, 1000)  — 205 nodes × 50 runs × 20 steps
u_snapshots_train.npy  (410, 1000)  — 410 DOFs  × 50 runs × 20 steps
```

### 2. POD-ROM (`data_driven/pod_rom.py`)

Singular Value Decomposition compresses the snapshot database:

```
T field: 205 DOFs → 3 modes  (333× compression, 99.999% energy)
u field: 410 DOFs → 3 modes  (333× compression, 99.999% energy)
```

Operator Inference identifies the reduced thermal operator from snapshots near k_nominal=150:

$$\frac{dq_T}{dt} = A_r q_T + b_r$$

Coupling operator K_cu maps reduced thermal to mechanical coordinates:

$$q_u \approx K_{cu} \cdot q_T$$

### 3. Kalman Filter (`data_driven/kalman_filter.py`)

Real-time field reconstruction from 8 sparse sensors (5 temperature + 3 strain):

**Linear Kalman Filter (LKF)** — reconstructs full field assuming known parameters. Achieves 16.5% temperature error (limited by ROM model bias at k≠k_nominal).

**Ensemble Kalman Filter (EnKF)** — augments state with log(k) for simultaneous field reconstruction and online parameter identification. Achieves **4.3% temperature error** by correcting ROM bias with sensor data.

### 4. GP Emulator (`data_driven/gp_emulator.py`)

Gaussian Process emulator trained directly on 50 FOM training snapshots:

- Input: (log k, log h, log Q₀) — 3 parameters
- Output: sensor readings flattened — 160 values (8 sensors × 20 steps)
- Kernel: Matérn-5/2
- Validation error: 0.1%

GP predictions at inference time take microseconds vs 7 seconds for a FOM run.

### 5. Bayesian Inversion (`data_driven/bayesian_inference.py`)

Sequential Monte Carlo (SMC) samples the posterior:

$$p(k, h, Q_0 \mid y_{obs}) \propto p(y_{obs} \mid k, h, Q_0) \cdot p(k, h, Q_0)$$

- **Prior**: log-Normal centered at geometric midpoints of parameter ranges
- **Likelihood**: Normal with Kennedy & O'Hagan model discrepancy inflation
- **Sampler**: PyMC SMC, 500 particles, chains=1

**Why k is identified but not h and Q₀:**
Temperature sensors strongly constrain k (controls profile shape and diffusion speed). h and Q₀ both affect temperature level but are non-identifiable from temperature sensors alone — they lie on a degeneracy ridge. Breaking this requires additional sensor types (calorimetry for Q₀, heat flux probes for h).

### 6. Field Reconstruction UQ (`data_driven/field_reconstruction.py`)

For each posterior sample (k, h, Q₀):
- GP predicts sensor readings
- Inverse-distance weighted interpolation over 50 FOM training snapshots
- Reconstructs full field with parameter-correct physics

Produces pointwise mean, std, and 95% credible intervals.

### 7. Dashboard (`data_driven/dashboard.py`)

Interactive Streamlit dashboard with:
- Real-time field heatmaps using actual node coordinates
- LKF/EnKF error curves and k-convergence
- POD singular value decay
- Prior vs posterior marginals
- Interactive corner plot (joint posteriors)
- Credible interval field maps

---

## Project Structure

```
thermo-mech-digital-twin/
├── simulation/
│   ├── Dockerfile
│   ├── config.py              # shared geometry and physics parameters
│   ├── fom_solver.py          # FEniCSx thermal + mechanical solver
│   └── run_parametric.py      # LHS sweep, snapshot collection
│
├── data_driven/
│   ├── Dockerfile
│   ├── config.py              # shared parameters (identical to simulation/)
│   ├── pod_rom.py             # POD basis + Operator Inference
│   ├── kalman_filter.py       # LKF + EnKF implementation
│   ├── observations.py        # sensor placement and noise model
│   ├── gp_emulator.py         # GP surrogate on FOM snapshots
│   ├── jax_surrogate.py       # JAX ROM rollout (used by Kalman)
│   ├── bayesian_inference.py  # PyMC SMC posterior sampling
│   ├── field_reconstruction.py# UQ field ensemble from posterior
│   ├── posterior_analysis.py  # ArviZ diagnostics and figures
│   ├── train_rom.py           # orchestrates ROM + Kalman pipeline
│   ├── sanity_check.py        # pre-SMC validation checks
│   └── dashboard.py           # Streamlit interactive dashboard
│
└── docker-compose.yml
```

---

## Getting Started

### Prerequisites

- Docker Desktop
- Docker Compose

### Run the full pipeline

```bash
# 1. Clone the repository
git clone git@github.com:hilary-anish/thermo-mech-digital-twin.git
cd thermo-mech-digital-twin

# 2. Build images
docker compose build

# 3. Run FOM parametric sweep (~15 seconds)
docker compose run --rm simulation

# 4. Run sanity checks before inference
docker compose --profile sanity run --rm sanity

# 5. Run full data-driven pipeline + launch dashboard
docker compose up data_driven
```

### Run only Bayesian inference (ROM already built)

```bash
docker compose --profile bayesian run --rm bayesian
```

### Run only ROM + Kalman (no Bayesian)

```bash
docker compose --profile train run --rm train
```

### Access the dashboard

Open `http://localhost:8501` in your browser.

---

## Configuration

All physics, geometry, and pipeline parameters are in `config.py` (shared between both containers):

```python
L  = 0.1       # fin length [m]
H  = 0.1       # fin height [m]
T_END = 100.0  # simulation end time [s]
DT    = 5.0    # time step [s]

K_RANGE  = (50.0,  200.0)   # thermal conductivity [W/mK]
H_RANGE  = (5.0,   80.0)    # convection coefficient [W/m²K]
Q0_RANGE = (1e4,   5e5)     # heat source amplitude [W/m³]

N_T_SENSORS = 5   # temperature sensors
N_U_SENSORS = 3   # strain sensors
```

---

## Tech Stack

| Component | Library |
|---|---|
| FEM solver | FEniCSx v0.8 |
| Reduced order model | NumPy (SVD + least squares) |
| Kalman filter | NumPy |
| GP emulator | scikit-learn |
| Bayesian inference | PyMC + ArviZ |
| JAX rollout | JAX |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |

---

## Author

**Anish Hilary Ignatius**  
MSc Computational Mechanics  

---

## License

MIT
