"""
jax_surrogate.py
================
JAX-differentiable wrapper around the existing POD-ROM.

Loads the pre-trained ROM matrices (plain NumPy .npy files saved by
train_rom.py) and re-implements the Euler roll-out using jax.numpy so
that JAX can autodiff through the entire forward pass.

The function signature uses log-space parameters for positivity:
    log_params = [log_k, log_h, log_Q0]

This is required because k, h, Q0 are strictly positive; sampling them
in unconstrained log-space prevents the sampler from proposing negative
values and stabilises the NUTS gradient computation.

NOTE on parameter scaling
-------------------------
The ROM operators A_r and b_r are identified by OpInf from FOM snapshots
at a single nominal parameter point (k_nominal=150, Q0_nominal=1e5).
OpInf produces a single discrete-time operator that captures the full
physics at that operating point — including boundary conditions, geometry,
and material properties — as a lumped black-box map.

Attempting to rescale A_r by k/k_nominal or b_r by Q0/Q0_nominal is
physically incorrect: the operators are NOT decomposable into separate
diffusion and source contributions post-identification. This was confirmed
experimentally: scaled ROM produces T > 2000K (above T_hot=600K boundary),
while unscaled ROM produces physically correct 300–600K range.

The ROM is therefore valid only near k_nominal. For parameter inference
across a wide range, the GP emulator is trained on 500 ROM evaluations
at the same nominal point and interpolates the sensor response surface.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from pathlib import Path

ROM_DIR = "/workspace/data/rom"

# ── Reference nominal (kept for backwards compatibility / documentation) ───
K_NOMINAL  = 150.0
Q0_NOMINAL = 1e5


# ── Load ROM matrices ──────────────────────────────────────────────────────

def load_rom(rom_dir: str = ROM_DIR):
    """
    Load pre-trained ROM matrices from disk and convert to JAX arrays.

    Returns
    -------
    Phi_T  : jnp.array (N_T, r_T)
    Phi_u  : jnp.array (N_u, r_u)
    A_r    : jnp.array (r_T, r_T)   — identified at k_nominal
    b_r    : jnp.array (r_T,)        — identified at k_nominal
    K_cu   : jnp.array (r_u, r_T)
    """
    p = Path(rom_dir)
    Phi_T = jnp.array(np.load(p / "Phi_T.npy"))
    Phi_u = jnp.array(np.load(p / "Phi_u.npy"))
    A_r   = jnp.array(np.load(p / "A_r.npy"))
    b_r   = jnp.array(np.load(p / "b_r.npy"))
    K_cu  = jnp.array(np.load(p / "K_cu.npy"))
    return Phi_T, Phi_u, A_r, b_r, K_cu


# ── Core JAX ROM prediction ────────────────────────────────────────────────

@partial(jax.jit, static_argnames=["n_steps"])
def jax_predict(log_params, Phi_T, Phi_u, A_r, b_r, K_cu,
                q0, n_steps: int, dt: float,
                k_nominal: float = K_NOMINAL,
                q0_nominal: float = Q0_NOMINAL):
    """
    Roll out the reduced thermal ODE using explicit Euler and return the
    full-field temperature and displacement.

    Parameters
    ----------
    log_params  : jnp.array (3,)  — [log_k, log_h, log_Q0]  (carried for
                  API compatibility; ROM does not use them for scaling)
    Phi_T       : jnp.array (N_T, r_T)
    Phi_u       : jnp.array (N_u, r_u)
    A_r         : jnp.array (r_T, r_T)  — OpInf operator (no rescaling)
    b_r         : jnp.array (r_T,)       — OpInf forcing  (no rescaling)
    K_cu        : jnp.array (r_u, r_T)
    q0          : jnp.array (r_T,)       — initial reduced state
    n_steps     : int  (static — needed for jax.lax.scan)
    dt          : float
    k_nominal, q0_nominal : unused, kept for call-site compatibility

    Returns
    -------
    T_rom : jnp.array (N_T, n_steps)
    u_rom : jnp.array (N_u, n_steps)
    """
    # No parameter-based rescaling — OpInf operators are valid as identified.
    # Explicit Euler transition: q[n+1] = q[n] + dt*(A_r @ q[n] + b_r)
    F = jnp.eye(A_r.shape[0]) + dt * A_r
    g = dt * b_r

    def step(q, _):
        q_next = F @ q + g
        return q_next, q_next

    _, q_T_traj = jax.lax.scan(step, q0, None, length=n_steps)
    # q_T_traj : (n_steps, r_T)  →  transpose to (r_T, n_steps)
    q_T_traj = q_T_traj.T

    T_rom = Phi_T @ q_T_traj          # (N_T, n_steps)
    u_rom = Phi_u @ (K_cu @ q_T_traj) # (N_u, n_steps)
    return T_rom, u_rom


@partial(jax.jit, static_argnames=["n_steps"])
def jax_predict_sensors(log_params, Phi_T, Phi_u, A_r, b_r, K_cu,
                         q0, T_sensor_nodes, u_sensor_nodes,
                         n_steps: int, dt: float):
    """
    Convenience wrapper: runs jax_predict and immediately extracts the
    predicted sensor time series as a flat 1-D vector.

    This is the function used as the forward model inside the PyMC / NumPyro
    likelihood — it maps (3,) log-parameters to (n_obs * n_steps,) predictions.

    Strain sensors observe the x-displacement component (even DOF indices).
    """
    T_rom, u_rom = jax_predict(log_params, Phi_T, Phi_u,
                                A_r, b_r, K_cu,
                                q0, n_steps, dt)

    y_T = T_rom[T_sensor_nodes, :]           # (n_T_sens, n_steps)
    y_u = u_rom[2 * u_sensor_nodes, :]       # (n_u_sens, n_steps)
    y   = jnp.vstack([y_T, y_u])             # (n_obs, n_steps)
    return y.flatten()                        # (n_obs * n_steps,)


# ── Numpy wrapper for GP emulator training ─────────────────────────────────

def rom_predict_scaled(k_val: float, h_val: float, Q0_val: float,
                       Phi_T, Phi_u, A_r, b_r, K_cu,
                       q0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """
    Pure-NumPy ROM prediction for a single (k, h, Q0) triple.

    Used in gp_emulator.py to generate the training dataset for the GP
    emulator (500 ROM evaluations).  Returns the sensor reading vector as
    a plain NumPy array (needed by scikit-learn).

    Parameters k_val, h_val, Q0_val are accepted for API compatibility
    but the ROM operators are NOT rescaled — see module docstring.

    Parameters
    ----------
    k_val, h_val, Q0_val : scalar physical parameter values (API compat)
    Phi_T, Phi_u, A_r, b_r, K_cu : NumPy ROM arrays (not JAX)
    q0          : (r_T,) initial reduced state (NumPy)
    n_steps     : number of time steps
    dt          : time step size

    Returns
    -------
    q_T : (r_T, n_steps+1)  reduced thermal trajectory
    T_r : (N_T, n_steps+1)  full thermal field
    u_r : (N_u, n_steps+1)  full displacement field
    """
    # No rescaling — OpInf operators are valid as identified (see module doc).
    r_T = len(b_r)
    q_T = np.zeros((r_T, n_steps + 1))
    q_T[:, 0] = q0

    for n in range(n_steps):
        q_T[:, n + 1] = q_T[:, n] + dt * (A_r @ q_T[:, n] + b_r)

    T_r = Phi_T @ q_T
    u_r = Phi_u @ (K_cu @ q_T)
    return q_T, T_r, u_r


# ── Smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading ROM from", ROM_DIR)
    Phi_T, Phi_u, A_r, b_r, K_cu = load_rom(ROM_DIR)
    print(f"  Phi_T : {Phi_T.shape}")
    print(f"  Phi_u : {Phi_u.shape}")
    print(f"  A_r   : {A_r.shape}")
    print(f"  K_cu  : {K_cu.shape}")

    from config import N_STEPS, DT
    import numpy as np
    q0_jax = jnp.zeros(A_r.shape[0])
    log_params = jnp.array([np.log(125.0), np.log(30.0), np.log(1e5)])

    T_r, u_r = jax_predict(log_params, Phi_T, Phi_u, A_r, b_r, K_cu,
                             q0_jax, N_STEPS, DT)
    print(f"\njax_predict smoke-test passed")
    print(f"  T_rom shape : {T_r.shape}   max T : {T_r.max():.1f} K")
    print(f"  u_rom shape : {u_r.shape}   max|u|: {jnp.abs(u_r).max()*1e6:.4f} µm")

    # verify NumPy wrapper gives same result
    Phi_T_np = np.array(Phi_T)
    Phi_u_np = np.array(Phi_u)
    A_r_np   = np.array(A_r)
    b_r_np   = np.array(b_r)
    K_cu_np  = np.array(K_cu)
    q0_np    = np.zeros(A_r_np.shape[0])
    _, T_np, _ = rom_predict_scaled(125.0, 30.0, 1e5,
                                     Phi_T_np, Phi_u_np, A_r_np, b_r_np, K_cu_np,
                                     q0_np, N_STEPS, DT)
    diff = np.abs(np.array(T_r) - T_np[:, 1:]).max()
    print(f"\n  NumPy vs JAX max diff : {diff:.2e}  "
          f"({'PASS' if diff < 1e-4 else 'FAIL — check scaling'})")
