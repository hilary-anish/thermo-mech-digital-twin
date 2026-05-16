"""
fom_solver.py
=============
Full-Order Model: coupled heat conduction + linear thermoelasticity
in a 2D fin geometry using FEniCSx.
"""

import numpy as np
from pathlib import Path

try:
    from mpi4py import MPI
    from dolfinx import default_scalar_type
    from dolfinx.fem import (
        Constant, Function, functionspace,
        dirichletbc, locate_dofs_topological
    )
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.mesh import (
        create_rectangle, CellType,
        locate_entities_boundary
    )
    import ufl
    from ufl import (
        TestFunction, TrialFunction, dx, ds,
        grad, inner, sym, Identity, tr
    )
    FENICS_OK = True
except ImportError as e:
    FENICS_OK = False
    print(f"FEniCSx not available: {e}")

from config import *


def build_mesh():
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [L, H]],
        [NX, NY],
        CellType.quadrilateral
    )
    fdim = domain.topology.dim - 1
    tol  = 1e-12
    boundaries = {
        "left":   locate_entities_boundary(
                      domain, fdim, lambda x: np.isclose(x[0], 0.0, atol=tol)),
        "right":  locate_entities_boundary(
                      domain, fdim, lambda x: np.isclose(x[0], L,   atol=tol)),
        "bottom": locate_entities_boundary(
                      domain, fdim, lambda x: np.isclose(x[1], 0.0, atol=tol)),
        "top":    locate_entities_boundary(
                      domain, fdim, lambda x: np.isclose(x[1], H,   atol=tol)),
    }
    return domain, boundaries


def run_simulation(k_val, h_val, Q0_val, verbose=False):
    domain, boundaries = build_mesh()
    fdim = domain.topology.dim - 1

    # ── Function spaces ────────────────────────────────────────────────────
    V_T = functionspace(domain, ("CG", 1))
    V_u = functionspace(domain, ("CG", 1, (2,)))

    # ── Persistent Functions — created once, updated in-place each step ────
    T_h = Function(V_T, name="T")
    T_n = Function(V_T, name="T_prev")
    T_n.x.array[:] = T_REF
    T_h.x.array[:] = T_REF

    # ── Constants ──────────────────────────────────────────────────────────
    k_c      = Constant(domain, default_scalar_type(k_val))
    rho_cp_c = Constant(domain, default_scalar_type(RHO * CP))
    dt_c     = Constant(domain, default_scalar_type(DT))
    Q0_c     = Constant(domain, default_scalar_type(Q0_val))
    h_c      = Constant(domain, default_scalar_type(h_val))
    T_inf_c  = Constant(domain, default_scalar_type(T_INF))

    lam = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
    mu  = E_MOD / (2 * (1 + NU))
    lam_c   = Constant(domain, default_scalar_type(lam))
    mu_c    = Constant(domain, default_scalar_type(mu))
    alpha_c = Constant(domain, default_scalar_type(ALPHA))
    T_ref_c = Constant(domain, default_scalar_type(T_REF))

    # ── Trial and test functions ───────────────────────────────────────────
    T_trial = TrialFunction(V_T)
    v_T     = TestFunction(V_T)
    u_trial = TrialFunction(V_u)
    v_u     = TestFunction(V_u)

    # ── Thermal forms — assembled fresh each step inside loop ──────────────
    a_T = (
        (rho_cp_c / dt_c) * inner(T_trial, v_T) * dx
        + k_c * inner(grad(T_trial), grad(v_T)) * dx
        + h_c * inner(T_trial, v_T) * ds
    )
    L_T = (
        (rho_cp_c / dt_c) * inner(T_n, v_T) * dx
        + Q0_c * v_T * dx
        + h_c * T_inf_c * v_T * ds
    )

    left_dofs = locate_dofs_topological(V_T, fdim, boundaries["left"])
    bc_T = dirichletbc(
        Constant(domain, default_scalar_type(T_HOT)), left_dofs, V_T
    )

    # ── Mechanical forms — T_h referenced as persistent Function ──────────
    # a_u has no dependence on T_h so the stiffness matrix is constant.
    # L_u references T_h directly — since T_h.x.array is updated in-place
    # before each mechanical solve, the assembled RHS always uses the
    # latest temperature field. This is the correct DOLFINx pattern.
    def eps(v):
        return sym(grad(v))

    def sigma_elastic(v):
        return lam_c * tr(eps(v)) * Identity(2) + 2 * mu_c * eps(v)

    # Plane strain thermal stress: (3*lam + 2*mu) * alpha * (T - T_ref)
    thermal_prestress = (3 * lam_c + 2 * mu_c) * alpha_c * (T_h - T_ref_c)

    a_u = inner(sigma_elastic(u_trial), eps(v_u)) * dx
    L_u = inner(thermal_prestress * Identity(2), eps(v_u)) * dx

    bottom_dofs = locate_dofs_topological(V_u, fdim, boundaries["bottom"])
    bc_u = dirichletbc(
        np.zeros(2, dtype=default_scalar_type), bottom_dofs, V_u
    )

    # Mechanical solver created once — stiffness matrix does not change
    petsc_opts = {"ksp_type": "preonly", "pc_type": "lu"}
    m_prob = LinearProblem(a_u, L_u, bcs=[bc_u], petsc_options=petsc_opts)

    # ── Snapshot storage ───────────────────────────────────────────────────
    N_T = V_T.dofmap.index_map.size_global
    N_u = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs

    T_snaps = np.zeros((N_T, N_STEPS))
    u_snaps = np.zeros((N_u, N_STEPS))

    # ── Time loop ──────────────────────────────────────────────────────────
    for step in range(N_STEPS):

        # Step 1: thermal solve — new LinearProblem each step
        # because L_T depends on T_n which changes every step
        t_prob = LinearProblem(a_T, L_T, bcs=[bc_T], petsc_options=petsc_opts)
        T_sol  = t_prob.solve()

        # Update T_h and T_n in-place
        T_h.x.array[:] = T_sol.x.array[:]
        T_h.x.scatter_forward()
        T_n.x.array[:] = T_sol.x.array[:]
        T_n.x.scatter_forward()

        # Step 2: mechanical solve — m_prob reuses stiffness matrix,
        # but reassembles RHS using updated T_h automatically
        u_sol = m_prob.solve()

        T_snaps[:, step] = T_h.x.array.copy()
        u_snaps[:, step] = u_sol.x.array.copy()

        if verbose:
            print(f"  step {step+1}/{N_STEPS} | "
                  f"T_max={T_h.x.array.max():.1f}K | "
                  f"|u|_max={np.abs(u_sol.x.array).max()*1e6:.3f}µm")

    return T_snaps, u_snaps, N_T, N_u


def get_dof_coordinates():
    domain, _ = build_mesh()
    V_T = functionspace(domain, ("CG", 1))
    coords = V_T.tabulate_dof_coordinates()[:, :2]
    return coords


def plot_validation(T_snaps, u_snaps, k_val, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = get_dof_coordinates()
    x_c, y_c = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

    sc0 = axes[0].scatter(x_c, y_c, c=T_snaps[:, -1],
                          cmap="hot", s=8, vmin=T_INF, vmax=T_HOT)
    plt.colorbar(sc0, ax=axes[0], label="T [K]")
    axes[0].set_title(f"Temperature at t={T_END}s  (k={k_val:.0f})")
    axes[0].set_aspect("equal")

    ux = u_snaps[0::2, -1]
    uy = u_snaps[1::2, -1]
    u_mag = np.sqrt(ux**2 + uy**2) * 1e6
    sc1 = axes[1].scatter(x_c, y_c, c=u_mag, cmap="viridis", s=8)
    plt.colorbar(sc1, ax=axes[1], label="|u| [µm]")
    axes[1].set_title(f"|u| at t={T_END}s")
    axes[1].set_aspect("equal")

    n_x = NX + 1
    sensor_x_idx = [n_x // 4, n_x // 2, 3 * n_x // 4]
    t_axis = np.arange(1, N_STEPS + 1) * DT
    for idx, col in zip(sensor_x_idx, ["tab:blue", "tab:orange", "tab:green"]):
        axes[2].plot(t_axis, T_snaps[idx, :], color=col,
                     label=f"x={x_c[idx]:.2f}m")
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("T [K]")
    axes[2].set_title("Temperature history")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / f"validation_k{k_val:.0f}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")