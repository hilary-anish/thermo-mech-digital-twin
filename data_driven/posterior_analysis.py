"""
posterior_analysis.py
=====================
Publication-quality posterior diagnostics for the MCMC trace.
Dark scientific theme matching the Streamlit dashboard.
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import arviz as az
from pathlib import Path

from config import SNAPSHOT_DIR, N_STEPS, DT, N_T_SENSORS, N_U_SENSORS
from kalman_filter import place_sensors
from observations import generate_synthetic_observations

POSTERIOR_DIR = "/workspace/data/posterior"
FIGURES_DIR   = f"{POSTERIOR_DIR}/figures"

# ── Dark theme setup ───────────────────────────────────────────────────────
BG     = "#0d1117"
BG2    = "#161b22"
BORDER = "#21262d"
TEXT   = "#adbac7"
TEXT_B = "#e6edf3"
BLUE   = "#388bfd"
AMBER  = "#f0883e"
GREEN  = "#56d364"
RED    = "#ff7b72"
PURPLE = "#d2a8ff"
TEAL   = "#39d353"

def apply_dark(fig, axes_list):
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        if ax is None:
            continue
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT_B)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.7)


def load_trace_and_truth():
    trace_path = Path(POSTERIOR_DIR) / "trace.nc"
    if not trace_path.exists():
        raise FileNotFoundError(f"trace.nc not found at {trace_path}")
    trace = az.from_netcdf(str(trace_path))
    with open(Path(POSTERIOR_DIR) / "posterior_summary.json") as f:
        meta = json.load(f)
    return trace, meta


# ── Figure 1: Trace plot ───────────────────────────────────────────────────

def plot_trace(trace):
    try:
        posterior = trace.posterior
        log_k  = posterior["log_k"].values.flatten()
        log_h  = posterior["log_h"].values.flatten()
        log_Q0 = posterior["log_Q0"].values.flatten()

        fig, axes = plt.subplots(3, 2, figsize=(11, 7),
                                  gridspec_kw={"width_ratios": [3, 1]})
        fig.patch.set_facecolor(BG)

        pairs = [
            (log_k,  r"log k",  BLUE),
            (log_h,  r"log h",  AMBER),
            (log_Q0, r"log Q₀", PURPLE),
        ]

        for i, (samples, label, color) in enumerate(pairs):
            ax_trace = axes[i, 0]
            ax_hist  = axes[i, 1]

            ax_trace.plot(samples, color=color, lw=0.7, alpha=0.8)
            ax_trace.set_ylabel(label, color=TEXT_B, fontsize=9)
            ax_trace.axhline(np.mean(samples), color="white", lw=1,
                              ls="--", alpha=0.5)

            ax_hist.hist(samples, bins=40, orientation="horizontal",
                         color=color, alpha=0.7, edgecolor="none")
            ax_hist.axhline(np.mean(samples), color="white", lw=1,
                             ls="--", alpha=0.5)
            ax_hist.set_xlabel("Count", fontsize=7)

        axes[-1, 0].set_xlabel("Sample index", color=TEXT, fontsize=8)
        apply_dark(fig, axes.flatten())

        fig.suptitle("SMC Trace — log-space parameters",
                     color=TEXT_B, fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out = Path(FIGURES_DIR) / "trace_plot.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight",
                    facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {out}")
        return fig
    except Exception as e:
        print(f"  WARNING: trace plot failed ({e})")
        plt.close("all")
        return None


# ── Figure 2: Marginal posteriors ─────────────────────────────────────────

def plot_marginals(trace, k_true, h_true, Q0_true):
    posterior = trace.posterior
    k_s  = posterior["k"].values.flatten()
    h_s  = posterior["h"].values.flatten()
    Q0_s = posterior["Q0"].values.flatten()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor(BG)

    for ax, samples, true_val, label, unit, color in [
        (axes[0], k_s,  k_true,  "Thermal conductivity",  "W/mK",  BLUE),
        (axes[1], h_s,  h_true,  "Convection coefficient","W/m²K", AMBER),
        (axes[2], Q0_s, Q0_true, "Heat source amplitude", "W/m³",  PURPLE),
    ]:
        counts, bins, patches = ax.hist(samples, bins=40, density=True,
                                         color=color, alpha=0.6,
                                         edgecolor=BG, linewidth=0.3)
        # HDI shading
        hdi = az.hdi(samples, hdi_prob=0.95)
        mask = (bins[:-1] >= hdi[0]) & (bins[:-1] <= hdi[1])
        for patch, m in zip(patches, mask):
            if m:
                patch.set_alpha(0.9)

        ax.axvline(true_val, color=GREEN, lw=2, ls="--",
                   label=f"True = {true_val:.3g}")
        ax.axvline(np.mean(samples), color="white", lw=1.5, ls="-",
                   label=f"Mean = {np.mean(samples):.3g}")

        ax.set_xlabel(f"{label}\n[{unit}]", fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{label}\n95% HDI: [{hdi[0]:.3g}, {hdi[1]:.3g}]",
                     fontsize=9, pad=8)
        ax.legend(fontsize=7.5)

    apply_dark(fig, axes)
    fig.suptitle("Marginal posterior distributions — 95% HDI",
                 color=TEXT_B, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = Path(FIGURES_DIR) / "posterior_marginals.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")
    return fig


# ── Figure 3: Corner plot ──────────────────────────────────────────────────

def plot_corner(trace, k_true, h_true, Q0_true):
    try:
        posterior = trace.posterior
        k_s  = posterior["k"].values.flatten()
        h_s  = posterior["h"].values.flatten()
        Q0_s = posterior["Q0"].values.flatten()

        fig = plt.figure(figsize=(9, 8), facecolor=BG)
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.05, wspace=0.05)

        params = [
            (k_s,  "k [W/mK]",  k_true,  BLUE),
            (h_s,  "h [W/m²K]", h_true,  AMBER),
            (Q0_s, "Q₀ [W/m³]", Q0_true, PURPLE),
        ]

        # Lower triangle: scatter + 2D density
        for row in range(3):
            for col in range(3):
                ax = fig.add_subplot(gs[row, col])
                ax.set_facecolor(BG2)
                for spine in ax.spines.values():
                    spine.set_edgecolor(BORDER)

                if col > row:
                    ax.axis("off")
                    continue

                x_data, x_label, x_true, x_color = params[col]
                y_data, y_label, y_true, y_color = params[row]

                if col == row:
                    # Diagonal: marginal histogram
                    ax.hist(x_data, bins=35, color=x_color, alpha=0.7,
                            density=True, edgecolor="none")
                    ax.axvline(x_true, color=GREEN, lw=1.5, ls="--")
                    ax.axvline(np.mean(x_data), color="white", lw=1, ls="-",
                               alpha=0.7)
                    ax.set_yticks([])
                else:
                    # Off-diagonal: scatter
                    ax.scatter(x_data, y_data, s=2, alpha=0.3,
                               color=BLUE, rasterized=True)
                    ax.scatter([x_true], [y_true], s=80, color=GREEN,
                               marker="X", zorder=5, linewidths=1.5,
                               edgecolors=BG)

                ax.tick_params(colors=TEXT, labelsize=7)
                ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.5)

                # Labels on edges only
                if row == 2:
                    ax.set_xlabel(x_label, color=TEXT, fontsize=8)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row > 0:
                    ax.set_ylabel(y_label, color=TEXT, fontsize=8)
                else:
                    ax.set_yticklabels([])

        fig.suptitle("Joint posterior distributions — parameter correlations",
                     color=TEXT_B, fontsize=12, fontweight="bold", y=0.98)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="X", color="w", markerfacecolor=GREEN,
                   markersize=8, label="True value"),
            Line2D([0], [0], color="white", lw=1, ls="-", label="Posterior mean"),
        ]
        fig.legend(handles=legend_elements, loc="upper right",
                   fontsize=8, facecolor=BG2, edgecolor=BORDER,
                   labelcolor=TEXT_B)

        out = Path(FIGURES_DIR) / "corner_plot.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {out}")
        return fig
    except Exception as e:
        print(f"  WARNING: corner plot failed ({e})")
        plt.close("all")
        return None


# ── Figure 4: Posterior predictive check ──────────────────────────────────

def plot_posterior_predictive(trace, y_obs_2d, t_axis, n_pp=150):
    try:
        from gp_emulator import load_gp_emulator, predict_gp
        gp, sx, sy = load_gp_emulator(POSTERIOR_DIR)
    except Exception as e:
        print(f"  WARNING: posterior predictive skipped ({e})")
        return None

    try:
        posterior = trace.posterior
        lk  = posterior["log_k"].values.flatten()
        lh  = posterior["log_h"].values.flatten()
        lQ0 = posterior["log_Q0"].values.flatten()
        idx = np.random.default_rng(0).choice(len(lk),
                                               min(n_pp, len(lk)),
                                               replace=False)

        n_obs = N_T_SENSORS + N_U_SENSORS
        n_t   = N_STEPS
        pp    = np.zeros((len(idx), n_obs, n_t))
        for j, i in enumerate(idx):
            y, _ = predict_gp(gp, sx, sy,
                               np.array([lk[i], lh[i], lQ0[i]]))
            pp[j] = y.reshape(n_obs, n_t)

        fig, axes = plt.subplots(n_obs, 1, figsize=(10, 1.9 * n_obs),
                                  sharex=True)
        fig.patch.set_facecolor(BG)
        if n_obs == 1:
            axes = [axes]

        labels = ([f"T sensor {i+1}" for i in range(N_T_SENSORS)] +
                  [f"Strain sensor {i+1}" for i in range(N_U_SENSORS)])
        colors = [BLUE] * N_T_SENSORS + [AMBER] * N_U_SENSORS

        t_plot = np.arange(1, n_t + 1) * DT

        for s_idx, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            pp_mean = pp[:, s_idx, :].mean(axis=0)
            pp_lo   = np.percentile(pp[:, s_idx, :], 2.5,  axis=0)
            pp_hi   = np.percentile(pp[:, s_idx, :], 97.5, axis=0)

            ax.fill_between(t_plot, pp_lo, pp_hi,
                            alpha=0.25, color=color)
            ax.plot(t_plot, pp_mean, color=color, lw=1.5,
                    label="PP mean")
            ax.plot(t_plot, y_obs_2d[s_idx, :], "o", ms=4,
                    color=RED, label="Observed", zorder=4)
            ax.set_ylabel(label, color=TEXT, fontsize=8)
            ax.set_facecolor(BG2)
            ax.tick_params(colors=TEXT, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.5)
            if s_idx == 0:
                ax.legend(fontsize=7.5, facecolor=BG2,
                          edgecolor=BORDER, labelcolor=TEXT_B)

        axes[-1].set_xlabel("t [s]", color=TEXT, fontsize=9)
        fig.suptitle("Posterior predictive check — sensor time series",
                     color=TEXT_B, fontsize=12, fontweight="bold", y=1.005)
        plt.tight_layout()

        out = Path(FIGURES_DIR) / "posterior_predictive.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {out}")
        return fig
    except Exception as e:
        print(f"  WARNING: posterior predictive failed ({e})")
        plt.close("all")
        return None


# ── Convergence diagnostics ────────────────────────────────────────────────

def print_diagnostics(trace):
    print("\n" + "=" * 60)
    print("Convergence diagnostics")
    print("=" * 60)
    summary = az.summary(trace,
                         var_names=["k", "h", "Q0", "log_k", "log_h", "log_Q0"])
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]])
    r_hat_max = summary["r_hat"].max()
    ess_min   = summary["ess_bulk"].min()
    print(f"\nr_hat max : {r_hat_max:.4f}  "
          f"({'CONVERGED' if r_hat_max < 1.05 else 'NOT CONVERGED'})")
    print(f"ESS min   : {ess_min:.0f}  "
          f"({'OK' if ess_min > 200 else 'LOW — run more draws'})")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────

def analyse_posterior():
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading trace …")
    trace, meta = load_trace_and_truth()
    k_true  = meta["k_true"]
    h_true  = meta["h_true"]
    Q0_true = meta["Q0_true"]

    print_diagnostics(trace)

    print("\nGenerating figures …")
    plot_trace(trace)
    plot_marginals(trace, k_true, h_true, Q0_true)
    plot_corner(trace, k_true, h_true, Q0_true)

    import json as _json
    with open(f"{SNAPSHOT_DIR}/metadata.json") as f:
        snap_meta = _json.load(f)
    N_T = snap_meta["N_T_dofs"]
    N_u = snap_meta["N_u_dofs"]
    T_true = np.load(f"{SNAPSHOT_DIR}/T_snapshots_test.npy")[:, :N_STEPS]
    u_true = np.load(f"{SNAPSHOT_DIR}/u_snapshots_test.npy")[:, :N_STEPS]
    T_nodes, u_nodes = place_sensors(N_T, N_u, seed=0)
    y_obs_2d = generate_synthetic_observations(T_true, u_true,
                                                T_nodes, u_nodes, seed=42)
    t_axis = np.arange(N_STEPS) * DT
    plot_posterior_predictive(trace, y_obs_2d, t_axis)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    return trace


if __name__ == "__main__":
    analyse_posterior()
