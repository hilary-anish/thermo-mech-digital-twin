"""
dashboard.py
============
Streamlit dashboard — Thermo-Mechanical Digital Twin
Anish Hilary Ignatius
"""

import numpy as np
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from config import K_RANGE, H_RANGE, Q0_RANGE, N_STEPS, DT

# ── Paths ──────────────────────────────────────────────────────────────────
SNAP_DIR      = "/workspace/data/snapshots"
ROM_DIR       = "/workspace/data/rom"
POSTERIOR_DIR = "/workspace/data/posterior"
FIGURES_DIR   = f"{POSTERIOR_DIR}/figures"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermo-Mechanical Digital Twin",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — dark scientific theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

h1 { 
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem !important;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
}

h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #79c0ff !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.3px;
}

.stMetric {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}

.stMetric label { color: #8b949e !important; font-size: 0.8rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #f0f6fc !important; font-family: 'IBM Plex Mono', monospace; }
.stMetric [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

.explanation-box {
    background: #161b22;
    border-left: 3px solid #388bfd;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    margin: 8px 0 20px 0;
    font-size: 0.88rem;
    color: #8b949e;
    line-height: 1.6;
}

.insight-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.875rem;
    color: #adbac7;
    line-height: 1.7;
}

.insight-box strong { color: #e6edf3; }
.insight-box .highlight { color: #56d364; font-family: 'IBM Plex Mono', monospace; }
.insight-box .warn { color: #f78166; font-family: 'IBM Plex Mono', monospace; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #388bfd;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 8px 0;
}

div[data-testid="stSidebarContent"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

.stSlider > div > div { background: #21262d; }
.stRadio > div { gap: 6px; }

hr { border-color: #21262d !important; }

.plotly-chart { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────

@st.cache_resource
def load_all():
    data = {}
    try:
        with open(f"{SNAP_DIR}/metadata.json") as f:
            data["meta"] = json.load(f)
        with open(f"{ROM_DIR}/rom_metadata.json") as f:
            data["rom_meta"] = json.load(f)
        with open(f"{ROM_DIR}/results_meta.json") as f:
            data["res_meta"] = json.load(f)
        for key, path in [
            ("coords",     f"{SNAP_DIR}/dof_coords.npy"),
            ("T_true",     f"{ROM_DIR}/T_true.npy"),
            ("u_true",     f"{ROM_DIR}/u_true.npy"),
            ("T_lkf",      f"{ROM_DIR}/T_lkf.npy"),
            ("u_lkf",      f"{ROM_DIR}/u_lkf.npy"),
            ("T_enkf",     f"{ROM_DIR}/T_enkf.npy"),
            ("u_enkf",     f"{ROM_DIR}/u_enkf.npy"),
            ("err_T_lkf",  f"{ROM_DIR}/err_T_lkf.npy"),
            ("err_u_lkf",  f"{ROM_DIR}/err_u_lkf.npy"),
            ("err_T_enkf", f"{ROM_DIR}/err_T_enkf.npy"),
            ("k_hist",     f"{ROM_DIR}/k_hist.npy"),
            ("t_axis",     f"{ROM_DIR}/t_axis.npy"),
            ("T_sensor",   f"{ROM_DIR}/T_sensor_nodes.npy"),
            ("sigma_T",    f"{ROM_DIR}/sigma_T.npy"),
            ("sigma_u",    f"{ROM_DIR}/sigma_u.npy"),
        ]:
            data[key] = np.load(path)
        data["loaded"] = True
    except FileNotFoundError as e:
        data["loaded"] = False
        data["error"]  = str(e)
    return data


@st.cache_resource
def load_bayesian():
    bay = {"loaded": False}
    try:
        with open(f"{POSTERIOR_DIR}/posterior_summary.json") as f:
            bay["summary"] = json.load(f)
        import arviz as az
        bay["trace"] = az.from_netcdf(f"{POSTERIOR_DIR}/trace.nc")
        for key, path in [
            ("T_mean",  f"{POSTERIOR_DIR}/T_posterior_mean.npy"),
            ("T_std",   f"{POSTERIOR_DIR}/T_posterior_std.npy"),
            ("T_lower", f"{POSTERIOR_DIR}/T_posterior_lower.npy"),
            ("T_upper", f"{POSTERIOR_DIR}/T_posterior_upper.npy"),
            ("u_mean",  f"{POSTERIOR_DIR}/u_posterior_mean.npy"),
            ("u_std",   f"{POSTERIOR_DIR}/u_posterior_std.npy"),
        ]:
            bay[key] = np.load(path)
        bay["loaded"] = True
    except Exception:
        pass
    return bay


D   = load_all()
BAY = load_bayesian()

# ── Plotly dark template ───────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#adbac7", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
)


def dark_layout(**kwargs):
    d = dict(**DARK)
    d.update(kwargs)
    return d


# ── Scatter-based field plot using actual node coordinates ─────────────────

def make_field_scatter(coords, field_vals, title, colorscale="Hot",
                       sensor_nodes=None, zmin=None, zmax=None,
                       colorbar_title="T [K]"):
    """Plot field using actual node x,y coordinates — no grid reshaping."""
    if zmin is None:
        zmin = field_vals.min()
    if zmax is None:
        zmax = field_vals.max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(
            color=field_vals,
            colorscale=colorscale,
            size=8,
            cmin=zmin, cmax=zmax,
            colorbar=dict(title=colorbar_title, thickness=12,
                          tickfont=dict(size=10)),
            line=dict(width=0),
        ),
        hovertemplate="x=%{x:.3f}m<br>y=%{y:.3f}m<br>val=%{marker.color:.2f}<extra></extra>",
        showlegend=False,
    ))
    if sensor_nodes is not None:
        fig.add_trace(go.Scatter(
            x=coords[sensor_nodes, 0],
            y=coords[sensor_nodes, 1],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#56d364",
                        line=dict(width=1.5, color="#0d1117")),
            name="Sensors",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#8b949e")),
        height=220,
        margin=dict(t=35, b=10, l=10, r=10),
        xaxis=dict(title="x [m]", gridcolor="#21262d", linecolor="#30363d",
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y [m]", gridcolor="#21262d", linecolor="#30363d"),
        **{k: v for k, v in DARK.items() if k not in ["xaxis", "yaxis"]},
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 Digital Twin Controls")
    st.markdown("<hr>", unsafe_allow_html=True)

    if D["loaded"]:
        meta     = D["meta"]
        rom_meta = D["rom_meta"]
        res_meta = D["res_meta"]

        st.markdown('<p class="section-header">Time control</p>', unsafe_allow_html=True)
        t_axis   = D["t_axis"]
        n_steps  = len(t_axis)
        step_idx = st.slider("Time step", 0, n_steps - 1, n_steps - 1,
                             format="step %d")
        st.caption(f"**t = {t_axis[step_idx]:.1f} s**")

        st.markdown('<p class="section-header">Assimilation</p>', unsafe_allow_html=True)
        filter_choice = st.radio(
            "Method",
            ["Linear Kalman Filter (LKF)", "Ensemble Kalman Filter (EnKF)"],
            index=1
        )

        st.markdown('<p class="section-header">ROM summary</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Thermal modes", rom_meta["r_T"])
        c2.metric("Mech. modes",   rom_meta["r_u"])
        c1.metric("T sensors", res_meta["n_T_sensors"])
        c2.metric("u sensors", res_meta["n_u_sensors"])

        st.markdown('<p class="section-header">Key results</p>', unsafe_allow_html=True)
        st.metric("Max LKF T error",   f"{res_meta['max_err_T_lkf_pct']:.2f}%")
        st.metric("True k",            f"{res_meta['k_true']:.1f} W/mK")
        st.metric("EnKF k (final)",    f"{res_meta['k_final_estimate']:.2f} W/mK",
                  delta=f"{abs(res_meta['k_final_estimate']-res_meta['k_true'])/res_meta['k_true']*100:.1f}% err")

        if BAY["loaded"]:
            st.markdown('<p class="section-header">Bayesian inference</p>', unsafe_allow_html=True)
            s = BAY["summary"]
            st.metric("Posterior k", f"{s['k_mean']:.1f} ± {s['k_sd']:.1f} W/mK")
            err_k = abs(s['k_mean'] - s['k_true']) / s['k_true'] * 100
            st.metric("k identification error", f"{err_k:.1f}%")
    else:
        st.error("Data not loaded.")
        st.code(D.get("error", "Unknown error"))


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("# Thermo-Mechanical Digital Twin")
st.markdown(
    "<small style='color:#8b949e'>POD-ROM · Ensemble Kalman Filter · "
    "Bayesian Inversion · FEniCSx · PyMC | "
    "Anish Hilary Ignatius</small>",
    unsafe_allow_html=True
)
st.markdown("---")

if not D["loaded"]:
    st.error("Snapshot/ROM data not found. Run `python3 train_rom.py` first.")
    st.stop()

meta   = D["meta"]
NX     = meta["geometry"]["NX"]
NY     = meta["geometry"]["NY"]
L_geo  = meta["geometry"]["L"]
H_geo  = meta["geometry"]["H"]
coords = D["coords"]

if "LKF" in filter_choice:
    T_est = D["T_lkf"];  u_est = D["u_lkf"]
    err_T = D["err_T_lkf"]; err_u = D["err_u_lkf"]; tag = "LKF"
else:
    T_est = D["T_enkf"]; u_est = D["u_enkf"]
    err_T = D["err_T_enkf"]; err_u = D["err_u_lkf"]; tag = "EnKF"

T_nodes = D["T_sensor"].astype(int)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Field reconstruction
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">§ 1 — Real-time Field Reconstruction</p>',
            unsafe_allow_html=True)
st.markdown(f"**{tag} reconstruction at t = {t_axis[step_idx]:.1f} s**")

col1, col2 = st.columns(2)

with col1:
    T_true_vals = D["T_true"][:, step_idx]
    T_est_vals  = T_est[:, step_idx]
    zmin_T = min(T_true_vals.min(), T_est_vals.min())
    zmax_T = max(T_true_vals.max(), T_est_vals.max())

    fig_T1 = make_field_scatter(coords, T_true_vals, "FOM ground truth",
                                 colorscale="Hot", sensor_nodes=T_nodes,
                                 zmin=zmin_T, zmax=zmax_T, colorbar_title="T [K]")
    fig_T2 = make_field_scatter(coords, T_est_vals,
                                 f"ROM + {tag} reconstruction",
                                 colorscale="Hot", sensor_nodes=T_nodes,
                                 zmin=zmin_T, zmax=zmax_T, colorbar_title="T [K]")
    st.markdown(f"**Temperature field** — range {zmin_T:.0f}–{zmax_T:.0f} K")
    st.plotly_chart(fig_T1, use_container_width=True)
    st.plotly_chart(fig_T2, use_container_width=True)

with col2:
    n_nodes = coords.shape[0]
    ux_true = D["u_true"][0::2][:n_nodes, step_idx]
    uy_true = D["u_true"][1::2][:n_nodes, step_idx]
    umag_true = np.sqrt(ux_true**2 + uy_true**2) * 1e6

    ux_est = u_est[0::2][:n_nodes, step_idx]
    uy_est = u_est[1::2][:n_nodes, step_idx]
    umag_est = np.sqrt(ux_est**2 + uy_est**2) * 1e6

    zmin_u = 0
    zmax_u = max(umag_true.max(), 1e-12)

    fig_u1 = make_field_scatter(coords, umag_true, "FOM ground truth",
                                 colorscale="Viridis",
                                 zmin=zmin_u, zmax=zmax_u, colorbar_title="|u| [µm]")
    fig_u2 = make_field_scatter(coords, umag_est,
                                 f"ROM + {tag} reconstruction",
                                 colorscale="Viridis",
                                 zmin=zmin_u, zmax=zmax_u, colorbar_title="|u| [µm]")
    st.markdown(f"**Displacement magnitude** — range {zmin_u:.0f}–{zmax_u:.0f} µm")
    st.plotly_chart(fig_u1, use_container_width=True)
    st.plotly_chart(fig_u2, use_container_width=True)

st.markdown("""<div class="explanation-box">
The temperature and displacement fields are reconstructed everywhere in the domain
from only <strong>8 sparse sensors</strong> (5 temperature + 3 strain) using a
POD-reduced model and a Kalman filter. The FOM (full-order FEniCSx simulation) serves
as ground truth. Green triangles mark sensor locations in the gradient zone (0 &lt; x &lt; 0.1 m)
where thermal gradients carry the most information about material properties.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Reconstruction errors + k identification
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<p class="section-header">§ 2 — Error Metrics & Online Parameter Identification</p>',
            unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    fig_eT = go.Figure()
    fig_eT.add_trace(go.Scatter(
        x=t_axis, y=err_T * 100, mode="lines",
        line=dict(color="#ff7b72", width=2),
        fill="tozeroy", fillcolor="rgba(255,123,114,0.08)"
    ))
    fig_eT.add_vline(x=t_axis[step_idx], line_dash="dash",
                     line_color="#388bfd", line_width=1)
    fig_eT.update_layout(
        title="Temperature reconstruction error",
        xaxis_title="t [s]", yaxis_title="Rel. L2 error [%]",
        yaxis_type="log", height=260,
        margin=dict(t=40, b=30, l=50, r=10),
        **DARK
    )
    st.plotly_chart(fig_eT, use_container_width=True)
    st.metric("Error at selected step", f"{err_T[step_idx]*100:.3f}%")

with col4:
    fig_eu = go.Figure()
    fig_eu.add_trace(go.Scatter(
        x=t_axis, y=err_u * 100, mode="lines",
        line=dict(color="#79c0ff", width=2),
        fill="tozeroy", fillcolor="rgba(121,192,255,0.08)"
    ))
    fig_eu.add_vline(x=t_axis[step_idx], line_dash="dash",
                     line_color="#388bfd", line_width=1)
    fig_eu.update_layout(
        title="Displacement reconstruction error",
        xaxis_title="t [s]", yaxis_title="Rel. L2 error [%]",
        yaxis_type="log", height=260,
        margin=dict(t=40, b=30, l=50, r=10),
        **DARK
    )
    st.plotly_chart(fig_eu, use_container_width=True)
    st.metric("Error at selected step", f"{err_u[step_idx]*100:.3f}%")

with col5:
    k_hist    = D["k_hist"]
    k_true_v  = D["res_meta"]["k_true"]
    fig_k = go.Figure()
    fig_k.add_hline(y=k_true_v, line_dash="dash", line_color="#56d364",
                    line_width=1.5,
                    annotation_text=f"True k = {k_true_v:.1f}",
                    annotation_font_color="#56d364",
                    annotation_position="bottom right")
    fig_k.add_trace(go.Scatter(
        x=t_axis, y=k_hist, mode="lines+markers",
        line=dict(color="#d2a8ff", width=2),
        marker=dict(size=4, color="#d2a8ff")
    ))
    fig_k.add_vline(x=t_axis[step_idx], line_dash="dash",
                    line_color="#388bfd", line_width=1)
    fig_k.update_layout(
        title="Online k identification (EnKF)",
        xaxis_title="t [s]", yaxis_title="k [W/mK]",
        height=260, margin=dict(t=40, b=30, l=50, r=10),
        **DARK
    )
    st.plotly_chart(fig_k, use_container_width=True)
    k_err = abs(k_hist[-1] - k_true_v) / k_true_v * 100
    st.metric("k error at final step", f"{k_err:.2f}%")

st.markdown("""<div class="explanation-box">
<strong>Temperature error</strong> reflects how accurately the ROM+Kalman system reconstructs the full thermal field
from sparse observations. The Ensemble Kalman Filter (EnKF) achieves ~4% error by simultaneously
estimating both the state and the unknown conductivity k.
<strong>Online k identification</strong>: the EnKF augments the state vector with log(k) and
updates it at each assimilation step using sensor innovations. Convergence to the true value is
limited by the ROM model bias (ROM built at k=150, true k=194).
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — POD singular value decay
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<p class="section-header">§ 3 — POD Model Compressibility</p>',
            unsafe_allow_html=True)

col6, col7 = st.columns(2)
for col_obj, sigma, name, color in [
    (col6, D["sigma_T"], "Thermal field (T)",    "#ff7b72"),
    (col7, D["sigma_u"], "Mechanical field (u)", "#79c0ff"),
]:
    with col_obj:
        r      = np.arange(1, len(sigma) + 1)
        energy = np.cumsum(sigma**2) / np.sum(sigma**2) * 100
        fig_sv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sv.add_trace(
            go.Scatter(x=r, y=sigma / sigma[0], mode="lines",
                       name="σᵢ/σ₁", line=dict(color=color, width=2)),
            secondary_y=False
        )
        fig_sv.add_trace(
            go.Scatter(x=r, y=energy, mode="lines", name="Cumul. energy [%]",
                       line=dict(color="#8b949e", width=1.5, dash="dash")),
            secondary_y=True
        )
        fig_sv.update_yaxes(title_text="Normalised σᵢ/σ₁",
                            type="log", secondary_y=False,
                            gridcolor="#21262d", linecolor="#30363d")
        fig_sv.update_yaxes(title_text="Cumulative energy [%]",
                            secondary_y=True,
                            gridcolor="#21262d", linecolor="#30363d")
        fig_sv.update_layout(
            title=name, xaxis_title="Mode index r",
            height=280, margin=dict(t=40, b=30, l=50, r=50),
            legend=dict(x=0.55, y=0.4, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
            **DARK
        )
        st.plotly_chart(fig_sv, use_container_width=True)

st.markdown("""<div class="explanation-box">
The rapid singular value decay confirms that both fields are highly compressible:
only <strong>3 POD modes</strong> capture 99.999% of the thermal and mechanical energy
across all 50 training simulations. This enables a 333× compression from 205 DOFs to 3
reduced coordinates, making real-time assimilation computationally feasible.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — BAYESIAN PANELS
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("# 🔬 Bayesian Inverse Problem — Posterior Analysis")
st.markdown(
    "<small style='color:#8b949e'>Sequential Monte Carlo (SMC) · "
    "GP Emulator on FOM snapshots · Kennedy & O'Hagan model discrepancy</small>",
    unsafe_allow_html=True
)

if not BAY["loaded"]:
    st.info("Bayesian results not yet available. Run `python3 bayesian_inference.py` "
            "and `python3 field_reconstruction.py`, then reload.")
else:
    s = BAY["summary"]

    # ── Posterior summary metrics ────────────────────────────────────────
    st.markdown('<p class="section-header">Posterior parameter estimates</p>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    err_k  = abs(s['k_mean']  - s['k_true'])  / s['k_true']  * 100
    err_h  = abs(s['h_mean']  - s['h_true'])  / s['h_true']  * 100
    err_Q0 = abs(s['Q0_mean'] - s['Q0_true']) / s['Q0_true'] * 100
    m1.metric("k [W/mK]",
              f"{s['k_mean']:.1f} ± {s['k_sd']:.1f}",
              delta=f"true {s['k_true']:.1f} | err {err_k:.1f}%")
    m2.metric("h [W/m²K]",
              f"{s['h_mean']:.1f} ± {s['h_sd']:.1f}",
              delta=f"true {s['h_true']:.1f} | err {err_h:.0f}%")
    m3.metric("Q₀ [W/m³]",
              f"{s['Q0_mean']:.2e} ± {s['Q0_sd']:.1e}",
              delta=f"true {s['Q0_true']:.2e} | err {err_Q0:.0f}%")
    m4.metric("SMC convergence",
              "✅ Good" if s.get("converged") else "⚠️ Check",
              delta=f"ESS {s['ess_min']:.0f} | r̂ {s['r_hat_max']:.3f}")

    # ── Panel A — Prior vs Posterior ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Panel A — Prior vs Posterior</p>',
                unsafe_allow_html=True)

    from config import K_RANGE, H_RANGE, Q0_RANGE
    PRIOR = {
        "log_k":  {"mu": (np.log(K_RANGE[1])  + np.log(K_RANGE[0]))  / 2,
                   "sigma": (np.log(K_RANGE[1])  - np.log(K_RANGE[0]))  / 4},
        "log_h":  {"mu": (np.log(H_RANGE[1])  + np.log(H_RANGE[0]))  / 2,
                   "sigma": (np.log(H_RANGE[1])  - np.log(H_RANGE[0]))  / 4},
        "log_Q0": {"mu": (np.log(Q0_RANGE[1]) + np.log(Q0_RANGE[0])) / 2,
                   "sigma": (np.log(Q0_RANGE[1]) - np.log(Q0_RANGE[0])) / 4},
    }

    param_defs = [
        ("k",  "k  [W/mK]",  K_RANGE,  s["k_true"],  s["k_mean"],  s["k_sd"],  PRIOR["log_k"]),
        ("h",  "h  [W/m²K]", H_RANGE,  s["h_true"],  s["h_mean"],  s["h_sd"],  PRIOR["log_h"]),
        ("Q0", "Q₀  [W/m³]", Q0_RANGE, s["Q0_true"], s["Q0_mean"], s["Q0_sd"], PRIOR["log_Q0"]),
    ]

    fig_prior = make_subplots(rows=1, cols=3,
                               subplot_titles=[p[1] for p in param_defs])

    for ci, (var, label, phys_range, true_val, post_mean, post_sd, prior_p) in \
            enumerate(param_defs, 1):
        x_lo = phys_range[0] * 0.3
        x_hi = phys_range[1] * 1.8
        x    = np.linspace(x_lo, x_hi, 400)
        # Prior: LogNormal
        prior_pdf = stats.lognorm.pdf(x, s=prior_p["sigma"],
                                       scale=np.exp(prior_p["mu"]))
        # Posterior: approximate Normal
        post_pdf  = stats.norm.pdf(x, loc=post_mean, scale=max(post_sd, 1e-6))
        prior_pdf /= (prior_pdf.max() + 1e-12)
        post_pdf  /= (post_pdf.max()  + 1e-12)

        fig_prior.add_trace(go.Scatter(
            x=x, y=prior_pdf, mode="lines",
            name="Prior" if ci == 1 else None,
            showlegend=(ci == 1),
            line=dict(color="#f0883e", width=2, dash="dot")
        ), row=1, col=ci)
        fig_prior.add_trace(go.Scatter(
            x=x, y=post_pdf, mode="lines",
            name="Posterior" if ci == 1 else None,
            showlegend=(ci == 1),
            fill="tozeroy", fillcolor="rgba(56,139,253,0.18)",
            line=dict(color="#388bfd", width=2.5)
        ), row=1, col=ci)
        fig_prior.add_vline(
            x=true_val, line_dash="dash", line_color="#56d364", line_width=1.5,
            annotation_text="true", annotation_font_color="#56d364",
            annotation_font_size=10, row=1, col=ci
        )

    fig_prior.update_layout(
        height=320,
        margin=dict(t=50, b=20, l=40, r=20),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10)),
        **DARK
    )
    for i in range(1, 4):
        fig_prior.update_xaxes(gridcolor="#21262d", linecolor="#30363d",
                               row=1, col=i)
        fig_prior.update_yaxes(gridcolor="#21262d", linecolor="#30363d",
                               row=1, col=i)
    st.plotly_chart(fig_prior, use_container_width=True)

    st.markdown("""<div class="explanation-box">
    Each panel compares the <strong>prior</strong> (what we believed before seeing data, orange dashed)
    with the <strong>posterior</strong> (updated belief after observing sensor data, blue filled).
    A narrow posterior relative to the prior indicates the data is highly informative.
    The green dashed line marks the true parameter value used to generate synthetic observations.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    <strong>Why does k recover well but not h and Q₀?</strong><br><br>
    The thermal conductivity <span class="highlight">k</span> governs how fast heat
    diffuses through the fin — it controls the <em>shape</em> of the temperature profile in space
    and the <em>speed</em> of transient evolution. With sensors in the gradient zone (x = 0–0.1 m),
    the data strongly constrains k: <span class="highlight">posterior k = {:.1f} ± {:.1f} W/mK
    ({:.1f}% error)</span>.<br><br>
    The convection coefficient <span class="warn">h</span> and heat source
    <span class="warn">Q₀</span> both affect the overall temperature <em>level</em> but
    in ways that are statistically indistinguishable from these sensors alone.
    A high Q₀ with high h produces nearly the same sensor readings as a low Q₀ with low h —
    they lie on a <em>non-identifiability ridge</em> in parameter space. This is a physical
    limitation: additional sensor types (e.g. calorimetry for Q₀, or surface heat flux probes
    for h) would be needed to break the degeneracy.
    </div>""".format(s["k_mean"], s["k_sd"],
                     abs(s['k_mean']-s['k_true'])/s['k_true']*100),
                unsafe_allow_html=True)

    # ── Panel B — Corner Plot ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Panel B — Joint Posterior Correlations</p>',
                unsafe_allow_html=True)

    # Build interactive Plotly corner plot
    try:
        posterior = BAY["trace"].posterior
        k_samp  = posterior["k"].values.flatten()
        h_samp  = posterior["h"].values.flatten()
        Q0_samp = posterior["Q0"].values.flatten()

        fig_corner = make_subplots(
            rows=2, cols=2,
            subplot_titles=["k vs h", "k vs Q₀", "h vs Q₀", ""],
            vertical_spacing=0.12, horizontal_spacing=0.08
        )

        scatter_kw = dict(mode="markers",
                          marker=dict(size=3, opacity=0.5,
                                      color="#388bfd",
                                      line=dict(width=0)))

        fig_corner.add_trace(go.Scatter(x=k_samp, y=h_samp,
                                         name="k vs h", **scatter_kw),
                              row=1, col=1)
        fig_corner.add_trace(go.Scatter(x=k_samp, y=Q0_samp,
                                         name="k vs Q₀", **scatter_kw),
                              row=1, col=2)
        fig_corner.add_trace(go.Scatter(x=h_samp, y=Q0_samp,
                                         name="h vs Q₀", **scatter_kw),
                              row=2, col=1)

        # Add true value markers
        for r, c, xv, yv in [
            (1, 1, s["k_true"],  s["h_true"]),
            (1, 2, s["k_true"],  s["Q0_true"]),
            (2, 1, s["h_true"],  s["Q0_true"]),
        ]:
            fig_corner.add_trace(go.Scatter(
                x=[xv], y=[yv], mode="markers",
                marker=dict(symbol="cross", size=12, color="#56d364",
                            line=dict(width=2, color="#56d364")),
                name="True value" if (r == 1 and c == 1) else None,
                showlegend=(r == 1 and c == 1),
            ), row=r, col=c)

        fig_corner.update_xaxes(title_text="k [W/mK]", row=1, col=1,
                                 gridcolor="#21262d", linecolor="#30363d")
        fig_corner.update_yaxes(title_text="h [W/m²K]", row=1, col=1,
                                 gridcolor="#21262d", linecolor="#30363d")
        fig_corner.update_xaxes(title_text="k [W/mK]", row=1, col=2,
                                 gridcolor="#21262d", linecolor="#30363d")
        fig_corner.update_yaxes(title_text="Q₀ [W/m³]", row=1, col=2,
                                 gridcolor="#21262d", linecolor="#30363d")
        fig_corner.update_xaxes(title_text="h [W/m²K]", row=2, col=1,
                                 gridcolor="#21262d", linecolor="#30363d")
        fig_corner.update_yaxes(title_text="Q₀ [W/m³]", row=2, col=1,
                                 gridcolor="#21262d", linecolor="#30363d")

        fig_corner.update_layout(
            height=500,
            margin=dict(t=50, b=20, l=60, r=20),
            showlegend=True,
            legend=dict(x=0.6, y=0.3, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=10)),
            **DARK
        )
        st.plotly_chart(fig_corner, use_container_width=True)

    except Exception as e:
        st.warning(f"Corner plot failed: {e}")

    st.markdown("""<div class="explanation-box">
    Each panel shows the <strong>joint posterior distribution</strong> of two parameters.
    The green cross marks the true parameter values. <br>
    <strong>k is tightly identified</strong> — it appears as a narrow vertical band at k ≈ 198.
    <strong>h and Q₀ are spread</strong> — the cloud extends broadly, confirming they are
    non-identifiable from temperature sensors alone. The positive h–Q₀ slope shows the
    non-identifiability ridge: high Q₀ and high h are compensating effects on the sensor readings.
    </div>""", unsafe_allow_html=True)

    # ── Panel C — Field UQ ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Panel C — Field Reconstruction with Credible Intervals</p>',
                unsafe_allow_html=True)

    T_mean_vals = BAY["T_mean"][:, step_idx]
    T_std_vals  = BAY["T_std"][:, step_idx]
    T_true_vals = D["T_true"][:, step_idx]

    zmin_all = min(T_true_vals.min(), T_mean_vals.min())
    zmax_all = max(T_true_vals.max(), T_mean_vals.max())

    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.markdown("**FOM ground truth**")
        st.plotly_chart(
            make_field_scatter(coords, T_true_vals, "",
                               colorscale="Hot", sensor_nodes=T_nodes,
                               zmin=zmin_all, zmax=zmax_all,
                               colorbar_title="T [K]"),
            use_container_width=True
        )
    with cB:
        st.markdown("**Posterior mean T**")
        st.plotly_chart(
            make_field_scatter(coords, T_mean_vals, "",
                               colorscale="Hot",
                               zmin=zmin_all, zmax=zmax_all,
                               colorbar_title="T [K]"),
            use_container_width=True
        )
    with cC:
        u_mean_vals = BAY["u_mean"][:, step_idx]
        n_nodes     = coords.shape[0]
        ux_m = u_mean_vals[0::2][:n_nodes]
        uy_m = u_mean_vals[1::2][:n_nodes]
        umag_m = np.sqrt(ux_m**2 + uy_m**2) * 1e6
        st.markdown("**Posterior mean |u|**")
        st.plotly_chart(
            make_field_scatter(coords, umag_m, "",
                               colorscale="Viridis",
                               zmin=0, zmax=max(umag_m.max(), 1e-12),
                               colorbar_title="|u| [µm]"),
            use_container_width=True
        )
    with cD:
        std_max = T_std_vals.max() if T_std_vals.max() > 0 else 1.0
        st.markdown("**Posterior σ_T (uncertainty)**")
        st.plotly_chart(
            make_field_scatter(coords, T_std_vals, "",
                               colorscale="Plasma",
                               zmin=0, zmax=std_max,
                               colorbar_title="σ_T [K]"),
            use_container_width=True
        )

    st.markdown(f"""<div class="explanation-box">
    The posterior mean temperature field closely matches the FOM ground truth —
    confirming that k is well identified. The pointwise standard deviation
    <strong>σ_T = {BAY['T_std'].max():.2f} K</strong> is small because the posterior
    on k is tight (± {s['k_sd']:.1f} W/mK). Regions of higher σ_T near the hot boundary
    indicate where uncertainty in h and Q₀ propagates most strongly into the field.
    High-uncertainty regions identify where placing additional sensors would most reduce
    overall parameter uncertainty.
    </div>""", unsafe_allow_html=True)

    # ── Panel D — Posterior Predictive ───────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Panel D — Posterior Predictive Check</p>',
                unsafe_allow_html=True)

    pp_path = Path(FIGURES_DIR) / "posterior_predictive.png"
    if pp_path.exists():
        col_d, col_exp = st.columns([0.65, 0.35])
        with col_d:
            st.image(str(pp_path), use_container_width=True)
        with col_exp:
            st.markdown("""<div class="insight-box">
            <strong>What this plot shows:</strong><br><br>
            For each posterior sample (k, h, Q₀), the GP emulator predicts
            what the sensors <em>should have</em> read. The blue band is the
            95% predictive interval across all posterior samples.
            Red dots are the actual (noisy) observations.<br><br>
            <strong>Temperature sensors</strong> (top rows): the posterior
            predictive closely follows the observations — k is well identified
            and the forward model is accurate.<br><br>
            <strong>Strain sensors</strong> (bottom rows): wider bands reflect
            the large uncertainty in h and Q₀, which both affect thermal
            expansion.
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Run `python3 posterior_analysis.py` to generate this figure.")


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<small style='color:#484f58'>Thermo-Mechanical Digital Twin · "
    "FEniCSx · NumPy · PyMC · ArviZ · Streamlit · Plotly · "
    "Anish Hilary Ignatius · 2025</small>",
    unsafe_allow_html=True
)
