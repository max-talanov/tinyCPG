#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_stdp_izhi_nest.py

NEST 3.9.0: Izhikevich neurons + plain STDP + dynamic weight tracking
EXTENSION: adds Ia (muscle afferent) generators driven by an estimated muscle force proxy.

Motivation:
NEST has no built-in muscle model. We approximate the muscle force from motor neuron firing:
  motor spikes -> motor rate (Hz) -> low-pass filter -> "force" (a.u.)
Then we drive Ia as an inhomogeneous Poisson generator:
  Ia_rate(t) = IA_BASE_HZ + IA_FORCE_GAIN_HZ * force(t)
Optionally with an "almost sinusoidal" modulation whose amplitude scales with force.

Run:
  python 2_stdp_izhi_nest.py
"""

import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================
# Sizes (requested / defaults)
# ============================
N_CUT = 100                 # cutaneous (stretch) afferents
N_BS = 100                  # optional second sensory stream
N_RG = 50 * 4               # 200
N_MOTOR = 200               # motor neurons
N_IA = 100                  # Ia afferents (set to N_MOTOR for 1:1)

# ============================
# Timing / stimulation
# ============================
SIM_MS = 1000.0
N_PHASES = 4
PHASE_MS = SIM_MS / N_PHASES

CUT_RATE_ON_HZ = 200.0
CUT_RATE_OFF_HZ = 0.0
BS_RATE_HZ = 200.0

# simulate in chunks so we can update Ia online
SAMPLE_DT_MS = 10.0

# ============================
# Connectivity
# ============================
cut2rg_p = 0.002
rg2mot_p = 0.02

stdp_p = 0.5
DELAY_MS = 1.0

# ============================
# STDP (plain, no DA)
# ============================
TAU_PLUS = 20.0
LAMBDA = 0.002
ALPHA = 1.05
W0_STDP = 5.0
WMAX = 50.0
MU_PLUS = 0.0
MU_MINUS = 0.0

# ============================
# Izhikevich params (RS-like)
# ============================
izh_params = {
    "a": 0.02,
    "b": 0.2,
    "c": -65.0,
    "d": 8.0,
    "V_th": 30.0,
    "V_min": -120.0,
}
I_E_RG = 0.0
I_E_MOTOR = 0.0

USE_STATIC_PATHS = True
W_STATIC = 15.0  # pA

# ============================
# Ia force-proxy parameters
# ============================
TAU_FORCE_MS = 50.0          # low-pass time constant for force proxy
FORCE_GAIN = 1.0             # motor rate -> force scale
FORCE_MAX = 1.0              # clamp force

IA_BASE_HZ = 5.0
IA_FORCE_GAIN_HZ = 150.0
IA_RATE_MAX_HZ = 400.0

# "almost sinusoidal" modulation (optional)
IA_SIN_MOD_HZ = 2.0          # Hz
IA_SIN_MOD_DEPTH = 0.6       # 0..1 (0 disables)

# ============================
# Plot settings
# ============================
HIST_BINS = 30
FPS = 20
CMAP = "viridis"

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def main():
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    # ----------------------------
    # Inputs: Poisson -> parrot
    # ----------------------------
    cut_pg = nest.Create("poisson_generator", N_CUT)
    cut_in = nest.Create("parrot_neuron", N_CUT)
    nest.Connect(cut_pg, cut_in, conn_spec={"rule": "one_to_one"})

    bs_pg = nest.Create("poisson_generator", N_BS)
    bs_in = nest.Create("parrot_neuron", N_BS)
    nest.Connect(bs_pg, bs_in, conn_spec={"rule": "one_to_one"})

    # NEW: Ia generator -> parrot (Ia afferents)
    ia_pg = nest.Create("poisson_generator", N_IA)
    ia_in = nest.Create("parrot_neuron", N_IA)
    nest.Connect(ia_pg, ia_in, conn_spec={"rule": "one_to_one"})

    nest.SetStatus(cut_pg, {"rate": CUT_RATE_OFF_HZ})
    nest.SetStatus(bs_pg, {"rate": BS_RATE_HZ})
    nest.SetStatus(ia_pg, {"rate": IA_BASE_HZ})

    # ----------------------------
    # Neuron populations
    # ----------------------------
    rg = nest.Create("izhikevich", N_RG)
    motor = nest.Create("izhikevich", N_MOTOR)

    nest.SetStatus(rg, izh_params)
    nest.SetStatus(motor, izh_params)

    nest.SetStatus(rg, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_RG})
    nest.SetStatus(motor, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_MOTOR})

    # ----------------------------
    # Recorders
    # ----------------------------
    rg_rec = nest.Create("spike_recorder")
    mot_rec = nest.Create("spike_recorder")
    ia_rec = nest.Create("spike_recorder")  # NEW: monitor Ia if needed
    nest.Connect(rg, rg_rec)
    nest.Connect(motor, mot_rec)
    nest.Connect(ia_in, ia_rec)

    # ----------------------------
    # Optional weight recorders (event-based) – must be attached via CopyModel
    # ----------------------------
    HAVE_WR = True
    try:
        wr_cut_rg = nest.Create("weight_recorder")
        wr_bs_rg = nest.Create("weight_recorder")
        wr_rg_mot = nest.Create("weight_recorder")
    except Exception:
        HAVE_WR = False
        wr_cut_rg = wr_bs_rg = wr_rg_mot = None

    # ----------------------------
    # STDP synapse models (cut->RG, BS->RG, RG->motor)
    # ----------------------------
    stdp_defaults = {
        "tau_plus": TAU_PLUS,
        "lambda": LAMBDA,
        "alpha": ALPHA,
        "mu_plus": MU_PLUS,
        "mu_minus": MU_MINUS,
        "Wmax": WMAX,
    }

    if HAVE_WR:
        nest.CopyModel("stdp_synapse", "stdp_cut_rg", {**stdp_defaults, "weight_recorder": wr_cut_rg})
        nest.CopyModel("stdp_synapse", "stdp_bs_rg", {**stdp_defaults, "weight_recorder": wr_bs_rg})
        nest.CopyModel("stdp_synapse", "stdp_rg_mot", {**stdp_defaults, "weight_recorder": wr_rg_mot})
    else:
        nest.CopyModel("stdp_synapse", "stdp_cut_rg", stdp_defaults)
        nest.CopyModel("stdp_synapse", "stdp_bs_rg", stdp_defaults)
        nest.CopyModel("stdp_synapse", "stdp_rg_mot", stdp_defaults)

    syn_cut = {"synapse_model": "stdp_cut_rg", "weight": W0_STDP, "delay": DELAY_MS}
    syn_bs = {"synapse_model": "stdp_bs_rg", "weight": W0_STDP, "delay": DELAY_MS}
    syn_rm = {"synapse_model": "stdp_rg_mot", "weight": W0_STDP, "delay": DELAY_MS}

    nest.Connect(cut_in, rg, conn_spec={"rule": "pairwise_bernoulli", "p": stdp_p}, syn_spec=syn_cut)
    nest.Connect(bs_in, rg, conn_spec={"rule": "pairwise_bernoulli", "p": stdp_p}, syn_spec=syn_bs)
    nest.Connect(rg, motor, conn_spec={"rule": "pairwise_bernoulli", "p": stdp_p}, syn_spec=syn_rm)

    # ----------------------------
    # NEW: Ia -> RG feedback (static excitatory)
    # ----------------------------
    IA2RG_P = 0.3
    IA2RG_W = 10.0
    nest.Connect(
        ia_in, rg,
        conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
        syn_spec={"synapse_model": "static_synapse", "weight": IA2RG_W, "delay": DELAY_MS},
    )

    # Optional sparse static paths (in parallel)
    if USE_STATIC_PATHS:
        nest.Connect(
            cut_in, rg,
            conn_spec={"rule": "pairwise_bernoulli", "p": cut2rg_p},
            syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC, "delay": DELAY_MS},
        )
        nest.Connect(
            rg, motor,
            conn_spec={"rule": "pairwise_bernoulli", "p": rg2mot_p},
            syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC, "delay": DELAY_MS},
        )

    # ----------------------------
    # Weight sampling helpers (for plots)
    # ----------------------------
    def sample_weights(syn_model_name: str) -> np.ndarray:
        conns = nest.GetConnections(synapse_model=syn_model_name)
        if len(conns) == 0:
            return np.array([], dtype=float)
        return np.asarray(nest.GetStatus(conns, "weight"), dtype=float)

    times = []
    cut_mean, cut_std = [], []
    bs_mean, bs_std = [], []
    rm_mean, rm_std = [], []
    cut_snaps, bs_snaps, rm_snaps = [], [], []

    # ----------------------------
    # NEW: closed-loop motor->force->Ia state
    # ----------------------------
    force = 0.0
    last_mot_event_len = 0

    force_trace = []
    ia_rate_trace = []
    mot_rate_trace = []

    def update_force_and_ia_rate(t_ms: float):
        """Called once per step after Simulate(dt)."""
        nonlocal force, last_mot_event_len

        dt_s = SAMPLE_DT_MS / 1000.0

        ev = nest.GetStatus(mot_rec, "events")[0]
        cur_len = len(ev["times"])
        new_spikes = cur_len - last_mot_event_len
        last_mot_event_len = cur_len

        # mean motor firing rate per neuron (Hz)
        mot_rate_hz = (new_spikes / max(1, N_MOTOR)) / dt_s

        # low-pass filter to "force" proxy
        tau_s = TAU_FORCE_MS / 1000.0
        target = FORCE_GAIN * mot_rate_hz
        force += (dt_s / tau_s) * (target - force)
        force = clamp(force, 0.0, FORCE_MAX)

        # Ia rate from force (+ optional sinusoidal modulation)
        t_s = t_ms / 1000.0
        sin_mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * IA_SIN_MOD_HZ * t_s))
        amp = 1.0 - IA_SIN_MOD_DEPTH + IA_SIN_MOD_DEPTH * sin_mod  # in [1-depth, 1]
        ia_rate_hz = (IA_BASE_HZ + IA_FORCE_GAIN_HZ * force) * amp
        ia_rate_hz = clamp(ia_rate_hz, 0.0, IA_RATE_MAX_HZ)

        # apply to Ia generators for next window
        nest.SetStatus(ia_pg, {"rate": ia_rate_hz})

        mot_rate_trace.append(mot_rate_hz)
        force_trace.append(force)
        ia_rate_trace.append(ia_rate_hz)

    def log_sample(t_ms: float):
        times.append(t_ms)

        w_cut = sample_weights("stdp_cut_rg")
        w_bs = sample_weights("stdp_bs_rg")
        w_rm = sample_weights("stdp_rg_mot")

        cut_snaps.append(w_cut)
        bs_snaps.append(w_bs)
        rm_snaps.append(w_rm)

        def push_stats(w, m_list, s_list):
            if w.size == 0:
                m_list.append(np.nan)
                s_list.append(np.nan)
            else:
                m_list.append(float(w.mean()))
                s_list.append(float(w.std()))

        push_stats(w_cut, cut_mean, cut_std)
        push_stats(w_bs, bs_mean, bs_std)
        push_stats(w_rm, rm_mean, rm_std)

    # ----------------------------
    # Run: 4 phases, chunked cut stimulation + Ia updates
    # ----------------------------
    chunk = int(N_CUT / N_PHASES)
    t = 0.0

    for phase in range(N_PHASES):
        nest.SetStatus(cut_pg, {"rate": CUT_RATE_OFF_HZ})
        start = phase * chunk
        end = (phase + 1) * chunk
        nest.SetStatus(cut_pg[start:end], {"rate": CUT_RATE_ON_HZ})

        n_steps = int(PHASE_MS / SAMPLE_DT_MS)
        for _ in range(n_steps):
            nest.Simulate(SAMPLE_DT_MS)
            t += SAMPLE_DT_MS

            update_force_and_ia_rate(t)
            log_sample(t)

    times_arr = np.asarray(times)

    # ----------------------------
    # Plots: mean±std and std-only
    # ----------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(times_arr, cut_mean, label="cut→RG mean", color="tab:blue")
    plt.fill_between(times_arr, np.asarray(cut_mean) - np.asarray(cut_std), np.asarray(cut_mean) + np.asarray(cut_std),
                     color="tab:blue", alpha=0.20)
    plt.plot(times_arr, bs_mean, label="BS→RG mean", color="tab:orange")
    plt.fill_between(times_arr, np.asarray(bs_mean) - np.asarray(bs_std), np.asarray(bs_mean) + np.asarray(bs_std),
                     color="tab:orange", alpha=0.20)
    plt.plot(times_arr, rm_mean, label="RG→motor mean", color="tab:green")
    plt.fill_between(times_arr, np.asarray(rm_mean) - np.asarray(rm_std), np.asarray(rm_mean) + np.asarray(rm_std),
                     color="tab:green", alpha=0.20)
    plt.xlabel("time (ms)")
    plt.ylabel("STDP weight (pA)")
    plt.title("Learning curves with dispersion band (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(times_arr, cut_std, label="cut→RG std", color="tab:blue")
    plt.plot(times_arr, bs_std, label="BS→RG std", color="tab:orange")
    plt.plot(times_arr, rm_std, label="RG→motor std", color="tab:green")
    plt.xlabel("time (ms)")
    plt.ylabel("STD of weight (pA)")
    plt.title("Dispersion of synaptic weights (std over time)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # NEW: motor rate -> force -> Ia rate
    plt.figure(figsize=(14, 7))
    plt.plot(times_arr, mot_rate_trace, label="motor mean rate (Hz/neuron)")
    plt.plot(times_arr, force_trace, label="force proxy (a.u.)")
    plt.plot(times_arr, ia_rate_trace, label="Ia generator rate (Hz)")
    plt.xlabel("time (ms)")
    plt.title("Closed-loop proprioception: motor rate → force → Ia rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Histogram movie of weights
    # ----------------------------
    def global_minmax(snaps):
        mn = np.inf
        mx = -np.inf
        for w in snaps:
            if w.size == 0:
                continue
            mn = min(mn, float(w.min()))
            mx = max(mx, float(w.max()))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            return 0.0, 1.0
        return mn, mx

    cut_min, cut_max = global_minmax(cut_snaps)
    bs_min, bs_max = global_minmax(bs_snaps)
    rm_min, rm_max = global_minmax(rm_snaps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    titles = ["cut→RG weights", "BS→RG weights", "RG→motor weights"]
    ranges = [(cut_min, cut_max), (bs_min, bs_max), (rm_min, rm_max)]
    snaps = [cut_snaps, bs_snaps, rm_snaps]
    cols = ["tab:blue", "tab:orange", "tab:green"]

    for ax, title, (r0, r1) in zip(axes, titles, ranges):
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.set_xlim(r0, r1)
    axes[-1].set_xlabel("weight (pA)")

    bars = []
    for ax, ws, (r0, r1), col in zip(axes, snaps, ranges, cols):
        w0 = ws[0] if len(ws) else np.array([])
        counts, bin_edges = np.histogram(w0, bins=HIST_BINS, range=(r0, r1))
        width = bin_edges[1] - bin_edges[0]
        bars.append(ax.bar(bin_edges[:-1], counts, width=width, align="edge", color=col, alpha=0.85))

    time_text = fig.text(0.78, 0.94, "", fontsize=12)

    def update(frame_idx):
        t_ms = times_arr[frame_idx]
        time_text.set_text(f"t = {t_ms:.1f} ms")
        for ax_idx in range(3):
            w = snaps[ax_idx][frame_idx]
            r0, r1 = ranges[ax_idx]
            counts, _ = np.histogram(w, bins=HIST_BINS, range=(r0, r1))
            for rect, h in zip(bars[ax_idx], counts):
                rect.set_height(h)
            axes[ax_idx].set_ylim(0, max(1, int(counts.max() * 1.1)))
        return (*bars[0], *bars[1], *bars[2], time_text)

    _anim = FuncAnimation(fig, update, frames=len(times_arr), interval=int(1000 / FPS), blit=False)
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Final weight maps (color-coded)
    # ----------------------------
    def plot_weight_map(syn_model: str, pre_nodes, post_nodes, title: str):
        conns = nest.GetConnections(source=pre_nodes, target=post_nodes, synapse_model=syn_model)
        if len(conns) == 0:
            print(f"{title}: no connections found")
            return

        src = np.asarray(nest.GetStatus(conns, "source"), dtype=int)
        tgt = np.asarray(nest.GetStatus(conns, "target"), dtype=int)
        w = np.asarray(nest.GetStatus(conns, "weight"), dtype=float)

        pre_ids = np.asarray(pre_nodes, dtype=int)
        post_ids = np.asarray(post_nodes, dtype=int)
        pre_index = {gid: i for i, gid in enumerate(pre_ids)}
        post_index = {gid: j for j, gid in enumerate(post_ids)}

        mat = np.full((len(pre_ids), len(post_ids)), np.nan, dtype=float)
        for s, t2, ww in zip(src, tgt, w):
            i = pre_index.get(int(s))
            j = post_index.get(int(t2))
            if i is not None and j is not None:
                mat[i, j] = ww

        plt.figure(figsize=(12, 6))
        im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=CMAP)
        plt.title(title)
        plt.xlabel("post index")
        plt.ylabel("pre index")
        cbar = plt.colorbar(im)
        cbar.set_label("weight (pA)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        pre_i = np.array([pre_index[int(s)] for s in src], dtype=int)
        post_j = np.array([post_index[int(t2)] for t2 in tgt], dtype=int)
        sc = plt.scatter(post_j, pre_i, c=w, s=10, cmap=CMAP, marker="s")
        plt.title(title + " (scatter)")
        plt.xlabel("post index")
        plt.ylabel("pre index")
        cbar2 = plt.colorbar(sc)
        cbar2.set_label("weight (pA)")
        plt.tight_layout()
        plt.show()

    plot_weight_map("stdp_cut_rg", cut_in, rg, "Final STDP weight map: cut→RG")
    plot_weight_map("stdp_bs_rg", bs_in, rg, "Final STDP weight map: BS→RG")
    plot_weight_map("stdp_rg_mot", rg, motor, "Final STDP weight map: RG→motor")

    # Spike sanity
    ev_rg = nest.GetStatus(rg_rec, "events")[0]
    ev_m = nest.GetStatus(mot_rec, "events")[0]
    ev_ia = nest.GetStatus(ia_rec, "events")[0]
    print("RG spikes:", len(ev_rg["times"]), "Motor spikes:", len(ev_m["times"]), "Ia spikes:", len(ev_ia["times"]))

if __name__ == "__main__":
    main()
