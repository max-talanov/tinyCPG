#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEST 3.9.0 script: Izhikevich neurons + plain STDP + dynamic weight tracking
+ histogram animation + color-coded synapse-weight maps + mean±std plots

Requested sizes:
- cut = 100
- bs  = 100
- rg  = 50*4 = 200
- motor = 200

Features:
- Real Izhikevich neurons (NEST model: "izhikevich")
- STDP everywhere it is possible: cut→RG, BS→RG, RG→motor (stdp_synapse)
- 4-phase chunked stimulation of cutaneous fibers
- Dynamic "training picture" of weights:
    A) learning curves (mean/std)
    B) mean±std (dispersion band)
    C) animated histograms of weight distributions over time
    D) color-coded synapse-weight maps (source vs target index colored by weight) at the end of training
    E) color-coded event-based weight updates (if weight_recorder is available)

Important NEST 3.9 note:
- synapse parameter "weight_recorder" cannot be passed via Connect(..., syn_spec=...).
  It must be set via CopyModel() / SetDefaults(). This script does that.

Run:
    python 2_stdp_izhi_memcpg_nest_color.py
"""

import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------
# Sizes (as requested)
# ----------------------------
N_CUT   = 100
N_BS    = 100
N_RG    = 50 * 4   # 200
N_MOTOR = 200

# ----------------------------
# Timing / stimulation
# ----------------------------
SIM_MS   = 1000.0
N_PHASES = 4
PHASE_MS = SIM_MS / N_PHASES

CUT_RATE_ON_HZ  = 200.0
CUT_RATE_OFF_HZ = 0.0
BS_RATE_HZ      = 200.0

# sample weights every dt ms (learning curves + histogram frames)
SAMPLE_DT_MS = 10.0

# ----------------------------
# Connectivity
# ----------------------------
# "anatomical" sparse static paths (optional, see below)
cut2rg_p = 0.002
rg2mot_p = 0.02

# STDP connections (plastic paths)
stdp_p   = 0.5
DELAY_MS = 1.0

# ----------------------------
# STDP params (start values; tune as needed)
# ----------------------------
TAU_PLUS = 20.0   # ms
LAMBDA   = 0.002  # learning rate
ALPHA    = 1.05   # LTD/LTP ratio
W0_STDP  = 5.0    # pA initial
WMAX     = 50.0   # pA cap
MU_PLUS  = 0.0    # additive
MU_MINUS = 0.0

# ----------------------------
# Izhikevich neuron params (RS-like defaults)
# ----------------------------
izh_params = {
    "a": 0.02,
    "b": 0.2,
    "c": -65.0,
    "d": 8.0,
    "V_th": 30.0,
    "V_min": -120.0,
}

# Optional baseline currents (set to 0.0 for purely input-driven)
I_E_RG = 0.0
I_E_MOTOR = 0.0

# Optional: keep sparse static paths in parallel with STDP
# If you want ONLY STDP connections, set USE_STATIC_PATHS = False
USE_STATIC_PATHS = True
W_STATIC = 15.0   # pA (tune)

# Histogram animation settings
HIST_BINS = 30
FPS = 20  # animation playback speed

# Color maps (user asked for color coding)
CMAP = "viridis"


def main():
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    # Inputs: poisson_generator -> parrot_neuron
    cut_pg = nest.Create("poisson_generator", N_CUT)
    cut_in = nest.Create("parrot_neuron", N_CUT)
    nest.Connect(cut_pg, cut_in, conn_spec={"rule": "one_to_one"})

    bs_pg = nest.Create("poisson_generator", N_BS)
    bs_in = nest.Create("parrot_neuron", N_BS)
    nest.Connect(bs_pg, bs_in, conn_spec={"rule": "one_to_one"})

    nest.SetStatus(cut_pg, {"rate": CUT_RATE_OFF_HZ})
    nest.SetStatus(bs_pg,  {"rate": BS_RATE_HZ})

    # Neurons
    rg = nest.Create("izhikevich", N_RG)
    motor = nest.Create("izhikevich", N_MOTOR)
    nest.SetStatus(rg, izh_params)
    nest.SetStatus(motor, izh_params)
    nest.SetStatus(rg, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_RG})
    nest.SetStatus(motor, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_MOTOR})

    # Spike recorders (sanity check)
    rg_rec  = nest.Create("spike_recorder")
    mot_rec = nest.Create("spike_recorder")
    nest.Connect(rg, rg_rec)
    nest.Connect(motor, mot_rec)

    # Optional weight recorders (event-based)
    HAVE_WR = True
    try:
        wr_cut_rg = nest.Create("weight_recorder")
        wr_bs_rg  = nest.Create("weight_recorder")
        wr_rg_mot = nest.Create("weight_recorder")
    except Exception:
        HAVE_WR = False
        wr_cut_rg = wr_bs_rg = wr_rg_mot = None

    # STDP synapses: 3 distinct models; attach weight_recorders via CopyModel
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
        nest.CopyModel("stdp_synapse", "stdp_bs_rg",  {**stdp_defaults, "weight_recorder": wr_bs_rg})
        nest.CopyModel("stdp_synapse", "stdp_rg_mot", {**stdp_defaults, "weight_recorder": wr_rg_mot})
    else:
        nest.CopyModel("stdp_synapse", "stdp_cut_rg", stdp_defaults)
        nest.CopyModel("stdp_synapse", "stdp_bs_rg",  stdp_defaults)
        nest.CopyModel("stdp_synapse", "stdp_rg_mot", stdp_defaults)

    syn_cut = {"synapse_model": "stdp_cut_rg", "weight": W0_STDP, "delay": DELAY_MS}
    syn_bs  = {"synapse_model": "stdp_bs_rg",  "weight": W0_STDP, "delay": DELAY_MS}
    syn_rm  = {"synapse_model": "stdp_rg_mot", "weight": W0_STDP, "delay": DELAY_MS}

    # Plastic connections
    nest.Connect(cut_in, rg,    conn_spec={"rule": "pairwise_bernoulli", "p": stdp_p}, syn_spec=syn_cut)
    nest.Connect(bs_in,  rg,    conn_spec={"rule": "pairwise_bernoulli", "p": stdp_p}, syn_spec=syn_bs)
    nest.Connect(rg,     motor, conn_spec={"rule": "pairwise_bernoulli", "p": stdp_p}, syn_spec=syn_rm)

    # Optional sparse static paths (in parallel)
    if USE_STATIC_PATHS:
        nest.Connect(cut_in, rg, conn_spec={"rule": "pairwise_bernoulli", "p": cut2rg_p},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC, "delay": DELAY_MS})
        nest.Connect(rg, motor, conn_spec={"rule": "pairwise_bernoulli", "p": rg2mot_p},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC, "delay": DELAY_MS})

    # Smooth sampling
    def sample_weights(syn_model_name: str) -> np.ndarray:
        conns = nest.GetConnections(synapse_model=syn_model_name)
        if len(conns) == 0:
            return np.array([], dtype=float)
        return np.asarray(nest.GetStatus(conns, "weight"), dtype=float)

    times = []
    cut_mean, cut_std = [], []
    bs_mean,  bs_std  = [], []
    rm_mean,  rm_std  = [], []

    cut_snapshots, bs_snapshots, rm_snapshots = [], [], []

    def log_sample(t_ms: float):
        times.append(t_ms)
        w_cut = sample_weights("stdp_cut_rg")
        w_bs  = sample_weights("stdp_bs_rg")
        w_rm  = sample_weights("stdp_rg_mot")

        cut_snapshots.append(w_cut)
        bs_snapshots.append(w_bs)
        rm_snapshots.append(w_rm)

        def stats(w):
            if w.size == 0:
                return np.nan, np.nan
            return float(w.mean()), float(w.std())

        m, s = stats(w_cut); cut_mean.append(m); cut_std.append(s)
        m, s = stats(w_bs);  bs_mean.append(m);  bs_std.append(s)
        m, s = stats(w_rm);  rm_mean.append(m);  rm_std.append(s)

    # Run: phases + chunked cut stimulation
    chunk = int(N_CUT / N_PHASES)
    t = 0.0
    for phase in range(N_PHASES):
        nest.SetStatus(cut_pg, {"rate": CUT_RATE_OFF_HZ})
        start = phase * chunk
        end   = (phase + 1) * chunk
        nest.SetStatus(cut_pg[start:end], {"rate": CUT_RATE_ON_HZ})

        n_steps = int(PHASE_MS / SAMPLE_DT_MS)
        for _ in range(n_steps):
            nest.Simulate(SAMPLE_DT_MS)
            t += SAMPLE_DT_MS
            log_sample(t)

    times_arr = np.asarray(times)

    # Mean + dispersion band
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
    plt.title("Mean and dispersion (mean ± std) of synaptic weights")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Std over time
    plt.figure(figsize=(14, 6))
    plt.plot(times_arr, cut_std, label="cut→RG std", color="tab:blue")
    plt.plot(times_arr, bs_std,  label="BS→RG std",  color="tab:orange")
    plt.plot(times_arr, rm_std,  label="RG→motor std", color="tab:green")
    plt.xlabel("time (ms)")
    plt.ylabel("STD of weight (pA)")
    plt.title("Dispersion of synaptic weights (std over time)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Animated histograms
    def global_minmax(snaps):
        mn = np.inf; mx = -np.inf
        for w in snaps:
            if w.size == 0: continue
            mn = min(mn, float(w.min()))
            mx = max(mx, float(w.max()))
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            return 0.0, 1.0
        return mn, mx

    cut_min, cut_max = global_minmax(cut_snapshots)
    bs_min,  bs_max  = global_minmax(bs_snapshots)
    rm_min,  rm_max  = global_minmax(rm_snapshots)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    titles = ["cut→RG weights", "BS→RG weights", "RG→motor weights"]
    ranges = [(cut_min, cut_max), (bs_min, bs_max), (rm_min, rm_max)]
    snaps  = [cut_snapshots, bs_snapshots, rm_snapshots]
    cols   = ["tab:blue", "tab:orange", "tab:green"]

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

    time_text = fig.text(0.8, 0.94, "", fontsize=12)

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

    # Color-coded synapse-weight maps (final snapshot)
    def plot_weight_map(syn_model: str, pre_nodes, post_nodes, title: str):
        conns = nest.GetConnections(source=pre_nodes, target=post_nodes, synapse_model=syn_model)
        if len(conns) == 0:
            print(f"{title}: no connections found")
            return

        src = np.asarray(nest.GetStatus(conns, "source"), dtype=int)
        tgt = np.asarray(nest.GetStatus(conns, "target"), dtype=int)
        w   = np.asarray(nest.GetStatus(conns, "weight"), dtype=float)

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

    plot_weight_map("stdp_cut_rg", cut_in, rg,    "Final STDP weight map: cut→RG")
    plot_weight_map("stdp_bs_rg",  bs_in,  rg,    "Final STDP weight map: BS→RG")
    plot_weight_map("stdp_rg_mot", rg,     motor, "Final STDP weight map: RG→motor")

    # Event-based plots color-coded
    if HAVE_WR:
        def plot_wr(wr, title):
            ev = nest.GetStatus(wr, "events")[0]
            if len(ev["times"]) == 0:
                print(f"{title}: no weight update events recorded")
                return
            plt.figure(figsize=(14, 4))
            plt.scatter(ev["times"], ev["weights"], s=3, c=ev["weights"], cmap=CMAP, alpha=0.8)
            plt.xlabel("time (ms)")
            plt.ylabel("weight (pA)")
            plt.title(title + " (event-based; color-coded)")
            cbar = plt.colorbar()
            cbar.set_label("weight (pA)")
            plt.tight_layout()
            plt.show()

        plot_wr(wr_cut_rg, "STDP cut→RG")
        plot_wr(wr_bs_rg,  "STDP BS→RG")
        plot_wr(wr_rg_mot, "STDP RG→motor")

    # Spike sanity check
    ev_rg = nest.GetStatus(rg_rec, "events")[0]
    ev_m  = nest.GetStatus(mot_rec, "events")[0]
    print("RG spikes:", len(ev_rg["times"]), " Motor spikes:", len(ev_m["times"]))


if __name__ == "__main__":
    main()
