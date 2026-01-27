#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_stdp_izhi_nest.py (v5) — two legs + Ia driven by muscles (no sinus modulation)

Changes vs v4:
1) Removed sinusoidal modulation of Ia entirely.
2) Added explicit MUSCLE relay populations (parrot_neuron) for each muscle group:
      M-E -> mus-E
      M-F -> mus-F
   Force/length proxies are computed from MUS spikes (not motor spikes),
   so Ia is now “connected to the proper muscles”.
3) Extended the model to TWO LEGS (Left + Right) with the same per-leg architecture.
4) Right leg BS drive is phase-shifted by pi relative to Left leg (diagonal alternation).
5) Kept reciprocal inhibition inside each leg (RG-E <-> RG-F).
6) Optional mild commissural diagonal inhibition between legs (enabled by default).

NEST: 3.9.0

Run:
  python 2_stdp_izhi_nest.py
"""

import nest
import numpy as np
import matplotlib.pyplot as plt


# ============================
# Sizes (your constraints + extension)
# ============================
N_CUT = 100
N_BS = 100

N_RG_TOTAL = 200
N_RG_E = N_RG_TOTAL // 2
N_RG_F = N_RG_TOTAL - N_RG_E

N_MOTOR_E = 100
N_MOTOR_F = 100

N_MUS_E = 100
N_MUS_F = 100

N_IA_E = 100
N_IA_F = 100

LEGS = ("L", "R")


# ============================
# Simulation timing
# ============================
SIM_MS = 6000.0
SAMPLE_DT_MS = 10.0


# ============================
# CUT stimulation (extensor only)
# ============================
N_PHASES = 6
PHASE_MS = SIM_MS / N_PHASES
CUT_RATE_ON_HZ = 200.0
CUT_RATE_OFF_HZ = 0.0


# ============================
# Brainstem drive
# ============================
BS_OSC_HZ = 1.0
BS_RATE_BASE_HZ = 0.0     # keep inactive side quiet
BS_RATE_AMP_HZ = 300.0
BS_RATE_MIN_HZ = 0.0

# Left/right phase shift for BS:
# Right is shifted by pi -> diagonal alternation (Left extensor ~ Right flexor)
BS_PHASE = {"L": 0.0, "R": np.pi}


# ============================
# Connectivity
# ============================
P_IN_STDP = 0.5
P_RG_REC = 0.12
DELAY_MS = 1.0

# Reciprocal inhibition inside leg (RG-E <-> RG-F)
P_RG_RECIP = 0.20
W_RG_RECIP = -18.0
DELAY_RECIP_MS = 1.0

# Motor -> muscle relay synapses (static)
W_M2MUS = 1.0
P_M2MUS = 0.8

# Muscle afferents -> RG feedback synapses (static excitatory)
IA2RG_P = 0.4
IA2RG_W = 12.0

# Baseline RG drive (insurance)
BASE_DRIVE_HZ = 10.0
BASE_DRIVE_W = 18.0
BASE_DRIVE_P = 0.08

# Optional static parallel paths (insurance)
USE_STATIC_PARALLEL = True
P_STATIC_IN = 0.03
P_STATIC_RM = 0.03
W_STATIC_IN = 22.0
W_STATIC_RM = 35.0


# ============================
# Commissural (between legs) — mild stabilizer
# ============================
ENABLE_COMMISSURAL = True
P_COMM = 0.08
W_COMM_INH = -10.0
DELAY_COMM_MS = 1.0
# Diagonal inhibition:
# RG-E(L) inhibits RG-F(R) and RG-E(R) inhibits RG-F(L),
# RG-F(L) inhibits RG-E(R) and RG-F(R) inhibits RG-E(L)


# ============================
# STDP params (plain, no DA)
# ============================
TAU_PLUS = 20.0
LAMBDA = 0.002
ALPHA = 1.05
MU_PLUS = 0.0
MU_MINUS = 0.0
WMAX = 120.0

W0_IN = 22.0
W0_RM = 30.0


# ============================
# Izhikevich neurons
# ============================
izh_params = {
    "a": 0.02,
    "b": 0.2,
    "c": -65.0,
    "d": 8.0,
    "V_th": 30.0,
    "V_min": -120.0,
}

I_E_RG = 1.0
I_E_MOTOR = 1.0


# ============================
# Muscle proxy: activation/force/length
# ============================
TAU_ACT_MS = 80.0
ACT_GAIN = 0.03
ACT_MAX = 1.2

TAU_FORCE_RISE_MS = 140.0
TAU_FORCE_DECAY_MS = 60.0
FORCE_MAX = 25.0
FORCE_SAT_K = 2.5

TAU_LENGTH_MS = 260.0
L0 = 1.0
L_MIN, L_MAX = 0.5, 2.0
SHORTEN_GAIN = 0.010
STRETCH_GAIN = 0.35  # extensor-only stretch from CUT fraction


# ============================
# Ia generator model (NO SINUS MODULATION)
# ============================
IA_BASE_HZ = 10.0
IA_K_FORCE = 6.0
IA_K_STRETCH = 250.0
IA_RATE_MAX_HZ = 500.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def bs_rates_counterphase(t_ms: float, leg: str) -> tuple[float, float]:
    """
    Counter-phase BS within a leg, with optional leg phase shift.
      E gets +sin half-wave
      F gets -sin half-wave
    """
    t_s = t_ms / 1000.0
    s = np.sin(2.0 * np.pi * BS_OSC_HZ * t_s + BS_PHASE[leg])
    e = max(0.0, s)
    f = max(0.0, -s)
    r_e = BS_RATE_BASE_HZ + BS_RATE_AMP_HZ * e
    r_f = BS_RATE_BASE_HZ + BS_RATE_AMP_HZ * f
    r_e = clamp(r_e, BS_RATE_MIN_HZ, BS_RATE_BASE_HZ + BS_RATE_AMP_HZ)
    r_f = clamp(r_f, BS_RATE_MIN_HZ, BS_RATE_BASE_HZ + BS_RATE_AMP_HZ)
    return r_e, r_f


def make_weight_recorder_safe():
    try:
        return nest.Create("weight_recorder")
    except Exception:
        return None


def main():
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    # ----------------------------
    # Build per-leg structures
    # ----------------------------
    leg = {}
    for side in LEGS:
        # CUT input (per leg)
        cut_pg = nest.Create("poisson_generator", N_CUT)
        cut_in = nest.Create("parrot_neuron", N_CUT)
        nest.Connect(cut_pg, cut_in, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(cut_pg, {"rate": CUT_RATE_OFF_HZ})

        # BS inputs (per leg; separate E and F)
        bs_pg_e = nest.Create("poisson_generator", N_BS)
        bs_in_e = nest.Create("parrot_neuron", N_BS)
        nest.Connect(bs_pg_e, bs_in_e, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(bs_pg_e, {"rate": BS_RATE_BASE_HZ})

        bs_pg_f = nest.Create("poisson_generator", N_BS)
        bs_in_f = nest.Create("parrot_neuron", N_BS)
        nest.Connect(bs_pg_f, bs_in_f, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(bs_pg_f, {"rate": BS_RATE_BASE_HZ})

        # Baseline drive
        base_pg = nest.Create("poisson_generator", N_BS)
        base_in = nest.Create("parrot_neuron", N_BS)
        nest.Connect(base_pg, base_in, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(base_pg, {"rate": BASE_DRIVE_HZ})

        # Ia devices
        ia_pg_e = nest.Create("poisson_generator", N_IA_E)
        ia_in_e = nest.Create("parrot_neuron", N_IA_E)
        nest.Connect(ia_pg_e, ia_in_e, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(ia_pg_e, {"rate": IA_BASE_HZ})

        ia_pg_f = nest.Create("poisson_generator", N_IA_F)
        ia_in_f = nest.Create("parrot_neuron", N_IA_F)
        nest.Connect(ia_pg_f, ia_in_f, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(ia_pg_f, {"rate": IA_BASE_HZ})

        # Neuron populations
        rg_e = nest.Create("izhikevich", N_RG_E)
        rg_f = nest.Create("izhikevich", N_RG_F)
        m_e = nest.Create("izhikevich", N_MOTOR_E)
        m_f = nest.Create("izhikevich", N_MOTOR_F)
        for pop in (rg_e, rg_f, m_e, m_f):
            nest.SetStatus(pop, izh_params)
        nest.SetStatus(rg_e, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_RG})
        nest.SetStatus(rg_f, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_RG})
        nest.SetStatus(m_e,  {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_MOTOR})
        nest.SetStatus(m_f,  {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_MOTOR})

        # MUSCLE relays
        mus_e = nest.Create("parrot_neuron", N_MUS_E)
        mus_f = nest.Create("parrot_neuron", N_MUS_F)

        # Recorders
        rec_muse = nest.Create("spike_recorder")
        rec_musf = nest.Create("spike_recorder")
        nest.Connect(mus_e, rec_muse)
        nest.Connect(mus_f, rec_musf)

        leg[side] = dict(
            cut_pg=cut_pg, cut_in=cut_in,
            bs_pg_e=bs_pg_e, bs_in_e=bs_in_e,
            bs_pg_f=bs_pg_f, bs_in_f=bs_in_f,
            base_pg=base_pg, base_in=base_in,
            ia_pg_e=ia_pg_e, ia_in_e=ia_in_e,
            ia_pg_f=ia_pg_f, ia_in_f=ia_in_f,
            rg_e=rg_e, rg_f=rg_f,
            m_e=m_e, m_f=m_f,
            mus_e=mus_e, mus_f=mus_f,
            rec_muse=rec_muse, rec_musf=rec_musf,
        )

    # ----------------------------
    # Synapse models with weight recorders (optional)
    # ----------------------------
    stdp_defaults = {
        "tau_plus": TAU_PLUS,
        "lambda": LAMBDA,
        "alpha": ALPHA,
        "mu_plus": MU_PLUS,
        "mu_minus": MU_MINUS,
        "Wmax": WMAX,
    }

    for side in LEGS:
        wr_cut = make_weight_recorder_safe()
        wr_bse = make_weight_recorder_safe()
        wr_bsf = make_weight_recorder_safe()
        wr_rge_me = make_weight_recorder_safe()
        wr_rgf_mf = make_weight_recorder_safe()

        def copy(name, wr):
            if wr is not None:
                nest.CopyModel("stdp_synapse", name, {**stdp_defaults, "weight_recorder": wr})
            else:
                nest.CopyModel("stdp_synapse", name, stdp_defaults)

        copy(f"stdp_cut_rge_{side}", wr_cut)
        copy(f"stdp_bs_rge_{side}", wr_bse)
        copy(f"stdp_bs_rgf_{side}", wr_bsf)
        copy(f"stdp_rge_me_{side}", wr_rge_me)
        copy(f"stdp_rgf_mf_{side}", wr_rgf_mf)

    # ----------------------------
    # Connect everything per leg
    # ----------------------------
    for side in LEGS:
        L = leg[side]

        # CUT -> RG-E
        nest.Connect(
            L["cut_in"], L["rg_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
            syn_spec={"synapse_model": f"stdp_cut_rge_{side}", "weight": W0_IN, "delay": DELAY_MS},
        )

        # BS -> RG-E / RG-F
        nest.Connect(
            L["bs_in_e"], L["rg_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
            syn_spec={"synapse_model": f"stdp_bs_rge_{side}", "weight": W0_IN, "delay": DELAY_MS},
        )
        nest.Connect(
            L["bs_in_f"], L["rg_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
            syn_spec={"synapse_model": f"stdp_bs_rgf_{side}", "weight": W0_IN, "delay": DELAY_MS},
        )

        # Baseline drive -> RG
        nest.Connect(
            L["base_in"], L["rg_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": BASE_DRIVE_P},
            syn_spec={"synapse_model": "static_synapse", "weight": BASE_DRIVE_W, "delay": DELAY_MS},
        )
        nest.Connect(
            L["base_in"], L["rg_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": BASE_DRIVE_P},
            syn_spec={"synapse_model": "static_synapse", "weight": BASE_DRIVE_W, "delay": DELAY_MS},
        )

        # RG -> motor
        nest.Connect(
            L["rg_e"], L["m_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
            syn_spec={"synapse_model": f"stdp_rge_me_{side}", "weight": W0_RM, "delay": DELAY_MS},
        )
        nest.Connect(
            L["rg_f"], L["m_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
            syn_spec={"synapse_model": f"stdp_rgf_mf_{side}", "weight": W0_RM, "delay": DELAY_MS},
        )

        # Motor -> muscle relay
        nest.Connect(
            L["m_e"], L["mus_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_M2MUS},
            syn_spec={"synapse_model": "static_synapse", "weight": W_M2MUS, "delay": DELAY_MS},
        )
        nest.Connect(
            L["m_f"], L["mus_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_M2MUS},
            syn_spec={"synapse_model": "static_synapse", "weight": W_M2MUS, "delay": DELAY_MS},
        )

        # Local RG recurrence
        nest.Connect(
            L["rg_e"], L["rg_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_REC},
            syn_spec={"synapse_model": "static_synapse", "weight": 8.0, "delay": DELAY_MS},
        )
        nest.Connect(
            L["rg_f"], L["rg_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_REC},
            syn_spec={"synapse_model": "static_synapse", "weight": 8.0, "delay": DELAY_MS},
        )

        # Reciprocal inhibition inside leg
        nest.Connect(
            L["rg_e"], L["rg_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_RECIP},
            syn_spec={"synapse_model": "static_synapse", "weight": W_RG_RECIP, "delay": DELAY_RECIP_MS},
        )
        nest.Connect(
            L["rg_f"], L["rg_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_RECIP},
            syn_spec={"synapse_model": "static_synapse", "weight": W_RG_RECIP, "delay": DELAY_RECIP_MS},
        )

        # Ia -> RG feedback
        nest.Connect(
            L["ia_in_e"], L["rg_e"],
            conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
            syn_spec={"synapse_model": "static_synapse", "weight": IA2RG_W, "delay": DELAY_MS},
        )
        nest.Connect(
            L["ia_in_f"], L["rg_f"],
            conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
            syn_spec={"synapse_model": "static_synapse", "weight": IA2RG_W, "delay": DELAY_MS},
        )

        # Optional static parallel paths
        if USE_STATIC_PARALLEL:
            nest.Connect(
                L["bs_in_e"], L["rg_e"],
                conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_IN},
                syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_IN, "delay": DELAY_MS},
            )
            nest.Connect(
                L["bs_in_f"], L["rg_f"],
                conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_IN},
                syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_IN, "delay": DELAY_MS},
            )
            nest.Connect(
                L["cut_in"], L["rg_e"],
                conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_IN},
                syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_IN, "delay": DELAY_MS},
            )
            nest.Connect(
                L["rg_e"], L["m_e"],
                conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_RM},
                syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_RM, "delay": DELAY_MS},
            )
            nest.Connect(
                L["rg_f"], L["m_f"],
                conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_RM},
                syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_RM, "delay": DELAY_MS},
            )

    # ----------------------------
    # Commissural coupling (between legs)
    # ----------------------------
    if ENABLE_COMMISSURAL:
        L = leg["L"]
        R = leg["R"]
        nest.Connect(L["rg_e"], R["rg_f"],
                     conn_spec={"rule": "pairwise_bernoulli", "p": P_COMM},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_COMM_INH, "delay": DELAY_COMM_MS})
        nest.Connect(R["rg_e"], L["rg_f"],
                     conn_spec={"rule": "pairwise_bernoulli", "p": P_COMM},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_COMM_INH, "delay": DELAY_COMM_MS})
        nest.Connect(L["rg_f"], R["rg_e"],
                     conn_spec={"rule": "pairwise_bernoulli", "p": P_COMM},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_COMM_INH, "delay": DELAY_COMM_MS})
        nest.Connect(R["rg_f"], L["rg_e"],
                     conn_spec={"rule": "pairwise_bernoulli", "p": P_COMM},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_COMM_INH, "delay": DELAY_COMM_MS})

    # ----------------------------
    # Weight sampling helper
    # ----------------------------
    def sample_w(model_name: str) -> np.ndarray:
        conns = nest.GetConnections(synapse_model=model_name)
        if len(conns) == 0:
            return np.array([], dtype=float)
        return np.asarray(nest.GetStatus(conns, "weight"), dtype=float)

    # ----------------------------
    # Closed-loop states per leg + logs
    # ----------------------------
    state = {}
    for side in LEGS:
        state[side] = dict(
            act_e=0.0, act_f=0.0,
            force_e=0.0, force_f=0.0,
            len_e=L0, len_f=L0,
            last_muse=0, last_musf=0,
        )

    times = []
    wstats = {side: {k: ([], []) for k in ["cut->rge", "bs->rge", "bs->rgf", "rge->me", "rgf->mf"]} for side in LEGS}

    bs_rate_e = {side: [] for side in LEGS}
    bs_rate_f = {side: [] for side in LEGS}
    mus_rate_e = {side: [] for side in LEGS}
    mus_rate_f = {side: [] for side in LEGS}
    act_e_tr = {side: [] for side in LEGS}
    act_f_tr = {side: [] for side in LEGS}
    force_e_tr = {side: [] for side in LEGS}
    force_f_tr = {side: [] for side in LEGS}
    len_e_tr = {side: [] for side in LEGS}
    len_f_tr = {side: [] for side in LEGS}
    ia_e_tr = {side: [] for side in LEGS}
    ia_f_tr = {side: [] for side in LEGS}

    def new_spikes(rec, last_len):
        ev = nest.GetStatus(rec, "events")[0]
        cur = len(ev["times"])
        return cur - last_len, cur

    def update_leg(side: str, t_ms: float, cut_active_frac: float):
        dt_s = SAMPLE_DT_MS / 1000.0
        L = leg[side]
        S = state[side]

        # BS rates
        r_e, r_f = bs_rates_counterphase(t_ms, side)
        nest.SetStatus(L["bs_pg_e"], {"rate": r_e})
        nest.SetStatus(L["bs_pg_f"], {"rate": r_f})
        bs_rate_e[side].append(r_e)
        bs_rate_f[side].append(r_f)

        # Muscle spikes -> rates
        sp_e, cur_e = new_spikes(L["rec_muse"], S["last_muse"])
        sp_f, cur_f = new_spikes(L["rec_musf"], S["last_musf"])
        S["last_muse"] = cur_e
        S["last_musf"] = cur_f

        r_muse = (sp_e / max(1, N_MUS_E)) / dt_s
        r_musf = (sp_f / max(1, N_MUS_F)) / dt_s
        mus_rate_e[side].append(r_muse)
        mus_rate_f[side].append(r_musf)

        # Activation LPF (from muscle relays)
        tauA_s = TAU_ACT_MS / 1000.0
        target_ae = clamp(ACT_GAIN * r_muse, 0.0, ACT_MAX)
        target_af = clamp(ACT_GAIN * r_musf, 0.0, ACT_MAX)
        S["act_e"] += (dt_s / tauA_s) * (target_ae - S["act_e"])
        S["act_f"] += (dt_s / tauA_s) * (target_af - S["act_f"])

        # Force
        target_fe = FORCE_MAX * (1.0 - np.exp(-FORCE_SAT_K * S["act_e"]))
        target_ff = FORCE_MAX * (1.0 - np.exp(-FORCE_SAT_K * S["act_f"]))

        tau_rise_s = TAU_FORCE_RISE_MS / 1000.0
        tau_decay_s = TAU_FORCE_DECAY_MS / 1000.0

        if target_fe > S["force_e"]:
            S["force_e"] += (dt_s / tau_rise_s) * (target_fe - S["force_e"])
        else:
            S["force_e"] += (dt_s / tau_decay_s) * (target_fe - S["force_e"])

        if target_ff > S["force_f"]:
            S["force_f"] += (dt_s / tau_rise_s) * (target_ff - S["force_f"])
        else:
            S["force_f"] += (dt_s / tau_decay_s) * (target_ff - S["force_f"])

        S["force_e"] = clamp(S["force_e"], 0.0, FORCE_MAX)
        S["force_f"] = clamp(S["force_f"], 0.0, FORCE_MAX)

        # Length
        tauL_s = TAU_LENGTH_MS / 1000.0
        S["len_e"] += (dt_s / tauL_s) * (L0 - S["len_e"])
        S["len_f"] += (dt_s / tauL_s) * (L0 - S["len_f"])

        S["len_e"] -= SHORTEN_GAIN * S["force_e"] * dt_s
        S["len_f"] -= SHORTEN_GAIN * S["force_f"] * dt_s

        if cut_active_frac > 0.0:
            S["len_e"] += STRETCH_GAIN * cut_active_frac * dt_s

        S["len_e"] = clamp(S["len_e"], L_MIN, L_MAX)
        S["len_f"] = clamp(S["len_f"], L_MIN, L_MAX)

        # Ia rates (NO SINUS)
        stretch_e = max(0.0, S["len_e"] - L0)
        stretch_f = max(0.0, S["len_f"] - L0)

        ia_e = IA_BASE_HZ + IA_K_FORCE * S["force_e"] + IA_K_STRETCH * stretch_e
        ia_f = IA_BASE_HZ + IA_K_FORCE * S["force_f"] + IA_K_STRETCH * stretch_f
        ia_e = clamp(ia_e, 0.0, IA_RATE_MAX_HZ)
        ia_f = clamp(ia_f, 0.0, IA_RATE_MAX_HZ)

        nest.SetStatus(L["ia_pg_e"], {"rate": ia_e})
        nest.SetStatus(L["ia_pg_f"], {"rate": ia_f})

        # logs
        act_e_tr[side].append(S["act_e"])
        act_f_tr[side].append(S["act_f"])
        force_e_tr[side].append(S["force_e"])
        force_f_tr[side].append(S["force_f"])
        len_e_tr[side].append(S["len_e"])
        len_f_tr[side].append(S["len_f"])
        ia_e_tr[side].append(ia_e)
        ia_f_tr[side].append(ia_f)

    def log_weights(t_ms: float):
        times.append(t_ms)
        for side in LEGS:
            def push(model, key):
                w = sample_w(model)
                if w.size == 0:
                    wstats[side][key][0].append(np.nan)
                    wstats[side][key][1].append(np.nan)
                else:
                    wstats[side][key][0].append(float(w.mean()))
                    wstats[side][key][1].append(float(w.std()))
            push(f"stdp_cut_rge_{side}", "cut->rge")
            push(f"stdp_bs_rge_{side}", "bs->rge")
            push(f"stdp_bs_rgf_{side}", "bs->rgf")
            push(f"stdp_rge_me_{side}", "rge->me")
            push(f"stdp_rgf_mf_{side}", "rgf->mf")

    # ----------------------------
    # Run: chunked CUT phases (applied to BOTH legs)
    # ----------------------------
    chunk = max(1, int(N_CUT / N_PHASES))
    t = 0.0

    for phase in range(N_PHASES):
        for side in LEGS:
            nest.SetStatus(leg[side]["cut_pg"], {"rate": CUT_RATE_OFF_HZ})
        start = phase * chunk
        end = min(N_CUT, (phase + 1) * chunk)
        for side in LEGS:
            nest.SetStatus(leg[side]["cut_pg"][start:end], {"rate": CUT_RATE_ON_HZ})

        cut_active_frac = float(end - start) / float(N_CUT)

        n_steps = int(PHASE_MS / SAMPLE_DT_MS)
        for _ in range(n_steps):
            nest.Simulate(SAMPLE_DT_MS)
            t += SAMPLE_DT_MS
            for side in LEGS:
                update_leg(side, t, cut_active_frac)
            log_weights(t)

    times_arr = np.asarray(times)

    # ----------------------------
    # Plots
    # ----------------------------
    plt.figure(figsize=(14, 5))
    plt.plot(times_arr, bs_rate_e["L"], label="BS E (L)")
    plt.plot(times_arr, bs_rate_f["L"], label="BS F (L)")
    plt.plot(times_arr, bs_rate_e["R"], label="BS E (R)")
    plt.plot(times_arr, bs_rate_f["R"], label="BS F (R)")
    plt.xlabel("time (ms)")
    plt.ylabel("Hz")
    plt.title("Brainstem drive (counter-phase within legs, phase-shift between legs)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for side in LEGS:
        plt.figure(figsize=(14, 7))
        for key in ["cut->rge", "bs->rge", "bs->rgf"]:
            m = np.asarray(wstats[side][key][0])
            s = np.asarray(wstats[side][key][1])
            plt.plot(times_arr, m, label=f"{key} mean ({side})")
            plt.fill_between(times_arr, m - s, m + s, alpha=0.15)
        plt.xlabel("time (ms)")
        plt.ylabel("weight (pA)")
        plt.title(f"STDP learning — inputs (leg {side}) mean ± std")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(times_arr, np.asarray(wstats["L"]["rge->me"][0]), label="L rge->me mean")
    plt.plot(times_arr, np.asarray(wstats["L"]["rgf->mf"][0]), label="L rgf->mf mean")
    plt.plot(times_arr, np.asarray(wstats["R"]["rge->me"][0]), label="R rge->me mean")
    plt.plot(times_arr, np.asarray(wstats["R"]["rgf->mf"][0]), label="R rgf->mf mean")
    plt.xlabel("time (ms)")
    plt.ylabel("weight (pA)")
    plt.title("STDP learning — motor synapses (means)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(times_arr, mus_rate_e["L"], label="mus-E rate L")
    plt.plot(times_arr, mus_rate_f["L"], label="mus-F rate L")
    plt.plot(times_arr, mus_rate_e["R"], label="mus-E rate R")
    plt.plot(times_arr, mus_rate_f["R"], label="mus-F rate R")
    plt.xlabel("time (ms)")
    plt.ylabel("Hz/neuron")
    plt.title("Muscle relay rates (from motor spikes)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(times_arr, act_e_tr["L"], label="Act E L")
    plt.plot(times_arr, act_f_tr["L"], label="Act F L")
    plt.plot(times_arr, act_e_tr["R"], label="Act E R")
    plt.plot(times_arr, act_f_tr["R"], label="Act F R")
    plt.xlabel("time (ms)")
    plt.ylabel("a.u.")
    plt.title("Activation proxies (from muscle relays)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(times_arr, force_e_tr["L"], label="Force E L")
    plt.plot(times_arr, force_f_tr["L"], label="Force F L")
    plt.plot(times_arr, force_e_tr["R"], label="Force E R")
    plt.plot(times_arr, force_f_tr["R"], label="Force F R")
    plt.xlabel("time (ms)")
    plt.ylabel("force (a.u.)")
    plt.title("Force proxies")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(times_arr, len_e_tr["L"], label="Len E L")
    plt.plot(times_arr, len_f_tr["L"], label="Len F L")
    plt.plot(times_arr, len_e_tr["R"], label="Len E R")
    plt.plot(times_arr, len_f_tr["R"], label="Len F R")
    plt.axhline(L0, linestyle="--", linewidth=1)
    plt.xlabel("time (ms)")
    plt.ylabel("length (a.u.)")
    plt.title("Length proxies (E also stretched by CUT)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(times_arr, ia_e_tr["L"], label="Ia-E L")
    plt.plot(times_arr, ia_f_tr["L"], label="Ia-F L")
    plt.plot(times_arr, ia_e_tr["R"], label="Ia-E R")
    plt.plot(times_arr, ia_f_tr["R"], label="Ia-F R")
    plt.xlabel("time (ms)")
    plt.ylabel("Hz")
    plt.title("Ia generator rates (no sinus; force + length only)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()