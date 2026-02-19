#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpg_2legs_nest_to_hdf5.py
Run the 2-leg CPG NEST simulation headlessly (HPC-friendly) and save all time-series
(and basic network stats) into an HDF5 file for later plotting on a local machine.

Example:
  python3 cpg_2legs_nest_to_hdf5.py --out cpg_run.h5 --sim-ms 60000 --dt-ms 10 --threads 10 --long-run

If your NEST build supports MPI, launch with mpirun/srun externally.
Only rank 0 writes the .h5 file.
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import h5py
import nest


def get_kernel_parallel_status(nest_mod):
    """Return (mpi_procs, local_threads) without assuming specific kernel keys."""
    ks = nest_mod.GetKernelStatus()
    mpi_procs = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
    local_threads = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
    return int(mpi_procs), int(local_threads)


LEGS = ("L", "R")

# ---------- sizes ----------
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

# Interneurons (per leg) to match physiological motifs in the schematic
N_IA_INT = 50  # inhibitory interneurons driven by Ia afferents
N_INE = 50  # inhibitory interneurons mediating RG-E -> RG-F inhibition
N_INF = 50  # inhibitory interneurons mediating RG-F -> RG-E inhibition

# Synaptic weights (tune as needed)
W_IA_IN2INT = 6.0  # Ia parrot -> Ia inhibitory interneuron (excitatory synapse)
W_IA_INT2ANT = -10.0  # Ia inhibitory interneuron -> antagonist motor pool (inhibitory synapse)

W_RG2IN = 8.0  # RG -> inhibitory interneuron (excitatory synapse)
W_IN2RG = -18.0  # inhibitory interneuron -> RG partner (inhibitory synapse)

# ---------- CUT training ----------
N_PHASES = 6
CUT_RATE_ON_HZ = 200.0
CUT_RATE_OFF_HZ = 0.0

# ---------- brainstem ----------
BS_OSC_HZ = 1.0
BS_RATE_BASE_HZ = 0.0
BS_RATE_AMP_HZ = 300.0
BS_RATE_MIN_HZ = 0.0
BS_PHASE = {"L": 0.0, "R": np.pi}  # left-right alternation

# ---------- connectivity ----------
P_IN_STDP = 0.5
P_RG_REC = 0.12
DELAY_MS = 1.0

P_RG_RECIP = 0.20
W_RG_RECIP = -18.0
DELAY_RECIP_MS = 1.0

# Motor-pool reciprocal inhibition (extra safeguard against E/F co-activation)
P_MOTOR_RECIP = 0.25
W_MOTOR_RECIP = -22.0
DELAY_MOTOR_RECIP_E2F_MS = 1.5
DELAY_MOTOR_RECIP_F2E_MS = 1.0

W_M2MUS = 1.0
P_M2MUS = 0.8

IA2RG_P = 0.4
IA2RG_W = 12.0

BASE_DRIVE_HZ = 2.0
BASE_DRIVE_W = 1.0
BASE_DRIVE_P = 0.10

USE_STATIC_PARALLEL = False
P_STATIC_IN = 0.03
P_STATIC_RM = 0.03
W_STATIC_IN = 22.0
W_STATIC_RM = 35.0

ENABLE_COMMISSURAL = True
P_COMM = 0.08
W_COMM_INH = -10.0
DELAY_COMM_MS = 1.0

# ---------- STDP ----------
TAU_PLUS = 20.0
# Lower learning rate and mild LTP/LTD balance + multiplicative STDP to reduce hard-boundary pileups at 0/Wmax
LAMBDA = 0.001
ALPHA = 0.95
MU_PLUS = 0.4
MU_MINUS = 0.4
WMAX = 120.0

W0_IN = 22.0
W0_RM = 30.0

# ---------- Izhikevich ----------
izh_params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, V_th=30.0, V_min=-120.0)
izh_inh_params = dict(a=0.1, b=0.2, c=-65.0, d=2.0, V_th=30.0, V_min=-120.0)  # UPDATED_v7
I_E_RG = 1.0

# Izhikevich "chattering" (bursting-like) parameters for RG-F excitatory neurons
# (Izhikevich 2003/2004 canonical set)
RGF_A = 0.02
RGF_B = 0.2
RGF_C = -50.0
RGF_D = 2.0
I_E_MOTOR = 1.0

# ---------- muscle proxies ----------
# Activation proxy: saturating nonlinearity + brainstem gating to avoid saturation & enforce timing
TAU_ACT_RISE_MS = 60.0
TAU_ACT_DECAY_MS = 35.0
ACT_MAX = 1.2
ACT_SAT_K = 5e-4          # slope for activation from muscle relay rate
ACT_GATE_POWER = 1.0      # >1.0 makes windows narrower around BS peaks

TAU_FORCE_RISE_MS = 140.0
TAU_FORCE_DECAY_MS = 60.0
FORCE_MAX = 25.0
FORCE_SAT_K = 2.5

TAU_LENGTH_MS = 260.0
L0 = 1.0
L_MIN, L_MAX = 0.5, 2.0
SHORTEN_GAIN = 0.010
STRETCH_GAIN = 0.35  # extensor-only stretch from CUT fraction

# ---------- Ia ----------
IA_BASE_HZ = 10.0
IA_K_FORCE = 6.0
IA_K_STRETCH = 250.0
IA_RATE_MAX_HZ = 500.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def bs_rates_counterphase(t_ms: float, leg: str) -> tuple[float, float]:
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


def safe_len_connections(**kwargs) -> int:
    try:
        return len(nest.GetConnections(**kwargs))
    except Exception:
        return -1


def synapse_sign_stats():
    conns = nest.GetConnections()
    if len(conns) == 0:
        return dict(total=0, exc=0, inh=0)
    w = np.array(nest.GetStatus(conns, "weight"), dtype=float)
    return dict(total=int(w.size), exc=int(np.sum(w >= 0.0)), inh=int(np.sum(w < 0.0)))


def node_model_counts(models):
    out = {}
    for m in models:
        try:
            out[m] = int(len(nest.GetNodes(properties={"model": m})[0]))
        except Exception:
            out[m] = -1
    return out


def sample_w(model_name: str) -> np.ndarray:
    conns = nest.GetConnections(synapse_model=model_name)
    if len(conns) == 0:
        return np.array([], dtype=float)
    return np.asarray(nest.GetStatus(conns, "weight"), dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="cpg_run.h5")
    ap.add_argument("--sim-ms", type=float, default=10000.0)
    ap.add_argument("--dt-ms", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--print-every", type=int, default=50, help="progress cadence in steps")
    ap.add_argument("--weight-sample-ms", type=float, default=100.0,
                    help="How often to sample STDP weights (ms). Larger = faster.")
    ap.add_argument("--rate-update-ms", type=float, default=20.0,
                    help="How often to push updated rates to poisson_generators (ms). Larger = faster.")
    ap.add_argument("--resolution-ms", type=float, default=0.2,
                    help="NEST kernel resolution (ms). Larger = faster, but less precise. Try 0.2, 0.5, 1.0.")
    ap.add_argument("--simulate-chunk-ms", type=float, default=50.0,
                    help="Simulate in larger chunks to reduce Python<->NEST call overhead (ms). Must be >= dt-ms. Try 50 or 100.")
    ap.add_argument("--long-run", action="store_true",
                    help="Enable long-run defaults (aimed at >=30s sims): coarser chunking, less frequent sampling, and weight downsampling for trend plots.")
    ap.add_argument("--max-weight-conns", type=int, default=0,
                    help="If >0, downsample each projection's connection list to at most this many connections when computing weight mean/std (trend mode speed-up).")
    ap.add_argument("--save-weights", type=str, default="snapshots", choices=["none", "final", "snapshots"],
                    help="Save full weight vectors for plastic projections: none=only mean/std; final=store initial+final full vectors; snapshots=store full vectors at each weight sample tick (can be large).")
    ap.add_argument("--stdp-winit-dist", type=str, default="lognormal",
                    choices=["const", "normal", "lognormal"],
                    help="Initial weight distribution for STDP synapses. const=all weights=W0_IN; normal/lognormal draw per-connection weights.")
    ap.add_argument("--stdp-winit-mean", type=float, default=W0_IN,
                    help="Target mean (approx.) for initial STDP weights.")
    ap.add_argument("--stdp-winit-std", type=float, default=0.5,
                    help="Std parameter for initial STDP weights. For normal: std in weight units. For lognormal: sigma of underlying normal.")
    ap.add_argument("--stdp-winit-min", type=float, default=0.0,
                    help="Lower bound for initial STDP weights (used via redraw/clipping).")
    ap.add_argument("--stdp-winit-max", type=float, default=WMAX,
                    help="Upper bound for initial STDP weights (used via redraw/clipping).")
    ap.add_argument("--nest-verbosity", type=str, default="M_ERROR",
                    help="NEST verbosity level to reduce slurmout I/O. Try M_ERROR or M_WARNING.")
    args = ap.parse_args()
    # --- STDP randomized initial weights helper ---
    def make_stdp_init_weight_param(dist: str, mean_w: float, std_w: float, wmin: float, wmax: float):
        """Return either a scalar or a NEST Parameter for per-connection initial weights.

        Bio-plausible default: lognormal (positive, heavy-tailed). We enforce bounds using
        clipping (not redraw) for speed and reliability during Connect.

        Notes:
          - For `normal`: std_w is in weight units.
          - For `lognormal`: std_w is sigma of the underlying normal distribution.
            We choose mu so that E[w] ~= mean_w (before redraw/clipping): mu = ln(mean_w) - 0.5*sigma^2.
        """
        dist = str(dist).lower().strip()
        mean_w = float(mean_w)
        std_w = float(std_w)
        wmin = float(wmin)
        wmax = float(wmax)

        if dist == "const":
            return float(mean_w)

        if dist == "normal":
            p = nest.random.normal(mean=mean_w, std=max(1e-12, std_w))

        elif dist == "lognormal":
            # IMPORTANT: NEST's lognormal parameterization can vary by version.
            # We try the underlying-normal (mu/sigma) form first, then fall back.
            if mean_w <= 0.0:
                return float(wmin)

            sigma = max(1e-12, std_w)
            mu = float(np.log(max(1e-12, mean_w)) - 0.5 * sigma * sigma)

            p = None
            try:
                # Newer-style / explicit parameterization
                p = nest.random.lognormal(mu=mu, sigma=sigma)
            except Exception:
                try:
                    # Older-style signature might still accept mean/std keywords
                    p = nest.random.lognormal(mean=mu, std=sigma)
                except Exception:
                    try:
                        # Last resort: interpret args as mean/std of the *lognormal* itself
                        p = nest.random.lognormal(mean=mean_w, std=sigma)
                    except Exception:
                        return float(mean_w)

        else:
            # Safe fallback
            return float(mean_w)

        # Enforce biologically sensible bounds without redraw (redraw can crash during Connect)
        # We prefer hard clipping via NEST math combinators so weight sampling never retries.
        if wmin > -np.inf or wmax < np.inf:
            try:
                # Try a dedicated clip if available (version-dependent)
                if hasattr(nest.math, "clip"):
                    p = nest.math.clip(p, min=wmin, max=wmax)
                else:
                    # Generic: p <- min(max(p, wmin), wmax)
                    if wmin > -np.inf:
                        p = nest.math.max(p, wmin)
                    if wmax < np.inf:
                        p = nest.math.min(p, wmax)
            except Exception:
                # As a last resort, skip bounding rather than failing the run
                pass

        return p

    # STDP initial weight parameter (scalar or per-connection random Parameter)
    W_INIT_IN = make_stdp_init_weight_param(
        args.stdp_winit_dist,
        args.stdp_winit_mean,
        args.stdp_winit_std,
        args.stdp_winit_min,
        min(float(args.stdp_winit_max), float(WMAX)),
    )
    # --- NEST verbosity (reduce log spam / slurmout I/O) ---
    try:
        nest.set_verbosity(str(args.nest_verbosity))
    except Exception:
        # Fall back silently if verbosity string is not supported
        pass

    # --- Long-run mode: reduce Python<->NEST overhead and weight sampling cost ---
    # For long simulations, the dominant cost is often weight sampling (GetStatus on many connections).
    # Long-run mode shifts toward "trend" logging rather than high-frequency snapshots.
    if args.long_run:
        # Coarser outer chunking reduces the number of nest.Simulate() calls.
        if args.simulate_chunk_ms < 100.0:
            args.simulate_chunk_ms = 100.0
        # Coarser rate updates reduce frequent SetStatus calls.
        if args.rate_update_ms < 100.0:
            args.rate_update_ms = 100.0
        # Coarser weight sampling for trend plots.
        if args.weight_sample_ms < 1000.0:
            args.weight_sample_ms = 1000.0
        # Default weight downsampling (if not explicitly set)
        if int(args.max_weight_conns) <= 0:
            args.max_weight_conns = 2000
        # Reduce progress print cadence (in steps) so output doesn't grow too much.
        if args.print_every < 200:
            args.print_every = 200

    SIM_MS = float(args.sim_ms)
    DT_MS = float(args.dt_ms)
    PHASE_MS = SIM_MS / int(N_PHASES)

    # We will call nest.Simulate() in larger chunks to reduce overhead.
    CHUNK_MS = float(args.simulate_chunk_ms)
    RES_MS = float(args.resolution_ms)

    def q_ms(x: float) -> float:
        """Quantize time to an integer multiple of the NEST resolution."""
        if RES_MS <= 0.0:
            return float(x)
        steps = int(round(float(x) / RES_MS))
        return steps * RES_MS

    # Ensure chunk is a clean multiple of resolution
    CHUNK_MS = q_ms(CHUNK_MS)
    if CHUNK_MS <= 0.0:
        raise ValueError(f"--simulate-chunk-ms quantized to {CHUNK_MS}, choose a larger value.")

    if CHUNK_MS < DT_MS:
        raise ValueError(f"--simulate-chunk-ms ({CHUNK_MS}) must be >= --dt-ms ({DT_MS}).")

    # Convert sampling cadences (ms) into "chunk steps"
    weight_every = max(1, int(round(float(args.weight_sample_ms) / CHUNK_MS)))
    rate_every = max(1, int(round(float(args.rate_update_ms) / CHUNK_MS)))

    nest.ResetKernel()
    nest.SetKernelStatus(
        {"resolution": float(args.resolution_ms), "local_num_threads": int(args.threads), "print_time": False})

    # Robust rank/proc detection:
    # - under Slurm, SLURM_PROCID/SLURM_NTASKS are the most reliable
    # - otherwise, fall back to NEST helpers if present
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        nproc = int(os.environ.get("SLURM_NTASKS", "1"))
    else:
        rank = getattr(nest, "Rank", lambda: 0)()
        nproc = getattr(nest, "NumProcesses", lambda: 1)()

    if rank == 0:
        print(f"[NEST] processes={nproc} | local_threads={nest.GetKernelStatus('local_num_threads')}")
        print(
            f"[Run] sim_ms={SIM_MS} dt_ms={DT_MS} chunk_ms={CHUNK_MS} resolution_ms={float(args.resolution_ms)} phases={N_PHASES} phase_ms={PHASE_MS:.2f}")
        print(f"[STDP init] dist={args.stdp_winit_dist} mean={args.stdp_winit_mean} std={args.stdp_winit_std} min={args.stdp_winit_min} max={min(float(args.stdp_winit_max), float(WMAX))}")

    # ---- build per-leg ----
    leg = {}
    for side in LEGS:
        cut_pg = nest.Create("poisson_generator", N_CUT)
        cut_in = nest.Create("parrot_neuron", N_CUT)
        nest.Connect(cut_pg, cut_in, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(cut_pg, {"rate": CUT_RATE_OFF_HZ})

        bs_pg_e = nest.Create("poisson_generator", N_BS)
        bs_in_e = nest.Create("parrot_neuron", N_BS)
        nest.Connect(bs_pg_e, bs_in_e, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(bs_pg_e, {"rate": BS_RATE_BASE_HZ})

        bs_pg_f = nest.Create("poisson_generator", N_BS)
        bs_in_f = nest.Create("parrot_neuron", N_BS)
        nest.Connect(bs_pg_f, bs_in_f, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(bs_pg_f, {"rate": BS_RATE_BASE_HZ})

        base_pg = nest.Create("poisson_generator", N_BS)
        base_in = nest.Create("parrot_neuron", N_BS)
        nest.Connect(base_pg, base_in, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(base_pg, {"rate": BASE_DRIVE_HZ})

        ia_pg_e = nest.Create("poisson_generator", N_IA_E)
        ia_in_e = nest.Create("parrot_neuron", N_IA_E)
        nest.Connect(ia_pg_e, ia_in_e, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(ia_pg_e, {"rate": IA_BASE_HZ})

        ia_pg_f = nest.Create("poisson_generator", N_IA_F)
        ia_in_f = nest.Create("parrot_neuron", N_IA_F)
        nest.Connect(ia_pg_f, ia_in_f, conn_spec={"rule": "one_to_one"})
        nest.SetStatus(ia_pg_f, {"rate": IA_BASE_HZ})

        rg_e = nest.Create("izhikevich", N_RG_E)
        rg_f = nest.Create("izhikevich", N_RG_F)
        m_e = nest.Create("izhikevich", N_MOTOR_E)
        m_f = nest.Create("izhikevich", N_MOTOR_F)
        # Record RG population spiking to compute population rates (like muscle relay rates)
        rec_rge = nest.Create("spike_recorder")
        rec_rgf = nest.Create("spike_recorder")
        nest.Connect(rg_e, rec_rge)
        nest.Connect(rg_f, rec_rgf)
        # Interneurons
        ia_int_e = nest.Create("izhikevich", N_IA_INT)  # inhibitory
        ia_int_f = nest.Create("izhikevich", N_IA_INT)  # inhibitory
        in_e = nest.Create("izhikevich", N_INE)  # inhibitory
        in_f = nest.Create("izhikevich", N_INF)  # inhibitory
        for pop in (rg_e, rg_f, m_e, m_f):
            nest.SetStatus(pop, izh_params)
        for pop in (ia_int_e, ia_int_f, in_e, in_f):
            nest.SetStatus(pop, izh_inh_params)
        nest.SetStatus(rg_e, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_RG})
        nest.SetStatus(rg_f, {"a": RGF_A, "b": RGF_B, "c": RGF_C, "d": RGF_D,
                              "V_m": -65.0, "U_m": RGF_B * (-65.0), "I_e": I_E_RG})
        nest.SetStatus(m_e, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_MOTOR})
        nest.SetStatus(m_f, {"V_m": -65.0, "U_m": 0.2 * (-65.0), "I_e": I_E_MOTOR})

        mus_e = nest.Create("parrot_neuron", N_MUS_E)
        mus_f = nest.Create("parrot_neuron", N_MUS_F)
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
            rg_e=rg_e, rg_f=rg_f, m_e=m_e, m_f=m_f,
            ia_int_e=ia_int_e, ia_int_f=ia_int_f, in_e=in_e, in_f=in_f,
            mus_e=mus_e, mus_f=mus_f,
            rec_muse=rec_muse, rec_musf=rec_musf,
            rec_rge=rec_rge, rec_rgf=rec_rgf
        )

    # ---- STDP models ----
    stdp_defaults = {
        "tau_plus": TAU_PLUS,
        "lambda": LAMBDA,  # <-- key is a string, so it's fine
        "alpha": ALPHA,
        "mu_plus": MU_PLUS,
        "mu_minus": MU_MINUS,
        "Wmax": WMAX,
    }

    for side in LEGS:
        def copy(name, wr):
            if wr is not None:
                nest.CopyModel("stdp_synapse", name, {**stdp_defaults, "weight_recorder": wr})
            else:
                nest.CopyModel("stdp_synapse", name, stdp_defaults)

        copy(f"stdp_cut_rge_{side}", make_weight_recorder_safe())
        copy(f"stdp_bs_rge_{side}", make_weight_recorder_safe())
        copy(f"stdp_bs_rgf_{side}", make_weight_recorder_safe())

    # ---- connect per leg ----
    for side in LEGS:
        L = leg[side]

        nest.Connect(L["cut_in"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
                     syn_spec={"synapse_model": f"stdp_cut_rge_{side}", "weight": W_INIT_IN, "delay": DELAY_MS})

        nest.Connect(L["bs_in_e"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
                     syn_spec={"synapse_model": f"stdp_bs_rge_{side}", "weight": W_INIT_IN, "delay": DELAY_MS})
        nest.Connect(L["bs_in_f"], L["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
                     syn_spec={"synapse_model": f"stdp_bs_rgf_{side}", "weight": W_INIT_IN, "delay": DELAY_MS})

        nest.Connect(L["base_in"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": BASE_DRIVE_P},
                     syn_spec={"synapse_model": "static_synapse", "weight": BASE_DRIVE_W, "delay": DELAY_MS})
        nest.Connect(L["base_in"], L["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": BASE_DRIVE_P},
                     syn_spec={"synapse_model": "static_synapse", "weight": BASE_DRIVE_W, "delay": DELAY_MS})

        nest.Connect(L["rg_e"], L["m_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W0_RM, "delay": DELAY_MS})
        nest.Connect(L["rg_f"], L["m_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_IN_STDP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W0_RM, "delay": DELAY_MS})

        # Motor-pool reciprocal inhibition (helps enforce E/F alternation)
        nest.Connect(L["m_e"], L["m_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_MOTOR_RECIP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_MOTOR_RECIP,
                               "delay": DELAY_MOTOR_RECIP_E2F_MS})
        nest.Connect(L["m_f"], L["m_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_MOTOR_RECIP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_MOTOR_RECIP,
                               "delay": DELAY_MOTOR_RECIP_F2E_MS})

        nest.Connect(L["m_e"], L["mus_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_M2MUS},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_M2MUS, "delay": DELAY_MS})
        nest.Connect(L["m_f"], L["mus_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_M2MUS},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_M2MUS, "delay": DELAY_MS})

        # Ia afferent pathways via inhibitory interneurons:
        # - Ia from extensor inhibits flexor motor pool
        # - Ia from flexor inhibits extensor motor pool
        nest.Connect(L["ia_in_e"], L["ia_int_e"], conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_IA_IN2INT, "delay": DELAY_MS})
        nest.Connect(L["ia_int_e"], L["m_f"], conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_IA_INT2ANT, "delay": DELAY_MS})

        nest.Connect(L["ia_in_f"], L["ia_int_f"], conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_IA_IN2INT, "delay": DELAY_MS})
        nest.Connect(L["ia_int_f"], L["m_e"], conn_spec={"rule": "pairwise_bernoulli", "p": IA2RG_P},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_IA_INT2ANT, "delay": DELAY_MS})

        nest.Connect(L["rg_e"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_REC},
                     syn_spec={"synapse_model": "static_synapse", "weight": 8.0, "delay": DELAY_MS})
        nest.Connect(L["rg_f"], L["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_REC},
                     syn_spec={"synapse_model": "static_synapse", "weight": 8.0, "delay": DELAY_MS})

        # Reciprocal inhibition mediated by inhibitory interneurons (InE, InF)
        nest.Connect(L["rg_e"], L["in_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_RECIP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_RG2IN, "delay": DELAY_RECIP_MS})
        nest.Connect(L["in_e"], L["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_RECIP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_IN2RG, "delay": DELAY_RECIP_MS})

        nest.Connect(L["rg_f"], L["in_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_RECIP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_RG2IN, "delay": DELAY_RECIP_MS})
        nest.Connect(L["in_f"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_RG_RECIP},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_IN2RG, "delay": DELAY_RECIP_MS})

        if USE_STATIC_PARALLEL:
            nest.Connect(L["bs_in_e"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_IN},
                         syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_IN, "delay": DELAY_MS})
            nest.Connect(L["bs_in_f"], L["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_IN},
                         syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_IN, "delay": DELAY_MS})
            nest.Connect(L["cut_in"], L["rg_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_IN},
                         syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_IN, "delay": DELAY_MS})
            nest.Connect(L["rg_e"], L["m_e"], conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_RM},
                         syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_RM, "delay": DELAY_MS})
            nest.Connect(L["rg_f"], L["m_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_STATIC_RM},
                         syn_spec={"synapse_model": "static_synapse", "weight": W_STATIC_RM, "delay": DELAY_MS})

        # ---- commissural ----
    if ENABLE_COMMISSURAL:
        LL = leg["L"];
        RR = leg["R"]
        # Physiological simplification: flexor rhythm generators mutually inhibit across the midline
        nest.Connect(LL["rg_f"], RR["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_COMM},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_COMM_INH, "delay": DELAY_COMM_MS})
        nest.Connect(RR["rg_f"], LL["rg_f"], conn_spec={"rule": "pairwise_bernoulli", "p": P_COMM},
                     syn_spec={"synapse_model": "static_synapse", "weight": W_COMM_INH, "delay": DELAY_COMM_MS})

    # ---- stats (pre-sim) ----

    stats_nodes = node_model_counts(
        ["izhikevich", "parrot_neuron", "poisson_generator", "spike_recorder", "weight_recorder"])
    stats_syn_sign = synapse_sign_stats()
    stats_syn_models = {
        "L_stdp_cut_rge": safe_len_connections(synapse_model="stdp_cut_rge_L"),
        "L_stdp_bs_rge": safe_len_connections(synapse_model="stdp_bs_rge_L"),
        "L_stdp_bs_rgf": safe_len_connections(synapse_model="stdp_bs_rgf_L"),
        "R_stdp_cut_rge": safe_len_connections(synapse_model="stdp_cut_rge_R"),
        "R_stdp_bs_rge": safe_len_connections(synapse_model="stdp_bs_rge_R"),
        "R_stdp_bs_rgf": safe_len_connections(synapse_model="stdp_bs_rgf_R"),
        "static_total": safe_len_connections(synapse_model="static_synapse"),
    }
    if rank == 0:
        print("[Stats] node_models:", stats_nodes)
        print("[Stats] syn_sign:", stats_syn_sign)
        print("[Stats] syn_models:", stats_syn_models)
    # ---- cache connection collections for faster weight sampling ----
    # NOTE: in MPI runs, each rank sees (and caches) its local connections.
    conns_cache = {side: {} for side in LEGS}
    for side in LEGS:
        L = leg[side]
        # Plastic (STDP) connections cached by synapse model
        for key, model in [
            ("cut->rge", f"stdp_cut_rge_{side}"),
            ("bs->rge", f"stdp_bs_rge_{side}"),
            ("bs->rgf", f"stdp_bs_rgf_{side}"),
        ]:
            try:
                conns_cache[side][key] = nest.GetConnections(synapse_model=model)
            except Exception:
                conns_cache[side][key] = []

    # Keep an unmodified cache for full-weight saving (not downsampled)
    conns_full_cache = {side: {k: conns_cache[side][k] for k in conns_cache[side].keys()} for side in LEGS}

    # Cache endpoints once for full-weight saving
    conns_endpoints = {side: {} for side in LEGS}
    for side in LEGS:
        for key, conns in conns_full_cache[side].items():
            try:
                if conns is None or len(conns) == 0:
                    conns_endpoints[side][key] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
                else:
                    src = np.asarray(nest.GetStatus(conns, "source"), dtype=np.int64)
                    tgt = np.asarray(nest.GetStatus(conns, "target"), dtype=np.int64)
                    conns_endpoints[side][key] = (src, tgt)
            except Exception:
                conns_endpoints[side][key] = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    # Optional connection downsampling for faster weight trend stats (mean/std)
    # This reduces the size of the weight arrays pulled via nest.GetStatus(conns, "weight")
    # without changing the simulated network.
    max_w_conns = int(getattr(args, "max_weight_conns", 0) or 0)
    if max_w_conns > 0:
        for side in LEGS:
            for key, conns in conns_cache[side].items():
                try:
                    if conns is not None and len(conns) > max_w_conns:
                        # ConnectionCollection supports slicing in NEST 3.x
                        conns_cache[side][key] = conns[:max_w_conns]
                except Exception:
                    # If slicing is not supported, keep original
                    pass
    # ---- storage ----
    times = []
    wstats = {side: {k: ([], []) for k in ["cut->rge", "bs->rge", "bs->rgf"]} for side in LEGS}
    logs = {side: dict(bs_e=[], bs_f=[], mus_e=[], mus_f=[],
                       rge=[], rgf=[],
                       act_e=[], act_f=[], force_e=[], force_f=[],
                       len_e=[], len_f=[], ia_e=[], ia_f=[]) for side in LEGS}
    state = {side: dict(act_e=0.0, act_f=0.0, force_e=0.0, force_f=0.0,
                        len_e=L0, len_f=L0,
                        last_muse=0, last_musf=0,
                        last_rge=0, last_rgf=0) for side in LEGS}

    # Optional full-weight storage (final or snapshots)
    wfull_times = []
    wfull = {side: {k: [] for k in ["cut->rge", "bs->rge", "bs->rgf"]} for side in LEGS}

    # If only "final" weights are requested, also capture the INITIAL full weight vectors at t=0.
    # This prevents downstream plotting code (e.g., quantile bands over time) from seeing only a
    # single timepoint and producing empty/degenerate plots.
    if rank == 0 and args.save_weights == "final":
        wfull_times.append(0.0)
        for side in LEGS:
            for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                conns = conns_full_cache[side][key]
                if conns is None or len(conns) == 0:
                    wfull[side][key].append(np.array([], dtype=np.float32))
                else:
                    w = np.asarray(nest.GetStatus(conns, "weight"), dtype=np.float32)
                    wfull[side][key].append(w)

    def new_spikes(rec, last_n):
        # Fast, constant-memory spike counting
        cur = int(nest.GetStatus(rec, "n_events")[0])
        return cur - last_n, cur

    def update_leg(side: str, t_ms: float, dt_ms_actual: float, cut_active_frac: float, do_rate_update: bool):
        dt_s = float(dt_ms_actual) / 1000.0
        L = leg[side]
        S = state[side]
        P = logs[side]

        r_e, r_f = bs_rates_counterphase(t_ms, side)
        if do_rate_update:
            nest.SetStatus(L["bs_pg_e"], {"rate": r_e})
            nest.SetStatus(L["bs_pg_f"], {"rate": r_f})
        P["bs_e"].append(r_e);
        P["bs_f"].append(r_f)

        sp_e, cur_e = new_spikes(L["rec_muse"], S["last_muse"])
        sp_f, cur_f = new_spikes(L["rec_musf"], S["last_musf"])
        S["last_muse"] = cur_e;
        S["last_musf"] = cur_f

        # RG population spike rates (Hz per neuron) for plotting
        sp_rge, cur_rge = new_spikes(L["rec_rge"], S["last_rge"])
        sp_rgf, cur_rgf = new_spikes(L["rec_rgf"], S["last_rgf"])
        S["last_rge"] = cur_rge
        S["last_rgf"] = cur_rgf

        # NOTE: muscle parrot neurons amplify spikes because each muscle cell can receive many motor spikes.
        # Normalize by expected motor->muscle fan-in so proxy activation doesn't saturate in both phases.
        dt_s_safe = max(1e-9, dt_s)
        fanin_e = max(1.0, float(N_MOTOR_E) * float(P_M2MUS))
        fanin_f = max(1.0, float(N_MOTOR_F) * float(P_M2MUS))

        r_muse = ((sp_e / max(1, N_MUS_E)) / dt_s_safe) / fanin_e
        r_musf = ((sp_f / max(1, N_MUS_F)) / dt_s_safe) / fanin_f
        P["mus_e"].append(r_muse)
        P["mus_f"].append(r_musf)

        r_rge = (sp_rge / max(1, N_RG_E)) / dt_s_safe
        r_rgf = (sp_rgf / max(1, N_RG_F)) / dt_s_safe
        P["rge"].append(r_rge)
        P["rgf"].append(r_rgf)

        # Activation proxy: (1) saturate muscle relay rate -> activation,
        # (2) gate by BS drive phase, (3) separate rise/decay taus so it won't linger > step.
        bs_den = max(1e-9, float(BS_RATE_AMP_HZ))
        d_e = clamp((r_e - float(BS_RATE_BASE_HZ)) / bs_den, 0.0, 1.0)
        d_f = clamp((r_f - float(BS_RATE_BASE_HZ)) / bs_den, 0.0, 1.0)
        d_e = float(d_e) ** float(ACT_GATE_POWER)
        d_f = float(d_f) ** float(ACT_GATE_POWER)

        # Saturating mapping from muscle relay rate to activation
        a_raw_e = ACT_MAX * (1.0 - np.exp(-ACT_SAT_K * float(r_muse)))
        a_raw_f = ACT_MAX * (1.0 - np.exp(-ACT_SAT_K * float(r_musf)))

        # Gate by brainstem drive (enforces correct phase and duration)
        target_ae = clamp(a_raw_e * d_e, 0.0, ACT_MAX)
        target_af = clamp(a_raw_f * d_f, 0.0, ACT_MAX)

        tau_rise_s = TAU_ACT_RISE_MS / 1000.0
        tau_decay_s = TAU_ACT_DECAY_MS / 1000.0
        tau_e = tau_rise_s if target_ae > S["act_e"] else tau_decay_s
        tau_f = tau_rise_s if target_af > S["act_f"] else tau_decay_s

        kAe = 1.0 - np.exp(-dt_s_safe / max(1e-9, tau_e))
        kAf = 1.0 - np.exp(-dt_s_safe / max(1e-9, tau_f))

        S["act_e"] += kAe * (target_ae - S["act_e"])
        S["act_f"] += kAf * (target_af - S["act_f"])
        S["act_e"] = clamp(S["act_e"], 0.0, ACT_MAX)
        S["act_f"] = clamp(S["act_f"], 0.0, ACT_MAX)

        target_fe = FORCE_MAX * (1.0 - np.exp(-FORCE_SAT_K * S["act_e"]))
        target_ff = FORCE_MAX * (1.0 - np.exp(-FORCE_SAT_K * S["act_f"]))
        tau_rise_s = TAU_FORCE_RISE_MS / 1000.0
        tau_decay_s = TAU_FORCE_DECAY_MS / 1000.0

        # Stable force dynamics (rise/decay) for large dt
        kFe = 1.0 - np.exp(-dt_s_safe / max(1e-9, (tau_rise_s if target_fe > S["force_e"] else tau_decay_s)))
        kFf = 1.0 - np.exp(-dt_s_safe / max(1e-9, (tau_rise_s if target_ff > S["force_f"] else tau_decay_s)))
        S["force_e"] += kFe * (target_fe - S["force_e"])
        S["force_f"] += kFf * (target_ff - S["force_f"])

        S["force_e"] = clamp(S["force_e"], 0.0, FORCE_MAX)
        S["force_f"] = clamp(S["force_f"], 0.0, FORCE_MAX)

        tauL_s = TAU_LENGTH_MS / 1000.0
        kL = 1.0 - np.exp(-dt_s_safe / max(1e-9, tauL_s))
        S["len_e"] += kL * (L0 - S["len_e"])
        S["len_f"] += kL * (L0 - S["len_f"])
        S["len_e"] -= SHORTEN_GAIN * S["force_e"] * dt_s
        S["len_f"] -= SHORTEN_GAIN * S["force_f"] * dt_s
        if cut_active_frac > 0.0:
            S["len_e"] += STRETCH_GAIN * cut_active_frac * dt_s
        S["len_e"] = clamp(S["len_e"], L_MIN, L_MAX)
        S["len_f"] = clamp(S["len_f"], L_MIN, L_MAX)

        stretch_e = max(0.0, S["len_e"] - L0)
        stretch_f = max(0.0, S["len_f"] - L0)
        ia_e = IA_BASE_HZ + IA_K_FORCE * S["force_e"] + IA_K_STRETCH * stretch_e
        ia_f = IA_BASE_HZ + IA_K_FORCE * S["force_f"] + IA_K_STRETCH * stretch_f
        ia_e = clamp(ia_e, 0.0, IA_RATE_MAX_HZ)
        ia_f = clamp(ia_f, 0.0, IA_RATE_MAX_HZ)
        if do_rate_update:
            nest.SetStatus(L["ia_pg_e"], {"rate": ia_e})
            nest.SetStatus(L["ia_pg_f"], {"rate": ia_f})

        P["act_e"].append(S["act_e"]);
        P["act_f"].append(S["act_f"])
        P["force_e"].append(S["force_e"]);
        P["force_f"].append(S["force_f"])
        P["len_e"].append(S["len_e"]);
        P["len_f"].append(S["len_f"])
        P["ia_e"].append(ia_e);
        P["ia_f"].append(ia_f)

    # Keep last sampled mean/std so we can append smoothly without resampling every step
    last_wstats = {side: {k: (np.nan, np.nan) for k in ["cut->rge", "bs->rge", "bs->rgf"]} for side in LEGS}

    def log_weights(t_ms: float, step_idx: int):
        """Append weight mean/std time series.
        Optionally store full weight vectors for plastic projections.

        - mean/std are appended every step (reusing last sampled values)
        - full vectors are stored only when sampling happens (snapshots)
        """
        times.append(t_ms)
        do_sample = (step_idx % weight_every == 0)

        for side in LEGS:
            for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                if do_sample:
                    conns = conns_cache[side][key]
                    if conns is None or len(conns) == 0:
                        last_wstats[side][key] = (np.nan, np.nan)
                    else:
                        w = np.asarray(nest.GetStatus(conns, "weight"), dtype=float)
                        last_wstats[side][key] = (float(w.mean()), float(w.std()))
                mval, sval = last_wstats[side][key]
                wstats[side][key][0].append(mval)
                wstats[side][key][1].append(sval)

        # Full weight storage (snapshots at sampling ticks)
        if args.save_weights == "snapshots" and do_sample:
            wfull_times.append(float(t_ms))
            for side in LEGS:
                for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                    conns = conns_full_cache[side][key]
                    if conns is None or len(conns) == 0:
                        wfull[side][key].append(np.array([], dtype=np.float32))
                    else:
                        w = np.asarray(nest.GetStatus(conns, "weight"), dtype=np.float32)
                        wfull[side][key].append(w)

    total_steps = int(np.ceil(SIM_MS / CHUNK_MS))
    if rank == 0 and (SIM_MS >= 30000.0) and (not args.long_run):
        print("[Hint] Long simulation detected. Consider adding --long-run "
              "(sets chunk>=200ms, weight_sample>=1000ms, rate_update>=100ms, and downsamples weight reads).")
    done_steps = 0
    chunk = max(1, int(N_CUT / N_PHASES))
    t_ms = 0.0

    t0 = time.time()
    sim_accum = 0.0
    book_accum = 0.0
    for phase in range(N_PHASES):
        for side in LEGS:
            nest.SetStatus(leg[side]["cut_pg"], {"rate": CUT_RATE_OFF_HZ})

        start = phase * chunk
        end = min(N_CUT, (phase + 1) * chunk)
        for side in LEGS:
            nest.SetStatus(leg[side]["cut_pg"][start:end], {"rate": CUT_RATE_ON_HZ})
        cut_active_frac = float(end - start) / float(N_CUT)

        # Simulate in larger chunks to reduce Python <-> NEST overhead.
        n_chunks = int(PHASE_MS // CHUNK_MS)
        tail_ms = q_ms(PHASE_MS - n_chunks * CHUNK_MS)
        n_chunks_total = n_chunks + (1 if tail_ms > 1e-9 else 0)

        for local_chunk in range(n_chunks_total):
            cur_chunk_ms = CHUNK_MS if local_chunk < n_chunks else tail_ms
            cur_chunk_ms = q_ms(cur_chunk_ms)
            if cur_chunk_ms <= 0.0:
                continue

            t_sim0 = time.perf_counter()
            nest.Simulate(cur_chunk_ms)
            sim_accum += (time.perf_counter() - t_sim0)

            t_ms += cur_chunk_ms
            done_steps += 1  # now counts "chunks"

            t_book0 = time.perf_counter()
            do_rate_update = (done_steps % rate_every == 0)
            for side in LEGS:
                update_leg(side, t_ms, cur_chunk_ms, cut_active_frac, do_rate_update)
            log_weights(t_ms, done_steps)
            book_accum += (time.perf_counter() - t_book0)

            if rank == 0 and (
                    (done_steps % int(args.print_every) == 0) or (done_steps == total_steps) or (local_chunk == 0)):
                print(f"[Sim] Phase {phase + 1}/{N_PHASES} | chunk {done_steps}/{total_steps} | "
                      f"phase_chunk {local_chunk + 1}/{n_chunks_total} | t={t_ms:.1f} ms | chunk_ms={cur_chunk_ms:.1f}")

    if rank == 0:
        wall = time.time() - t0
        print(f"[Done] wall={wall:.1f}s, out={args.out}")
        tot = max(1e-9, (sim_accum + book_accum))
        print(
            f"[Timing] nest.Simulate: {sim_accum:.1f}s ({100.0 * sim_accum / tot:.1f}%) | bookkeeping: {book_accum:.1f}s ({100.0 * book_accum / tot:.1f}%)")

    # Capture final full weight vectors once (after simulation) if requested.
    # Note: in "final" mode we store both initial (t=0) and final (t=end) snapshots.
    if rank == 0 and args.save_weights == "final":
        wfull_times.append(float(t_ms))
        for side in LEGS:
            for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                conns = conns_full_cache[side][key]
                if conns is None or len(conns) == 0:
                    wfull[side][key].append(np.array([], dtype=np.float32))
                else:
                    w = np.asarray(nest.GetStatus(conns, "weight"), dtype=np.float32)
                    wfull[side][key].append(w)

    # ---- write HDF5 (rank 0 only) ----
    if rank != 0:
        return

    times_arr = np.asarray(times, dtype=np.float32)

    with h5py.File(args.out, "w") as h5:
        h5.attrs["created_utc"] = datetime.utcnow().isoformat() + "Z"
        h5.attrs["nest_version"] = str(nest.__version__)
        h5.attrs["sim_ms"] = SIM_MS
        h5.attrs["dt_ms"] = CHUNK_MS
        h5.attrs["inner_dt_ms"] = DT_MS
        h5.attrs["resolution_ms"] = float(args.resolution_ms)
        h5.attrs["phases"] = int(N_PHASES)
        h5.attrs["bs_osc_hz"] = float(BS_OSC_HZ)
        h5.attrs["local_threads"] = int(args.threads)
        h5.attrs["mpi_processes"] = int(nproc)
        h5.attrs["save_weights_mode"] = str(args.save_weights)

        gstats = h5.create_group("stats")
        for k, v in stats_nodes.items():
            gstats.attrs[f"nodes_{k}"] = int(v)
        for k, v in stats_syn_sign.items():
            gstats.attrs[f"syn_{k}"] = int(v)
        for k, v in stats_syn_models.items():
            gstats.attrs[k] = int(v)

        h5.create_dataset("times_ms", data=times_arr, compression="gzip")
        if len(wfull_times) > 0:
            h5.create_dataset("weights_times_ms", data=np.asarray(wfull_times, dtype=np.float32), compression="gzip")

        for side in LEGS:
            g = h5.create_group(f"leg_{side}")
            for key, arr in logs[side].items():
                g.create_dataset(key, data=np.asarray(arr, dtype=np.float32), compression="gzip")

            gw = g.create_group("weights")
            for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                gw.create_dataset(f"{key}_mean", data=np.asarray(wstats[side][key][0], dtype=np.float32),
                                  compression="gzip")
                gw.create_dataset(f"{key}_std", data=np.asarray(wstats[side][key][1], dtype=np.float32),
                                  compression="gzip")

            # Optional: full weight vectors (shape: [T_samples, N_connections])
            if len(wfull_times) > 0:
                gfw = g.create_group("full_weights")
                for key in ["cut->rge", "bs->rge", "bs->rgf"]:
                    src, tgt = conns_endpoints[side].get(key, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                    gk = gfw.create_group(key.replace("->", "_to_"))
                    gk.create_dataset("source", data=src, compression="gzip")
                    gk.create_dataset("target", data=tgt, compression="gzip")

                    if len(wfull[side][key]) > 0:
                        if wfull[side][key][0].size == 0:
                            wmat = np.zeros((len(wfull[side][key]), 0), dtype=np.float32)
                        else:
                            wmat = np.stack(wfull[side][key], axis=0).astype(np.float32, copy=False)
                        gk.create_dataset("w", data=wmat, compression="gzip")

    print(f"[HDF5] saved {args.out}")


if __name__ == "__main__":
    main()