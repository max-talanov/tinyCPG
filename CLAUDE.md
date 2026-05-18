# tinyCPG — Two-Leg Rat Spinal CPG (Debug Workspace)

NEST-based spinal central pattern generator for rat locomotion. Two legs (left/right),
each with extensor + flexor half-centers, motor pools, muscle proxies, Ia afferents,
cutaneous afferents, and tonic brainstem drive. Trained with STDP on BS→RG and CUT→RG
synapses. Production runs on the MN5 supercomputer; this workspace is for fast local
iteration before submitting array jobs.

## Goal of the current debug pass

Get cleaner E/F counter-phase and self-sustained rhythm with **reduced BS dependence**.
Specifically: the model should keep alternating when BS_REGULAR_HZ is dropped from 60 to
20 Hz, relying on Ia closed-loop feedback into the spinal reciprocal-inhibition core
instead of the brainstem.

This is bio-plausibility-motivated: deafferented fictive locomotion in rat preparations
runs on weak tonic drive plus intrinsic INaP bursting plus reciprocal inhibition. We
don't have INaP in Izhikevich neurons, so we approximate it with closed-loop sensory
feedback (Ia → InE/InF → RG).

## Quick start

```bash
# Fast local iteration (~30 s wall clock)
./debug.sh

# Plot results
python3 cpg_plot_from_hdf5.py --in results/debug.h5 --save-prefix debug

# Production run on MN5 (don't touch unless ready)
sbatch run.sh
```

## File map

| File | Purpose |
|---|---|
| `cpg_2legs_fast.py` | The model. Neurons, connections, sim loop, HDF5 export. |
| `cpg_plot_from_hdf5.py` | Reads HDF5, makes per-leg PNGs. |
| `run.sh` | MN5 SLURM array script (N=100, 10-point μ:CV sweep). |
| `debug.sh` | Local single-config run with `--debug-small`. |
| `CLAUDE.md` | This file. |

## Current model state (as of this debug session)

- Cleanly alternates at ~1.7 Hz (600 ms cycles), rat trot range.
- E/F counter-phase: RG correlation ~−0.65 to −0.85, force correlation similar.
- Force returns to 0 between bursts (no more flat-topping).
- L vs R desynchronised via commissural inhibition.
- **Known residual**: RG-F peaks ~2× RG-E amplitude (intrinsic-bursting Izhikevich on F
  side, regular-spiking on E side). User has decided to keep this asymmetry.

## Architecture

```
              CUT (cutaneous, phasic)         BS (brainstem, tonic)
               │ static + STDP                 │ STDP, Wmax=30
               ▼                               ▼
              RG-E ◄──── InF ◄───── RG-F   (asymmetric: F→E STRONG, E→F WEAK)
               │          ▲          │
               │          │          │
               ▼     (Ia loop)       ▼
              M-E                   M-F        (motor pools, reciprocal inhibition)
               │                     │
               ▼                     ▼
              mus-E                 mus-F      (parrot relays → activation proxy)
               │                     │
               └───── force, length ─┘
                          │
                          ▼
                         Ia-E, Ia-F  (rate-coded, force + stretch)
                          │
                          └──→ InE, InF, ia_int → motor antagonist
```

Cross-leg: L↔R commissural inhibition on RG-F (strong) and RG-E (weak).

## Key constants in `cpg_2legs_fast.py`

| Constant | Where | Notes |
|---|---|---|
| `BS_REGULAR_HZ = 60` | line ~80 | Tonic BS rate. `--debug-small` drops to 20. |
| `CUT_RATE_ON_HZ = 100` | line ~66 | Rat Group-II/Aβ peak rate. Don't push >100 Hz. |
| `W_INF2RGE = -32`, `W_INE2RGF = -5` | lines ~67-72 | **Asymmetric reciprocal inhibition (Zhang 2022). KEEP THIS RATIO.** F→E is ~6× stronger than E→F. |
| `WMAX_BS = 30` | line ~236 | BS STDP weight cap, prevents BS-alone runaway. |
| `W_CUT2INE = 6`, `P_CUT2INE = 0.30` | line ~266 | Stance-phase cutaneous reflex. |
| `W_IA2IN = 5`, `P_IA2IN = 0.25` | lines ~69-70 | **NEW** — Ia → RG reciprocal interneurons. This is the closed-loop knob for debug. |
| `RGF_C = -55, RGF_D = 4` | line ~288 | Intrinsically bursting Izhikevich for RG-F. |
| `rg_ref = 250` | line ~1148 | Activation gate reference Hz. |
| `FORCE_SAT_K = 1.0` | line ~298 | Force saturation. K=1 keeps force linear. |

## Modification history (grep-friendly)

| Tag | What it does |
|---|---|
| `MOD_TONIC_BS` | BS is constant-rate, identical for both legs (not phase-gated). |
| `MOD_COACT` | BS subthreshold alone; CUT co-activates RG via static pathway. |
| `MOD_ZHANG_ASYM` | F→E inhibition ~6× stronger than E→F (Zhang/Shevtsova/Rybak 2022). |
| `MOD_CUT_REFLEX` | CUT → InE (stance-phase cutaneous reflex). |
| `MOD_ACT_GATE` | Activation gated by RG rate (ref 250 Hz), not BS phase. |
| `MOD_FORCE_LINEAR` | FORCE_SAT_K=1.0 — force linear in working range. |
| `MOD_DEBUG_SMALL` | Small-N + low-BS local debug mode. |
| `MOD_IA_LOOP` | **NEW** — Ia → InE/InF closed-loop sensory drive into CPG core. |

## Bio-plausibility constraints (rat)

| Quantity | Range | Source |
|---|---|---|
| BS reticulospinal | 20–80 Hz | Drew, Rossignol |
| CUT Group-II / Aβ | 80–100 Hz peak | Loeb, Pearson |
| Locomotor cycle | 400–700 ms | Bellardita & Kiehn 2015 |
| Rat trot speed | ~30 cm/s | Lemieux et al. 2016 |

Don't push values outside these ranges without flagging it.

## What "good" looks like in debug output

- Clean alternation: RG-E vs RG-F correlation < −0.85
- Force-E vs Force-F correlation < −0.80
- Force minima < 2, peaks > 12, both half-centers
- L vs R legs not perfectly synchronised
- Cycle period 400–700 ms
- Activation reaches 0 cleanly between bursts

## What "broken" looks like

- Everything dies → BS too low for current Ia loop strength → bump `W_IA2IN` or `P_IA2IN`
- One half-center locked permanently → inhibition asymmetry too extreme → reduce `|W_INF2RGE|`
- Both legs synchronised → commissural too weak → bump `W_COMM_F_INH`
- Force flat-tops at 17 → `FORCE_SAT_K` regressed; should be 1.0
- Activation rides at 0.5 constantly → `rg_ref` clamped; should be ~250 Hz

## Debug iteration pattern

1. Edit one knob in `cpg_2legs_fast.py` (typically `W_IA2IN`, `P_IA2IN`, `BS_REGULAR_HZ`, or one of the `W_*2*` weights)
2. `./debug.sh`
3. `python3 cpg_plot_from_hdf5.py --in results/debug.h5 --save-prefix debug`
4. Inspect `debug_legL_rg_rate.png`, `debug_legL_force.png`, `debug_legL_activation.png`
5. Repeat

Each iteration is ~30 s. Don't edit `run.sh` during debug.

## When ready for MN5

1. Verify alternation works at BS=20 Hz with `./debug.sh`
2. Verify it still works at BS=60 Hz: remove `--debug-small` from `debug.sh` and re-run with `--sim-ms 5000`
3. `sbatch run.sh`
4. After completion, plot one HDF5 to confirm: `python3 cpg_plot_from_hdf5.py --in results/cpg_bursting_commfix_idx04_*.h5 --save-prefix solid_bs`

## Things NOT to touch without flagging the user

- The asymmetric inhibition ratio (`W_INF2RGE` / `W_INE2RGF` ≈ 6:1) — this is the Zhang 2022 finding
- The `--enforce-tonic-bs` semantics — bio-plausibility commitment
- `bs_rates_tonic` — should return identical values for both legs
- `BS_REGULAR_HZ` upper bound — keep below 80 Hz (rat reticulospinal)
- `CUT_RATE_ON_HZ` upper bound — keep below 100 Hz (rat Group-II/Aβ)
