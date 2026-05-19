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

- **Paced-gait mode** (`--paced-gait`): explicit 1 s trot cycle (L/R 180° offset). Force-E peaks ~17 a.u.
  with clean flat-top 500 ms stance windows, drops to ~0 in swing. Force-F peaks ~5–7 a.u. in debug
  (limited by RGF burst rate at BS=20 Hz); production will be higher.
- Activation-E: square-wave plateau at ~1.2, clean reset to 0 each swing.
- L vs R desynchronised via commissural inhibition + paced external drive.
- **Known debug-mode limitation for F**: FF force limited to ~7 a.u. at BS=20 Hz; production (BS=60 Hz,
  N=100) expected to reach >12 a.u.
- Without `--paced-gait`: cleanly alternates in debug mode; corr(RGE,RGF) ~−0.71 to −0.73.

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
| `W_INF2RGE = -48`, `W_INE2RGF = -8` | lines ~75-82 | **Asymmetric reciprocal inhibition (Zhang 2022). KEEP THIS RATIO.** F→E is 6× stronger (48:8). |
| `WMAX_BS = 30` | line ~236 | BS STDP weight cap, prevents BS-alone runaway. |
| `W_CUT2INE = 6`, `P_CUT2INE = 0.30` | line ~266 | Stance-phase cutaneous reflex. |
| `W_IA2IN = 6`, `P_IA2IN = 0.25` | lines ~69-70 | Ia → RG reciprocal interneurons. Closed-loop knob. 5 too weak; 8 over-speeds cycle. |
| `RGF_C = -55, RGF_D = 4` | line ~288 | Intrinsically bursting Izhikevich for RG-F. |
| `rg_ref = 100` | line ~1190 | Activation gate reference Hz. 100 Hz calibrated for debug-mode burst peaks; clamps to 1 in production (300+ Hz). |
| `ACT_SAT_K = 0.02` | line ~291 | Activation saturation slope. **Was 5e-4 (40× too small) — regression fixed.** |
| `TAU_ACT_RISE/DECAY_MS = 20/20` | line ~288 | Activation time constants. Tuned for 150–200 ms debug cycles. **Overridden to 40/40 by `--paced-gait`.** |
| `TAU_FORCE_RISE/DECAY_MS = 30/30` | line ~294 | Force time constants. Rat fast-twitch range. **Overridden to 80/80 by `--paced-gait`.** |
| `FORCE_SAT_K = 1.0` | line ~301 | Force saturation. K=1 keeps force linear. |
| `N_INF = 40` (debug-small) | line ~792 | Doubled in debug-small to give 12 InF connections per RGE vs 6 at N=20. |
| `--step-period-ms 1000` | `debug.sh` | Full gait cycle period. HALF_MS=500ms per leg. |
| `--n-ia-groups 3` | `debug.sh` | Heel/mid/toe sequential Ia-E groups (60/80/100 Hz). Each active 167ms. |

## Modification history (grep-friendly)

| Tag | What it does |
|---|---|
| `MOD_TONIC_BS` | BS is constant-rate, identical for both legs (not phase-gated). |
| `MOD_COACT` | BS subthreshold alone; CUT co-activates RG via static pathway. |
| `MOD_ZHANG_ASYM` | F→E inhibition 6× stronger than E→F (W_INF2RGE=-48, W_INE2RGF=-8). |
| `MOD_CUT_REFLEX` | CUT → InE (stance-phase cutaneous reflex). |
| `MOD_ACT_GATE` | Activation gated by RG rate (rg_ref=100 Hz, ACT_GATE_POWER=2). |
| `MOD_FORCE_LINEAR` | FORCE_SAT_K=1.0 — force linear in working range. |
| `MOD_DEBUG_SMALL` | Small-N + low-BS local debug mode; N_INF=40 (doubled). |
| `MOD_IA_LOOP` | Ia → InE/InF closed-loop sensory drive into CPG core (W_IA2IN=6). |
| `MOD_PACED_GAIT` | Explicit 1-s trot cycle: L/R 180° offset, sequential Ia-E heel→toe during stance. |

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
- Activation rides at 0.5 constantly → `rg_ref` too low; should be ~100 Hz (debug) — gate always clamped to 1 means no burst/trough discrimination

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
