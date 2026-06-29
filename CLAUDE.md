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
| `run_speed_stdp.sh` | Phase A: 3 speeds × 3 λ {1e-5,1e-4,1e-3}, 120 s. (descending/BS-plastic arm) |
| `run_sensory_stdp.sh` | Sensory-learning arm: same 3×3 matrix but `--freeze-bs-rg --stdp-ia-rg --wmax-ia 10`. Pair with `run_speed_stdp.sh` for descending-vs-sensory contrast. |
| `run_ablation_graded.sh` | Phase B: 3 Ia gains × 3 λ, 120 s. |
| `run_frozen.sh` | Frozen-weight control: STDP off, air stepping, (mean,CV) sweep. |
| `debug.sh` | Local single-config run with `--debug-small`. |
| `CLAUDE.md` | This file. |

## Frozen-weight control (`run_frozen.sh`)

Tests the Phase B two-regime hypothesis: that the descending-weight
*distribution* (not the learning dynamics) carries the counter-phase quality.
Sets STDP frozen (`--stdp-lambda 0`) and imposes a lognormal CUT→RGE
distribution of prescribed (mean, CV) via `--stdp-winit-dist lognormal_cv
--stdp-winit-mean <M> --stdp-winit-std <CV> --stdp-winit-bs-mean-mul 0.25`,
all at air stepping (`--ia-feedback-gain 0.1`). Two sweeps isolate the
mechanisms: **mean** sweep at fixed CV=0.02 (weakness), **CV** sweep at fixed
mean=63 (heterogeneity). Plot with `cpg_frozen_figure.py`. No `--sweep-pairs`
(non-sweep mode uses `--out` directly).

## Current model state (as of this debug session)

- **Paced-gait mode** (`--paced-gait`): explicit 1 s trot cycle (L/R 180° offset). Force-E peaks ~17 a.u.
  with clean flat-top 500 ms stance windows, drops to ~0 in swing. Force-F peaks ~5–7 a.u. in debug
  (limited by RGF burst rate at BS=20 Hz); production will be higher.
- Activation-E: square-wave plateau at ~1.2, clean reset to 0 each swing.
- L vs R desynchronised via commissural inhibition + paced external drive.
- **Known debug-mode limitation for F**: FF force limited to ~7 a.u. at BS=20 Hz; production (BS=60 Hz,
  N=100) expected to reach >12 a.u.
- Without `--paced-gait`: cleanly alternates in debug mode; corr(RGE,RGF) ~−0.71 to −0.73.

### Sensory-driven mode (`--freeze-bs-rg --stdp-ia-rg`, WMAX_IA=10)

Learning shifted from descending (BS) to sensory (muscle-Ia) pathway: BS→RG frozen at
weak init, plastic homonymous Ia→RG added. **Validated to outperform the BS-plastic
control** (debug-small, paced, 25 s): corr(Force-E,Force-F) **−0.978 vs −0.955**,
corr(RGE,RGF) −0.813 vs −0.788, and the **weak flexor is fixed** — Force-F peak rises
**11→17 a.u.** with clean troughs (F-E min 0.16). Mechanism: a light phased Ia→RG loop
reinforces each burst without filling the inter-burst trough; raising WMAX_IA saturates
it into tonic co-excitation that destroys counter-phase (monotonic in the sweep). This is
the bio-plausibility win the debug goal was after: self-sustained counter-phase on weak
tonic BS + closed-loop proprioception.

**Confirmed at production scale** (full N, BS=60 Hz, paced, 15 s): corr(RGE,RGF) **−0.965**,
corr(F-E,F-F) **−0.983**, Force-E/F peaks **16.7/16.7** (fully balanced), troughs ~1.0–1.1,
CUT→RGE→62.4, Ia→RG self-stabilises at **~4.5 pA** (same as debug — robust, sub-cap).
Cleaner than debug-small. Production sweep: `run_sensory_stdp.sh` (sensory-learning arm,
mirrors `run_speed_stdp.sh` for a descending-vs-sensory paired contrast).

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
| `MOD_FREEZE_BS` | `--freeze-bs-rg`: BS→RG-E/RG-F static (no STDP), held at weak lognormal init (W_INIT_BS). BS becomes fixed tonic drive. |
| `MOD_IA_RG_STDP` | `--stdp-ia-rg`: plastic homonymous Ia→RG (Ia-E→RG-E, Ia-F→RG-F, Wmax=WMAX_IA=10, density P_IA2RG_STDP=0.5). Muscle afferents become the learning drive to the RGs. **WMAX_IA=10 validated as sweet spot** (see below); `--wmax-ia`/`--p-ia2rg` to sweep. |
| `--ia-feedback-gain` | Multiplicative gain on closed-loop Ia rate. 1.0 baseline / 0.5 toe stepping / 0.1 air stepping (Courtine/Lavrov SCI paradigm). |
| `--cut-feedback-gain` | Multiplicative gain on cutaneous CUT stance drive (loading-dependent paw contact). Scaled with loading alongside `--ia-feedback-gain`; the external Ia-E heel→toe ramp (stim pacing) stays at full. |
| `--ia-ext-f-hz` | MOD_FLEXOR_AFFERENT: rate (Hz) of the external flexor swing-afferent (hip/flexor-stretch signal; Grillner & Rossignol 1978). Drives RG-F directly + InF during swing, clocking the flexor symmetrically to the stance Ia-E ramp. 0 = off (intrinsic-only flexor); 80 = on. Un-gated by loading (joint-position, not load-based). |
| `--stdp-lambda` | Override STDP LAMBDA (default 1e-3). Bio-plausible range 5e-4 to 5e-3 (Bi & Poo 1998; Morrison 2007). |
| `--dump-connectivity` | Build the network, write per-connection WEIGHT + DELAY arrays for all 18 named projections to an HDF5, then exit (no sim — runs in seconds at production N). Feeds the connectivity-statistics figure (`cpg_connectivity_figure.py`) and CSV table. Static weights are delta-valued; plastic are lognormal-init; delays follow the rat `length_velocity` preset + 0.2 ms jitter. |
| `--freeze-bs-rg` | MOD_FREEZE_BS: freeze BS→RG (static at weak init). Removes descending plasticity; pair with `--stdp-ia-rg`. Frozen runs drop `bs->rge`/`bs->rgf` from the tracked plastic-weight keys. |
| `--stdp-ia-rg` | MOD_IA_RG_STDP: add plastic homonymous Ia→RG. Adds `ia->rge`/`ia->rgf` weight keys. Shifts learning from descending (BS) to sensory (Ia) pathway. Pair with `--freeze-bs-rg`. |
| `--wmax-ia` | Weight cap for Ia→RG STDP (default 10). **Low cap is critical**: homonymous Ia→RG is in-phase positive feedback — light (≤10) reinforces bursts without filling troughs; high (≥60) saturates into tonic co-excitation that destroys counter-phase. |
| `--p-ia2rg` | Connection probability of the Ia→RG projection (default 0.5). |

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
