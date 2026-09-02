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
python3 scripts/cpg_plot_from_hdf5.py --in results/debug.h5 --save-prefix debug

# Production run on MN5 (don't touch unless ready)
sbatch run.sh
```

## File map

| File | Purpose |
|---|---|
| `cpg_2legs_fast.py` | The model. Neurons, connections, sim loop, HDF5 export. |
| `scripts/` | Current figure/analysis scripts (feed `paper/figures/`), `cpg_param_table.py`, `cpg_plot_from_hdf5.py`, `build_deck.py`. |
| `scripts/legacy/` | Superseded figure scripts, kept for reference only — not used by the current paper. |
| `scripts/cpg_plot_from_hdf5.py` | Reads HDF5, makes per-leg PNGs. |
| `scripts/cpg_cutforce_diagnostics.py` | Pass/fail check for `--cut-trigger force` sweep outputs: corr(Force-E,Force-F) plus `frac_at_cap` (is the failsafe timer doing the work, or genuine force-threshold crossings?). Run before trusting any correlation number from this mode. |
| `run.sh` | MN5 SLURM array script (N=100, 10-point μ:CV sweep). |
| `run_speed_stdp.sh` | Phase A: 3 speeds × 3 λ {1e-5,1e-4,1e-3}, 120 s. (descending/BS-plastic arm) |
| `run_sensory_stdp.sh` | Sensory-learning arm: same 3×3 matrix but `--freeze-bs-rg --stdp-ia-rg --wmax-ia 10`. Pair with `run_speed_stdp.sh` for descending-vs-sensory contrast. Outputs `cpg_sensory_stdp_*`. |
| `run_ablation_sensory.sh` | Sensory-learning ablation arm: graded loading × λ with frozen BS + plastic Ia→RG (Ia is the *gated* learning drive). Outputs `cpg_ablsens_*`. Plot with `--mode ablsens`. |
| `run_ablation_graded.sh` | Phase B: 3 Ia gains × 3 λ, 120 s. |
| `run_frozen.sh` | Frozen-weight control: STDP off, air stepping, (mean,CV) sweep. |
| `debug.sh` | Local single-config run with `--debug-small`. |
| `debug_force.sh` | Local single-config run with `--debug-small --cut-trigger force` (closed-loop, force-triggered CUT — see below). |
| `run_cutforce_sweep.sh` | EXPLORATORY MN5 sweep round 1 (9 tasks): fatigue-onset-τ {200,400,600} × cap {500,800,1100}. Superseded by round 2 — its apparent "best" result turned out 100% cap-dominated on re-diagnosis, see "Force-triggered CUT" below. |
| `run_cutforce_sweep2.sh` | EXPLORATORY MN5 sweep round 2 (9 tasks): fatigue-onset-τ {400,600,800} × tighter, bio-plausible cap {300,450,600}. Superseded — 100% cap-dominated on all 9 configs, see "Force-triggered CUT" below. |
| `run_cutforce_sweep3.sh` | EXPLORATORY MN5 sweep round 3 (9 tasks): fast fatigue-onset-τ {100,150,250} × looser `--cut-force-off-frac` {0.30,0.40,0.50}, cap fixed at 450ms. First round to escape cap-domination (frac_at_cap ~0 across the whole grid) — see "Force-triggered CUT" below. |
| `run_cutforce_sweep4.sh` | EXPLORATORY MN5 sweep round 4 (9 tasks): narrows around round 3's working region — fatigue-onset-τ {250,300,350} × `--cut-force-off-frac` {0.25,0.30,0.35}, cap still fixed at 450ms. 6/9 configs reverted to cap-domination (incl. the numerically best-looking correlation); best genuine result τ=250/off=0.35 — see "Force-triggered CUT" below. |
| `run_cutforce_sweep5.sh` | CONFIRMATION refinement (9 tasks, not a new exploration): brackets round 4's τ=250/off=0.35 optimum — fatigue-onset-τ {240,250,260} × `--cut-force-off-frac` {0.35,0.375,0.40}, cap still fixed at 450ms. **Confirmed: frac_at_cap=0.00 on all 9 configs** — best point τ=260/off=0.35, see "Force-triggered CUT" below. |
| `CLAUDE.md` | This file. |

## Frozen-weight control (`run_frozen.sh`)

Tests the Phase B two-regime hypothesis: that the descending-weight
*distribution* (not the learning dynamics) carries the counter-phase quality.
Sets STDP frozen (`--stdp-lambda 0`) and imposes a lognormal CUT→RGE
distribution of prescribed (mean, CV) via `--stdp-winit-dist lognormal_cv
--stdp-winit-mean <M> --stdp-winit-std <CV> --stdp-winit-bs-mean-mul 0.25`,
all at air stepping (`--ia-feedback-gain 0.1`). Two sweeps isolate the
mechanisms: **mean** sweep at fixed CV=0.02 (weakness), **CV** sweep at fixed
mean=63 (heterogeneity). Plot with `scripts/legacy/cpg_frozen_figure.py`. No `--sweep-pairs`
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

### Force-triggered CUT (`--cut-trigger force`)

Replaces the paced-gait *clock* with a closed-loop stance detector: CUT (cutaneous/
paw-contact) firing is gated directly on each leg's own `force_e`, not on a fixed
timer — "foot touches down" (CUT ON) when `force_e` rises through
`--cut-force-on-frac` (default 0.80) of that leg's current-bout running peak, "foot
lifts off" (CUT OFF) when it falls through `--cut-force-off-frac` (default 0.20). The
peak is **not** time-decayed — it resets to a seed value at each new stance onset and
then holds monotonically (grows through stance, frozen through the following swing).
A time-decaying peak was tried first and rejected: with `--muscle-fatigue` on, a
slowly-fatiguing force and a slowly-decaying peak converge together and the *relative*
OFF threshold never actually gets crossed — the leg locks at a permanently-reduced-but-
still-"on" plateau instead of releasing.

**Symmetry-breaking**: real gait doesn't start from identical L/R initial conditions —
one leg is already planted. `--leading-leg` (default `R`) seeds that leg into stance at
t=0. `--lead-offset-ms` (default 150 ms) is also a *symmetric priming window*: **both**
legs' CUT is ON for this duration (not just the leader's) so both sides' plastic
CUT→RG-E synapse gets co-activation training before the split — at low initial STDP
weight (production sweeps start as low as mean=0–3.5) the lagging leg's synapse
otherwise never gets its first potentiation and that leg struggles to ever mount a real
stance (confirmed by direct test: without priming, leading leg reached corr(Force-E,
Force-F) −0.92, lagging leg only −0.20). At the end of the window the lagging leg is
cut back to swing immediately; too long a window desyncs this into leg synchronisation
instead (confirmed at 400 ms: corr(Force-E_L, Force-E_R) flipped to **+0.41**, legs
moving together — a real failure mode, not just a weaker one).

**Failsafe timeout (required, not optional)**: RG-E has no INaP-style self-terminating
burst mechanism — only RG-F got the intrinsically-bursting Izhikevich treatment
(`RGF_C`/`RGF_D`). So `CUT → RG-E → force_e → CUT` is a pure positive-feedback loop:
force saturates near its ceiling and **just sits there** — confirmed by direct test
(production N, BS=60Hz, cap disabled): R stayed in stance with force_e flat at ~17.5
for 4+ continuous seconds, L stayed in swing at ~0 the whole time. `--cut-max-stance-ms`
/ `--cut-max-swing-ms` (both default 600 ms) cap each phase and force the transition
regardless of force — the endogenous-timer backstop for when peripheral gating alone
stalls (bio: hip-extension/limb-position limit triggers swing even under continued
loading, Grillner & Rossignol 1978; matches the two-level sensory-gated +
endogenous-timer picture, Rybak/McCrea unit-burst-generator model). **Do not remove
this timeout.**

**Debug-scale validated** (`debug_force.sh`, debug-small, BS=20 Hz, 10 s): both legs
alternate stance/swing continuously for the full run (7-8 bouts/leg, no lock-in),
corr(Force-E,Force-F) **−0.967 (L) / −0.967 (R)**, corr(RGE,RGF) **−0.83 (L) / −0.88
(R)**, corr(Force-E_L, Force-E_R) **−0.80**, Force-E peaks ~17.5 a.u. with clean
near-0 troughs.

**Production scale is NOT yet at the same bar.** At full N, BS=60Hz, step_period=520ms,
sweep-pairs 3.5:0.30 (the established operating point for `run_speed_stdp.sh` etc.),
the debug-tuned defaults only reach corr(Force-E,Force-F) ≈ **−0.65 (L) / −0.77 to
−0.85 (R)** over 8-20s, and bout-duration analysis showed transitions landing almost
exactly at the `--cut-max-stance-ms` value every cycle — i.e. the failsafe was doing
essentially *all* the work, not genuine force-threshold crossings (this is true at
debug scale too, on closer inspection — the "validated" debug numbers above are real
and clean, but likely cap-dominated rather than proof the pure threshold mechanism
alone is what's producing them).

**Correlation target recalibrated.** −0.85+ (the timer-based debug bar) is the wrong
target for this mode — that number is an artifact of the clock imposing a literal
square wave. A genuinely emergent, force-triggered gait should look more like
**−0.7 to −0.8**, with real cycle-to-cycle variability, once STDP has saturated. The
number that actually matters is **frac_at_cap** (below), not the correlation.

**`--muscle-fatigue` round 1** (`run_cutforce_sweep.sh`, results/2026-08-25, 9 tasks:
fatigue-onset-τ {200,400,600} × cap {500,800,1100}) found slower fatigue → better
force amplitude and correlation, apparently plateauing around τ=600/cap=800
(corr(Force-E,Force-F) ≈ −0.57 L / −0.69 R). **Re-diagnosed with exact ground-truth
`cut_on` logging (added after round 1 — see MOD_CUT_FORCE_TRIGGER) and that "best"
result turned out to be 100% cap-dominated on both legs** — stance duration exactly
800.0ms, zero variance, every single bout. Round 1's files predate the `cut_on`
array and can only be re-checked by reconstructing bouts from a force threshold,
which is unreliable (confirmed: reconstruction gave different at-cap verdicts on the
*same* file depending on the threshold chosen — see `scripts/cpg_cutforce_diagnostics.py`
docstring). **Trust `cut_on`-based ("exact") diagnostics only; treat any
pre-2026-08-27 result as unverified.**

**`--muscle-fatigue` round 2** (`run_cutforce_sweep2.sh`) inverts the round-1 fix
direction. Round 1's implied fix (bigger cap) is in tension with bio-plausibility
anyway — cap=800ms already exceeds the paper's own locomotor-cycle constraint
(400-700ms for a *full* stride, Bellardita & Kiehn 2015) for a single half-cycle.
Round 2 instead holds fatigue-onset-τ in the range that gave good amplitude/quality
(400-800ms) but *tightens* the cap toward bio-plausible half-cycle durations
(300-600ms), to test directly whether genuine crossings emerge under a realistic
time budget. **Result: 100% cap-dominated on both legs, at every one of the 9
tested combinations** (durations exactly equal to the cap, zero variance,
confirmed with exact `cut_on` ground truth — results/2026-08-27). A fatigue/force
overlay at the best-correlation config (τ=800/cap=600) explains why: `fatigue_e`
only reaches **~0.62 of its 0.95 ceiling** by the time the cap fires — force is
still ~70-80% of peak, nowhere near the 0.20 (`--cut-force-off-frac`) crossing
target. Neither axis tried in rounds 1-2 (fatigue speed, cap duration) alone
gets there — ruling out "hold τ in the good range, shrink the cap" as a fix.

**Round 3** (`run_cutforce_sweep3.sh`) tests the two remaining untried levers
together, cap held **fixed** at 450ms so any drop in `frac_at_cap` is
unambiguously attributable to them: fatigue-onset-τ pushed much faster (100,
150, 250ms — below round 1-2's 200-800ms floor) × `--cut-force-off-frac` loosened
(0.30, 0.40, 0.50 — vs 0.20 throughout rounds 1-2, so the crossing target no
longer requires near-complete decay). A 4s local smoke-test at τ=150/off=0.40
did escape cap-domination (`frac_at_cap`=0.00 both legs) but showed legs
synchronising (corr(Force-E_L,Force-E_R) = +0.88, the same failure mode seen at
`--lead-offset-ms 400`) — too short a run to judge properly, but a concrete
thing to watch for in the full 60s sweep.

**Round 3 result (results/2026-08-30): first round to escape cap-domination.**
`frac_at_cap` ≈ 0 on both legs across **all 9 configs**, with genuine bout-duration
variability (std up to ±52ms, vs. the flat zero of rounds 1-2) — confirmed with
exact `cut_on` ground truth. Quality still varies sharply within the grid: τ=100-150ms
gives short (~100ms), weak bouts and **legs synchronise in 5 of 6 of those configs**
(corr(Force-E_L,Force-E_R) up to +0.47 — the failure mode flagged above, now
confirmed for real, not just in a short smoke-test). τ=250ms (this round's ceiling)
gives the best results and stays anti-phase: best config τ=250/off=0.30 —
corr(Force-E,Force-F) −0.61(L)/−0.66(R), corr(Force-E_L,Force-E_R) −0.67, bout
duration 341±52/343±49ms. Still short of the −0.7/−0.8 recalibrated target, and
quality was still climbing with τ at the top of the tested range — round 3 ran out
of grid before finding a ceiling, not because τ=250 is optimal.

**Round 4** (`run_cutforce_sweep4.sh`) narrows into the region that actually
worked: fatigue-onset-τ {250, 300, 350} × `--cut-force-off-frac` {0.25, 0.30, 0.35},
cap still fixed at 450ms (round 3's value, the one that actually escaped
cap-domination — not re-testing the cap axis). Extends past round 3's τ=250
ceiling while staying well below round 1-2's τ=400+ floor where cap-domination
returned.

**Round 4 result (results/2026-08-31): the mechanism is more brittle than round 3's
trend implied.** 6 of the 9 configs **reverted to cap-domination** (frac_at_cap
0.97-1.00), including the numerically best-looking correlation in the whole grid
(τ=350/off=0.35: corr(Force-E,Force-F) −0.70(L)/−0.78(R) — but 97-100% cap-dominated,
a disguised clock exactly like round 1's trap, visibly confirmed by a perfectly
regular force waveform). Round 3's "higher τ → better" trend does **not** simply
continue — off-frac has to loosen *together* with τ, not independently: at τ=250,
off=0.30 and off=0.35 both stay genuine (frac_at_cap=0.00 both legs); at τ=300, only
off=0.35 is even mostly genuine (0.26/0.17, not clean); at τ=350, nothing in this
grid escapes the cap. **Best genuine result across all four rounds: τ=250/off=0.35**
— corr(Force-E,Force-F) −0.59(L)/−0.65(R), corr(Force-E_L,Force-E_R) **−0.72** (inside
the −0.7/−0.8 target), frac_at_cap=0.00 both legs, bout duration 292±45/292±56ms
(genuine ~16-19% cycle-to-cycle variability, visibly irregular waveform unlike the
τ=350 trap). τ=250/off=0.30 is the second genuine candidate, slightly weaker
(corr(Force-E_L,Force-E_R) −0.69).

**Round 5** (`run_cutforce_sweep5.sh`) is a small confirmation refinement, not a new
exploration: brackets the τ=250/off=0.35 optimum tightly — fatigue-onset-τ
{240, 250, 260} × `--cut-force-off-frac` {0.35, 0.375, 0.40} — to check the winning
point isn't a lucky single grid cell (i.e. small perturbations either side stay
genuine and don't collapse back into cap-domination the way τ=300/350 did just
0.05-0.10 higher on off-frac). Same cap=450ms, same operating point, same 60s length
as rounds 1-4.

**Round 5 result (results/2026-09-01): confirmed — the whole neighborhood is
genuine, not just one lucky cell.** `frac_at_cap` = **0.00 on both legs, all 9
configs**, exact ground truth. Correlation is stable and good across the whole
grid: corr(Force-E,Force-F) −0.52 to −0.63 (L) / −0.60 to −0.67 (R), corr(Force-E_L,
Force-E_R) **−0.65 to −0.73** (every config strongly anti-phase, no synchronisation
anywhere in this grid). **Best point: τ=260/off=0.35** — corr(Force-E,Force-F)
−0.63(L)/−0.67(R), corr(Force-E_L,Force-E_R) −0.71, bout duration 308±27/310±29ms
(tightest, most consistent spread of any genuine result so far, ~9% relative
variability). This closes out the "fix cap-dominance" step of the maturation plan:
`--cut-trigger force --muscle-fatigue` with τ≈240-260ms, off-frac≈0.35-0.40, cap=450ms
is a demonstrated, robust, genuinely closed-loop operating point — not confirmed
across other STDP init points yet (Phase 3, next).

**Required workflow from now on**: run `scripts/cpg_cutforce_diagnostics.py` on every
sweep output before trusting any correlation number. `frac_at_cap` near 1.0 on either
leg means the result is a disguised clock, regardless of how clean the correlation
looks.

### Muscle fatigue (`--muscle-fatigue`)

Opt-in (OFF by default — existing timer-based paced-gait runs are unaffected). Adds a
slow activity-dependent attenuation to the force proxy (both E and F): fatigue builds
toward `--fatigue-max-frac` (default 0.95) with time constant `--fatigue-tau-onset-ms`
(default 400 ms) while activation is high, and clears with `--fatigue-tau-recovery-ms`
(default 600 ms) while activation is low. This is what lets `force_e` actually decay
during a sustained stance bout instead of sitting flat at its ceiling forever (see
"Failsafe timeout" above) — the closest local analogue to the INaP-driven burst
termination the project's Izhikevich neurons don't have.

**`--fatigue-max-frac` must leave the fatigued force floor comfortably below the OFF
threshold**, or the leg locks at a reduced-but-still-"on" plateau instead of actually
releasing — confirmed at 0.85: force settled at a stable floor (~2.6, from residual
activation even at full fatigue) against an off-threshold of ~1.9 and never crossed it.
0.95 leaves a floor of ~0.9, safely below a typical off-threshold — this is why the
default was raised from the first value tried.

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
| `MOD_CUT_FORCE_TRIGGER` | `--cut-trigger force`: replaces the paced-gait clock with a per-leg Schmitt trigger on `force_e` (CUT ON/OFF at `--cut-force-on-frac`/`--cut-force-off-frac` of a per-bout running peak — "foot touches"/"foot lifts"). `--leading-leg`/`--lead-offset-ms` break initial L/R symmetry (the offset window is also a symmetric CUT→RG-E STDP priming window). `--cut-max-stance-ms`/`--cut-max-swing-ms` are a required failsafe timeout (RG-E has no self-terminating burst mechanism and locks permanently without it — see "Force-triggered CUT" above). Logs a per-leg ground-truth `cut_on` (0/1) array to the output HDF5 so `scripts/cpg_cutforce_diagnostics.py` can measure exact bout durations instead of reconstructing them from force. Requires `--paced-gait`. Production-scale tuning not yet confirmed — see "Force-triggered CUT" above and `run_cutforce_sweep2.sh`. |
| `MOD_MUSCLE_FATIGUE` | `--muscle-fatigue`: opt-in (OFF by default) slow activity-dependent force attenuation (`--fatigue-tau-onset-ms`/`--fatigue-tau-recovery-ms`/`--fatigue-max-frac`), so `force_e` can decay on its own during sustained activation instead of relying entirely on the `--cut-trigger force` failsafe cap. Only affects the force proxy, not the neural circuit. |
| `MOD_FREEZE_BS` | `--freeze-bs-rg`: BS→RG-E/RG-F static (no STDP), held at weak lognormal init (W_INIT_BS). BS becomes fixed tonic drive. |
| `MOD_IA_RG_STDP` | `--stdp-ia-rg`: plastic homonymous Ia→RG (Ia-E→RG-E, Ia-F→RG-F, Wmax=WMAX_IA=10, density P_IA2RG_STDP=0.5). Muscle afferents become the learning drive to the RGs. **WMAX_IA=10 validated as sweet spot** (see below); `--wmax-ia`/`--p-ia2rg` to sweep. |
| `--ia-feedback-gain` | Multiplicative gain on closed-loop Ia rate. 1.0 baseline / 0.5 toe stepping / 0.1 air stepping (Courtine/Lavrov SCI paradigm). |
| `--cut-feedback-gain` | Multiplicative gain on cutaneous CUT stance drive (loading-dependent paw contact). Scaled with loading alongside `--ia-feedback-gain`; the external Ia-E heel→toe ramp (stim pacing) stays at full. |
| `--ia-ext-f-hz` | MOD_FLEXOR_AFFERENT: rate (Hz) of the external flexor swing-afferent (hip/flexor-stretch signal; Grillner & Rossignol 1978). Drives RG-F directly + InF during swing, clocking the flexor symmetrically to the stance Ia-E ramp. 0 = off (intrinsic-only flexor); 80 = on. Un-gated by loading (joint-position, not load-based). |
| `--stdp-lambda` | Override STDP LAMBDA (default 1e-3). Bio-plausible range 5e-4 to 5e-3 (Bi & Poo 1998; Morrison 2007). |
| `--dump-connectivity` | Build the network, write per-connection WEIGHT + DELAY arrays for all 18 named projections to an HDF5, then exit (no sim — runs in seconds at production N). Feeds the connectivity-statistics figure (`scripts/cpg_connectivity_figure.py`) and CSV table. Static weights are delta-valued; plastic are lognormal-init; delays follow the rat `length_velocity` preset + 0.2 ms jitter. |
| `--freeze-bs-rg` | MOD_FREEZE_BS: freeze BS→RG (static at weak init). Removes descending plasticity; pair with `--stdp-ia-rg`. Frozen runs drop `bs->rge`/`bs->rgf` from the tracked plastic-weight keys. |
| `--stdp-ia-rg` | MOD_IA_RG_STDP: add plastic homonymous Ia→RG. Adds `ia->rge`/`ia->rgf` weight keys. Shifts learning from descending (BS) to sensory (Ia) pathway. Pair with `--freeze-bs-rg`. |
| `--wmax-ia` | Weight cap for Ia→RG STDP (default 10). **Low cap is critical**: homonymous Ia→RG is in-phase positive feedback — light (≤10) reinforces bursts without filling troughs; high (≥60) saturates into tonic co-excitation that destroys counter-phase. |
| `--p-ia2rg` | Connection probability of the Ia→RG projection (default 0.5). |
| `--static-weight-cv` | **Bio-plausibility (default 0.5):** per-connection lognormal weight heterogeneity on all static synapses (mean/sign preserved). Biological weights are lognormal (Song 2005; Buzsáki & Mizuseki 2014). `0` = legacy delta weights (used by the frozen-weight control). |
| `--cut-static-w` | **Bio-plausibility (default 0 = dropped):** weight of the fixed CUT→RG-E co-activation pathway. Default leaves a single plastic cutaneous projection; set `14` to restore the legacy bootstrap. |

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
3. `python3 scripts/cpg_plot_from_hdf5.py --in results/debug.h5 --save-prefix debug`
4. Inspect `debug_legL_rg_rate.png`, `debug_legL_force.png`, `debug_legL_activation.png`
5. Repeat

Each iteration is ~30 s. Don't edit `run.sh` during debug.

## When ready for MN5

1. Verify alternation works at BS=20 Hz with `./debug.sh`
2. Verify it still works at BS=60 Hz: remove `--debug-small` from `debug.sh` and re-run with `--sim-ms 5000`
3. `sbatch run.sh`
4. After completion, plot one HDF5 to confirm: `python3 scripts/cpg_plot_from_hdf5.py --in results/cpg_bursting_commfix_idx04_*.h5 --save-prefix solid_bs`

## Things NOT to touch without flagging the user

- The asymmetric inhibition ratio (`W_INF2RGE` / `W_INE2RGF` ≈ 6:1) — this is the Zhang 2022 finding
- The `--enforce-tonic-bs` semantics — bio-plausibility commitment
- `bs_rates_tonic` — should return identical values for both legs
- `BS_REGULAR_HZ` upper bound — keep below 80 Hz (rat reticulospinal)
- `CUT_RATE_ON_HZ` upper bound — keep below 100 Hz (rat Group-II/Aβ)
