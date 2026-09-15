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
| `run_cutforce_sweep6.sh` | PHASE 3 — seed/init robustness (10 tasks, 120s each): holds round 5's winning config fixed (τ=260, off=0.35, cap=450ms), sweeps the same 10-point (μ,CV) STDP-init grid as `run.sh`/paper Algorithm 1. Tests whether the operating point found in rounds 1-5 holds away from μ=3.5. |
| `run_cutforce_sensory_unload.sh` | EXPLORATORY, round 1 (9 tasks, 120s each): production-scale test of the unloading-rescue mechanism (Ia→RG-E/F now loading-dependent, see "Core architecture fix" below) — sensory arm (`--freeze-bs-rg`), 3×3 grid of `--cut-feedback-gain` (loading) × `--ia-feedback-gain` (compensation). Local debug-scale tuning plateaued at weak counter-phase across a wide search; suspected debug-scale population-size ceiling (N_IA_E/F=30 vs 100 production), not a broken mechanism — see "Force-triggered CUT" below. |
| `CLAUDE.md` | This file. |
| `spinal_plasticity_as_learning_spec.md` | Literature-grounded spec for the `--consolidate` tag-and-capture mechanism (see "Tag-and-capture consolidation" below) — spinal-cord analogue of hippocampal synaptic tagging and capture, written by direct analogy to [`hippocampal_timescales_as_circuit_spec.md`](https://github.com/max-talanov/tinyHippo/blob/main/hippocampal_timescales_as_circuit_spec.md). |

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

**Phase 3 — seed/init robustness** (`run_cutforce_sweep6.sh`) holds the round-5
winning config fixed (τ=260, off-frac=0.35, cap=450ms) and instead varies the STDP
initial-weight distribution across the same 10-point (μ,CV) diagnostic grid the
base timer-based model already uses for its own robustness claim (paper Algorithm 1
/ `run.sh`) — reusing the project's established methodology rather than inventing a
new one, so the two are directly comparable. Every round so far (1-5) tested only
μ=3.5,CV=0.30; this asks whether the mechanism holds at μ=0 and μ=16 too, the real
stress tests. 120s sim (matching Algorithm 1's own duration, not the 60s first-pass
length used in rounds 1-5) since this is confirmatory, not exploratory.

**Round 6 — partial result, 8/10 in (results/2026-09-07), very good so far.**
μ=0 through μ=9 (idx00-07) all confirmed genuine: `frac_at_cap` ≈ 0.00-0.01 on both
legs at every point, corr(Force-E,Force-F) tightly clustered −0.63 to −0.69 regardless
of initial weight, corr(Force-E_L,Force-E_R) −0.68 to −0.76 (tighter than round 5).
Bout-duration variability *shrinks* as μ increases (±39-46ms at μ=0-1 down to
±20-27ms at μ=5-9) — more initial synaptic drive needs less from the stochastic
bootstrap. STDP weight trends confirm the same initialization-independence the base
timer-based model already shows (paper §4.1): μ=0 (CUT→RG-E starts at 0 pA) and μ=9
(starts ~10-11 pA) converge to the identical ~62 pA plateau. **μ=12 (idx08) and μ=16
(idx09) — the two highest-weight stress tests — are still pending on MN5; do not
treat Phase 3 as closed until those land.**

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

### Core architecture fix: Ia→RG-E/F now a standing plastic pathway (2026-09-14)

While chasing why force-triggered CUT can't be rescued by more Ia drive under
simulated unloading (see "Force-triggered CUT" above), a reference architecture
diagram (`CPG_feedback_loops_mems.png`) surfaced a real gap: it shows a **direct
excitatory Ia→RG-E/F projection**, separate from the existing `Ia→InE/InF`
reciprocal-inhibition loop (MOD_IA_LOOP). The code already had this exact synapse
(`stdp_ia_rge`/`stdp_ia_rgf`) but only wired it behind `--stdp-ia-rg`, i.e. only in
the sensory-learning arm. Decision (explicit user call, given the choice between
scoping this to force-trigger only vs. universal): **make it universal** — Ia→RG-E/F
is now always wired and always plastic, in every mode, alongside BS→RG and CUT→RG
(BS→RG still drops out under `--freeze-bs-rg`; Ia→RG and CUT→RG never do). A fixed
weight here was considered and rejected — the paper's whole point is
plasticity/rehabilitation, so a static baseline couldn't represent that; three
simultaneously plastic pathways is the new standing model.

**Consequence: every existing figure/sweep in the paper was generated on a circuit
missing this pathway** and needs regenerating — this is now "regenerate the paper's
evidence base," not a small patch. See `~/.claude/plans/virtual-forging-owl.md` for
the full rollout.

**Smoke test — base timer-based descending arm (`debug.sh`, unchanged flags, just
the new circuit):** results improved, didn't regress. corr(Force-E,Force-F)
−0.70(L)/−0.78(R), corr(RGE,RGF) −0.72 both legs, corr(Force-E_L,Force-E_R) **−0.98**.
Notably, **the long-standing weak-flexor debug problem looks fixed for free**:
Force-F peak was previously capped ~5-7 a.u. at BS=20 Hz (see "Known debug-mode
limitation for F" above) — with the new pathway it reaches **17.2-17.3**, matching
Force-E, without touching `--freeze-bs-rg`/`--stdp-ia-rg` at all. Converged weights
sane: bs→rge/rgf ~18, cut→rge ~63 (consistent with prior baselines), ia→rge ~5.2,
ia→rgf ~5.9-6.1 (well under WMAX_IA=10 cap).

**Smoke test — force-triggered CUT at the confirmed operating point (τ=260/off=0.35/
cap=450), descending arm, new circuit, 60s debug:** corr(Force-E,Force-F)
**−0.81(L)/−0.84(R)** (better than the old-circuit confirmed values of −0.63/−0.67),
corr(Force-E_L,Force-E_R) −0.25 (anti-phase, not synchronized), `frac_at_cap`
0.06-0.07 (close to genuine, not the old circuit's clean 0.00 — expected, since this
operating point was tuned on the old circuit; Phase 1 of the rollout plan is to
re-confirm/re-tune it on the new one, not assume it transfers exactly).

**Re-confirmation, steady-state (2026-09-15) — the operating point holds; the
0.06-0.07 above was the same recovery-transient artifact found throughout the
`--consolidate` tuning work, not a persisting regression.** Re-ran the identical
operating point at two seeds (12345, 54321) and applied
`scripts/cpg_cutforce_diagnostics.py --steady-from-ms 30000` (built during the
`--consolidate` tuning rounds specifically because whole-run `frac_at_cap`
conflates an early settling period with steady-state behavior — see "Tag-and-
capture consolidation" below). Steady-state: `frac_at_cap` **0.00/0.00 in both
seeds** (even cleaner than the whole-run 0.01/0.00-0.01), corr(Force-E_L,
Force-E_R) reproducibly negative in both (−0.286, −0.529) — no synchronization.
**This operating point is confirmed still genuine on the current (post-2026-09-14)
circuit at debug scale**, closing the "not yet re-tuned" flag above. Not yet
checked at production N/BS=60Hz (see CLAUDE.md's own MN5 checklist), and this
confirmation is a prerequisite for, not a substitute for, defining a force-trigger
speed axis (see `~/.claude/plans/resilient-soaring-flamingo.md` Stage 1) — this
single point remains one fixed timing, not a speed sweep.

**Sensory arm re-tested at baseline loading, new circuit:** stable and improved — 60s
debug, corr(Force-E,Force-F) −0.81(L)/−0.79(R), `frac_at_cap` 0.05-0.06,
corr(Force-E_L,Force-E_R) properly anti-phase (though its exact value bounces between
runs at debug scale, consistent with the already-noted L/R-metric instability at this
scale — per-leg metrics are the stable/trustworthy ones here).

**Unloading-rescue attempt — round 1: loading-dependent Ia→RG weight cap
(`--wmax-ia-unloaded`, MOD_IA_RG_LOADING_GAIN).** Boosting `--ia-feedback-gain` alone
(up to 8x) never rescued force under `--cut-feedback-gain 0.1` because the real
bottleneck wasn't the input rate, it was the Ia→RG-E STDP weight ceiling (WMAX_IA=10,
deliberately low so Ia stays a light boost when CUT is present at full strength —
raising it there destroys counter-phase, already validated). Fix: `WMAX_IA` now
linearly relaxes toward a new `--wmax-ia-unloaded` (default 60) as `--cut-feedback-gain`
drops from 1→0, so Ia can only take over more excitatory drive when cutaneous input is
genuinely reduced. Bio-plausible framing: post-SCI/deafferentation upregulation of
spinal sensory gain (central sensitization), not an arbitrary knob. At full loading the
effective cap is unchanged (=10, confirmed). Result at gain=0.1 (effective cap→55):
Ia→RG-E weight grew from ~4.5 (fixed-cap case) to ~30 pA, force_e max rose from ~1.6 to
~8 (vs. the normal ~17 ceiling) — real progress, not yet a clean rescue.

**Unloading-rescue attempt — round 2: loading-dependent peak-force seed
(`CUT_FORCE_PEAK_SEED_MIN_FRAC`).** With force now reaching ~8 but the Schmitt
trigger's adaptive-peak tracker still seeded at a fixed 10 (calibrated for the normal
~17 ceiling), the seed sat permanently *above* the achievable peak, so it never adapted
and on/off thresholds were meaningless relative to the leg's actual force scale (bout
durations were a degenerate 50±0ms — chattering every tick). Fix: the seed itself now
scales down with loading too (`CUT_FORCE_PEAK_SEED_FRAC * (0.5 + 0.5*cut_feedback_gain)`
— floor at half its full-loading value, not fully to zero, to avoid a pathologically
noise-sensitive trigger). Result: bout durations became real again (257±166ms/
295±159ms, still noisy) instead of degenerate chatter, `frac_at_cap` 0.23-0.33 (down
from chattering, but not yet genuine), corr(Force-E,Force-F) still weak (−0.18 to
−0.22) — **the mechanism now runs instead of degenerating, but hasn't converged to a
clean rescue within 60s debug scale.** This needs its own proper tuning round (longer
duration, on/off-frac retuning for the lower force ceiling, maybe a gain/cap-ratio
sweep) — the same kind of multi-round search Phase 3 rounds 1-6 needed, not a
one-shot fix.

**Local tuning attempted, did not converge — moved to production-scale test
(`run_cutforce_sensory_unload.sh`) instead of continuing to guess locally.**
Systematically varied duration (60/120s), `--ia-feedback-gain` (1/2/4/6/8/12),
`--cut-feedback-gain` (0.1/0.5, air/toe), and the on/off-frac hysteresis band
(0.80/0.35 baseline, 0.85/0.25, 0.90/0.20, 0.80/0.20) at debug-small scale.
Findings: (1) `frac_at_cap` goes low (0.00-0.04) at longer duration (120s) with the
*original* on=0.80/off=0.35 thresholds — widening or narrowing the hysteresis band
made it worse (0.29-0.94), so retuning that axis isn't the lever; (2) force ceiling
converges to ~7-9 (about half the normal ~17) regardless of gain 4-12 or loading
0.1-0.5 — a real plateau, not a transient; (3) `ia->rge` weight converges to ~29-30 pA
regardless of gain, well below the relaxed cap (35-55 depending on loading) — so it's
not cap-limited either, it's a genuine dynamical fixed point of the current STDP
setup; (4) **corr(Force-E,Force-F) stayed weak (−0.13 to −0.26) across every
combination tried**, including toe-stepping (a much milder condition than air —
essentially the same weak result, −0.13/−0.15). A flat, unmoving result across such a
wide parameter search, right after two structural fixes that were each individually
necessary just to get the mechanism running at all, looks like a debug-scale ceiling,
not a dead end: **`N_IA_E`/`N_IA_F` drops from 100 (production) to 30 at
`--debug-small`** (cpg_2legs_fast.py:1116-1117) — 3.3x fewer Ia units at the same
connection density feeding the new pathway, which may cap the aggregate Ia→RG-E
current well below what production N could deliver, independent of weight/gain
tuning. This is the same debug/production divergence pattern this project has hit
repeatedly (see the original force-trigger debug-vs-production gap noted earlier in
this section). `run_cutforce_sensory_unload.sh` tests this directly: production N,
3×3 grid of loading (`--cut-feedback-gain` 1.0/0.5/0.1) × Ia compensation
(`--ia-feedback-gain` 1.0/4.0/8.0), sensory arm (`--freeze-bs-rg`), 120s, otherwise
the confirmed operating point unchanged. Not yet submitted.

### Tag-and-capture consolidation (`--consolidate`, 2026-09-15)

Motivated by the unloading-rescue plateau immediately above and the round 1-6
tuning history: every search so far looked for a **static** fixed point of
gain/cap parameters, and rounds 1, 2 and 4 kept reverting to
`frac_at_cap`-dominated (disguised-clock) results under small perturbations.
[`spinal_plasticity_as_learning_spec.md`](spinal_plasticity_as_learning_spec.md)
(literature review, written this session) argues the mechanistic reason:
vanilla STDP with a fixed `Wmax` has no way to tell "genuine progress" apart
from "an artifact of a degenerate (failsafe-forced) bout" and un-learn the
latter — every potentiation is kept permanently. Real spinal plasticity has an
explicit retention gate absent here: Sandkühler's spinal dorsal-horn E-LTP/
L-LTP (protein-synthesis-dependent, BDNF/D1-D5-gated), Grau's contingency-
gated spinal instrumental learning (non-contingent outcomes actively
*suppress*, not just fail to reinforce), and Wolpaw's two-phase H-reflex
conditioning (a fast Phase I, a slow multi-site Phase II) on exactly the
Ia→motor pathway this model has.

**Mechanism** (full design: `~/.claude/plans/resilient-soaring-flamingo.md`):
NEST's native `stdp_synapse` keeps driving `weight` exactly as before (the
fast, local, per-synapse tag-setting process). A new per-connection
`baseline` is the captured/stable component; the live tag
(`weight − baseline`) decays toward it with time constant
`--consolidate-tau-tag-ms` at every gate tick, unless a shared per-leg
PRP-pool-like accumulator crosses `--consolidate-prp-threshold` first — genuine
(real force-threshold) bout endings push the pool up via
`--consolidate-prp-gain-genuine`, failsafe-forced ones push it down (steeper,
via `--consolidate-prp-gain-forced`), and crossing the threshold freezes
`baseline := weight` for every connection in that pathway/leg (a capture
event). `Wmax` is untouched throughout — this governs retention *within* the
existing ceiling, not the ceiling itself. Applies to `CUT→RG-E` and
`Ia→RG-E/F`; `BS→RG` gets identical bookkeeping logged for measurement
symmetry only and is never written back (weak literature support for
touching `WMAX_BS`'s documented anti-runaway role — spec doc §3). Scoped to
`--cut-trigger force` only (the only mode with a genuine-vs-failsafe-forced
bout-boundary signal to gate on); raises at start-up if passed without it.

**First-pass verification (debug-small, this session) — mechanism confirmed
working, default parameters do not yet show a self-correction benefit.**
Three checks:

1. *Regression*: `debug_force.sh` unmodified (no `--consolidate`) is
   byte-for-byte unaffected — confirmed, the flag is a true no-op when absent.
2. *Mechanism sanity* (a naturally-occurring falsification-test case): the
   plain `debug_force.sh` config has no `--muscle-fatigue`, so every bout is
   failsafe-forced (`frac_at_cap`=1.00 both legs, confirmed) — a run where
   `prp_pool` can only ever decrease. With `--consolidate` on, `baseline`
   stayed exactly flat at its t=0 init the entire 10s run on all three
   pathways (`prp_pool` never left 0) while `weight` visibly drifted away
   from it (`cut→rge`: baseline 22.2, live weight 34.6) — directly confirming
   the tag/capture split is doing real, inspectable work: an unreinforced
   potentiation shows up as a persistent gap from baseline instead of being
   silently retained the way vanilla STDP would.
3. *Self-correction hypothesis* (the actual target): re-ran the round-5
   operating point (τ=260/off=0.35/cap=450, 60s, new Ia-direct-pathway
   circuit) and round 4's brittle τ=300/off=0.30 point, with vs. without
   `--consolidate`. Results were **mixed, not positive**: at the round-5
   point (already near-genuine post-architecture-fix, `frac_at_cap`
   0.00-0.01 without consolidation), turning consolidation on made it
   slightly *worse* (0.10-0.12) — with the default gain ratio
   (`prp_gain_forced`=0.30 vs `prp_gain_genuine`=0.15) and this config's
   roughly even genuine/forced mix (~69/71 events each over the run),
   `prp_pool` net-decays to 0 almost every cycle and **capture never once
   triggered** on any pathway, so `Ia→RG` sat capture-starved near its low
   init the whole run instead of being allowed to reach the level that
   otherwise helps stabilize genuine crossings. At round 4's fully-degenerate
   point (100% `frac_at_cap` from the start, zero genuine bouts ever),
   consolidation made **no difference** (still 100% both legs) — with no
   genuine bouts to ever seed a PRP increment, there is nothing for the
   mechanism to bootstrap from; it cannot rescue a starting point that never
   produces the signal it depends on.

**Conclusion: implementation is correct and behaves exactly as designed
(confirmed by direct state inspection, not just aggregate correlation
numbers), but the first-pass default constants
(`tau_tag_ms`=2000, `prp_threshold`=1.0, `prp_gain_genuine`=0.15,
`prp_gain_forced`=0.30) do not yet demonstrate the hoped-for self-correction
benefit and need their own local tuning round** — the same multi-round
process Phase 3 (rounds 1-6) needed, not a one-shot fix. Two concrete levers
for that round: (a) the genuine/forced gain ratio is currently the most
aggressive part of the default (2:1) and may be actively starving capture at
borderline operating points — worth trying a shallower ratio or a lower
`prp_threshold` first; (b) the mechanism has no way to help a 100%-forced
starting point recover on its own — if that turns out to matter, it would
need either an exploration term (occasional stochastic relaxation of the
failsafe) or accepting that this mechanism only refines already-partially-
working operating points rather than rescuing fully broken ones. Also note:
bookkeeping overhead roughly doubled with `--consolidate` on (77-83% of wall
time vs. ~50% without, at 60s debug-small) from the added per-tick
`GetStatus`/`SetStatus` round trips — fine at debug scale, but worth
profiling before any production-scale use.

**Round 1 tuning (2026-09-15) — gain ratio, not threshold or `tau_tag`, is the
lever.** All runs at the round-5 operating point (τ=260/off=0.35/cap=450,
60s, new circuit), varying `--consolidate-prp-gain-genuine`/`-forced`/
`-prp-threshold` against the no-consolidate baseline (`frac_at_cap`
0.01/0.00, corr(F-E,F-F) −0.655/−0.661, corr(F-E_L,F-E_R) −0.263) and the
shipped defaults (0.15/0.30/1.0 — confirmed above to never capture):

| genuine/forced/threshold | captures L/R | `cut→rge` gap (weight−baseline) L/R | `frac_at_cap` L/R | corr(F-E,F-F) L/R | corr(F-E_L,F-E_R) |
|---|---|---|---|---|---|
| 0.15/0.30/1.0 (shipped default) | 0 / 0 | +12.1 / +12.3 | 0.12 / 0.10 | −0.540 / −0.703 | −0.491 |
| 0.15/0.15/1.0 (symmetric) | 0 / 0 | +12.2 / +12.7 | **0.32** / 0.12 | −0.522 / −0.609 | **+0.009** |
| **0.20/0.15/1.0** | 3 / 4 | +4.9 / +1.1 | 0.10 / 0.13 | −0.541 / −0.606 | **−0.724** |
| 0.25/0.15/1.0 | 7 / 7 | +0.5 / +0.0 | 0.07 / 0.03 | −0.583 / −0.688 | **+0.273** |
| 0.20/0.10/1.0 | 7 / 7 | +1.0 / +0.0 | 0.06 / 0.06 | −0.548 / −0.660 | −0.462 |
| 0.20/0.15/0.5 | 9 / 9 | +0.1 / −0.2 | 0.07 / 0.03 | −0.476 / −0.674 | **+0.242** |

Three findings, none of them "just raise the gain":

1. **Symmetric gain (1:1) does not fix the never-captures problem** and makes
   `frac_at_cap` *worse* (0.32) than the 2:1-suppressive shipped default —
   at this operating point's roughly-even genuine/forced mix (~68 genuine /
   71 forced events over 60s), even-money gain still nets slightly negative
   most of the time, so this isn't a knob that can be nudged gently; it needs
   to cross into genuine-favoring territory before anything changes.
2. **A mildly genuine-favoring ratio (0.20/0.15, i.e. ~1.3:1) is the best
   single point found**: captures actually happen (3-4, not 0), `cut→rge`'s
   baseline moves to ~55-57 pA (up from stuck at its ~22 pA init — real
   consolidation, not a rounding artifact), and `frac_at_cap` stays
   comparable to the shipped default (no worse). It also gives by far the
   best L/R desynchronization of everything tested (corr(F-E_L,F-E_R)
   −0.724, vs. −0.263 with no consolidation at all).
3. **Pushing further in the same direction (more genuine bias, or a lower
   threshold) is not monotonically better — it actively synchronizes the
   legs.** 7-9 captures converges `cut→rge`'s baseline to ~63 pA (matching
   this pathway's known natural STDP plateau almost exactly — the mechanism
   is doing something coherent, not just drifting), but corr(F-E_L,F-E_R)
   flips **positive** at every more-aggressive setting tried (+0.273, +0.242)
   except 0.20/0.10 (−0.462, still worse than 0.20/0.15's −0.724). The
   likely mechanism: capturing too easily and too often lets both legs'
   `cut→rge` converge to the *same* stable plateau independently, removing
   the run-to-run asymmetry that keeps the two legs desynchronized — a
   genuine over-consolidation failure mode, not a tuning artifact to shrug
   off.

**Status after round 1: promising lead, not yet confirmed.** 0.20/0.15/1.0
was the best point from a single round, at a single operating point and
seed — not yet bracketed or re-checked at a second seed, the same standard
every other constant in this file was held to before being called
"confirmed" (cf. Phase 3 rounds 1-6).

**Round 2 confirmation (2026-09-15) — bracket 0.15-0.25 (genuine) ×
0.10-0.20 (forced) at a second seed (54321 vs. round 1's 12345), same
round-5 operating point, `tau_tag_ms` still untested/held at 2000 (not part
of this round's scope):**

| genuine/forced | captures L/R | `frac_at_cap` L/R | corr(F-E,F-F) L/R | corr(F-E_L,F-E_R) |
|---|---|---|---|---|
| no-consolidate (reference) | n/a | 0.03 / 0.01 | −0.607 / −0.663 | −0.286 |
| 0.15/0.10 | 3 / 3 | 0.06 / 0.06 | −0.487 / −0.653 | −0.559 |
| 0.15/0.15 (symmetric-ish) | 0 / 0 | 0.14 / 0.09 | −0.417 / −0.570 | **+0.194** |
| 0.15/0.20 | 0 / 0 | 0.10 / 0.13 | −0.374 / −0.486 | −0.637 |
| 0.20/0.10 | 7 / 7 | 0.06 / 0.03 | −0.416 / −0.601 | −0.027 |
| **0.20/0.15 (round-1 winner)** | 4 / 4 | 0.03 / 0.06 | −0.501 / −0.557 | **−0.688** |
| 0.20/0.20 | 0 / 0 | 0.19 / 0.17 | −0.377 / −0.530 | −0.356 |
| 0.25/0.10 | 11 / 11 | 0.03 / 0.06 | −0.579 / −0.605 | −0.761 |
| 0.25/0.15 | 7 / 7 | 0.03 / 0.04 | −0.670 / −0.732 | −0.656 |
| 0.25/0.20 | 4 / 4 | 0.10 / 0.06 | −0.514 / −0.648 | +0.003 |

Two findings, one confirming round 1 and one qualifying it:

1. **Round 1's two structural findings replicate exactly.** Ratios at or
   below 1:1 (0.15/0.15, 0.15/0.20, 0.20/0.20) again either never capture at
   all or capture zero times, and 0.15/0.15 again gives a desynchronized/
   positive corr(F-E_L,F-E_R) (+0.194, vs. +0.009 at seed 1) — the same
   failure mode, reproduced with an independent seed. Genuine-favoring gain
   is confirmed to be the real lever, not a seed-1 artifact.
2. **But most individual points in the bracket are seed-sensitive — only
   0.20/0.15 is not.** 0.25/0.15 scored corr(F-E_L,F-E_R) **+0.273** (bad,
   synchronized) at seed 1 and **−0.656** (good) at seed 2 — the sign
   flips on the same config with only the seed changed. 0.20/0.10 similarly
   degrades from −0.462 (seed 1) to −0.027 (seed 2). **0.20/0.15 is the one
   point that stayed good in both**: −0.724 (seed 1) → −0.688 (seed 2), a
   ~5% difference, not a coin flip. Higher-capture-count configs (7-11
   captures, at 0.25/0.10 or 0.25/0.15) can look excellent at a given seed
   (0.25/0.10 hits the best corrLR of the whole seed-2 grid, −0.761) but
   without a second seed there was no way to tell that apart from noise —
   which is exactly why this confirmation step existed.

**0.20/0.15 is now confirmed and promoted to the shipped CLI defaults**
(`--consolidate-prp-gain-genuine 0.20`, `--consolidate-prp-gain-forced 0.15`,
replacing the original guessed 2:1 ratio of 0.15/0.30, which is now confirmed
across two seeds to never capture at all at this operating point). Still
open for a future round: `tau_tag_ms` was never varied (held at 2000ms
throughout both rounds).

**Sensory-arm generalization check (2026-09-15) — the confirmed default does
NOT transfer; this is a real, arm-specific negative result, not noise.**
Re-ran the identical 9-point bracket (genuine 0.15-0.25 × forced 0.10-0.20,
seed 12345) at the same round-5 timing config (τ=260/off=0.35/cap=450) but
with `--freeze-bs-rg` added (BS→RG frozen at weak init instead of plastic —
see "Sensory-driven mode" below):

| genuine/forced | captures L/R | `frac_at_cap` L/R | corr(F-E,F-F) L/R | corr(F-E_L,F-E_R) |
|---|---|---|---|---|
| no-consolidate (reference) | n/a | 0.04 / 0.03 | −0.676 / −0.785 | −0.267 |
| 0.15/0.10 | 0 / 3 | **0.89** / 0.28 | −0.402 / −0.711 | **+0.035** |
| 0.15/0.15 | 0 / 0 | **0.89** / 0.52 | −0.466 / −0.687 | +0.137 |
| 0.15/0.20 | 0 / 0 | **1.00** / 0.89 | −0.472 / −0.593 | **+0.564** |
| 0.20/0.10 | 7 / 7 | 0.22 / 0.13 | −0.585 / −0.748 | **+0.754** |
| **0.20/0.15 (descending-arm-confirmed default)** | 3 / 3 | 0.40 / 0.32 | −0.448 / −0.711 | **+0.522** |
| 0.20/0.20 | 0 / 0 | 0.91 / 0.60 | −0.453 / −0.700 | +0.129 |
| 0.25/0.10 | 11 / 11 | 0.23 / 0.07 | −0.683 / −0.725 | −0.147 |
| 0.25/0.15 | 6 / 7 | 0.38 / 0.17 | −0.631 / −0.731 | −0.440 |
| 0.25/0.20 | 4 / 4 | 0.30 / 0.22 | −0.534 / −0.655 | +0.655 |

Every single point in the bracket makes `frac_at_cap` **worse than the
no-consolidate reference**, several catastrophically (0.15/0.20: 1.00/0.89 —
essentially fully cap-dominated), and most give a **positive**
corr(F-E_L,F-E_R) — synchronized legs, the failure mode flagged as a risk
back in the original design. The descending-arm-confirmed default (0.20/0.15)
is squarely in the bad range here (0.40/0.32 `frac_at_cap`, +0.522 corrLR).
Only 0.25/0.10 and 0.25/0.15 (both high-genuine, low-forced) come out
directionally reasonable (negative corrLR, `frac_at_cap` still elevated but
not collapsed) — a different corner of the grid than the descending arm's
best point, not the same one holding up more weakly.

**This is not a mechanism bug** — the weight-trajectory plots
(`scripts/cpg_consolidate_weights_grid.py` output) show the identical, clean
capture staircase in the sensory arm as in the descending arm; `baseline`
correctly freezes at `weight` on each capture event in both. Per-leg
force traces (`scripts/cpg_consolidate_force_stages.py`) also look
qualitatively fine throughout (corr(F-E,F-F) −0.47 to −0.84) — this
generalization failure is invisible to eyeballing single-leg force plots,
which is exactly why `frac_at_cap` and corr(F-E_L,F-E_R) exist as the
decision metrics rather than a visual check. The likely reason it differs
from the descending arm: with `BS→RG` frozen at a weak init instead of
growing toward its own ~18 pA STDP plateau, the sensory arm has less tonic
excitatory buffering, so consolidating (permanently locking in) `CUT→RG-E`'s
weight growth pushes the loop into the same positive-feedback
force-saturation regime the failsafe timeout exists to catch (see "Force-
triggered CUT" above) more readily than when BS is also plastic and sharing
the excitatory load.

**Consequence: `--consolidate`'s shipped defaults are validated for the
descending arm only.** Do not use `--consolidate` on `--freeze-bs-rg` runs
with the current defaults without its own tuning round — this bracket found
a *different* promising corner (high-genuine/low-forced, e.g. 0.25/0.10-0.15).

**Sensory-arm tuning round (2026-09-15) — the 0.25/0.10-0.15 corner has a
reproducible bad zone in its middle, not a smooth trade-off.** Fine-swept
forced ∈ {0.10, 0.1125, 0.125, 0.1375, 0.15} at genuine=0.25 fixed, first at
seed 12345 then repeated at seed 54321 to separate real structure from
per-seed noise (the same check that mattered for the descending arm):

| forced | seed1 `frac_at_cap` L/R | seed1 corrLR | seed2 `frac_at_cap` L/R | seed2 corrLR |
|---|---|---|---|---|
| no-consolidate | 0.04 / 0.03 | −0.267 | 0.03 / 0.04 | −0.670 |
| **0.10** | 0.23 / 0.07 | **−0.147** | 0.10 / 0.09 | **−0.262** |
| 0.1125 | 0.20 / 0.07 | +0.299 | 0.23 / 0.16 | +0.527 |
| 0.125 | 0.12 / 0.07 | +0.257 | 0.22 / 0.13 | +0.714 |
| 0.1375 | 0.23 / 0.20 | −0.072 | 0.23 / 0.17 | +0.318 |
| 0.15 | 0.38 / 0.17 | −0.440 | 0.28 / 0.23 | +0.076 |

Two-seed comparison separates a real effect from what first looked like
random scatter: **0.10 is the only forced value that stays negative
(anti-phase) in both seeds** — the interior of the box (0.1125-0.1375) gives
a **positive** (synchronized) corrLR in *both* seeds, a reproducible bad zone,
not noise. 0.15, which looked like the round's best point at seed 1
(−0.440), flips to positive (+0.076) at seed 2 — unreliable, same pattern as
0.25/0.15 and 0.20/0.10 in the earlier descending-arm confirmation. Nothing
in this box fully restores `frac_at_cap` to the no-consolidate level (best
case 0.10/0.09 at seed 2, still ~3x worse than baseline) — this remains a
real, open regression for the sensory arm, not a solved problem.

**Working recommendation for `--freeze-bs-rg` runs: `--consolidate-prp-gain-
genuine 0.25 --consolidate-prp-gain-forced 0.10`** — the only point in the
tested region that is reproducibly better than doing nothing on
corr(F-E_L,F-E_R) without a compensating cap-domination regression as bad as
the rest of the box.

**Methodological correction (2026-09-15) — the "regression" above was
measuring the wrong window; the user's own read of it turned out to be
right.** The user pointed out that a `frac_at_cap` regression under
`--consolidate` is exactly what they'd expect, framed against the opposite
problem this project has previously had: vanilla STDP recovering the walking
pattern *too fast* to be a plausible stand-in for real rehabilitation timescales.
That prompted checking directly, with `bouts_from_cut_on` windowed into
0-20s/20-40s/40-60s, whether the whole-run `frac_at_cap` numbers above were
reporting a genuine steady-state problem or an extended-but-resolving early
transient. They were overwhelmingly the latter: e.g. sensory-arm 0.25/0.10
shows `frac_at_cap` 0.73/0.23 in the first 20s but 0.00/0.00 for the rest of
the run — statistically indistinguishable from the no-consolidate reference's
own steady state (also 0.00/0.00 after its first 20s) — and this holds for
essentially every genuine=0.25 point tested (0.10 through 0.20), not just the
working recommendation. The whole-run average was reporting the length of a
recovery period, not a persistent failure — which is the bio-plausible
behavior `--consolidate` was introduced to get (a genuine settling-in period
instead of instant convergence), not a bug.

This has a real consequence beyond the sensory arm: **the whole-run
`frac_at_cap`/corr(F-E_L,F-E_R) numbers used to score every bracket point in
both this round and the round-2 descending-arm confirmation are contaminated
by this same transient**, and restricting to steady-state
(`--steady-from-ms 30000`, now supported directly by
`scripts/cpg_cutforce_diagnostics.py`) changes some of those numbers
substantially:

| config (arm, gains) | whole-run corrLR | **steady-state (t≥30s) corrLR** |
|---|---|---|
| descending, 0.20/0.15, seed 12345 | −0.724 | **−0.839** |
| descending, 0.20/0.15, seed 54321 | −0.688 | **−0.839** |
| descending, 0.25/0.15, seed 12345 | +0.273 | +0.297 (still bad) |
| descending, 0.25/0.15, seed 54321 | −0.656 | −0.720 (still good — genuinely bistable) |
| sensory, 0.25/0.10, seed 12345 | −0.147 | **−0.286** (≈ no-consolidate's −0.285) |
| sensory, 0.25/0.10, seed 54321 | −0.262 | **−0.280** (≈ no-consolidate's −0.720... note seed2's own no-consolidate baseline is a stronger −0.720, so 0.25/0.10 is *not* quite matching baseline at seed 2, unlike seed 1) |
| sensory, 0.25/0.15, seed 12345 | −0.440 | −0.829 (excellent) |
| sensory, 0.25/0.15, seed 54321 | +0.076 | +0.316 (still bad — genuinely bistable, not a transient artifact) |
| sensory, 0.25/0.1125-0.1375 (both seeds, 6 runs) | −0.07 to +0.53 | +0.01 to +1.00 (positive in all 6 at steady state, incl. the one whole-run value that had looked negative — the "bad zone" is real, not a transient) |

**Two corrected conclusions:**
1. **The descending-arm confirmed default is confirmed more strongly than
   originally reported**, not less: 0.20/0.15's steady-state
   corr(F-E_L,F-E_R) is −0.839 in *both* seeds — not just same-sign but
   numerically identical, the tightest replication of any point tested in
   either arm. No change to the shipped default.
2. **Cap-domination is not the sensory arm's real, persistent problem — L/R
   phase-locking is.** Every tested genuine=0.25 point reaches clean,
   non-cap-dominated bout timing at steady state; what actually
   distinguishes them is whether the legs settle into anti-phase (0.10: good
   in both seeds) or in-phase (0.1125-0.1375: bad in both seeds) lock, with
   0.15 sitting on a genuine bistable boundary (excellent at seed 1, bad at
   seed 2) rather than being noise either way. The working recommendation
   (genuine=0.25, forced=0.10) is unchanged by this correction — if anything
   it's better supported now, since its steady-state numbers show it
   converging to a real anti-phase attractor rather than merely averaging out
   noise — but the previous framing ("`frac_at_cap` stays ~3x worse than
   no-consolidate, a real open regression") was measuring recovery-period
   length, not steady-state quality, and should not be read as a persisting
   defect. **Whole-run metrics should not be used alone for `--consolidate`
   tuning going forward — always pair with `--steady-from-ms` on
   `scripts/cpg_cutforce_diagnostics.py`.**

**Full 3×3 sensory-arm bracket, both seeds, steady-state (t≥30s) — completes
the grid the working recommendation was drawn from.** The earlier
sensory-arm bracket (round labeled "generalization check") only had seed-1
coverage across the full 3×3 grid; seed 54321 was filled in for the
remaining 7 cells to check the same two-seed standard applied everywhere
else in this file:

| genuine/forced | seed1 `atCap` L/R | seed1 corrLR | seed2 `atCap` L/R | seed2 corrLR | verdict |
|---|---|---|---|---|---|
| no-consolidate | 0.00/0.00 | −0.285 | 0.00/0.00 | −0.720 | reference |
| 0.15/0.10 | 0.82/0.03 | −0.147 | 0.20/0.00 | −0.316 | still cap-dominated (seed1 L) |
| 0.15/0.15 | 0.85/0.35 | −0.151 | 0.91/0.85 | +0.866 | cap-dominated both seeds, corrLR flips |
| 0.15/0.20 | 1.00/0.91 | +0.891 | 0.94/0.88 | +0.902 | cap-dominated **and** synchronized, both seeds |
| 0.20/0.10 | 0.00/0.00 | +1.000 | 0.00/0.00 | −0.527 | genuine both seeds, corrLR flips |
| 0.20/0.15 | 0.00/0.09 | +0.938 | 0.20/0.00 | +0.605 | genuine, but **synchronized in both seeds** |
| 0.20/0.20 | 0.85/0.47 | −0.297 | 0.71/0.82 | −0.200 | cap-dominated, both seeds |
| **0.25/0.10** | **0.00/0.00** | **−0.286** | **0.00/0.00** | **−0.280** | **genuine + anti-phase, both seeds** |
| 0.25/0.15 | 0.00/0.00 | −0.829 | 0.00/0.00 | +0.316 | genuine both seeds, corrLR flips (bistable) |
| 0.25/0.20 | 0.00/0.00 | +1.000 | 0.03/0.00 | +0.632 | genuine, but **synchronized in both seeds** |

The completed grid separates into three clean groups rather than a noisy
scatter: **genuine=0.15 is simply too weak** to reliably escape
cap-domination in this arm (still cap-dominated in 5 of 6 seed×forced
combinations); **genuine=0.20/0.25 paired with forced=0.15 or 0.20 reliably
escapes cap-domination but reliably synchronizes the legs instead**
(positive corrLR in *both* seeds at all four such cells — a real,
reproducible failure mode, not scatter); and **0.25/0.10 is the only cell in
the entire 3×3 grid that is both genuine and anti-phase in both seeds**, with
its two corrLR values (−0.286, −0.280) nearly identical — the same tight
cross-seed replication quality that confirmed 0.20/0.15 for the descending
arm. This is the strongest evidence yet for the working recommendation and
completes the sensory-arm bracket to the same two-seed standard as the
descending arm, though it remains a working recommendation, not a promoted
CLI default — see caveats above (one timing operating point, `tau_tag_ms`
unswept).

**Summary: every `--consolidate` config tested this session, sorted into
healthy vs. pathological.** The round-by-round tables above are the lab
notebook (chronological, includes dead ends); this collects the same numbers
by outcome instead, using the two failure modes this document already
established (cap-domination: `frac_at_cap` elevated; leg-synchronization:
corr(F-E_L,F-E_R) positive) plus two outcomes that are neither: a config that
never captures at all (**inert** — the mechanism is a no-op, not a failure,
but doesn't do anything either), and a config whose verdict **flips sign
between seeds** (**bistable** — confirmed reproducible in both directions,
not noise, but unusable as a fixed default either way). Steady-state
(t≥30s) numbers are used wherever computed; whole-run numbers are used
otherwise and marked accordingly — see the methodological correction above
for why that distinction matters.

*Descending arm (τ=260/off=0.35/cap=450):*

| genuine/forced/threshold | seed(s) | corrLR | Verdict |
|---|---|---|---|
| **0.20/0.15/1.0** | 12345 **and** 54321 | **−0.839 / −0.839** (steady-state, identical) | **HEALTHY — shipped CLI default** |
| 0.15/0.10/1.0 | 54321 only | −0.559 (whole-run) | Healthy-looking, single-seed only — not cross-confirmed |
| 0.25/0.10/1.0 | 54321 only | −0.761 (whole-run) | Healthy-looking, single-seed only — not cross-confirmed |
| 0.15/0.15/1.0 | 12345 **and** 54321 | +0.009 / +0.194 | **PATHOLOGICAL — synchronized, both seeds** |
| 0.25/0.20/1.0 | 54321 only | +0.003 | Pathological (borderline-synchronized) |
| 0.20/0.15/**0.5** | 12345 only | +0.242 | **PATHOLOGICAL — synchronized** (threshold, not just gain ratio, matters) |
| 0.25/0.15/1.0 | 12345 **and** 54321 | +0.297 (steady-state) / −0.720 (steady-state) | **BISTABLE — flips sign between seeds, do not use** |
| 0.20/0.10/1.0 | 12345 **and** 54321 | −0.462 / −0.027 | Unreliable — collapses toward zero at the second seed |
| 0.15/0.30/1.0 (original guess) | 12345 only | −0.491 | Inert — 0 captures, mechanism never engages |
| 0.15/0.20/1.0 | 54321 only | −0.637 | Inert — 0 captures despite an incidentally-OK corrLR |
| 0.20/0.20/1.0 | 54321 only | −0.356 | Inert — 0 captures |

*Sensory arm (`--freeze-bs-rg`, same timing config), full 3×3 bracket +
genuine=0.25 fine-sweep, steady-state, both seeds throughout:*

| genuine/forced | corrLR (seed1 / seed2) | Verdict |
|---|---|---|
| **0.25/0.10** | **−0.286 / −0.280** (nearly identical) | **HEALTHY — working recommendation, confirmed both seeds** |
| 0.15/0.10 | −0.147 / −0.316 | Pathological — cap-dominated at seed1 (`atCap` 0.82) |
| 0.15/0.15 | −0.151 / +0.866 | Pathological — cap-dominated both seeds, synchronizes at seed2 |
| 0.15/0.20 | +0.891 / +0.902 | **PATHOLOGICAL — worst case: cap-dominated AND synchronized, both seeds** |
| 0.20/0.15 | +0.938 / +0.605 | Pathological — genuine bout timing, but synchronized in both seeds |
| 0.20/0.20 | −0.297 / −0.200 | Pathological — cap-dominated both seeds |
| 0.25/0.1125 | positive, both seeds | Pathological — synchronized both seeds (interior "bad zone") |
| 0.25/0.125 | positive, both seeds | Pathological — synchronized both seeds (interior "bad zone") |
| 0.25/0.1375 | positive, both seeds | Pathological — synchronized both seeds (interior "bad zone") |
| 0.25/0.20 | +1.000 / +0.632 | Pathological — genuine bout timing, but synchronized both seeds |
| 0.20/0.10 | +1.000 / −0.527 | **BISTABLE — flips sign between seeds, do not use** |
| 0.25/0.15 | −0.829 / +0.316 | **BISTABLE — flips sign between seeds, do not use** |

**Reading the two tables together**: exactly one point per arm is healthy —
0.20/0.15 (descending, shipped) and 0.25/0.10 (sensory, working
recommendation) — and both are healthy for the same reason, tight
same-sign, near-identical corrLR across two independent seeds, not just a
good number once. Everything else sorts into one of three unhealthy
buckets, and they are different failures needing different fixes: inert
configs need a stronger genuine-favoring push before they'll do anything at
all; pathological configs need to move *away* from wherever they are,
generally toward less capture (sensory arm) or more (descending arm's
symmetric case) or a different threshold entirely; bistable configs are the
most dangerous of the three to mistake for progress, since a single-seed run
can make one look like either a clean win or a clean failure at random.

### Force-trigger speed axis (Stage 1, 2026-09-15) — partial progress, fast side only

Prompted by the question of whether the project is ready for a production
MN5 run across "all speeds" with `--consolidate`: force-trigger mode has no
existing speed concept at all. `--step-period-ms` (the timer-mode speed
knob) only paces the within-bout Ia-E heel→toe ramp here, not cycle length;
bout duration is emergent from `--fatigue-tau-onset-ms` × `--cut-force-off-
frac`, and rounds 1-6 only ever searched for **one** good point (τ=260/
off=0.35/cap=450, re-confirmed above), never a speed family. This is Stage 1
of `~/.claude/plans/resilient-soaring-flamingo.md`'s staged roadmap:
establish 2-4 more (τ, off-frac) points bracketing the confirmed one,
descending arm, 2 seeds, steady-state metrics.

**Fast direction — one clean point found, but the achievable range looks
narrow.** τ=200/off=0.30 is genuine and reproducible in both seeds
(`frac_at_cap` 0.00/0.00 and 0.01/0.00; stance duration tight at 350±0-12ms,
tighter than the confirmed point's own ±27ms; corr(F-E_L,F-E_R) −0.399 and
−0.619, both anti-phase). But measuring **full gait-cycle period**
(stance-onset to stance-onset, not just stance duration) shows it's only
modestly faster than the confirmed point: **800ms vs. 850ms, ≈6%** — not the
kind of spread timer mode's 1200/520/350ms (3.4×) speed grid covers. Two
intermediate attempts on the way there failed outright: τ=200/off=0.35 gave
genuine bouts (`frac_at_cap`=0.00) but wildly variable duration (±135-159ms,
~60% relative) and inconsistent corrLR across seeds (+0.165, −0.017) — not
usable despite passing the cap-domination check.

**Slow direction — two attempts, both failed, harder than the fast side.**
τ=280/off=0.35 (a modest +20ms step from the confirmed τ=260) **collapsed to
100% cap-domination in both seeds** (`frac_at_cap` 1.00/1.00, duration
exactly 450±0ms — a disguised clock). τ=300/off=0.40 (loosening off-frac
further, per round 4's old-circuit finding that off must loosen alongside
higher τ) didn't fix it either — **~50% cap-domination in both seeds**
(0.50/0.51, 0.49/0.48), a reproducible bimodal mix of genuine and
failsafe-capped bouts, not an improvement. The medium operating point
appears to sit much closer to its slow-side failure boundary than its
fast-side one on the current circuit — a real, currently unexplained
asymmetry, not yet root-caused.

**Status: Stage 1 not complete.** One additional confirmed point (fast,
modest speedup) plus two failed slow attempts. Before a 3+-point speed axis
can be called established: (a) the slow direction needs a different lever
than "more of the same" — candidates not yet tried: loosening `--cut-force-
on-frac` (never swept, fixed at 0.80 through every round including this
one), or accepting that a genuinely slower bout may require raising
`--cut-max-stance-ms` itself (currently untouched by design, since a moving
cap was previously how "disguised clock" results were diagnosed — raising it
deliberately as part of defining "slow" is a different, defensible use, but
changes what `frac_at_cap` even means for that point and should be flagged
explicitly if done); (b) even the fast side's ~6% spread needs a second,
more distinct fast point before "fast" is a meaningfully different speed
rather than a slightly-tighter version of medium.

**Swing has no closed-loop signal at all — it is a pure timer, always
(2026-09-15).** Checked directly: `cut_force_gate()` never reads `force_f`;
the only signal that ends swing is `fe >= on_thr` (extensor force rising),
which has nothing to do with the flexor/swing side itself. Measuring stance
and swing bout durations *separately* (the existing diagnostics script only
ever reported stance) shows **swing sits at exactly `--cut-max-swing-ms`
with zero variance in every config tested, including the confirmed
"genuine" medium point** — swing has always been failsafe-timed, not
force-triggered; there is no bug here, the model simply has no sensory
variable analogous to hip/limb position that would let swing end any other
way. This means `--cut-max-swing-ms` is not a backstop to avoid touching for
swing the way it is for stance — it is the *only* thing that sets swing
duration, so it looked like a natural, low-risk lever for the speed axis.

**Tested that directly — it isn't low-risk, because stance and swing are
coupled through muscle fatigue *recovery*, not independent.** Two attempts,
stance parameters held at their already-confirmed values in both cases:
- **Slow attempt 3**: medium's exact stance config (τ=260/off=0.35/stance-cap
  450, unchanged) + swing cap raised 450→600ms alone. Expected stance to be
  unaffected since nothing about it changed. Instead **stance itself
  collapsed to 100% cap-domination in both seeds** — a longer swing gives
  `fatigue_e` more time to clear via `--fatigue-tau-recovery-ms` before the
  next stance, so that stance starts less fatigued and takes measurably
  longer to re-fatigue down through `off_thr`, past the unchanged 450ms
  stance cap. The two phases are coupled through shared fatigue state, not
  independent just because they're gated by separate flags.
- **Fast attempt 3**: the confirmed fast stance config (τ=200/off=0.30) +
  swing cap lowered 450→350ms alone. Stance stayed genuine (`frac_at_cap`
  0.00/0.00 both seeds) but reintroduced the same failure as the earlier
  τ=200/off=0.35 attempt — high bout-duration variance (±144-160ms) and
  inconsistent corrLR across seeds (−0.235, +0.249) — plausibly the same
  coupling in reverse (less recovery time causing bout-to-bout drift).

**Revised status: Stage 1 needs a "scale the whole clock together" approach,
not single-parameter nudges.** Changing stance-side timing (τ, off-frac)
alone breaks stance directly; changing swing-side timing (swing cap) alone
breaks stance indirectly through fatigue recovery. The remaining untried
approach is scaling `--fatigue-tau-onset-ms`, `--fatigue-tau-recovery-ms`,
`--cut-max-stance-ms`, and `--cut-max-swing-ms` together (proportionally,
preserving the confirmed point's ratios) rather than moving one axis at a
time — this has not yet been attempted and is a reasonable next step, not
a confirmed fix.

**Tested — the scaling approach works cleanly for slow, but not for fast
(2026-09-15).** Confirms the hypothesis for one direction and rules it out
as a general fix for the other:

- **Slow, confirmed: 1.3× scale** (τ=340, recovery=780, both caps=585,
  off-frac unchanged at 0.35). Genuine and tight in both seeds
  (`frac_at_cap` 0.00/0.00; stance 535±23/537±22ms seed 1, 513±22/513±22ms
  seed 2 — **~4% relative variability, tighter than the confirmed medium
  point's own ~9%**), and corr(F-E_L,F-E_R) is nearly identical across seeds
  (−0.825, −0.832 — the same tight cross-seed match quality that confirmed
  0.20/0.15 for `--consolidate` and the medium point itself). Full gait
  cycle: **1135/1113ms vs. medium's 850ms, ≈31-34% slower** — a genuinely
  distinct speed, not a marginal nudge. **This is now a second confirmed
  force-trigger operating point** (slow), alongside the original medium one.
- **Fast, still not solved: 0.75× scale failed.** (τ=195, recovery=450,
  caps=340). Steady-state `frac_at_cap` 0.32-0.59 (partial cap-domination,
  mixed genuine/capped bouts) and corr(F-E_L,F-E_R) **positive in both
  seeds** (+0.342, +0.111 — synchronized, the same failure mode as every
  other fast attempt). Would-be full cycle ≈630-646ms (≈25-26% faster,
  a meaningfully distinct speed *if* it had worked) — but it isn't genuine,
  so it doesn't count.

**Net Stage 1 status: 2 of 3 speed points now confirmed (medium, slow);
fast remains open after 5 distinct attempts** (τ=200/off=0.35;
swing-cap-350-alone; 0.75× uniform scale; plus the earlier τ=200/off=0.30,
which is genuine but only a ~6% speedup, not a distinct fast point). The
fast direction is consistently harder than the slow direction across every
method tried so far (single-axis and proportional-scaling alike) — a
real, reproducible asymmetry in this circuit that a future round should
treat as the object of study itself (why does speeding up specifically
desynchronize the legs?), rather than keep attacking with the same class of
parameter nudge.

**Investigating the asymmetry: why speeding up desynchronizes the legs
(2026-09-15).** Two absolute-time constants are held fixed across every
Stage 1 attempt while bout duration shrinks for faster configs — and both
turn out to matter, for different reasons.

**Confirmed factor 1 — `--lead-offset-ms` doesn't scale with cycle period.**
It's a fixed 150ms absolute value throughout. As a *fraction of stance
duration* it stays a safe 28-38% for the two confirmed-good points (medium
400ms stance, slow 535ms) but balloons to 53-61% for every failed fast
attempt (245-283ms stance) — directly the failure zone this file already
documented for oversized priming windows (§ "Force-triggered CUT": 400ms
priming against a ~1000ms cycle flipped corr(F-E_L,F-E_R) to +0.41). Testing
it directly on the 0.75× scaled-fast config (150→112ms, same proportion):
steady-state `frac_at_cap` dropped from 0.32-0.59 to 0.22-0.36, and one
seed's corr(F-E_L,F-E_R) flipped from badly-synchronized (+0.342/+0.111) to
strongly anti-phase (−0.700) — a real, substantial improvement, confirming
this is a genuine contributing cause, not coincidence. Not a full fix on its
own: residual cap-domination and one seed's near-zero (ambiguous) corrLR
remained.

**Confirmed factor 2 — `--rate-update-ms` (the gate-check tick) doesn't scale
down either, but the "obvious" fix backfires.** Inspecting the raw
steady-state bout-duration sequence for the lead-offset-corrected config
showed durations locked to *only two discrete values*, 300 or 350ms — never
anything between — because the failsafe check only runs at 50ms ticks, and
340ms's failsafe fires at the first tick ≥340ms, which lands on 350
regardless of how close to genuinely completing the bout actually was. At
medium/slow speeds this tick is a fine fraction of bout duration (~9-12%);
at this fast config's ~300ms bouts it's a much coarser ~17%, so a
near-miss and a comfortable margin both round to the identical "at cap"
verdict. **Tested the obvious fix directly (`--rate-update-ms`/
`--simulate-chunk-ms` 50→20ms) and it made things dramatically worse, not
better**: mean bout duration collapsed to 40-84ms with variance exceeding
the mean (±78-128ms) — not finer resolution of the same rhythm, but outright
Schmitt-trigger **chattering**. `peak_e_est = max(fe, peak_e_est)` and the
on/off-threshold check both run every rate-update tick; sampling `force_e`
more often lets the per-bout running peak track force noise more precisely,
which makes the relative on/off thresholds cross spuriously on tiny
fluctuations instead of the real envelope. The coarse tick isn't just a
measurement artifact to fix by sampling faster — it's doing real,
load-bearing noise-averaging that a naive resolution increase removes.

**Answer to "why does speeding up desynchronize the legs": at least two
independent, confirmed mechanisms, not one.** (1) A fixed-duration symmetric
priming window becomes a larger fraction of a shorter cycle, pushing toward
the already-known over-priming synchronization failure. (2) A fixed gate
tick becomes a coarser fraction of a shorter cycle, inflating apparent
cap-domination — but the fix isn't simply finer sampling, since the same
tick also low-pass-filters the peak-tracking Schmitt trigger against noise,
and removing that naively causes chattering instead. A genuine fast
operating point likely needs `--lead-offset-ms` scaled down (factor 1,
confirmed to help) *and* a noise-robust way to shrink the effective tick
period (factor 2 — not yet solved; simple down-scaling doesn't work, and
whatever replaces it needs to preserve peak-tracking noise rejection while
still resolving a shorter bout).

**Factor 2 follow-up: an explicit noise filter, decoupled from tick rate
(2026-09-15) — implemented, tested, still doesn't solve it.** Added
`--cut-force-filter-tau-ms` (`MOD_CUT_FORCE_TRIGGER`, default 0 = off, exact
original behaviour): an exponential low-pass on `force_e` feeding both
`peak_e_est` and the on/off threshold comparisons, with its own time
constant independent of `--rate-update-ms`, so the tick can shrink without
losing noise rejection. Two problems found in sequence, both real:

1. **First test (τ_filter=30ms at rate-update=20ms) still chattered** —
   durations ~24-32ms, barely above the tick floor. Root cause found by
   inspection: the filter state was only re-seeded at *stance* onset
   (reusing the existing `peak_e_est` reset point), so a swing phase
   shorter than the filter's own settling time left `force_e_filt` carrying
   a stale, lagging estimate from the *previous* stance into the new bout
   — filter lag dominating a bout shorter than itself, not noise rejection.
   Fixed: re-seed `force_e_filt[side] = None` at *every* phase transition,
   not just stance onset.
2. **A stronger filter (τ=100ms) made it worse before the fix** (durations
   collapsed to exactly the 20ms tick floor, corr(F-E,F-F) degraded to
   ~-0.05, essentially no rhythm) — consistent with (1): a filter slower
   than the bout it's supposed to smooth doesn't stabilize the loop, it
   destabilizes it (lagged feedback into a hysteresis/relay controller is a
   classic route to a new oscillation mode, not noise suppression).
3. **After the fix, re-tested τ=30ms at rate-update=20ms: still broken**,
   though less severely — durations 28-90ms (still nowhere near the ~280
   -300ms target), std comparable to or exceeding the mean, and markedly
   asymmetric between legs in one seed (L=90ms vs R=28ms). corr(F-E,F-F)
   improved to -0.32 to -0.59 (better than the pre-fix -0.05 to -0.36, but
   still well short of working configs' -0.6 to -0.8).

**Conclusion: the filter approach is real, the bug fix was necessary, and
neither is sufficient on its own.** Something beyond peak-tracking noise is
also unscaled at fine ticks — a plausible next suspect, not yet tested: the
Ia-E heel→toe sub-group pacing (`SUB_STANCE_MS`, derived from
`--step-period-ms`/`--stance-fraction`/`--n-ia-groups`, still at its
original ~167ms value throughout every Stage 1 attempt) is now much larger
than the collapsed ~20-90ms bouts, so only the first (weakest, 60Hz)
sub-group ever fires — a third absolute-time constant that was never scaled
alongside the others, on top of `--lead-offset-ms` (factor 1) and the tick/
filter (factor 2). The pattern across all three is the same: this circuit
has more independent absolute-time constants than were ever exercised by
rounds 1-6's single-operating-point search, and a coherent fast operating
point likely needs all of them scaled together, not one or two at a time.

**Also surfaced during this work, orthogonal to the filter itself but
consequential for the rest of this file's methodology: identical code, same
seed, same flags, produced different corr(F-E_L,F-E_R) on two consecutive
runs** (-0.318 vs +0.298, medium point, filter off) — confirmed not a code
regression (per-leg `frac_at_cap` and corr(F-E,F-F) matched the established
range in both runs; only corrLR differed) but genuine run-to-run
nondeterminism, most likely from NEST's multi-threaded execution affecting
spike-arrival order in ways a fixed `--seed` doesn't fully pin down. This is
a re-surfacing of an already-documented caution in this file ("L/R-metric
instability at debug scale... per-leg metrics are the stable/trustworthy
ones here"), but this session's `--consolidate` and Stage 1 tuning leaned on
corrLR more heavily than that caution suggests was warranted. Doesn't
invalidate prior 2-seed-confirmed results (per-leg metrics were consistent
throughout, and the confirmed points' corrLR matches were tight enough to
be real signal, not just luck — e.g. 0.20/0.15's steady-state -0.839 in
*both* seeds), but any *single* corrLR data point, including ones in this
file, should be read with this in mind, and a genuine future confirmation
pass would benefit from more than 2 repeats given now-demonstrated
same-seed variance.

### Sensory-driven mode (`--freeze-bs-rg`, now just freezing BS since Ia→RG is always on — WMAX_IA=10)

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
| `MOD_FREEZE_BS` | `--freeze-bs-rg`: BS→RG-E/RG-F static (no STDP), held at weak lognormal init (W_INIT_BS). BS becomes fixed tonic drive; Ia→RG and CUT→RG keep training regardless (see MOD_IA_RG_STDP). |
| `MOD_IA_RG_STDP` | **Always wired, always plastic homonymous Ia→RG** (Ia-E→RG-E, Ia-F→RG-F, Wmax=WMAX_IA=10, density P_IA2RG_STDP=0.5) — matches the reference architecture diagram's direct excitatory Ia→RG projection (distinct from the Ia→InE/InF reciprocal-inhibition loop, MOD_IA_LOOP). A third standing plastic pathway alongside BS→RG and CUT→RG in every mode (2026-09-14 — previously gated behind `--stdp-ia-rg`, opt-in only for the sensory-learning arm; see "Core architecture fix" below for why). `--wmax-ia`/`--p-ia2rg` still override the cap/density. |
| `MOD_CONSOLIDATE` | `--consolidate`: opt-in (OFF by default), requires `--cut-trigger force`. Replaces vanilla STDP's "every potentiation kept forever, up to Wmax" retention with tag-and-capture consolidation on `CUT→RG-E` and `Ia→RG-E/F`: `weight` still evolves via native `stdp_synapse` (unchanged, the fast/local tag-setting process); a new per-connection `baseline` is the captured/stable component, and the live tag (`weight − baseline`) decays toward it with time constant `--consolidate-tau-tag-ms` unless a shared per-leg PRP-pool-like accumulator crosses `--consolidate-prp-threshold` first (genuine force-threshold bout endings push it up via `--consolidate-prp-gain-genuine`, failsafe-forced endings push it down via `--consolidate-prp-gain-forced`, matching Grau's finding that non-contingent outcomes actively suppress rather than merely fail to reinforce). `Wmax` is untouched — this governs retention *within* the existing ceiling, not the ceiling itself. `BS→RG` (when not frozen) gets identical bookkeeping logged for measurement symmetry only and is never written back to NEST — literature support for touching `WMAX_BS`'s documented anti-runaway role is weak (see [`spinal_plasticity_as_learning_spec.md`](spinal_plasticity_as_learning_spec.md) §3). See "Tag-and-capture consolidation" below for the literature basis and first-pass verification results. |
| `--ia-feedback-gain` | Multiplicative gain on closed-loop Ia rate. 1.0 baseline / 0.5 toe stepping / 0.1 air stepping (Courtine/Lavrov SCI paradigm). |
| `--cut-feedback-gain` | Multiplicative gain on cutaneous CUT stance drive (loading-dependent paw contact). Scaled with loading alongside `--ia-feedback-gain`; the external Ia-E heel→toe ramp (stim pacing) stays at full. |
| `--ia-ext-f-hz` | MOD_FLEXOR_AFFERENT: rate (Hz) of the external flexor swing-afferent (hip/flexor-stretch signal; Grillner & Rossignol 1978). Drives RG-F directly + InF during swing, clocking the flexor symmetrically to the stance Ia-E ramp. 0 = off (intrinsic-only flexor); 80 = on. Un-gated by loading (joint-position, not load-based). |
| `--stdp-lambda` | Override STDP LAMBDA (default 1e-3). Bio-plausible range 5e-4 to 5e-3 (Bi & Poo 1998; Morrison 2007). |
| `--dump-connectivity` | Build the network, write per-connection WEIGHT + DELAY arrays for all 18 named projections to an HDF5, then exit (no sim — runs in seconds at production N). Feeds the connectivity-statistics figure (`scripts/cpg_connectivity_figure.py`) and CSV table. Static weights are delta-valued; plastic are lognormal-init; delays follow the rat `length_velocity` preset + 0.2 ms jitter. |
| `--freeze-bs-rg` | MOD_FREEZE_BS: freeze BS→RG (static at weak init). Removes descending plasticity only; Ia→RG and CUT→RG keep training (always on, see MOD_IA_RG_STDP). Frozen runs drop `bs->rge`/`bs->rgf` from the tracked plastic-weight keys. |
| `--stdp-ia-rg` | **DEPRECATED/no-op** (2026-09-14): plastic homonymous Ia→RG is now always wired (MOD_IA_RG_STDP). Flag kept only so old scripts passing it don't break. |
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
