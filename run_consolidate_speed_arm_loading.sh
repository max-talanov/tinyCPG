#!/bin/bash -l
#SBATCH --job-name=CPG_CONSOL_SAL
#SBATCH --output=Nest_consol_sal_%A_%a.slurmout
#SBATCH --error=Nest_consol_sal_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-11
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=acc
#
# STAGE 4 of ~/.claude/plans/resilient-soaring-flamingo.md -- the first
# MN5 submission for --consolidate under --cut-trigger force. Do not extend
# this script's scope without re-reading the CLAUDE.md sections below; it
# is deliberately narrower than "all speeds and sensory ablation modes"
# because the underlying force-trigger speed axis (Stage 1) and per-speed/
# per-loading consolidate re-tuning (Stages 2-3) are not both complete --
# see "MN5 readiness verdict" in CLAUDE.md for the full reasoning.
#
# SCOPE (4 cells x 3 seeds = 12 tasks):
#   Speeds:  medium (step-period=520ms, the paper's ~13.5cm/s anchor) and
#            slow (step-period=1200ms, ~6cm/s) -- FAST IS EXCLUDED. Six
#            distinct local attempts (single-axis nudges, proportional
#            0.75x fatigue-only scaling, lead-offset correction, two filter
#            strengths, full time-constant-family scaling incl.
#            SUB_STANCE_MS) all failed to find a fast point that is both
#            genuine (non-cap-dominated) AND non-bistable in L/R phase
#            locking across seeds -- see "Fast direction, attempt 6" in
#            CLAUDE.md. Do not add a fast row to this script until that is
#            solved in its own pass.
#   Arms:    descending (BS->RG plastic, default) and sensory
#            (--freeze-bs-rg). Each uses its OWN confirmed --consolidate
#            gain pair -- 0.20/0.15 descending, 0.25/0.10 sensory -- these
#            do NOT transfer between arms (confirmed, two seeds, see
#            "Sensory-arm generalization check" in CLAUDE.md).
#   --consolidate is MEDIUM-SPEED ONLY. Stage 2 tested the medium-confirmed
#            descending gain pair at the slow point across 12 configs (gain
#            ratio, capture threshold, tag time constant) and found it
#            actively harmful there in every case -- cap-domination,
#            synchronization, or degenerate chattering, never genuine +
#            stable (see "Stage 2" in CLAUDE.md). The two slow-speed cells
#            below therefore run WITHOUT --consolidate, as a no-consolidate
#            control (already confirmed excellent at slow: steady-state
#            corr(F-E_L,F-E_R) -0.822) -- not a placeholder, a deliberate
#            choice not to ship a gain pair known to regress that speed.
#   Loading: FULL WEIGHT-BEARING ONLY -- toe and air stepping are excluded.
#            Stage 3's seed-1 screen found the confirmed medium timing
#            config (tau=260/off=0.35/cap=450) fails EVEN WITHOUT
#            --consolidate at toe stepping (steady-state atCap 1.00/1.00,
#            both arms -- a disguised clock) and at air stepping (bout
#            duration collapses to the 50ms rate-update-ms tick floor with
#            zero variance, both arms -- degenerate chattering, not a
#            genuine rhythm at all). This is a timing-mechanism failure at
#            reduced loading, independent of --consolidate entirely -- see
#            "Stage 3" in CLAUDE.md. A previous whole-run-correlation-only
#            check had read this as the paper's expected qualitative
#            unloading degradation; it wasn't checked against frac_at_cap or
#            steady-state windowing until now, and that check shows the
#            mechanism itself is broken there, not just weaker. Toe/air stay
#            out of this script until the medium timing config (or a
#            loading-specific alternative) is re-tuned for the reduced force
#            ceiling under partial/full unloading -- a "Stage 1 for the
#            loading axis," not yet attempted.
#   Seeds:   12345 / 54321 / 98765 -- three, not two, given the documented
#            same-seed/same-config run-to-run nondeterminism finding (NEST
#            multi-threaded execution can flip corr(F-E_L,F-E_R) sign with
#            everything else held fixed -- see "Also surfaced during this
#            work" in CLAUDE.md's Stage 1 section).
#
# NOT in scope for this submission (deliberately -- do not silently add):
#   - Fast speed (unresolved, see above).
#   - --consolidate at the slow point (confirmed harmful, see above) --
#     the two slow cells are no-consolidate controls, not a TODO.
#   - Toe/air loading at any speed (confirmed broken even without
#     --consolidate, see above) -- needs its own timing re-tune first.
#   - STDP initial-weight (mu, CV) robustness grid (Phase 3's own 10-point
#     sweep, never repeated with --consolidate on).
#
# 120s sim, matching run_cutforce_sweep6.sh's own precedent for a
# confirmatory (not exploratory) production submission -- more gait cycles
# per run than the 60s local tuning rounds used.
#
# TIME BUDGET: same 12h precedent as run_cutforce_sweep6.sh (force-trigger
# + muscle-fatigue + consolidate bookkeeping is measurably slower per
# simulated second than the timer-only path -- do not shrink this).
#
# After completion, run BOTH diagnostics on every output, same as every
# prior force-trigger round:
#   python3 scripts/cpg_cutforce_diagnostics.py --steady-from-ms 30000 results/cpg_consol_sal_*.h5
# Read per cell across its 3 seeds: frac_at_cap should stay near 0.00 and
# corr(F-E_L,F-E_R) should be reproducibly negative in at least 2 of 3 --
# a cell that behaves like the excluded fast point (deterministic but
# seed-flipping) should be flagged, not averaged over silently.
#
# Output: results/cpg_consol_sal_<cell-label>_idx00_mu03.50_cv00.30_seed<SEED>.h5
# (auto-named by --tag/--outdir via the sweep-pairs machinery, same pattern
# as run_cutforce_sweep6.sh -- idx/mu/cv are fixed since this run doesn't
# sweep the STDP init grid, see "NOT in scope" above.)

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "[ConsolSAL] ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK array_task=${SLURM_ARRAY_TASK_ID:-NA}"

OUTDIR="results/"
SIM_MS=120000

SEEDS=(12345 54321 98765)

# cell index 0-3 -- see header comment for what each one is. Full weight-
# bearing only -- toe/air excluded, see header "Loading" note above.
CELL_LABELS=(medium_desc_full medium_sens_full slow_desc_full_noconsolidate slow_sens_full_noconsolidate)
CELL_SPEED=(medium medium slow slow)
CELL_ARM=(desc sens desc sens)

TASK=${SLURM_ARRAY_TASK_ID:-0}
CELL_IDX=$(( TASK / 3 ))
SEED_IDX=$(( TASK % 3 ))
SEED=${SEEDS[$SEED_IDX]}
LABEL=${CELL_LABELS[$CELL_IDX]}
SPEED=${CELL_SPEED[$CELL_IDX]}
ARM=${CELL_ARM[$CELL_IDX]}

# ---- speed -> confirmed force-trigger timing (Stage 1: medium, slow only) ----
if [ "$SPEED" = "medium" ]; then
  PERIOD=520
  FAT_ONSET=260
  FAT_RECOVERY=600
  CAP=450
  OFFFRAC=0.35
  LEAD_OFFSET=150
else # slow
  PERIOD=1200
  FAT_ONSET=340
  FAT_RECOVERY=780
  CAP=585
  OFFFRAC=0.35
  LEAD_OFFSET=150
fi

# ---- arm -> BS plasticity + confirmed --consolidate gain pair (do not transfer between arms) ----
# NOTE: flags are built as arrays, not space-joined strings -- a space-joined
# string handed to "${VAR}" (unquoted) is not guaranteed to word-split back
# into separate argv tokens on every shell/IFS configuration (confirmed to
# silently fail, landing as one unrecognized argparse token, in local testing
# for this exact pattern -- see "Stage 3" in CLAUDE.md). Arrays sidestep the
# ambiguity entirely.
ARM_FLAGS=()
if [ "$ARM" = "sens" ]; then
  ARM_FLAGS=(--freeze-bs-rg)
  GAIN_GENUINE=0.25
  GAIN_FORCED=0.10
else
  GAIN_GENUINE=0.20
  GAIN_FORCED=0.15
fi

# loading is full weight-bearing only in this script (see header) -- default
# --ia-feedback-gain/--cut-feedback-gain (1.0) apply, no flags needed.

# ---- --consolidate is medium-speed only (Stage 2: harmful at slow, see header) ----
if [ "$SPEED" = "medium" ]; then
  CONSOLIDATE_FLAGS=(--consolidate --consolidate-prp-gain-genuine "${GAIN_GENUINE}" --consolidate-prp-gain-forced "${GAIN_FORCED}")
  echo "[ConsolSAL] task=$TASK cell=$LABEL speed=$SPEED(period=${PERIOD}ms) arm=$ARM loading=full seed=$SEED gains=${GAIN_GENUINE}/${GAIN_FORCED}"
else
  CONSOLIDATE_FLAGS=()
  echo "[ConsolSAL] task=$TASK cell=$LABEL speed=$SPEED(period=${PERIOD}ms) arm=$ARM loading=full seed=$SEED consolidate=OFF (no-consolidate control, see header)"
fi

srun --cpu-bind=cores --distribution=block:block \
  python3 -u cpg_2legs_fast.py \
    --tag "consol_sal_${LABEL}" \
    --out cpg_run.h5 \
    --outdir "$OUTDIR" \
    --seed "$SEED" \
    --sweep-pairs "3.5:0.30" \
    --sweep-run-idx 0 \
    --sweep-dist lognormal_cv \
    --sim-ms "$SIM_MS" \
    --dt-ms 10 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --nest-verbosity M_ERROR \
    --max-weight-conns 2000 \
    --save-weights snapshots \
    --delay-model length_velocity \
    --species rat \
    --delay-jitter-ms 0.2 \
    --weight-sample-ms 1000 \
    --rate-update-ms 100 \
    --simulate-chunk-ms 100 \
    --bs-base-hz 6 \
    --bs-noise-std-hz 0.25 \
    --enforce-tonic-bs \
    --paced-gait \
    --step-period-ms "$PERIOD" \
    --stance-fraction 0.5 \
    --n-ia-groups 3 \
    --ia-ext-hz 60 80 100 \
    --ia-ext-f-hz 80 \
    --cut-trigger force \
    --leading-leg R \
    --lead-offset-ms "$LEAD_OFFSET" \
    --cut-force-on-frac 0.80 \
    --cut-force-off-frac "$OFFFRAC" \
    --cut-max-stance-ms "$CAP" \
    --cut-max-swing-ms "$CAP" \
    --muscle-fatigue \
    --fatigue-tau-onset-ms "$FAT_ONSET" \
    --fatigue-tau-recovery-ms "$FAT_RECOVERY" \
    --fatigue-max-frac 0.95 \
    "${CONSOLIDATE_FLAGS[@]}" \
    "${ARM_FLAGS[@]}" \
    --long-run
