#!/bin/bash -l
#SBATCH --job-name=CPG_CONSOL_SAL
#SBATCH --output=Nest_consol_sal_%A_%a.slurmout
#SBATCH --error=Nest_consol_sal_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-23
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
# SCOPE (8 cells x 3 seeds = 24 tasks):
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
#   Loading: full weight-bearing / toe stepping / air stepping, but ONLY at
#            medium speed -- the paper defines the loading axis at the
#            baseline speed anchor only, not as an independent grid over
#            every speed (see "Correction: use the paper's own
#            5-locomotion-mode terminology" in CLAUDE.md). Slow-speed rows
#            are full weight-bearing only.
#   Seeds:   12345 / 54321 / 98765 -- three, not two, given the documented
#            same-seed/same-config run-to-run nondeterminism finding (NEST
#            multi-threaded execution can flip corr(F-E_L,F-E_R) sign with
#            everything else held fixed -- see "Also surfaced during this
#            work" in CLAUDE.md's Stage 1 section).
#
# NOT in scope for this submission (deliberately -- do not silently add):
#   - Fast speed (unresolved, see above).
#   - Per-speed or per-loading --consolidate gain re-tuning (Stages 2-3).
#     Every cell below reuses the single confirmed gain pair for its arm,
#     unmodified by speed or loading -- treat this run's results as a
#     TRANSFER TEST of that pair, not as already-validated for these cells.
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

# cell index 0-7 -- see header comment for what each one is
CELL_LABELS=(medium_desc_full medium_desc_toe medium_desc_air medium_sens_full medium_sens_toe medium_sens_air slow_desc_full slow_sens_full)
CELL_SPEED=(medium medium medium medium medium medium slow slow)
CELL_ARM=(desc desc desc sens sens sens desc sens)
CELL_LOADING=(full toe air full toe air full full)

TASK=${SLURM_ARRAY_TASK_ID:-0}
CELL_IDX=$(( TASK / 3 ))
SEED_IDX=$(( TASK % 3 ))
SEED=${SEEDS[$SEED_IDX]}
LABEL=${CELL_LABELS[$CELL_IDX]}
SPEED=${CELL_SPEED[$CELL_IDX]}
ARM=${CELL_ARM[$CELL_IDX]}
LOADING=${CELL_LOADING[$CELL_IDX]}

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
ARM_FLAGS=""
if [ "$ARM" = "sens" ]; then
  ARM_FLAGS="--freeze-bs-rg"
  GAIN_GENUINE=0.25
  GAIN_FORCED=0.10
else
  GAIN_GENUINE=0.20
  GAIN_FORCED=0.15
fi

# ---- loading -> Ia/CUT feedback gain (full = defaults, no flags needed) ----
LOADING_FLAGS=""
if [ "$LOADING" = "toe" ]; then
  LOADING_FLAGS="--ia-feedback-gain 0.5 --cut-feedback-gain 0.5"
elif [ "$LOADING" = "air" ]; then
  LOADING_FLAGS="--ia-feedback-gain 0.1 --cut-feedback-gain 0.1"
fi

echo "[ConsolSAL] task=$TASK cell=$LABEL speed=$SPEED(period=${PERIOD}ms) arm=$ARM loading=$LOADING seed=$SEED gains=${GAIN_GENUINE}/${GAIN_FORCED}"

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
    --consolidate \
    --consolidate-prp-gain-genuine "$GAIN_GENUINE" \
    --consolidate-prp-gain-forced "$GAIN_FORCED" \
    $ARM_FLAGS \
    $LOADING_FLAGS \
    --long-run
