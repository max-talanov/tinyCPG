#!/bin/bash -l
#SBATCH --job-name=CPG_FTGRID
#SBATCH --output=Nest_ftgrid_%A_%a.slurmout
#SBATCH --error=Nest_ftgrid_%A_%a.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=gp_bsccs
#
# FORCE-TRIGGERED LOCOMOTION-MODE x STDP-RATE GRID (paper main results).
# 4 modes x 5 lambda = 20 tasks, descending arm (BS->RG and CUT->RG-E plastic,
# the same circuit as round 6 / run_cutforce_sweep6.sh), closed-loop
# force-triggered stance/swing (--cut-trigger force --muscle-fatigue),
# STDP init mu=3.5, CV=0.30, 120 s. Replaces the timer-based (paced-clock)
# sensory-arm 5x5 grid (run_sensory_stdp.sh + run_ablation_sensory.sh) as the
# paper's main result. Air stepping is excluded: it has no genuine
# force-triggered operating point (chatters at the rate-update tick floor).
#
# task -> (mode, lambda): mode = task / 5, lambda = task % 5
#   modes:   0 slow, 1 medium, 2 fast, 3 toe
#   lambdas: 0 1e-2, 1 1e-3, 2 1e-4, 3 1e-5, 4 1e-6
#
# Per-mode timing: medium is round 6's production-confirmed point; slow/fast/toe
# are the debug-scale operating points from the force-trigger speed/loading work
# (slow = 1.3x scaled medium; fast/toe need --leg-fatigue-asym-frac to break an
# L/R phase-locking bistability). Screened at production N, lambda=1e-3, 60 s
# (results/screen_2026-09-25, steady state t>=30 s): slow stance 403+-18 ms,
# frac_at_cap 0.00, corr(F_E,F_F) -0.74, corr(F_E,L,F_E,R) -0.82; fast 176+-43 ms,
# 0.00, -0.43, -0.55; toe (tau=260/cap=450) 355+-57 ms, 0.00, -0.57, -0.69.
# Medium = round 6 at mu=3.5: 307+-29 ms, 0.00, -0.65, -0.75.
#
# Local use (no SLURM): TASK=<0-19> THREADS=4 [SIM_MS=60000] [OUTDIR=...] [TAG_PREFIX=...] ./run_forcetrig_grid.sh
#
# After completion:
#   python3 scripts/cpg_cutforce_diagnostics.py --steady-from-ms 30000 results/<dated>/cpg_ftgrid_*.h5

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

TASK=${SLURM_ARRAY_TASK_ID:-${TASK:-0}}
THREADS=${SLURM_CPUS_PER_TASK:-${THREADS:-4}}
SIM_MS=${SIM_MS:-120000}
OUTDIR=${OUTDIR:-results/}
TAG_PREFIX=${TAG_PREFIX:-ftgrid}

MODES=(slow medium fast toe)
LAMBDAS=(1e-2 1e-3 1e-4 1e-5 1e-6)
LAMTAGS=(lam1em2 lam1em3 lam1em4 lam1em5 lam1em6)
MODE=${MODES[$(( TASK / 5 ))]}
LAMBDA=${LAMBDAS[$(( TASK % 5 ))]}
LAMTAG=${LAMTAGS[$(( TASK % 5 ))]}

LOADING_FLAGS=()
ASYM=0.0
case "$MODE" in
  slow)    # 1.3x scaled medium
    PERIOD=1200; FAT_ONSET=340; FAT_RECOVERY=780; CAP=585; OFFFRAC=0.35; LEAD_OFFSET=150 ;;
  medium)  # round 6 (run_cutforce_sweep6.sh), confirmed at production N
    PERIOD=520;  FAT_ONSET=260; FAT_RECOVERY=600; CAP=450; OFFFRAC=0.35; LEAD_OFFSET=150 ;;
  fast)
    PERIOD=600;  FAT_ONSET=156; FAT_RECOVERY=360; CAP=380; OFFFRAC=0.30; LEAD_OFFSET=90; ASYM=0.04 ;;
  toe)     # partial unloading; debug-scale tau=100/cap=330 flickered at the 100 ms tick at production N
    PERIOD=730;  FAT_ONSET=260; FAT_RECOVERY=438; CAP=450; OFFFRAC=0.35; LEAD_OFFSET=110; ASYM=0.12
    LOADING_FLAGS=(--ia-feedback-gain 0.5 --cut-feedback-gain 0.5) ;;
esac
# Optional per-run overrides (screening only).
FAT_ONSET=${FAT_ONSET_OVERRIDE:-$FAT_ONSET}
CAP=${CAP_OVERRIDE:-$CAP}
OFFFRAC=${OFFFRAC_OVERRIDE:-$OFFFRAC}
ASYM=${ASYM_OVERRIDE:-$ASYM}

ASYM_FLAGS=()
if [ "$ASYM" != "0.0" ]; then
  ASYM_FLAGS=(--leg-fatigue-asym-frac "$ASYM")
fi

echo "[FTGrid] task=$TASK mode=$MODE lambda=$LAMBDA threads=$THREADS sim=${SIM_MS}ms period=$PERIOD tau=$FAT_ONSET rec=$FAT_RECOVERY cap=$CAP off=$OFFFRAC lead=$LEAD_OFFSET asym=$ASYM"

LAUNCH=()
if [ -n "$SLURM_JOB_ID" ]; then
  LAUNCH=(srun --cpu-bind=cores --distribution=block:block)
fi

"${LAUNCH[@]}" python3 -u cpg_2legs_fast.py \
    --tag "${TAG_PREFIX}_${MODE}_${LAMTAG}" \
    --out cpg_run.h5 \
    --outdir "$OUTDIR" \
    --seed 12345 \
    --sweep-pairs "3.5:0.30" \
    --sweep-run-idx 0 \
    --sweep-dist lognormal_cv \
    --sim-ms "$SIM_MS" \
    --stdp-lambda "$LAMBDA" \
    --dt-ms 10 \
    --threads "$THREADS" \
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
    "${ASYM_FLAGS[@]}" \
    "${LOADING_FLAGS[@]}" \
    --long-run
