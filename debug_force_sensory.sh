#!/bin/bash
# debug_force_sensory.sh — local smoke test: force-triggered CUT (confirmed
# Phase 3 operating point, rounds 1-6) combined with the SENSORY-LEARNING arm
# (--freeze-bs-rg --stdp-ia-rg) for the first time. This combination has never
# been run before — rounds 1-6 only validated force-trigger with descending
# (BS) STDP. Purpose: catch gross failure (permanent lock-in, immediate
# cap-domination, L/R sync) cheaply before submitting run_sensory_stdp_force.sh
# / run_ablation_sensory_force.sh to MN5.
#
# Mirrors debug_force.sh but layers on:
#   - the confirmed round 5/6 operating point (off-frac=0.35, cap=450ms,
#     --muscle-fatigue --fatigue-tau-onset-ms 260), not debug_force.sh's older
#     defaults (off-frac=0.20, cap=600ms, no fatigue)
#   - --step-period-ms 520 (rounds 1-6's operating point, not debug_force.sh's
#     1000ms) so Ia-E sub-group timing matches what was actually tuned
#   - --freeze-bs-rg --stdp-ia-rg --wmax-ia 10 --stdp-lambda 1e-3 (sensory arm)
#
# After running, plot with:
#   python3 scripts/cpg_plot_from_hdf5.py --in results/debug_force_sensory.h5 --save-prefix debug_force_sensory
# and diagnose with:
#   python3 scripts/cpg_cutforce_diagnostics.py results/debug_force_sensory.h5

set -e

mkdir -p results

THREADS=${THREADS:-4}

python3 -u cpg_2legs_fast.py \
    --debug-small \
    --paced-gait \
    --cut-trigger force \
    --leading-leg R \
    --lead-offset-ms 150 \
    --cut-force-on-frac 0.80 \
    --cut-force-off-frac 0.35 \
    --cut-max-stance-ms 450 \
    --cut-max-swing-ms 450 \
    --muscle-fatigue \
    --fatigue-tau-onset-ms 260 \
    --fatigue-tau-recovery-ms 600 \
    --step-period-ms 520 \
    --stance-fraction 0.5 \
    --n-ia-groups 3 \
    --ia-ext-hz 60 80 100 \
    --ia-ext-f-hz 80 \
    --stdp-lambda 1e-3 \
    --freeze-bs-rg \
    --stdp-ia-rg \
    --wmax-ia 10 \
    --out results/debug_force_sensory.h5 \
    --sim-ms 60000 \
    --dt-ms 10 \
    --threads "$THREADS" \
    --sweep-pairs "3.5:0.30" \
    --sweep-run-idx 0 \
    --sweep-dist lognormal_cv \
    --seed 12345 \
    --nest-verbosity M_WARNING \
    --max-weight-conns 1000 \
    --save-weights snapshots \
    --delay-model length_velocity \
    --species rat \
    --delay-jitter-ms 0.2 \
    --weight-sample-ms 500 \
    --rate-update-ms 50 \
    --simulate-chunk-ms 50 \
    --bs-base-hz 6 \
    --bs-noise-std-hz 0.25 \
    --enforce-tonic-bs

echo ""
echo "=========================================="
echo "Done. To plot:"
echo "  python3 scripts/cpg_plot_from_hdf5.py --in results/debug_force_sensory.h5 --save-prefix debug_force_sensory"
echo "To diagnose (frac_at_cap must be ~0 before trusting anything):"
echo "  python3 scripts/cpg_cutforce_diagnostics.py results/debug_force_sensory.h5"
echo "=========================================="
