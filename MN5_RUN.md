# MN5 run manifest — descending vs sensory learning + ablation

What to upload to MN5, what to submit, and what to bring back for local plotting.
Plotting is done **locally** (after `scp`-ing results back), not on MN5.

> **NOTE — 5×5 matrix + logistic gate (post 2026-07-07).** Two changes require a
> re-run of the sensory arms: (i) the activation gate is now a smooth logistic
> (`MOD_LOGISTIC_GATE`, replaces the hard clamp — bio-plausibility) so all figures
> should be regenerated against the new equations; (ii) `run_sensory_stdp.sh` and
> `run_ablation_sensory.sh` now sweep **5 STDP rates** (λ = 1e-6 … 1e-2, arrays
> `0-14`) for the 5 modes × 5 λ comparison. Submit both, bring back
> `cpg_sensory_stdp_*` and `cpg_ablsens_*`, then run
> `scripts/cpg_mode_lambda_summary.py --indir <dated>` (heatmaps + trends + table, auto-detects
> the 5 λ) plus the per-λ figure scripts (which loop `lam1em2..lam1em6`).

> **NOTE — bio-plausibility defaults changed (post 2026-06-30).** The model now
> defaults to lognormal static-weight heterogeneity (`--static-weight-cv 0.5`) and
> a single plastic cutaneous projection (`--cut-static-w 0`, the static co-activation
> pathway dropped). These improve counter-phase but change the canonical numbers, so
> **all arms must be re-run** to regenerate a single-version result set. The run
> scripts need no edits — they pick up the new defaults automatically; the HDF5
> attrs `static_weight_cv` / `cut_static_w` record the configuration.

## 1. Files to upload

The model is standalone (no local imports), so MN5 needs only the model + the
SLURM scripts you intend to submit.

**Option A — git (cleanest, if MN5 has a clone):**
```bash
# on MN5, in the repo clone
git pull origin main
```

**Option B — scp/rsync the minimal set:**
```bash
# from this repo root, on your laptop
rsync -av \
  cpg_2legs_fast.py \
  run_speed_stdp.sh \
  run_sensory_stdp.sh \
  run_ablation_stim.sh \
  run_ablation_graded.sh \
  run_ablation_sensory.sh \
  run_frozen.sh \
  <user>@mn5:/path/to/tinyCPG/
```

| File | Role |
|---|---|
| `cpg_2legs_fast.py` | The model (the only code file needed). |
| `run_speed_stdp.sh` | Phase A — **descending** arm: speed × λ (BS→RG plastic). |
| `run_sensory_stdp.sh` | Phase A — **sensory** arm: speed × λ (frozen BS + plastic Ia→RG). |
| `run_ablation_stim.sh` | Phase B — epidural-**stim** arm (CUT intact). |
| `run_ablation_graded.sh` | Phase B — **natural** arm (CUT + Ia gated by loading). |
| `run_ablation_sensory.sh` | Phase B — **sensory** arm: loading × λ, Ia is the gated learning drive. |

`make sure they're executable: chmod +x run_*.sh` (already +x in git).

## 2. What to run on MN5

Each script is a 9-task array (`--array=0-8` = 3 conditions × 3 λ), 120 s/task,
64 cpus/task, partition `acc`.

**New / required (the sensory-learning results don't exist yet):**
```bash
sbatch run_sensory_stdp.sh       # -> results/cpg_sensory_stdp_<spd>_<lam>_*.h5
sbatch run_ablation_sensory.sh   # -> results/cpg_ablsens_<gain>_<lam>_*.h5
```

**Descending / ablation arms — only if not already produced with the current
model** (these scripts are unchanged in behaviour; the freeze/Ia flags are OFF
by default, so existing `cpg_speed_stdp_*`, `cpg_ablstim_*`, `cpg_ablgrad_*`
outputs are still valid). Re-run for single-version consistency if you prefer:
```bash
sbatch run_speed_stdp.sh         # -> results/cpg_speed_stdp_<spd>_<lam>_*.h5
sbatch run_ablation_stim.sh      # -> results/cpg_ablstim_<gain>_<lam>_*.h5
sbatch run_ablation_graded.sh    # -> results/cpg_ablgrad_<gain>_<lam>_*.h5
```

**Frozen-weight control (§3.6) — re-run required at BASELINE loading.**
`run_frozen.sh` now runs at full weight-bearing (`IA_GAIN=1.0`) and inherits
the bio-plausible defaults; it tests whether imposing the converged
CUT$\to$RG-E marginal distribution by hand reproduces the clean baseline
rhythm (corr $-0.90$). The old air-stepping frozen data is superseded.
```bash
sbatch run_frozen.sh             # -> results/cpg_frozen_m<M>_cv<CV3>_baseline_*.h5
```

Check progress: `squeue -u <user>`. Each array job writes its tasks into
`results/`. Logs: `Nest_*_<jobid>_<task>.slurmout/.slurmerr`.

> Connectivity figure (`--dump-connectivity`) is **already generated locally at
> production N** (`results/connectivity/conn_dump.h5`, committed) — no MN5 run needed.

## 3. What to bring back

```bash
# from your laptop — pull just the HDF5s into a dated folder
mkdir -p results/$(date +%F)
rsync -av '<user>@mn5:/path/to/tinyCPG/results/cpg_sensory_stdp_*.h5'  results/$(date +%F)/
rsync -av '<user>@mn5:/path/to/tinyCPG/results/cpg_ablsens_*.h5'       results/$(date +%F)/
# (and cpg_speed_stdp_* / cpg_ablstim_* / cpg_ablgrad_* if you re-ran them)
```
Each HDF5 is ~14 MB at production N — verify sizes after transfer (a truncated
scp shows up as a few MB and breaks the plotters).

## 4. Plot locally (after transfer)

Point `--indir` at the dated results folder. These are the **current** generators
(all write straight into `paper/figures/`, the single source of truth for the
manuscript — see `paper/README.md`). Superseded generators live in
`scripts/legacy/` and are not part of this pipeline.

```bash
INDIR=results/$(date +%F)

# Architecture + connectivity (Methods) — only need re-running after a --dump-connectivity change
python3 scripts/cpg_architecture_diagram.py
python3 scripts/cpg_connectivity_figure.py --in results/connectivity/conn_dump_sensory.h5 \
        --out paper/figures/fig_connectivity.png

# STDP weights: both legs x 3 projections x 5 modes, and 5 modes x 5 lambda
python3 scripts/cpg_stdp_weight_matrix.py --indir $INDIR --out paper/figures/fig_stdp_weight_matrix.png
python3 scripts/cpg_stdp_weights_grid.py  --indir $INDIR --out paper/figures/fig_stdp_weights_grid.png

# Force at 3 learning stages x 5 modes, one figure per lambda
for lam in lam1em2 lam1em3 lam1em4 lam1em5 lam1em6; do
  python3 scripts/cpg_force_stages.py --indir $INDIR --lambda-tag $lam \
          --out paper/figures/fig_force_stages_$lam.png
done

# Full-circuit population activity x 5 modes, one figure per lambda
for lam in lam1em2 lam1em3 lam1em4 lam1em5 lam1em6; do
  python3 scripts/cpg_network_matrix.py --indir $INDIR --lambda-tag $lam \
          --out paper/figures/fig_network_matrix_$lam.png
done

# 5x5 mode x lambda comparison: heatmaps + trends + table (auto-detects available lambda tags)
python3 scripts/cpg_mode_lambda_summary.py --indir $INDIR --out paper/figures/fig_mode_lambda
cp paper/figures/fig_mode_lambda_table.tex paper/mode_lambda_table.tex

# Epidural-stim vs natural loading contrast (needs cpg_ablstim_* and cpg_ablgrad_*)
python3 scripts/cpg_epidural_contrast.py --stim-dir $INDIR --natural-dir $INDIR \
        --out paper/figures/fig9_epidural_contrast.png

# Per-run gait/force/weights for any single file (debug tool, writes to results/)
python3 scripts/cpg_plot_from_hdf5.py --in $INDIR/cpg_sensory_stdp_13_5cms_lam1em3_*.h5 --save-prefix sensory_med
```
