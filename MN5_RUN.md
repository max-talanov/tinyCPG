# MN5 run manifest — descending vs sensory learning + ablation

What to upload to MN5, what to submit, and what to bring back for local plotting.
Plotting is done **locally** (after `scp`-ing results back), not on MN5.

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

Point `--indir` at the dated results folder. Sensory arm uses the new modes:
```bash
INDIR=results/$(date +%F)

# Paper figures with committed generators (all recompute metrics from the HDF5s)
python3 cpg_anchor_figure.py          --indir $INDIR --out plots/paper/fig2_main.png
python3 cpg_speed_stdp_figure.py      --indir $INDIR --outdir plots/paper          # fig5 + CSV
python3 cpg_ablation_graded_figure.py --indir $INDIR --outdir plots/paper          # fig6 + CSV
python3 cpg_epidural_contrast.py --stim-dir $INDIR --natural-dir $INDIR --lambda-tag lam1em3 \
        --out plots/paper/fig9_epidural_contrast.png                                # fig9 + CSV
python3 cpg_descending_vs_sensory.py  --indir $INDIR --out plots/paper/fig_descending_vs_sensory.png
python3 cpg_frozen_figure.py          --indir $INDIR --outdir plots/paper           # fig8 (needs cpg_frozen_*_baseline_*)

# STDP weight matrices (sensory tracks CUT + Ia->RG; descending tracks CUT + BS->RG)
python3 cpg_stdp_matrix.py --mode speed    --indir $INDIR --out plots/paper/fig_stdp_descending.png
python3 cpg_stdp_matrix.py --mode sensory  --indir $INDIR --out plots/paper/fig_stdp_sensory.png
python3 cpg_stdp_matrix.py --mode ablsens  --indir $INDIR --out plots/paper/fig_stdp_ablsens.png

# Network activity (descending vs sensory at matched speeds)
python3 cpg_network_activity.py --mode speed --speed-prefix cpg_speed_stdp   --indir $INDIR --out plots/paper/fig_net_descending.png
python3 cpg_network_activity.py --mode speed --speed-prefix cpg_sensory_stdp --indir $INDIR --out plots/paper/fig_net_sensory.png
python3 cpg_network_activity.py --mode ablation --ablation-prefix cpg_ablsens --indir $INDIR --out plots/paper/fig_net_ablsens.png

# Ia afferent activity
python3 cpg_ia_figure.py --mode sensory --indir $INDIR --out plots/paper/fig_ia_sensory.png
python3 cpg_ia_figure.py --mode ablsens --indir $INDIR --out plots/paper/fig_ia_ablsens.png

# Per-run gait/force/weights for any single file
python3 cpg_plot_from_hdf5.py --in $INDIR/cpg_sensory_stdp_13_5cms_lam1em3_*.h5 --save-prefix sensory_med
```
