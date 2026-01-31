#!/bin/bash -l
#SBATCH --job-name=CPG_NEST
#SBATCH --output=Nest_processing_%j.slurmout
#SBATCH --error=Nest_processing_%j.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
# #SBATCH --partition=acc   # if needed on MN5

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# OpenMP setup (crucial for NEST)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "[Slurm] ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$OMP_NUM_THREADS"
python3 -c "import nest; print('nest', nest.__version__)"

# Bind tasks to cores (this matters a lot)
srun --cpu-bind=cores --distribution=block:block \
  python3 -u cpg_2legs_nest_to_hdf5_fast.py \
    --out cpg_${SLURM_JOB_ID}.h5 \
    --sim-ms 10000 \
    --dt-ms 10 \
    --threads $SLURM_CPUS_PER_TASK \
    --weight-sample-ms 100 \
    --rate-update-ms 20 \
    --print-every 50