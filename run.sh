#!/bin/bash -l
#SBATCH --job-name=Nest_processing
#SBATCH --output=Nest_processing.slurmout
#SBATCH --error=Nest_processing.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --ntasks-per-node=50
#SBATCH --cpus-per-task=1

# Strongly recommended on HPC: ensure UTF-8 output + unbuffered python prints
export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

# OpenMP parameters
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Nest + MPI
srun python3 -u cpg_2legs_nest_to_hdf5.py \
  --out cpg_${SLURM_JOB_ID}.h5 \
  --sim-ms 10000 \
  --dt-ms 10 \
  --threads $SLURM_CPUS_PER_TASK \
  --print-every 50