#!/bin/bash -l
#SBATCH --job-name=CPG_NEST
#SBATCH --output=Nest_processing_%j.slurmout
#SBATCH --error=Nest_processing_%j.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --mem=0

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module load miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME:-nest}

# Quick sanity print
echo "[Slurm] ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK"
which python3
python3 -c "import nest; print('nest', nest.__version__)"

srun --mpi=pmix python3 -u cpg_2legs_nest_to_hdf5.py \
  --out cpg_${SLURM_JOB_ID}.h5 \
  --sim-ms 10000 \
  --dt-ms 10 \
  --threads $SLURM_CPUS_PER_TASK \
  --print-every 50