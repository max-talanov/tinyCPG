#!/bin/bash -l
#SBATCH --job-name=CPG_NEST
#SBATCH --output=Nest_processing_%j.slurmout
#SBATCH --error=Nest_processing_%j.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --partition=acc

export LANG=${LANG:-C.UTF-8}
export LC_ALL=${LC_ALL:-C.UTF-8}
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "[Slurm] ntasks=$SLURM_NTASKS cpus-per-task=$SLURM_CPUS_PER_TASK"

python3 - <<'PY'
import nest
ks = nest.GetKernelStatus()
mpi = ks.get("mpi_num_processes", ks.get("num_processes", ks.get("total_num_processes", 1)))
thr = ks.get("local_num_threads", ks.get("num_threads", ks.get("threads", 1)))
print("nest", nest.__version__, "mpi_procs", mpi, "local_threads", thr)
PY

srun --cpu-bind=cores --distribution=block:block \
  python3 -u cpg_2legs_nest_to_hdf5_fast.py \
    --out cpg_${SLURM_JOB_ID}.h5 \
    --sim-ms 10000 \
    --dt-ms 10 \
    --threads $SLURM_CPUS_PER_TASK \
    --long-run \
    --nest-verbosity M_ERROR \
    --max-weight-conns 2000
