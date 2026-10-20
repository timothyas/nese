#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=dataprep
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:30:00
#SBATCH --qos=devel

# Stage 1 took 18 minutes with 128 tasks
# Stage 2 took 31 minutes with 64 tasks

set -euo pipefail

# --- one thread per rank (nese-dataprep sets no thread caps; ranks are packed) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_NUM_WORKERS=1

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese
srun nese-dataprep create data.stage1.yaml
srun nese-dataprep create data.stage2.yaml
