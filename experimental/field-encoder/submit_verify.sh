#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=verify-v0
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

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
srun nese-verify metrics verify.validation.yaml
srun nese-verify metrics verify.2015.yaml
nese-verify movie verify.2015.yaml
