#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=olivia-sla-baseline
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

# Note, this took ~13 minutes

set -euo pipefail

# --- one thread per rank (nese-dataprep sets no thread caps; ranks are packed) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_NUM_WORKERS=1

module load nese
srun nese-dataprep create data.yaml
