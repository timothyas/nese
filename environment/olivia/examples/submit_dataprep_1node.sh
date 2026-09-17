#!/bin/bash -l

### Why 128 tasks per node
#
# Olivia CPU nodes have 256 cores (no SMT (Simultaneous Multi Thread)) and ~740 GiB RAM.
# Per-rank memory here is
# small (rank 0 peaks ~1-2 GiB for the one-time GEBCO regrid), so memory does not
# limit the count. 128 balances download concurrency against proxy connections and
# Lustre metadata load (per-rank log + zarr chunk writes). Tune down to 64 if CMEMS
# throttles; 256 costs nothing extra (nodes are billed whole) if profiling shows
# bandwidth headroom.

#SBATCH --account=nn2993k
#SBATCH --job-name=olivia-sla-baseline
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

# Using the data.yaml in ../../experimental/olivia-sla-baseline took ~13 minutes

set -euo pipefail

# --- one thread per rank (nese-dataprep sets no thread caps; ranks are packed) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_NUM_WORKERS=1

module load nese
srun nese-dataprep create data.yaml
