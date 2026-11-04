#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=nese-training-tests
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00

# The nese_training test suite, in the arm64 container that the login node cannot run.

set -euo pipefail
module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

srun apptainer exec --nv --bind /cluster --pwd "$PWD" "$NESE_ANEMOI_SIF" \
    python -m pytest -q -p no:cacheprovider /cluster/projects/nn2993k/timith/nese/training/tests
