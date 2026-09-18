#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=inference-v0.10k
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=00:20:00

set -euo pipefail

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

srun --overlap -n 1 --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
    "$NESE_ANEMOI_SIF" \
    nese-inference run forecast.yaml
