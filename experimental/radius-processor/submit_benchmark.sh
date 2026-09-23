#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=benchmark-block-size
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=01:00:00

set -euo pipefail
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

apptainer exec --nv --bind /cluster --pwd "$PWD" "$NESE_ANEMOI_SIF" python benchmark_block_size.py
