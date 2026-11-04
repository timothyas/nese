#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=training-v2p3
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=nn2993k
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=808G
#SBATCH --time=04:00:00

# v2p2 (6912 steps) took 1.5 hours total; 2x steps expected ~3 hours.
# devel QOS caps walltime at 2h, hence the project QOS.

set -euo pipefail

# An aborting rank dumps a multi-GB core file into the experiment directory.
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

# Per-job matplotlib cache dir -- avoids 4 ranks racing on ~/.config/matplotlib.
MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

# One process builds the graph; the training ranks load it. A concurrent
# torch.save from every rank corrupts the file.
apptainer exec --nv --bind /cluster --pwd "$PWD" "$NESE_ANEMOI_SIF" nese-resolve-graph training.stage2.yaml -o graph.stage2.yaml
apptainer exec --nv --bind /cluster --pwd "$PWD" "$NESE_ANEMOI_SIF" \
    anemoi-graphs create --overwrite --no-description graph.stage2.yaml graph.stage2.pt

srun --overlap --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
    anemoi-training train --config-name=training.stage2
