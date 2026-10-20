#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=training
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=808G
#SBATCH --time=01:00:00

# Arguments are Hydra overrides for training.yaml, see sweep.sh. 6912 steps took ~30 min in
# all-altimeter v1; give a longer run more --time.

set -euo pipefail

# An aborting rank dumps a multi-GB core file into the experiment directory.
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

# Per-job matplotlib cache dir -- avoids 4 ranks racing on ~/.config/matplotlib.
MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

srun --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
    anemoi-training train --config-name=training "$@"
