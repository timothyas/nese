#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=smoke-unseen-ft
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=00:30:00

# Two training steps on one GPU, into runs/smoke: checks that training.yaml resolves, the v2p2
# weights load, and both loss terms (and their unseen_* field losses) are finite. Submit by hand,
# not through nese-submit, so it never touches runs/v0.

set -euo pipefail

# Apptainer variables from a `nese` (Tykky) shell would reach the apptainer exec below and fail
# its mounts ("mount source /node doesn't exist"); nese-submit strips them, a bare sbatch does not.
for name in ${!APPTAINER@} ${!SINGULARITY@}; do unset "$name"; done
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

srun --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
    anemoi-training train --config-name=training \
    diagnostics.log.mlflow.experiment_name=smoke \
    training.max_steps=2 \
    training.optimization.lr_scheduler.warmup_t=1 \
    dataloader.limit_batches.training=2 \
    dataloader.limit_batches.validation=2 \
    dataloader.num_workers.training=2 \
    diagnostics.plot.callbacks=[]
