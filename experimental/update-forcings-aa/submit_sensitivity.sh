#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=sensitivity
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=01:00:00

# Globe-wide loss sensitivities for sensitivity_global.ipynb, one sensitivity/global/<t0>/ per date.

set -euo pipefail

# Apptainer variables from a `nese` (Tykky) shell would reach the apptainer exec below and fail
# its mounts ("mount source /node doesn't exist"); a bare sbatch passes them on.
for name in ${!APPTAINER@} ${!SINGULARITY@}; do unset "$name"; done
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

# Each checkpoint with the training yaml it was trained with
RUNS=(
    --run v2p2 ../all-altimeter/training.stage2.yaml
        ../all-altimeter/runs/v2p2/checkpoint/ae31444877904a0bab02b1d25a173aa9/last.ckpt
    --run v0 training.yaml runs/v0/checkpoint/512afc621bf04413861cfc567d0befe5/last.ckpt
)

# The dates of ../unseen-ft-all-altimeter's global sensitivities
for t0 in 2020-09-13T12 2015-05-31T12; do
    srun --export=ALL --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
        --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
        python sensitivity.py "${RUNS[@]}" --date "$t0"
done
