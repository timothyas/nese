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

# Loss sensitivities for sensitivity_global.ipynb (sensitivity/global/<t0>/, one per date) and
# sensitivity_cell.ipynb (sensitivity/<case>/, one per case), with the cases and dates of
# ../unseen-ft-all-altimeter and ../update-forcings-aa.

set -euo pipefail

# Apptainer variables from a `nese` (Tykky) shell would reach the apptainer exec below and fail
# its mounts ("mount source /node doesn't exist"); a bare sbatch passes them on.
for name in ${!APPTAINER@} ${!SINGULARITY@}; do unset "$name"; done
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

# Each checkpoint with the training yaml it was trained with: ../update-forcings-aa v0, which has
# the latent skip, and this experiment's v0, the same model without it
RUNS=(
    --run skip ../update-forcings-aa/training.yaml
        ../update-forcings-aa/runs/v0/checkpoint/512afc621bf04413861cfc567d0befe5/last.ckpt
    --run no_skip training.yaml runs/v0/checkpoint/6623840749ad4fcabb8374ed713b1f0e/last.ckpt
)

sensitivity() {
    srun --export=ALL --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
        --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
        python sensitivity.py "${RUNS[@]}" "$@"
}

GULF_STREAM_T0=2020-09-13T12
EASTERN_ATLANTIC_T0=2015-05-31T12

# The turbulent Gulf Stream extension, and the open eastern subtropical Atlantic between the
# Canary and Cape Verde Islands
sensitivity --scalars cell --case gulf_stream --date $GULF_STREAM_T0 --near 38.5 -63
sensitivity --scalars cell --case eastern_atlantic --date $EASTERN_ATLANTIC_T0 --near 22 -28

for t0 in $GULF_STREAM_T0 $EASTERN_ATLANTIC_T0; do
    sensitivity --scalars global --date "$t0"
done
