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

# Loss sensitivities for sensitivity_cell.ipynb (sensitivity/<case>/, one per case below) and
# sensitivity_global.ipynb (sensitivity/global/<t0>/, one per date below).
#
# An optional first argument names a variant, written one directory deeper, and the remaining
# arguments go to every sensitivity.py run. The noise-floor comparison uses
#   sbatch submit_sensitivity.sh scale256 --loss-scale 256
#   sbatch submit_sensitivity.sh bf16 --precision bf16-mixed

set -euo pipefail

# Apptainer variables from a `nese` (Tykky) shell would reach the apptainer exec below and fail
# its mounts ("mount source /node doesn't exist"); nese-submit strips them, a bare sbatch does not.
for name in ${!APPTAINER@} ${!SINGULARITY@}; do unset "$name"; done
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

CHECKPOINTS=(
    --checkpoint v2p2=../all-altimeter/runs/v2p2/checkpoint/ae31444877904a0bab02b1d25a173aa9/last.ckpt
    --checkpoint v0=runs/v0/checkpoint/c4fbe8c1bc4144a9bdd7ebcb3a4a540b/last.ckpt
    --checkpoint control=runs/control/checkpoint/d4e1a4f61f874367b2db7467886a4768/last.ckpt
)

VARIANT=${1:-}
EXTRA=("${@:2}")

sensitivity() {
    srun --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
        --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
        python sensitivity.py "${CHECKPOINTS[@]}" ${VARIANT:+--variant "$VARIANT"} "$@" "${EXTRA[@]}"
}

# The turbulent Gulf Stream extension, on the validation date the first runs drew (seed 0)
GULF_STREAM_T0=2020-09-13T12
sensitivity --scalars cell --case gulf_stream --date $GULF_STREAM_T0 --near 38.5 -63

# The open eastern subtropical Atlantic between the Canary and Cape Verde Islands, on a date when
# no input bin crossed much of it (the 2015 movie)
EASTERN_ATLANTIC_T0=2015-05-31T12
sensitivity --scalars cell --case eastern_atlantic --date $EASTERN_ATLANTIC_T0 --near 22 -28

# The globe-wide losses depend only on the date
for t0 in $GULF_STREAM_T0 $EASTERN_ATLANTIC_T0; do
    sensitivity --scalars global --date "$t0"
done
