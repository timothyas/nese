#!/bin/bash
#
# Submit one round of the training-schedule sweep, one job per setting, each logging to
# runs/<label>. See README.md.
#
#   bash sweep.sh round1
#   bash sweep.sh round2 <lr> <tau>     # the round 1 winner
#
# lr is the config value; anemoi multiplies it by the 4 GPUs, so AdamW's own lr is 4x this.
# Weight decay is set through tau = 1 / (AdamW lr * weight_decay), the number of steps AdamW's
# weights average over (Wang & Aitchison, arXiv:2405.13698), so each tau means the same
# averaging window at every lr.

set -euo pipefail
cd "$(dirname "$0")"

GPUS=4

submit() {  # lr tau max_steps warmup_steps [sbatch time]
    local label="lr$1_tau$2_s$3"
    if [ -e "runs/$label" ]; then
        echo "skip $label: runs/$label exists"
        return
    fi
    local wd
    wd=$(awk -v lr="$1" -v tau="$2" -v gpus="$GPUS" 'BEGIN { printf "%.4g", 1 / (lr * gpus * tau) }')
    sbatch --job-name="$label" --time="${5:-01:00:00}" submit_training.sh \
        training.optimization.lr="$1" \
        training.optimization.optimizer.weight_decay="$wd" \
        training.max_steps="$3" \
        training.optimization.lr_scheduler.warmup_t="$4" \
        diagnostics.log.mlflow.experiment_name="$label"
}

case "${1:-}" in
    round1)
        for lr in 6.25e-5 1.25e-4 2.5e-4 5e-4; do
            for tau in 2500 5000 10000; do
                submit "$lr" "$tau" 6912 500
            done
        done
        ;;
    round2)
        lr="${2:?round2 needs the winning lr}"
        tau="${3:?round2 needs the winning tau}"
        submit "$lr" "$tau" 3456 250
        submit "$lr" "$tau" 13824 1000 01:50:00
        ;;
    *)
        echo "usage: bash sweep.sh round1 | round2 <lr> <tau>" >&2
        exit 1
        ;;
esac
