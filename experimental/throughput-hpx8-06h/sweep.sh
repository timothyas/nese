#!/bin/bash
#
# Submit the throughput sweep, one profile job per setting, each logging to runs/<label>. See
# README.md.
#
#   bash sweep.sh round1              # bs1 builds graph.pt and doubles as the smoke test; bs2, bs4 wait for it
#   bash sweep.sh round2 <batch_size>
#   bash sweep.sh round3 <batch_size> <workers> <prefetch>   # flex_attention block size
#   bash sweep.sh one <batch_size> <workers> <prefetch> [block_size]
#   bash sweep.sh verify24h           # the chosen settings on the full 1993-2020 hpx8/24h store, after the tests pass
#   bash sweep.sh debug24h            # NaN losses at 24h: 16-mixed (with the loss's own NaN report) vs 32-bit, after the tests pass
#
# Batch size is per GPU. Validation uses the same batch size, so an OOM there counts too.

set -euo pipefail
cd "$(dirname "$0")"

DEPENDENCY=""
SUFFIX=""
EXTRA=()

submit() {  # batch_size num_workers prefetch_factor [block_size]
    local label="bs$1_w$2_pf$3${4:+_blk$4}$SUFFIX"
    if [ -e "runs/$label" ]; then
        echo "skip $label: runs/$label exists" >&2
        return
    fi
    sbatch --parsable --job-name="$label" ${DEPENDENCY:+--dependency=afterok:$DEPENDENCY} \
        submit_profile.sh \
        dataloader.batch_size.training="$1" \
        dataloader.batch_size.validation="$1" \
        dataloader.num_workers.training="$2" \
        dataloader.prefetch_factor="$3" \
        diagnostics.log.mlflow.experiment_name="$label" \
        ${4:+model.processor.block_size=$4} \
        "${EXTRA[@]}"
}

case "${1:-}" in
    round1)
        DEPENDENCY=$(submit 1 8 2)
        echo "bs1: $DEPENDENCY"
        for batch_size in 2 4; do
            echo "bs$batch_size: $(submit "$batch_size" 8 2)"
        done
        ;;
    round2)
        batch_size="${2:?round2 needs the chosen batch size}"
        for workers in 2 4 8 16; do
            for prefetch in 1 2; do
                echo "bs${batch_size}_w${workers}_pf${prefetch}: $(submit "$batch_size" "$workers" "$prefetch")"
            done
        done
        ;;
    round3)
        batch_size="${2:?round3 needs the chosen batch size}"
        workers="${3:?round3 needs the chosen number of workers}"
        prefetch="${4:?round3 needs the chosen prefetch factor}"
        for block_size in 32 64 128 256; do
            echo "blk$block_size: $(submit "$batch_size" "$workers" "$prefetch" "$block_size")"
        done
        ;;
    verify24h | debug24h)
        # Same cells as the 6h store, so graph.pt is reused. The statistics cover every mission,
        # so the normaliser is recomputed from this store, as in the real stage 2 run.
        SUFFIX=_24h
        EXTRA=(
            system.input.dataset='${system.input.root}/anemoi-hpx8-24h/dataset.zarr'
            data.frequency=24h
            task.timestep=24H
            training.update_ds_stats_on_ckpt_load.states=true
            dataloader.training.datasets.ocean.start=1993-11-01
            dataloader.training.datasets.ocean.end=2018-12-31
            dataloader.validation.datasets.ocean.start=2019-01-01
            dataloader.validation.datasets.ocean.end=2020-12-31
            dataloader.test.datasets.ocean.start=2019-01-01
            dataloader.test.datasets.ocean.end=2020-12-31
        )
        DEPENDENCY=$(sbatch --parsable submit_tests.sh)  # nese_training changed since the last runs
        echo "tests: $DEPENDENCY"
        if [ "$1" = verify24h ]; then
            for workers in 4 8; do
                echo "bs1_w${workers}_pf1_24h: $(submit 1 "$workers" 1)"
            done
        else
            # Store vs checkpoint statistics made no difference (2512563-4); now precision.
            SUFFIX=_24h_losscheck
            echo "16-mixed, loss check: $(submit 1 8 1)"
            SUFFIX=_24h_fp32
            EXTRA+=(training.precision=32)
            echo "32-bit: $(submit 1 8 1)"
        fi
        ;;
    one)
        echo "$(submit "${2:?}" "${3:?}" "${4:?}" "${5:-}")"
        ;;
    *)
        echo "usage: bash sweep.sh round1 | round2 <batch_size> | round3 <batch_size> <workers> <prefetch> | one <bs> <w> <pf> [blk] | verify24h | debug24h" >&2
        exit 1
        ;;
esac
