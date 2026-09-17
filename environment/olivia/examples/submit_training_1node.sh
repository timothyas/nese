#!/bin/bash -l
#
# Single-node anemoi-training run on one Olivia GH200 node (4 GPUs).
# Copy next to a flat training recipe (`training.yaml`) and `sbatch` from there.
#
# Single-node uses NVLink-C2C only, so `apptainer exec --nv` is all that is
# needed -- no libfabric / NCCL host binds (that is the multi-node script).
#
# The stock experimental recipes (e.g. experimental/local-sla-16yr/training.yaml)
# are Mac-pathed and `accelerator: cpu`; the paths + hardware are overridden on the
# `anemoi-training` line below. `graph.overwrite: true` builds the graph from the
# dataset on the fly, so no prebuilt graph is needed.

#SBATCH --account=nn2993k
#SBATCH --job-name=nese-anemoi-train
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=808G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p slurm

RECIPE_DIR="$PWD"                                  # dir holding training.yaml
DATASET_ZARR="$RECIPE_DIR/anemoi-hpx6-24h/dataset.zarr"   # anemoi dataset from nese-dataprep
GRAPH_PT="$RECIPE_DIR/graph.pt"                    # built on the fly if graph.overwrite: true
RUN_ROOT="$RECIPE_DIR/runs"                        # checkpoints / logs / plots
# ^ confirm these key names against your recipe (`grep -n 'system:' -A20 training.yaml`)

module load nese-anemoi          # sets NESE_ANEMOI_SIF + proxy

# Per-job matplotlib cache dir -- avoids 4 ranks racing on ~/.config/matplotlib.
MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

# Optional: stage the dataset onto node-local NVMe for faster epochs.
#   JOBSCRATCH="/cluster/gpuscratch/jobs/$SLURM_JOB_ID"
#   cp -r "$(dirname "$DATASET_ZARR")" "$JOBSCRATCH/" && DATASET_ZARR="$JOBSCRATCH/$(basename "$(dirname "$DATASET_ZARR")")/dataset.zarr"

# For a quick smoke test add: training.max_steps=4 diagnostics.plot.enabled=false
srun apptainer exec --nv --bind /cluster --pwd "$RECIPE_DIR" \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
    anemoi-training train --config-name=training \
        system.hardware.accelerator=cuda \
        system.hardware.num_gpus_per_node=4 \
        system.hardware.num_nodes=1 \
        system.input.dataset="$DATASET_ZARR" \
        system.input.graph="$GRAPH_PT" \
        system.output.root="$RUN_ROOT"
