#!/bin/bash -l
#
# Multi-node anemoi-training run on Olivia GH200 nodes (4 GPUs/node).
# Copy next to a training recipe and `sbatch` it from there.
#
# Multi-node NCCL rides HPE Slingshot, which needs host libfabric / NCCL /
# aws-ofi-nccl / OpenMPI bound into the container (the NGC container's bundled
# copies are build-time only). Bind list + module set from
# https://documentation.sigma2.no/hpc_machines/olivia/nccl_apptainer.html
# Confirm the exact module versions with `module spider` -- the ones below match
# the container's CUDA 12.9 as of 2026-09.

#SBATCH --account=nn2993k
#SBATCH --job-name=nese-anemoi-train-mn
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=808G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p slurm

RECIPE_DIR="$PWD"
DATASET_ZARR="$RECIPE_DIR/anemoi-hpx6-24h/dataset.zarr"
GRAPH_PT="$RECIPE_DIR/graph.pt"
RUN_ROOT="$RECIPE_DIR/runs"

module load nese-anemoi          # sets NESE_ANEMOI_SIF + proxy
module load NRIS/GPU
module load OpenMPI/5.0.8-GCC-14.3.0
module load NCCL/2.29.2-GCCcore-14.3.0-CUDA-12.9.1
module load aws-ofi-nccl/1.17.3-GCCcore-14.3.0-CUDA-12.9.1
module load libfabric/2.3.1-GCCcore-14.3.0

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29500

# Per-job matplotlib cache dir -- avoids ranks racing on ~/.config/matplotlib.
MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"

srun apptainer exec --nv \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --bind /cluster \
    --bind "$EBROOTCUDA":/usr/local/cuda \
    --bind "$EBROOTOPENMPI":/opt/openmpi \
    --bind "$EBROOTNCCL":/opt/nccl \
    --bind "$EBROOTLIBFABRIC":/opt/libfabric \
    --bind "$EBROOTAWSMINOFIMINNCCL":/opt/aws-ofi-nccl \
    --bind /usr/lib64 \
    --pwd "$PWD" \
    "$NESE_ANEMOI_SIF" \
    anemoi-training train --config-name=training \
        system.hardware.accelerator=cuda \
        system.hardware.num_gpus_per_node=4 \
        system.hardware.num_nodes="$SLURM_NNODES" \
        system.input.dataset="$DATASET_ZARR" \
        system.input.graph="$GRAPH_PT" \
        system.output.root="$RUN_ROOT"
