#!/usr/bin/env bash
#
# Rebuild both environments as Slurm jobs, from the login node:
#
#   bash environment/olivia/submit-rebuild.sh
#
# The two builds need different nodes: `nese` (Tykky) on a CPU node, `nese-anemoi` on an
# aarch64 `accel` node with no GPU. They are independent, so they run side by side. Logs go to
# ~/nese-build-<job name>-<job id>.out
#
# --export=NONE with a login shell, so a `nese` module loaded in this shell (whose Tykky
# variables break apptainer) does not reach the builds.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON=(--account=nn2993k --qos=devel --nodes=1 --time=01:00:00 --export=NONE --output="$HOME/nese-build-%x-%j.out")

sbatch "${COMMON[@]}" --job-name=nese --partition=small \
    --ntasks=1 --cpus-per-task=16 --mem=32G \
    --wrap="bash -l $SCRIPT_DIR/build-env.sh"

sbatch "${COMMON[@]}" --job-name=nese-anemoi --partition=accel --gpus-per-node=0 \
    --ntasks=1 --cpus-per-task=64 --mem=200G \
    --wrap="bash -l $SCRIPT_DIR/build-anemoi-env.sh"
