#!/bin/bash -l
#
# Parallel `nese-inference` sweep on one Olivia GH200 node (4 GPUs).
# Copy next to an inference config carrying a `date_range` block and `sbatch` from
# there. One MPI rank per GPU; each rank takes a stride of the initial conditions
# and fills its own `t0` rows of the shared zarr store. See nese-inference
# docs/parallel.md.
#
# The launch is one Slurm task running the *container's own* `mpirun`, not
# `srun -n 4`: the container ships HPC-X OpenMPI 4.1, and letting it fork the four
# ranks inside itself avoids having to make it talk to the host's Slurm PMI.

#SBATCH --account=nn2993k
#SBATCH --job-name=nese-anemoi-infer
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288   # the whole Grace node; drop this and use --exclusive if refused
#SBATCH --mem=808G
#SBATCH --time=06:00:00

set -euo pipefail
mkdir -p slurm

CONFIG="$PWD/config.yaml"        # anemoi run config + nese `date_range` / `logistics` blocks

# One rank per GPU, taken from the allocation rather than hardcoded: changing
# --gpus-per-node above is then the only edit needed. SLURM_GPUS_ON_NODE is the
# count allocated to this batch step, however the GPUs were requested.
#
# Node-LOCAL on purpose. `mpirun` runs inside one container on one task and
# OMPI_MCA_plm=isolated keeps it there, so a total-GPU count (SLURM_GPUS) would
# pile every rank onto this one node's GPUs. Multi-node needs a different launch
# entirely -- see "Running parallel inference" in ../README.md.
if [ "${SLURM_JOB_NUM_NODES:-1}" -ne 1 ]; then
    echo "!! $SLURM_JOB_NUM_NODES nodes allocated, but this script only ever uses one." >&2
    echo "!! For a bigger sweep, submit one job per date_range chunk, each with its" >&2
    echo "!! own store_dir, and concatenate the stores on t0 afterwards." >&2
    exit 1
fi
NRANKS="${SLURM_GPUS_ON_NODE:?no GPUs allocated -- this script expects --gpus-per-node}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / NRANKS ))

module load nese-anemoi          # sets NESE_ANEMOI_SIF + proxy

# One Slurm task, every GPU visible to it. nese-logistics derives each rank's
# `local_rank` by splitting COMM_WORLD on shared memory, and nese-inference maps
# that to `cuda:{local_rank}`, so the ranks land on distinct GPUs.
#
# The two OMPI_MCA settings are not optional:
#   plm=isolated  -- OpenMPI 4.1 sees SLURM_* in the env and would otherwise try to
#                    launch its `orted` helper via `srun`, which does not exist
#                    inside the container. 'isolated' keeps the launch local.
#   btl=self,vader -- 4.1 spells shared memory 'vader'; the 'sm' the CPU `nese`
#                    module sets is the OpenMPI 5 name. No fabric is involved here:
#                    the ranks only ever barrier.
#
# --oversubscribe is not really oversubscribing: OpenMPI asks Slurm how many slots
# it has and is told 1, because this is one task -- it counts tasks, not the 288
# cores the task holds. Without it mpirun refuses with "not enough slots available".
# --bind-to none then stops it binding all the ranks to the one core it thinks it
# was given.
srun apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env OMP_NUM_THREADS="$THREADS" \
    --env OMPI_MCA_plm=isolated \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,vader \
    "$NESE_ANEMOI_SIF" \
    mpirun -n "$NRANKS" --oversubscribe --bind-to none nese-inference run "$CONFIG"
