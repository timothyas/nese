#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=inference-v2p0
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=808G
#SBATCH --time=01:00:00

# This took 11 minutes total
# 7 min validation inference
# 3.5 min 2015 inference

set -euo pipefail

# An aborting rank dumps a multi-GB core file into the experiment directory.
ulimit -c 0

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

# Per-job matplotlib cache dir -- avoids 4 ranks racing on ~/.config/matplotlib.
MPLCONFIGDIR="/cluster/gpuscratch/jobs/$SLURM_JOB_ID/mpl"; mkdir -p "$MPLCONFIGDIR"
NRANKS="${SLURM_GPUS_ON_NODE:?no GPUs allocated -- this script expects --gpus-per-node}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / NRANKS ))

# mpirun here only ever launches ranks on this node (plm=isolated), but by default it also
# reads the Slurm allocation and then has to match Slurm's node name to its own. When that
# match fails, it aborts with "No nodes are available for this job". This happened on
# ~1 in 4 runs. Disabling the slurm allocation component leaves it just the local host.

srun --ntasks=1 apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env OMP_NUM_THREADS="$THREADS" \
    --env OMPI_MCA_plm=isolated \
    --env OMPI_MCA_ras=^slurm \
    --env PRTE_MCA_ras=^slurm \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,vader \
    "$NESE_ANEMOI_SIF" \
    mpirun -n "$NRANKS" --oversubscribe --bind-to none nese-inference run inference.validation.yaml

srun --ntasks=1 apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env OMP_NUM_THREADS="$THREADS" \
    --env OMPI_MCA_plm=isolated \
    --env OMPI_MCA_ras=^slurm \
    --env PRTE_MCA_ras=^slurm \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,vader \
    "$NESE_ANEMOI_SIF" \
    mpirun -n "$NRANKS" --oversubscribe --bind-to none nese-inference run inference.2015.yaml
