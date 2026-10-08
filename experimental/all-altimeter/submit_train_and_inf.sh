#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=training-inference-v0
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=808G
#SBATCH --time=01:00:00

# This took 33 minutes total
# <1 min for the graph
# 29 min training
# 2.5 min validation inference
# 1.5 min 2015 inference

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

# One process builds the graph; the training ranks load it. A concurrent
# torch.save from every rank corrupts the file.
apptainer exec --nv --bind /cluster --pwd "$PWD" "$NESE_ANEMOI_SIF" python resolve_graph.py
apptainer exec --nv --bind /cluster --pwd "$PWD" "$NESE_ANEMOI_SIF" \
    anemoi-graphs create --overwrite --no-description graph.resolved.yaml graph.pt

srun --overlap --gpu-bind=none apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" "$NESE_ANEMOI_SIF" \
    anemoi-training train --config-name=training

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
