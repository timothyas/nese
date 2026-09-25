#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=inference-v2
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=808G
#SBATCH --time=00:10:00

set -euo pipefail
NRANKS="${SLURM_GPUS_ON_NODE:?no GPUs allocated -- this script expects --gpus-per-node}"
THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / NRANKS ))

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese-anemoi

# mpirun here only ever launches ranks on this node (plm=isolated), but by default it also
# reads the Slurm allocation and then has to match Slurm's node name to its own. When that
# match fails, it aborts with "No nodes are available for this job". This happened on
# ~1 in 4 runs. Disabling the slurm allocation component leaves it just the local host.

srun apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env OMP_NUM_THREADS="$THREADS" \
    --env OMPI_MCA_plm=isolated \
    --env OMPI_MCA_ras=^slurm \
    --env PRTE_MCA_ras=^slurm \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,vader \
    "$NESE_ANEMOI_SIF" \
    mpirun -n "$NRANKS" --oversubscribe --bind-to none nese-inference run inference.validation.yaml

srun apptainer exec --nv --bind /cluster --pwd "$PWD" \
    --env OMP_NUM_THREADS="$THREADS" \
    --env OMPI_MCA_plm=isolated \
    --env OMPI_MCA_ras=^slurm \
    --env PRTE_MCA_ras=^slurm \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,vader \
    "$NESE_ANEMOI_SIF" \
    mpirun -n "$NRANKS" --oversubscribe --bind-to none nese-inference run inference.2015.yaml
