#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=nese-olivia-baseline
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=large
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

set -euo pipefail

# --- one thread per rank (nese-dataprep sets no thread caps; ranks are packed) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_NUM_WORKERS=1

# --- multi-node OpenMPI ---
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid

module load nese
srun nese-dataprep create data.yaml
