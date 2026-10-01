#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=check-masks
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00

set -euo pipefail
export OMP_NUM_THREADS=1

module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese

# A short slice first, so a bug fails in seconds rather than after the full pass
python check_masks.py 10
python check_masks.py
