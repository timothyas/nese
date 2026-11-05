#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=seen-fraction
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00

# The fraction of target observations unseen in the input window, over 100 training samples.

set -euo pipefail
module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese

python seen_fraction.py training.yaml verify.validation.yaml
