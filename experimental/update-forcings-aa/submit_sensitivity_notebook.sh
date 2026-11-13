#!/bin/bash -l

#SBATCH --account=nn2993k
#SBATCH --job-name=sensitivity-notebook
#SBATCH --output=slurm/%x-%j.out
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Executes sensitivity_global.ipynb once per date into sensitivity_global.<t0>.executed.ipynb,
# leaving the source notebook clean.

set -euo pipefail
module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese

for t0 in sensitivity/global/*/; do
    t0=$(basename "$t0")
    SENSITIVITY_DATE=$t0 jupyter nbconvert --to notebook --execute sensitivity_global.ipynb \
        --ExecutePreprocessor.timeout=1800 --output "sensitivity_global.$t0.executed.ipynb"
done
