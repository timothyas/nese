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

# Executes sensitivity_cell.ipynb once per case and sensitivity_global.ipynb once per date, into
# sensitivity_cell.<case>.executed.ipynb and sensitivity_global.<t0>.executed.ipynb, leaving the
# source notebooks clean.

set -euo pipefail
module purge
module use /cluster/projects/nn2993k/timith/sw/modules
module load nese

run() {
    jupyter nbconvert --to notebook --execute "$1.ipynb" --ExecutePreprocessor.timeout=1800 --output "$1.$2.executed.ipynb"
}

for case in gulf_stream eastern_atlantic; do
    SENSITIVITY_CASE=$case run sensitivity_cell $case
done
for t0 in sensitivity/global/*/; do
    t0=$(basename "$t0")
    SENSITIVITY_DATE=$t0 run sensitivity_global $t0
done
