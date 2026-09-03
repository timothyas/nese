#!/usr/bin/env bash
# Regenerate every table and check in ../main.tex.
set -euo pipefail
cd "$(dirname "$0")"
for script in table_phi table_pixel_shape check_weights check_analytic table_sigma table_coast table_validation \
              check_normalisation check_alternatives; do
    conda run -n nese-anemoi python "$script.py"
done
