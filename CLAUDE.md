# CLAUDE.md

## Rules

- All Python commands must be run in the `seso` conda environment (e.g. `conda run -n seso python ...`).
- Prefer installing packages from conda-forge (`conda install -n seso -c conda-forge <pkg>`) over pip when a conda-forge build is available.
- Prefer staying in xarray natively rather than dropping to numpy, unless absolutely necessary. Libraries like xesmf naturally handle extra dimensions (e.g. ensemble members), so avoid manual loops with numpy when xarray broadcasting will work.
- Avoid writing excessive comments, and instead write readable code
