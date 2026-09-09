# CLAUDE.md

## Rules

- All Python commands must be run in the `nese` conda environment (e.g. `conda run -n nese python ...`).
  On the Olivia HPC machine there is no `conda run`/`conda activate`: the env is a Tykky
  container, used via `module load nese`. Build/rebuild it and mutate its packages only
  through `environment/olivia/` (`build-env.sh` + `environment.yaml`); see
  `environment/olivia/README.md`.
- Prefer installing packages from conda-forge (`conda install -n nese -c conda-forge <pkg>`) over pip when a conda-forge build is available.
- Prefer staying in xarray natively rather than dropping to numpy, unless absolutely necessary. Libraries like xesmf naturally handle extra dimensions (e.g. ensemble members), so avoid manual loops with numpy when xarray broadcasting will work.
- Avoid writing excessive comments, and instead write readable code
- Never use git commands that are not read-only unless explicitly instructed to do so
