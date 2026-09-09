# CLAUDE.md

## Rules

- All Python commands must be run in the `nese` or `nese-anemoi` conda environments,
  the latter used whenever an anemoi package is needed (e.g. `conda run -n nese python ...`).
  On the Olivia HPC machine there is no `conda run`/`conda activate`: the env is a Tykky
  container, used via `module load nese`. Build/rebuild it and mutate its packages only
  through `environment/olivia/` (`build-env.sh` + `environment.yaml`); see
  `environment/olivia/README.md`.
- Prefer installing packages from conda-forge (e.g. `conda install -n nese -c conda-forge <pkg>`) over pip when a conda-forge build is available.
- Prefer staying in xarray natively rather than dropping to numpy, unless absolutely necessary.
  Avoid manual loops with numpy when xarray broadcasting will work.
  Do not extract a numpy array from xarray, instead use the named coordinates.
- Avoid writing excessive comments, and instead write readable code
- Never use git commands that are not read-only unless explicitly instructed to do so

## Project Context

### Overarching goals and scope

* The overall goal is to predict dense fields from sparse observations, like
  data assimilation does
    * Note: in the training.yaml configurations, this is why the sla diagnostic
      is added, in order to get a full field prediction that is penalized
      against the separate obs missions
* The work is motivated by AIFS-DOP http://arxiv.org/abs/2606.19093, which shows
  that something like this (although they go straight to 10 day forecasting) is
  possible in the atmosphere
* Right now, we are making prototypes to understand if the algorithm and code is doing what I want.
* The full version will use ~30 years of training data from all missions,
  include more variables, and atmospheric forcing (or maybe we will predict
  atmospheric state variables directly)

### Engineering

* For now, we are spatiotemporally binning the data. Because of that binning process,
  the problem is setup more like a forecast problem (predict state at t+dt from state at t & t-dt).
* If this works with binned data and a few variables, we will proceed to a more DA-true framework,
  taking in raw observations from a prior window and creating a full state
  estimate. This requires significant engineering we don't want to do now.
* We are using the anemoi framework with small extensions on top just to get a
  prototype going
    * anemoi-core repo is at $WORK/anemoi-core, with
      anemoi-models, anemoi-graphs, anemoi-training packages locally installed
      into the nese-anemoi conda environment
    * Local modifications are being made in $WORK/nese/training/src for now, and
      extensions can be made as needed. General philosophy is to use as much of
      anemoi as possible before modifying
    * We have created $WORK/nese-dataprep to prepare data
    * We have created $WORK/nese-inference, which is a very thin layer over
      inference to run anemoi-inference, and easy options to pivot as needed
