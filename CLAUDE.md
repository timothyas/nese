# CLAUDE.md

## Context

### Overarching Goals and Scope

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
* We are using the anemoi framework with small extensions on top just to get
  prototypes going. All packages are pip editable installed into the appropriate
  environment
    * We have created $WORK/nese-dataprep to prepare data
    * For model development, we are using the $WORK/anemoi-core repo with the
      python packages:
        * anemoi-graphs - graph creation and manipulation
        * anemoi-models - neural network model construction
        * anemoi-training - everything for ML training
    * Local modifications to anemoi-core are being made in $WORK/nese/training/src for now.
      General philosophy is to use as much of anemoi as possible before modifying
    * We have created $WORK/nese-inference to run inference, which is a thin
      layer over the anemoi-inference package
    * We have created $WORK/nese-verify for evaluation and visualization

## Environment Rules

General headline rules:
- Prefer installing packages from conda-forge over pip if possible

When running on MacOS (environment variable $nickname="bucharest"):
- All Python commands must be run in the `nese` or `nese-anemoi` conda environments,
  the latter used whenever an anemoi package is needed, e.g. for
  (e.g. `conda run -n nese python ...`).

When running on Olivia HPC machine (environment variable $nickname="olivia"):
- Do not run heavy commands on the login nodes
- There is no `conda run`/`conda activate`: the env is a Tykky
  container
- use `module load nese` for CPU work (e.g. nese-dataprep)
- use `module load nese-anemoi` for GPU work (e.g. nese-inference or anemoi-training)

See more environment notes at `nese/environment/olivia/README.md`

## Software Rules

- Prefer staying in xarray natively rather than dropping to numpy, unless absolutely necessary.
    * Do not extract a numpy array from xarray unless explicitly told to do so or
      unless absolute performance is necessary.
    * Instead, use the named dimensions of xarray Datasets and DataArrays. This is
      one of the main purposes of xarray.
    * Do not manually loop over dimensions when xarray can naturally broadcast
      (and arrays fit in memory)
- Avoid writing excessive comments, and instead write readable code with
  intuitive variable names
- Write human readable code while limiting the number of added lines of code
- Cover what you write with tests.
    * See `codecov.yaml` for specific repo targets
- For documentation: keep most descriptions to the docs/
  pages, whereas python module headers should only contain useful notes for
  developers. Do not duplicate information between the two.

## Permissions

- Never use git commands that are not read-only unless explicitly instructed to do so
