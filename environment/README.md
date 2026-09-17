# NESE Environments

The two conda environment files should work anywhere that accepts conda
natively.
They are:
* `environment.yaml`: CPU NESE environment for nese-dataprep, nese-verify, and
  general analysis
* `environment-anemoi.yaml`: NESE environment for nese-inference and the
  anemoi package suite. GPU on HPC, but only CPU on MacOS

To install in a developer-like setup:

```bash
conda env create -f environment.yaml
conda activate nese
pip install -e /path/to/nese-logistics --no-deps
pip install -e /path/to/nese-dataprep --no-deps
pip install -e /path/to/nese-verify --no-deps
```

and
```bash
conda env create -f environment-anemoi.yaml
conda activate nese-anemoi
pip install -e /path/to/anemoi-core/graphs --no-deps --no-build-isolation
pip install -e /path/to/anemoi-core/models --no-deps --no-build-isolation
pip install -e /path/to/anemoi-core/training --no-deps --no-build-isolation
pip install -e /path/to/nese/training --no-deps
pip install -e /path/to/nese-inference --no-deps
```

## Olivia

The Olivia HPC machine needs extra care, see `olivia/` for more details.
