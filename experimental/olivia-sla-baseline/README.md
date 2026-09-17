# Olivia SLA Only Baseline

The purpose is to simply scale up `../local-sla-16yr` to more data and a bigger
network.
This will serve as a baseline for experimentation including
* more variables
* problem formulation
* uncertainty representations
* matern loss penalty
* other architectural changes

See `../../environment/olivia` for scripts that create the `nese` environment,
and a README with tips (e.g., including file system notes).

## Building the dataset

With 1 node and 128 tasks, it took ~13 minutes to build the whole dataset.
The job is network-bound: every rank streams along-track altimetry from CMEMS
through the shared Olivia proxy, so more ranks/nodes help only up to the proxy's
aggregate throughput.
Both `small` and `large` allocate and bill whole nodes
here (`sbatch` prints a note), so prefer 1 node unless the single-node run is
clearly proxy-bound below full speed.


## Model

Same as `../local-sla-16yr` except...
* `num_channels = 512`
* 16 processor layers
* 16 attention heads everywhere

With forcings, `trainable_size = 0` for all four connections (no learnable node
embeddings or edge attributes), no dropout.

Graph: 33,469 ocean nodes (land removed in dataprep) to a 12,288-cell HEALPix
hidden mesh (nside=32, ~1.8 deg).

### Version 0

* Includes computable forcings
* No graph-dependent trainable parameters
* No dropout

Model size: **~57.4 M trainable parameters, 230 MB of weights (fp32)**.

| Stage | Trainable params | MB |
|---|---:|---:|
| Encoder (`GraphTransformerForwardMapper`, 1 block) | **3,499,520** | 14.0 |
| Processor (16-layer `TransformerProcessor`, 3,150,848 / layer) | **50,413,568** | 201.7 |
| Decoder (`GraphTransformerBackwardMapper`, 1 block) | **3,498,497** | 14.0 |
| **Total** | **57,411,585** | **229.6** |

The processor is 88% of the weights. Each processor layer is a plain
pre-norm transformer block: bias-free QKV + output projection (4 x 512x512), two
LayerNorms, and a 512->2048->512 MLP.

## Training

* Total batch size 16, with 4 GPUs x 4 samples per GPU
* Effective Learning Rate 1.0e-3
* Wall-clock time:
    * 30k training iterations = 1.8 hours (overfit)
    * 10k training iterations = 35 minutes

Opted not to write inputs or outputs to local / job file systems, since the data
is very small.
Outputs (checkpoints, plots) are written once per epoch so it's not overwhelming
anything.
