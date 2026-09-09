# Local SLA-Only Prototype

A slightly larger laptop prototype compared to `../local-sla-02yr`.
Again this is just to build out initial data preparation, training, inference,
evaluation, and visualization pipelines and to make sure things are working.

Goals:
* Ensure the data prep is good to go before burning CPU hours. We've run
  into a number of issues scaling up in @$WORK/nese-dataprep and fixed
  those. 15 years is apparently a good stress test, even with only sla.
* Make sure the neural network is or can actually learn something, having more data
  helps confidence here
* Solve some issues related to loss function definition, config details, etc, when more
  missions are present.
* Establish the initial evaluation pipeline

Trained on laptop, only predicting SLA.
Uses:
* 15 years of training data: 2000--2014 (inclusive)
* 1 year of validation data: 2015
* Deterministic MSE loss, uniformly weighted, normalized by live missions
  per-sample
* Common network specs:
    * `num_channels = 128`
    * MLP-to-hidden ratio = 4
      Defining the size of hidden to in & out:
      ```
      Linear(in → hidden), Activation, Linear(hidden → out)
      ```
    * Processor:
        * 4 layers
        * 4 attention heads
        * `dropout_p = 0`
    * Encoder & Decoder:
        * 4 attention heads


## Versions

### Version 0

* No forcings (e.g., cos/sin julian day, insolation)
    * Note that cos & sin lat & lon are used for the GraphTransformer operators,
      but they are not explicitly added as features
* No dropout
* `trainable_size = 2` for all (data, hidden, data2hidden, hidden2data)

Model size: **~1.79 M trainable parameters, 7.2 MB of weights (fp32)**.

| Stage | Trainable params | MB |
|---|---:|---:|
| Node attributes (learnable node embeddings, data + hidden) | 122,880 | 0.49 |
| Encoder (`GraphTransformerForwardMapper` + trainable edge attrs) | 228,352 + 215,900 = **444,252** | 1.78 |
| Processor (4-layer `TransformerProcessor`) | **791,552** | 3.17 |
| Decoder (`GraphTransformerBackwardMapper` + trainable edge attrs) | 227,841 + 200,814 = **428,655** | 1.71 |
| **Total** | **1,787,339** | **7.15** |

On top of the weights there are 1.55 M non-trainable buffers (6.9 MB): edge
lengths/directions and the fixed node coordinate features.

### Version 1

* No forcings
* `trainable_size = 0` ... get rid of these and see how it impacts "lines" in
  predictions

Model size: **~1.25 M trainable parameters, 5.0 MB of weights (fp32)**.

| Stage | Trainable params | MB |
|---|---:|---:|
| Node attributes (no learnable node embeddings) | 0 | 0.00 |
| Encoder (`GraphTransformerForwardMapper`, no trainable edge attrs) | **227,584** | 0.91 |
| Processor (4-layer `TransformerProcessor`) | **791,552** | 3.17 |
| Decoder (`GraphTransformerBackwardMapper`, no trainable edge attrs) | **227,329** | 0.91 |
| **Total** | **1,246,465** | **4.99** |

Non-trainable buffers are unchanged from version 0 at 1.55 M (6.9 MB): edge
lengths/directions and the fixed node coordinate features.
Dropping
`trainable_size 2->0` removes 540,874 trainable parameters (30% of version 0),
entirely from the node embeddings and the trainable edge attributes.


### Version 2

* With forcings
* `trainable_size = 0`

Model size: **~1.25 M trainable parameters, 5.0 MB of weights (fp32)**.

| Stage | Trainable params | MB |
|---|---:|---:|
| Node attributes (no learnable node embeddings) | 0 | 0.00 |
| Encoder (`GraphTransformerForwardMapper`, no trainable edge attrs) | **230,272** | 0.92 |
| Processor (4-layer `TransformerProcessor`) | **791,552** | 3.17 |
| Decoder (`GraphTransformerBackwardMapper`, no trainable edge attrs) | **230,017** | 0.92 |
| **Total** | **1,251,841** | **5.01** |

Non-trainable buffers are unchanged from version 1 at 1.55 M (6.9 MB): the seven new
forcings add only their normalisation statistics.

The forcings cost 5,376 trainable parameters over version 1, split evenly between the two
input embeddings: 7 forcings x 3 input steps x 128 channels = 2,688 on each side. The
processor is untouched, as it only ever sees the 128 hidden channels.

### Version 3

* With forcings
* `trainable_size = 0`
* Remove land points with nese-dataprep, rather than with masking
    * Keep `land_fraction` with `min-max` normalization,
      even though this just indicates coastal points

Model size: **~1.25 M trainable parameters, 5.0 MB of weights (fp32)** -- identical to
version 2.

| Stage | Trainable params | MB |
|---|---:|---:|
| Node attributes (no learnable node embeddings) | 0 | 0.00 |
| Encoder (`GraphTransformerForwardMapper`, no trainable edge attrs) | **230,272** | 0.92 |
| Processor (4-layer `TransformerProcessor`) | **791,552** | 3.17 |
| Decoder (`GraphTransformerBackwardMapper`, no trainable edge attrs) | **230,017** | 0.92 |
| **Total** | **1,251,841** | **5.01** |

Since there are no trainable parameters on the mesh, parameter count doesn't change from
v2 to v3, since nothing depends on the number of cells in the graph (only goes
with channel count and latent size).
Non-trainable buffers change, as there are fewer nodes and no boolean `sea_mask`
anymore.
No substantive changes to the graph, just removed dead weight.

### Version 3 Matern fine tuned

`v3mft` is same as v3, but with a Matern penalty, same as described in v4, but
added on after v3 training.
Fine tuning consisted of
* Max LR = 5e-5
* 50 linear warm up steps
* 300 training steps

All of these values are 1 order of magnitude smaller than the pretraining run.

### Version 4

Same as verison 3, but with a Matern loss penalty turned on, defined with

* Loss weights amplitudes:
    * 1 on misfit term
    * 0.1 on Matern term
* Matern parameters:
    * `order` = M = 2
    * `rho_hat` = 8 ~= 815 km to correlation drops to 0.14
    * only turned on at points with at least 3 neighbors


### Version Np

Capturing epistemic uncertainty with dropout.
Note: the dropout implementation blows up memory on the CPU, but is basically
free on the GPU.
So, this will wait for now.

* With forcings
* `trainable_size = 0`
* Remove land points with nese-dataprep
* `dropout_p = 0.1`

### Version Nt

* With forcings
* `trainable_size = 0`
* Remove land points with nese-dataprep
* Randomly remove 30% of input and target observations

### Version Nb


### After that ...

* Predict full state at multiple time steps, or `obs(t-win, t) -> x(t)` with
  removed points
* Tackle some of the network oddities noted when visualizing the architecture
* Use a "pooled" error metric for the loss, normalizing over observation counts
  rather than just num missions (probably not impactful, but cleaner)
* dropout > 0, note this only affects training. anemoi-inference is
  deterministic


## Step-by-step

### Pull in the data

```
HYDRA_IFACE=lo0 FI_PROVIDER=tcp mpirun -n 8 nese-dataprep create data.yaml
```

### Train the model

```
MPLBACKEND=Agg anemoi-training train --config-name=training
```
