# Local SLA-Only Prototype

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

On top of the weights there are 1.50 M non-trainable buffers (6.8 MB): edge
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

Non-trainable buffers are unchanged from version 0 at 1.50 M (6.8 MB): edge
lengths/directions and the fixed node coordinate features.
Dropping
`trainable_size 2->0` removes 540,874 trainable parameters (30% of version 0),
entirely from the node embeddings and the trainable edge attributes.


### Version 2

* With forcings
* `trainable_size = 0`

### Version 3

* test removing some fraction of data at prediction time

### After that ...

* Predict full state at multiple time steps, or `obs(t-win, t) -> x(t)` with
  removed points
* Tackle some of the network oddities noted when visualizing the architecture


## Step-by-step

### Pull in the data

```
HYDRA_IFACE=lo0 FI_PROVIDER=tcp mpirun -n 8 nese-dataprep create data.yaml
```

### Train the model

```
MPLBACKEND=Agg anemoi-training train --config-name=training
```
