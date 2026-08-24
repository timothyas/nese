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
    * Encoder & Decoder:
        * 4 attention heads


## Prototype 1: sea-mask-in-model

* No forcings (e.g., cos/sin julian day, insolation)
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
