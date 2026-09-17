# Local SLA Only
## Well, also with bathymetry/elevation, land fraction, and masking

This is just a simple test I can run on my macbook to see if the engineering is
working.

## Pull in the data

```
HYDRA_IFACE=lo0 FI_PROVIDER=tcp mpirun -n 8 nese-dataprep create data.yaml
nese-dataprep create gebco.yaml
```


## Train the model

```
MPLBACKEND=Agg anemoi-training train --config-name=training
```

## Create graphs

Use the utility

```
python resolve_graph.py
```

To extract the relevant information from `training.yaml` into
`graph.resolved.yaml`, which can be used with the `anemoi-graphs` CLI to create and
inspect graphs.
For example:

```
anemoi-graphs create graph.resolved.yaml graph.pt
anemoi-graphs inspect graph.pt figures/
```
