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

## Build the dataset

See `submit_1node.sh` and `submit_4node.sh` for 1 and 4 node versions on the
`small` and `large` partitions, respectively.

```bash
sbatch submit_1node.sh
```

The job is network-bound: every rank streams along-track altimetry from CMEMS
through the shared Olivia proxy, so more ranks/nodes help only up to the proxy's
aggregate throughput.
Both `small` and `large` allocate and bill whole nodes
here (`sbatch` prints a note), so prefer 1 node unless the single-node run is
clearly proxy-bound below full speed.

### Why 128 tasks per node

Olivia CPU nodes have 256 cores (no SMT (Simultaneous Multi Thread)) and ~740 GiB RAM.
Per-rank memory here is
small (rank 0 peaks ~1-2 GiB for the one-time GEBCO regrid), so memory does not
limit the count. 128 balances download concurrency against proxy connections and
Lustre metadata load (per-rank log + zarr chunk writes). Tune down to 64 if CMEMS
throttles; 256 costs nothing extra (nodes are billed whole) if profiling shows
bandwidth headroom.
