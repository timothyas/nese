"""The anemoi dataset behind a run, and the observation spec convention shared with training.

Inference output is no longer opened here. `nese-inference`'s `nese-zarr` output writes a
self-describing store, so a forecast is just

    xr.open_zarr(f"{run_dir}/inference-validation/forecast.zarr")

with `(t0, fhr, cell)` dims, latitude/longitude/lead_time/valid_time as coordinates, HEALPix
parent-mesh ids on `cell`, and the model's own account of itself in `.attrs`. Everything this
module used to reconstruct -- cracking the checkpoint zip for `variables_metadata`,
`name_to_index`, `output_mask`, `multistep_input` and `timestep`, borrowing the `healpix_*`
attrs off the training store, renaming `values` to `cell`, promoting lat/lon to coordinates,
splitting the two-stamp `time` into `t0` and `fhr`, and `open_mfdataset`-ing several hundred
netcdfs -- is written down at inference time instead. See `nese-inference/docs/output.md`.

What is left is the part that was never about inference output:

    the observations themselves    The truth at a forecast's valid time is not in the
      and start_date/end_date      forecast store, which holds the prediction only.
    the counts behind the          The run's `training-statistics.zarr` sidecar carries
      statistics                   mean/stdev/minimum/maximum, which is everything the
                                   pooling needs, but not `sums`, `squares`, `count` or
                                   `has_nans`, which are not in anemoi-datasets'
                                   `.statistics` either. Only `open_anemoi_statistics` has
                                   those, and only for columns the dataset actually holds.
    normalise_spec, resolve_groups Ported from nese_training.observations, so the block
                                   accepted here is the one training.yaml hands the loss.
                                   Not imported: nese_training pulls in torch and anemoi.

Runs written before the zarr output are netcdf and do not open here any more; re-run
inference against their checkpoint rather than converting. A model that predicts on the
whole mesh and masks its output (v1, v2) needs `post_processors: [{extract_mask: output_mask}]`
in the inference config to land on the same cells as one whose mesh was pruned at dataprep
time (v3).
"""

import numpy as np
import xarray as xr

RUN_DIR = "runs/v3"
ANEMOI_DATASET = "anemoi-hpx6-24h/dataset.zarr"

INFERENCE = "inference-validation"
FORECASTS = f"{INFERENCE}/forecast.zarr"
TRAINING_STATISTICS = f"{INFERENCE}/training-statistics.zarr"

DEFAULT_COUNT = "obs_count_{group}"

DATASET_STATISTICS = ("mean", "stdev", "minimum", "maximum",
                      "sums", "squares", "count", "has_nans")


def open_forecast(run_dir=RUN_DIR, forecasts=FORECASTS):
    """A run's forecasts: ``xr.open_zarr``, with the coordinates read in.

    Dims ``(t0, fhr, cell)``, ``cell`` holding parent-mesh HEALPix ids, and
    latitude/longitude/lead_time/valid_time as coordinates. Nothing is passed to
    ``open_zarr``: the store carries its own CF metadata, so ``lead_time`` decodes to a
    timedelta and ``fhr`` stays the plain integer hour that plots without a datetime
    locator to fight.

    The one thing worth a wrapper is eager coordinates. xarray leaves every non-index
    coordinate of a zarr store dask-backed, and ``valid_time`` is a coordinate that gets
    compared against timestamps and used as a vectorised indexer, neither of which works
    lazily. They are a few kB, so they are simply read.
    """
    xds = xr.open_zarr(f"{run_dir}/{forecasts}")
    return xds.assign_coords({name: c.compute() for name, c in xds.coords.items()})


def open_training_statistics(run_dir=RUN_DIR, statistics=TRAINING_STATISTICS):
    """The statistics the model was trained against, from the run's own sidecar.

    ``mean``, ``stdev``, ``minimum`` and ``maximum`` on a ``variable`` axis, with
    ``statistics_start_date`` and ``statistics_end_date`` in the attrs. nese-inference
    writes these from the *training* dataset recorded in the checkpoint, not from whatever
    the inference config was pointed at, which is what makes them the right frame for a
    normalised score.

    Be careful with the diagnostic's row. ``sla`` is created by a rename in the dataset
    config, and anemoi-datasets carries the source column's statistics through a rename
    untouched, so ``sla`` here is a verbatim copy of ``sla_filtered_jason_2`` rather than
    anything about the field the model emits. The statistics the model was actually
    normalised with are built later, by ``nese_training.observations.pool_statistics``,
    which overwrites mean/stdev/minimum/maximum for the field and every mission observing
    it and never writes the result back to a dataset. :func:`pooled_statistics` rebuilds
    that pooling; this function is its input, not a substitute for it.
    """
    return xr.open_zarr(f"{run_dir}/{statistics}")


def _open_store(path):
    """Every array in the store, lazily, with the anemoi conventions applied.

    The store carries ``_ARRAY_DIMENSIONS``, so open_zarr recovers
    ``(time, variable, ensemble, cell)`` and the store's own chunking on its own. What it
    cannot recover is the convention layered on top: the ``variable`` axis is labelled in
    ``attrs["variables"]`` rather than by a coordinate, and the time and grid arrays are
    stored as ordinary data variables rather than coordinates.
    """
    xds = xr.open_zarr(path)
    xds = xds.assign_coords(
        variable=list(xds.attrs["variables"]),
        time=xds["dates"],
        cell=xds["healpix_cell_index"],
        latitude=xds["latitudes"],
        longitude=xds["longitudes"],
    )
    return xds.drop_vars(["dates", "healpix_cell_index", "latitudes", "longitudes"])


def open_anemoi_dataset(path=ANEMOI_DATASET):
    """The anemoi dataset as xarray, one data variable per channel.

    Lazy: dask-backed, chunked one timestep at a time as stored. The ensemble axis is
    length 1 here and is dropped.
    """
    xds = _open_store(path)
    result = xds["data"].squeeze("ensemble", drop=True).to_dataset(dim="variable")
    result.attrs = dict(xds.attrs)
    return result


def open_anemoi_statistics(path=ANEMOI_DATASET, return_pooled=True):
    """The dataset's per-channel statistics, indexed by variable name."""
    sds = _open_store(path)[list(DATASET_STATISTICS)]
    sds = sds.load()
    if not return_pooled:
        return sds
    else:
        spec = normalise_spec("sla_filtered")
        pooled = pooled_statistics(spec)
        pooled = pooled.expand_dims(
            {
                "variable": np.asarray(["pooled"]).astype(sds.variable.dtype),
            },
        )
        sds = xr.concat([sds, pooled], dim="variable")
        return sds


def normalise_spec(spec):
    """Expand a bare ``observed_as`` string, or a partial mapping, into a full spec.

    Ported from nese_training.observations so the block accepted here is the same one
    training.yaml hands the loss. Not imported: nese_training pulls in torch and anemoi.
    """
    if isinstance(spec, str):
        spec = {"observed_as": spec}
    spec = dict(spec)
    if "observed_as" not in spec:
        raise ValueError(f"Observation spec {spec!r} is missing 'observed_as'.")
    spec.setdefault("groups", "auto")
    spec.setdefault("count", DEFAULT_COUNT)
    return spec


def resolve_groups(spec, metadata):
    """The observing groups of a field, ordered by where their columns sit.

    ``groups: auto`` reads ``variables_metadata``, where every observation channel
    carries its own ``param`` and ``group``, rather than splitting names on the
    ``observed_as`` prefix. The prefix rule cannot tell ``sla_filtered_jason_2``'s group
    from a hypothetical ``sla_filtered_jason`` at level 2; the metadata can.

    ``metadata`` is either a forecast store's attrs or an anemoi dataset's attrs -- they
    carry the same two keys.
    """
    if spec["groups"] != "auto":
        return list(spec["groups"])

    prefix = f"{spec['observed_as']}_"
    order = metadata["name_to_index"]
    known = metadata.get("variables_metadata") or {}

    def group_of(name):
        """The group a channel observes, or None."""
        meta = known.get(name)
        if meta is not None and meta.get("param") == spec["observed_as"]:
            return meta.get("group")
        if not known and name.startswith(prefix):
            return name[len(prefix):]
        return None

    # An alias inherits the metadata of the channel it copies -- inference renames
    # sla_filtered_jason_2 to sla, and the checkpoint sees both -- so a channel only
    # counts as observing the field while it still carries the name its count channel
    # pairs with. Reading the group from metadata rather than from the name is still
    # what keeps sla_filtered_jason_2 out of a hypothetical sla_filtered_jason level 2.
    found = sorted(
        (pos, group) for name, pos in order.items()
        if (group := group_of(name)) is not None and name == f"{prefix}{group}"
    )
    if not found:
        raise ValueError(f"No channels observe '{spec['observed_as']}' here.")
    return [group for _, group in found]


def pooled_statistics(spec, run_dir=RUN_DIR):
    """The statistics a field and all its observing groups share.

    Mirrors ``nese_training.observations.pool_statistics``, which is what training
    normalises with, so a reference line drawn from this sits in the same space the model
    was fit in. Groups are weighted by their obs-count mean, which is proportional to how
    many observations each contributed -- anemoi hands a processor only mean/stdev/
    minimum/maximum, never the counts behind them, and that weighting reproduces true
    count-weighted pooling closely.

    Per-mission statistics are not the neutral choice they look like: a mission's mean is
    the sea level of the years it flew, so it carries the trend rather than the field.

    Read entirely out of the run: the per-mission statistics from its
    ``training-statistics.zarr``, the channel groups from its forecast store's attrs. That
    is what the model was trained against, over the window the sidecar's
    ``statistics_start_date`` and ``statistics_end_date`` name -- not whatever window the
    dataset being evaluated happens to cover.

    Returns a dimensionless Dataset: mean, stdev, minimum and maximum. The total
    observation count behind them is not here, because the counts are not in the
    checkpoint's statistics; :func:`open_anemoi_statistics` is the only source of those.
    """
    stats = open_training_statistics(run_dir).compute()
    groups = resolve_groups(spec, open_forecast(run_dir).attrs)

    by_group = lambda names: (
        stats.sel(variable=names).assign_coords(variable=groups).rename(variable="group")
    )
    observed = by_group([f"{spec['observed_as']}_{group}" for group in groups])
    weight = by_group([spec["count"].format(group=group) for group in groups])["mean"]
    if float(weight.sum()) <= 0:
        raise ValueError(f"Every group observing '{spec['observed_as']}' has a zero obs count.")
    weight = weight/weight.sum()

    mean = (weight*observed["mean"]).sum("group")
    # Pool the second moment about zero, then shift: E[x^2] = var + mean^2.
    variance = (weight*(observed["stdev"]**2 + observed["mean"]**2)).sum("group") - mean**2

    return xr.Dataset({
        "mean": mean,
        "stdev": np.sqrt(np.maximum(variance, 0.0)),
        "minimum": observed["minimum"].min("group"),
        "maximum": observed["maximum"].max("group"),
    }, attrs={
        "description": f"statistics pooled over every group observing {spec['observed_as']}",
        "groups": groups,
        "statistics_start_date": stats.attrs.get("statistics_start_date"),
        "statistics_end_date": stats.attrs.get("statistics_end_date"),
    })


def scatter_to_mesh(xda, attrs):
    """Put a field on the stored cells back onto the full HEALPix mesh.

    The dropped cells come back NaN, which is what plotting routines paint as land.
    Assumes ``cell`` already holds parent-mesh indices, as both the forecast store and
    :func:`open_anemoi_dataset` do; ``attrs`` is that object's, for ``healpix_cells_total``.
    """
    return xda.reindex(cell=np.arange(attrs["healpix_cells_total"]))
