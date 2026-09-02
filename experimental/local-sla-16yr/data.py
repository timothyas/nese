"""Open anemoi-inference output, the checkpoint that produced it, and the dataset behind it.

anemoi-inference writes a bare netcdf: the grid dimension is named ``values``, which
collides with ``xr.Dataset.values`` and disagrees with anemoi-datasets, where the same
axis is ``cell``; latitude and longitude ship as data variables rather than coordinates;
``time`` is two stamps mixing the initial and the valid time; and the global attributes
are empty. Everything below that is not arithmetic exists to put that back.

Most of it comes from the **checkpoint**, not from an anemoi dataset. anemoi writes its
metadata into the checkpoint archive as plain JSON next to the pickled weights, so it
reads with nothing but zipfile and json, and it is the model's own account of itself:
the dataset used for evaluation may not be the one used for training, and if it is not,
the dataset's variables, groups and grid are not the ones the model knows.

    variables, variables_metadata,   Each channel's param and group, which is how
      name_to_index                  ``groups: auto`` resolves without parsing names.
    latitudes, longitudes            Present in the netcdf, but as data variables.
    output_mask                      v1 and v2 predict on the whole hpx6 mesh and mask
                                     the output; v3's mesh was pruned at dataprep time
                                     instead. Applying the mask makes the two directly
                                     comparable, and the netcdf itself does not say
                                     which convention it was written under.
    multistep_input, timestep        The input window, needed to know what the model had
                                     already been told about a cell.
    run_id, uuid, checkpoint path    Provenance, and the only copy of it: the netcdf's
                                     global attributes are empty.

Three things still have to come from the anemoi dataset:

    the observations              The truth at the forecast's valid time is not in the
      themselves, and end_date    netcdf -- only the initial condition is.
    mean/stdev/minimum/maximum    Statistics are not carried in the checkpoint metadata.
    healpix_cell_index and the    v1 and v2 can recover the parent-mesh indices from
      healpix_* attrs             ``output_mask``, but v3's pruning moved that
                                  information out of the checkpoint, so plotting on the
                                  full mesh needs the store. Worth putting back.

The inference config is deliberately not read. It is per-run, only the newest is on
disk, and its key names have changed between versions; the written files carry their own
initial times in their names.

Note ``ufs2arco.utils.convert_anemoi_inference_dataset`` does the values -> cell rename
too, but it also regexes ``^(base)_(\\d+)$`` to stack pressure levels, which would split
``sla_filtered_jason_2`` into ``sla_filtered_jason`` at "level 2". Hence the local copy.
"""

import glob
import json
import os
import zipfile

import numpy as np
import xarray as xr
import pandas as pd
import zarr

RUN_DIR = "runs/v3"
ANEMOI_DATASET = "anemoi-hpx6-24h/dataset.zarr"
DATE_FORMAT = "%Y-%m-%dT%H"

FORECASTS = "inference-validation"
CHECKPOINT = "checkpoint/*/inference-last.ckpt"
METADATA = "anemoi.json"

DEFAULT_COUNT = "obs_count_{group}"


def _only(mapping, what):
    """The single entry of a mapping anemoi keys by dataset name."""
    if len(mapping) != 1:
        raise ValueError(f"Expected one {what} in the checkpoint, found {list(mapping)}.")
    return next(iter(mapping.values()))


def open_checkpoint(run_dir=RUN_DIR):
    """What the model knows about itself, read out of the checkpoint archive.

    anemoi stores its metadata as JSON inside the checkpoint zip, with the supporting
    arrays alongside as raw buffers, so none of this needs torch or anemoi installed.
    """
    paths = sorted(glob.glob(f"{run_dir}/{CHECKPOINT}"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one {CHECKPOINT} under {run_dir}, found {len(paths)}: {paths}"
        )

    with zipfile.ZipFile(paths[0]) as archive:
        entry = next(n for n in archive.namelist() if n.endswith(f"/{METADATA}"))
        meta = json.loads(archive.read(entry))
        arrays = {
            key: np.frombuffer(archive.read(spec["path"]), dtype=spec["dtype"]).reshape(spec["shape"])
            for key, spec in _only(meta["supporting_arrays_paths"], "supporting array set").items()
        }

    dataset, task = _only(meta["dataset"], "dataset"), meta["config"]["task"]
    return {
        "checkpoint": paths[0],
        "run_id": meta["run_id"],
        "uuid": meta["uuid"],
        "variables": list(dataset["variables"]),
        "variables_metadata": dataset["variables_metadata"],
        "name_to_index": _only(meta["data_indices"], "data index")["data"]["input"]["name_to_index"],
        "frequency": dataset["frequency"],
        "multistep_input": task["multistep_input"],
        # anemoi writes pandas' old uppercase offset aliases, which pandas 3 rejects.
        "timestep": task["timestep"].lower(),
        "latitudes": arrays["latitudes"],
        "longitudes": arrays["longitudes"],
        "output_mask": arrays.get("output_mask"),
    }


def forecast_paths(run_dir=RUN_DIR):
    """The initial times and the netcdf written for each of them, in time order.

    Read off the directory rather than out of the inference config: the files name their
    own initial time, and the config that made them is not guaranteed to still be here.
    """
    paths = glob.glob(f"{run_dir}/{FORECASTS}/*.nc")
    if not paths:
        raise FileNotFoundError(f"No forecasts under {run_dir}/{FORECASTS}.")

    dates = pd.to_datetime([os.path.basename(p)[:-3] for p in paths], format=DATE_FORMAT)
    order = np.argsort(dates.values)
    return dates[order], [paths[i] for i in order]


STATISTICS = ("mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans")


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


def open_anemoi_statistics(path=ANEMOI_DATASET):
    """The dataset's per-channel statistics, indexed by variable name."""
    return _open_store(path)[list(STATISTICS)]


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

    ``metadata`` is either a checkpoint from :func:`open_checkpoint` or an anemoi
    dataset's attrs -- they carry the same two keys.
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


def pooled_statistics(spec, path=ANEMOI_DATASET):
    """The statistics a field and all its observing groups share.

    Mirrors ``nese_training.observations.pool_statistics``, which is what training
    normalises with, so a reference line drawn from this sits in the same space the model
    was fit in. Groups are weighted by their obs-count mean, which is proportional to how
    many observations each contributed -- anemoi hands a processor only mean/stdev/
    minimum/maximum, never the counts behind them, and that weighting reproduces true
    count-weighted pooling closely.

    Per-mission statistics are not the neutral choice they look like: a mission's mean is
    the sea level of the years it flew, so it carries the trend rather than the field.

    Note the statistics on disk are computed over ``statistics_start_date`` ..
    ``statistics_end_date`` -- the training period -- not over whatever window you are
    evaluating.

    Returns a dimensionless Dataset: mean, stdev, minimum, maximum, and the total
    observation count behind them.
    """
    stats = open_anemoi_statistics(path).compute()
    groups = resolve_groups(spec, stats.attrs)

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
        "count": observed["count"].sum("group"),
    }, attrs={
        "description": f"statistics pooled over every group observing {spec['observed_as']}",
        "groups": groups,
        "statistics_start_date": stats.attrs.get("statistics_start_date"),
        "statistics_end_date": stats.attrs.get("statistics_end_date"),
    })


# Grid metadata the checkpoint does not carry, needed by anything that draws or regrids.
BORROWED_ATTRS = (
    "healpix_nside",
    "healpix_level",
    "healpix_ordering",
    "healpix_cells_total",
    "field_shape",
    "resolution",
)


def grid(checkpoint, dataset=ANEMOI_DATASET):
    """The model's output cells as parent-mesh indices, and the metadata to describe them.

    Checks the checkpoint's own coordinates against the store's before trusting either.
    Nothing else would catch a checkpoint being scored against a dataset it was not
    trained on, which is the whole reason the model metadata is read from the checkpoint.
    """
    store = zarr.open(dataset, mode="r")
    mask = checkpoint["output_mask"]
    take = slice(None) if mask is None else mask

    for name in ("latitudes", "longitudes"):
        if not np.allclose(checkpoint[name][take], store[name][:]):
            raise ValueError(
                f"The {name} in {checkpoint['checkpoint']} do not match {dataset}. The "
                "model predicts on a different grid than it is being scored against."
            )

    attrs = dict(store.attrs)
    attrs = {key: attrs[key] for key in BORROWED_ATTRS if key in attrs}
    attrs.update({key: checkpoint[key] for key in
                  ("checkpoint", "run_id", "uuid", "frequency", "multistep_input", "timestep")})
    attrs["variables_metadata"] = checkpoint["variables_metadata"]
    return store["healpix_cell_index"][:], attrs


def _add_forecast_time(xds):
    """Split the netcdf's two-element ``time`` into t0 and lead time.

    ``time`` is [initial, valid]; ``fhr`` becomes the dimension so forecasts from
    different initial conditions stack, with the wall-clock time kept as ``valid_time``.
    """
    t0 = pd.Timestamp(xds["time"].values[0])
    lead_time = xds["time"] - xds["time"][0]

    xds = xds.assign_coords(
        valid_time=xds["time"],
        lead_time=("time", lead_time.values),
        fhr=(
            "time",
            lead_time.values.astype("timedelta64[h]").astype(int),
            {"description": "forecast hour, aka lead time in hours"},
        ),
    )
    xds = xds.swap_dims({"time": "fhr"}).drop_vars("time")
    return xds.expand_dims(t0=[t0])


def _fixup(xds, mask, cells, attrs, variables=None):
    """Everything an anemoi-inference netcdf is missing, put back.

    ``mask`` reduces a whole-mesh forecast to the cells the model actually predicts,
    which is how v1 and v2 land on the same 33469 cells v3 stores directly.
    """
    if mask is not None:
        xds = xds.isel(values=mask)
    xds = xds.rename({"values": "cell"})
    xds = xds.set_coords(["latitude", "longitude"]).assign_coords(cell=cells)
    if variables is not None:
        xds = xds[list(variables)]

    xds = _add_forecast_time(xds)
    xds.attrs = dict(attrs)
    return xds


def open_forecast(path, checkpoint=None, dataset=ANEMOI_DATASET, variables=None):
    """One anemoi-inference netcdf, with what it is missing put back.

    Returns dims ``(t0, fhr, cell)``, ``cell`` holding parent-mesh HEALPix indices, and
    latitude/longitude/valid_time/lead_time as coordinates.
    """
    checkpoint = checkpoint if checkpoint is not None else open_checkpoint()
    cells, attrs = grid(checkpoint, dataset)
    return _fixup(xr.open_dataset(path, chunks="auto"), checkpoint["output_mask"],
                  cells, attrs, variables)


def open_forecasts(variables=None, run_dir=RUN_DIR, checkpoint=None, dataset=ANEMOI_DATASET):
    """Every initial condition written under ``run_dir``, stacked on ``t0``.

    The grid is read once rather than per file: ``preprocess`` runs on all several
    hundred of them.
    """
    checkpoint = checkpoint if checkpoint is not None else open_checkpoint(run_dir)
    _, paths = forecast_paths(run_dir)
    cells, attrs = grid(checkpoint, dataset)
    mask = checkpoint["output_mask"]
    # Note: this operation looks like it will not scale.
    return xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="t0",
        # valid_time differs file to file and has to be concatenated with them, not
        # taken from the first; xarray's default for that is changing, so pin it.
        coords="different",
        compat="no_conflicts",
        preprocess=lambda xds: _fixup(xds, mask, cells, attrs, variables),
    )


def scatter_to_mesh(xda, dataset=ANEMOI_DATASET):
    """Put a field on the stored cells back onto the full HEALPix mesh.

    The dropped cells come back NaN, which is what plotting routines paint as land.
    Assumes ``cell`` already holds parent-mesh indices, as everything here returns.
    """
    npix = 12*zarr.open(dataset, mode="r").attrs["healpix_nside"]**2
    return xda.reindex(cell=np.arange(npix))
