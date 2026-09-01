"""Open anemoi-inference output, and the anemoi dataset it came from.

anemoi-inference writes a bare netcdf. Everything below that is not arithmetic exists
because that file is missing something the source anemoi dataset has.

    values -> cell          The grid dimension is named ``values``, which collides with
                            ``xr.Dataset.values`` and disagrees with anemoi-datasets,
                            where the same axis is ``cell``.
    healpix_cell_index      data.yaml prunes the mesh to its ocean cells, so a stored
                            field is 33469 long rather than hpx6's 49152. Nothing in the
                            netcdf says which cells survived.
    healpix_nside/_level/   Grid metadata. Needed by anything that draws or regrids.
      _ordering,
      field_shape,
      resolution
    latitude, longitude     Present, but as data variables rather than coordinates.
    variables_metadata      Each channel's ``param`` and ``group``, which is how
                            ``groups: auto`` resolves without parsing names.
    dates, frequency,       The truth at the forecast's valid time is not in the netcdf
      start_date/end_date,  -- only the initial condition is -- so it has to be read
      missing_dates         back out of the dataset, and we need to know where the
                            dataset stops.
    mean/stdev/minimum/     Normalised scores and climatology references later.
      maximum/count
    t0, lead_time, fhr      Derived. The file carries a two-element ``time`` mixing the
                            initial time and the valid time.
    provenance              Unavailable anywhere: global attrs are empty. Checkpoint and
                            run id would have to come from forecast.yaml.

Note ``ufs2arco.utils.convert_anemoi_inference_dataset`` does the values -> cell rename
too, but it also regexes ``^(base)_(\\d+)$`` to stack pressure levels, which would split
``sla_filtered_jason_2`` into ``sla_filtered_jason`` at "level 2". Hence the local copy.

Note that on second thought we should probably get this from the model checkpoint if
possible, rather than from an anemoi dataset.
That is because we may want to use a different anemoi dataset for testing than for
training, and the different dataset may have different groups, and hence different things like
variables_metadata and statistics.
But, lets reserve modifications here until this gets generalized in nese-inference or in a
new package like nese-tools or nese-val or whatever we call it.
It could be that we just store the necessary information at inference time, so that
downstream tasks are simpler.
"""

import numpy as np
import xarray as xr
import pandas as pd
import zarr
from omegaconf import OmegaConf

FORECAST_CONFIG = "forecast.yaml"
ANEMOI_DATASET = "anemoi-hpx6-24h/dataset.zarr"
DATE_FORMAT = "%Y-%m-%dT%H"

DEFAULT_COUNT = "obs_count_{group}"


def forecast_paths(config_file=FORECAST_CONFIG):
    """The initial times and netcdf path written for each of them.

    Rebuilds each path the way nese-inference does: set ``date`` and let OmegaConf
    resolve the ``${date}`` in ``output.netcdf``.
    """
    config = OmegaConf.load(config_file)
    dates = pd.date_range(**OmegaConf.to_container(config.date_range, resolve=True))

    paths = []
    for date in dates:
        config.date = date.strftime(config.get("date_format", DATE_FORMAT))
        paths.append(OmegaConf.to_container(config, resolve=True)["output"]["netcdf"])
    return dates, paths


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


def resolve_groups(spec, attrs):
    """The observing groups of a field, ordered by where their columns sit.

    ``groups: auto`` reads ``variables_metadata``, where every observation channel
    carries its own ``param`` and ``group``, rather than splitting names on the
    ``observed_as`` prefix. The prefix rule cannot tell ``sla_filtered_jason_2``'s group
    from a hypothetical ``sla_filtered_jason`` at level 2; the metadata can.
    """
    if spec["groups"] != "auto":
        return list(spec["groups"])

    order = attrs["name_to_index"]
    metadata = attrs.get("variables_metadata")
    if metadata is not None:
        found = [
            (order[name], meta["group"])
            for name, meta in metadata.items()
            if meta.get("param") == spec["observed_as"] and meta.get("group") is not None
        ]
    else:
        prefix = f"{spec['observed_as']}_"
        found = [
            (pos, name[len(prefix):]) for name, pos in order.items() if name.startswith(prefix)
        ]

    if not found:
        raise ValueError(f"No channels observe '{spec['observed_as']}' in this dataset.")
    return [group for _, group in sorted(found)]


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


# The metadata a forecast needs to stand on its own, once it is no longer next to the
# dataset it came from.
BORROWED_ATTRS = (
    "healpix_nside",
    "healpix_level",
    "healpix_ordering",
    "healpix_cells_total",
    "field_shape",
    "resolution",
    "frequency",
    "variables_metadata",
)


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


def _fixup(xds, cells, attrs, variables=None):
    """Everything an anemoi-inference netcdf is missing, put back."""
    xds = xds.rename({"values": "cell"})
    xds = xds.set_coords(["latitude", "longitude"]).assign_coords(cell=cells)
    if variables is not None:
        xds = xds[list(variables)]

    xds = _add_forecast_time(xds)
    xds.attrs = {key: attrs[key] for key in BORROWED_ATTRS if key in attrs}
    return xds


def _grid(dataset):
    """The cell indices and metadata a forecast has to borrow from its dataset.

    zarr.open rather than :func:`_open_store`: this wants two small arrays and the attrs,
    and has no use for the 27 GB ``data`` array that opening the store as a Dataset would
    bring into scope.
    """
    group = zarr.open(dataset, mode="r")
    return group["healpix_cell_index"][:], dict(group.attrs)


def open_forecast(path, dataset=ANEMOI_DATASET, variables=None):
    """One anemoi-inference netcdf, with what it is missing put back.

    Returns dims ``(t0, fhr, cell)``, ``cell`` holding parent-mesh HEALPix indices, and
    latitude/longitude/valid_time/lead_time as coordinates.
    """
    cells, attrs = _grid(dataset)
    return _fixup(xr.open_dataset(path, chunks="auto"), cells, attrs, variables)


def open_forecasts(variables=None, config_file=FORECAST_CONFIG, dataset=ANEMOI_DATASET):
    """Every initial condition in the inference config, stacked on ``t0``.

    The grid is read once rather than per file: ``preprocess`` runs on all several
    hundred of them.
    """
    _, paths = forecast_paths(config_file)
    cells, attrs = _grid(dataset)
    # Note: this operation looks like it will not scale.
    return xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="t0",
        # valid_time differs file to file and has to be concatenated with them, not
        # taken from the first; xarray's default for that is changing, so pin it.
        coords="different",
        compat="no_conflicts",
        preprocess=lambda xds: _fixup(xds, cells, attrs, variables),
    )


def scatter_to_mesh(xda, dataset=ANEMOI_DATASET):
    """Put a field on the stored cells back onto the full HEALPix mesh.

    The dropped cells come back NaN, which is what plotting routines paint as land.
    Assumes ``cell`` already holds parent-mesh indices, as everything here returns.
    """
    npix = 12*zarr.open(dataset, mode="r").attrs["healpix_nside"]**2
    return xda.reindex(cell=np.arange(npix))
