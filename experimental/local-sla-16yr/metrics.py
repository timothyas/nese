"""Deterministic RMSE, MAE and bias against the observations themselves.

The model emits one dense ``sla`` field; the only truth is sparse per-mission
observations, so every score is masked to the cells that mission actually saw in the
target bin -- the same ``obs_count > 0`` mask ``nese_training.losses.ObservationMaskedMSE``
uses during training. HEALPix cells are equal area, so a plain mean over cells is
already area weighted and no Voronoi weights are needed.

Results come at two granularities, both keeping every initial condition unless
``AGGREGATE_T0``:

    per group      dims (t0, fhr, group). NaN where a mission was not live.
    all groups     dims (t0, fhr), by two rules side by side:
                     - live-mission mean, matching the training loss: score each group
                       over its own observed cells, then average over the groups that
                       observed anything in this bin
                     - obs-count pooled (``*_pooled``): every observed (group, cell)
                       pair as one sample, so a wide-swath mission carries more weight

and at three conditionings, which split the same sums without changing them:

    (none)         every observed cell
    by_region      + dim ``region``, where the cell sat in the model's input window
    by_overlap     + dim ``n_overlap``, how many missions saw the cell at the valid time

The regions ask whether the model is interpolating or recalling. Only about a fifth of
the cells scored at a valid time were observed at the initial time, so averaging over
all of them hides which. Note the split is *within* the observed cells: error where
nothing was observed at the valid time has no target and is not measurable here.

``n_overlap`` is the compact form of a pairwise mission-vs-mission mask: one dim of size
``n_live`` rather than ``n_live**2``, answering the same question of whether error drops
where missions corroborate one another. The pairwise matrix itself is one more ``where=``
mask away if it is ever wanted -- ``observed.rename(group="other")``.

All 13 missions are training loss targets, so none of them is held out and no group's
score is out of sample in that sense. The only held-out axis here is time. (The
``sla_filtered_jason_2`` -> ``sla`` alias in forecast.yaml does not make jason_2 special:
``sla`` is a diagnostic, so the aliased column is never an input, never a loss target,
and never in the statistics pool -- ``pool_statistics`` overwrites its mean and stdev
with the pooled ones.)
"""

import os

import numpy as np
import xarray as xr
import pandas as pd

import data

RUN_DIR = "runs/v3mft"
ANEMOI_DATASET = "anemoi-hpx6-24h/dataset.zarr"
OUTPUT_DIR = None  # defaults to {run_dir}/metrics

REGIONS = ("seen_at_t0", "seen_earlier", "unseen")

# Average over initial conditions instead of keeping one row per t0.
AGGREGATE_T0 = False

# What is predicted, and what it is scored against. Same block training.yaml hands the
# loss, so the mission list is recovered rather than restated.
OBSERVATIONS = {
    "sla": {
        "observed_as": "sla_filtered",
        "groups": "auto",
        "count": "obs_count_{group}",
    },
}


def _stack(ads, names, groups):
    """The named channels as one array with a ``group`` dim."""
    return xr.concat([ads[name] for name in names], dim=pd.Index(groups, name="group"))


def _at(xda, times, when):
    """``xda`` sampled at ``when``, a (t0, ...) array of timestamps.

    The unique timestamps are selected first so the vectorised indexing that follows
    runs on a small subset rather than on the whole store. The resulting ``time`` coord
    duplicates ``when`` and is dropped.
    """
    return xda.sel(time=times).sel(time=when).drop_vars("time")


def _verifiable(predicted, dataset_attrs, model):
    """Drop initial conditions the dataset cannot support at either end.

    Late ones verify past ``end_date`` and have no truth; early ones reach back past
    ``start_date`` for their input window. Both would otherwise be silent all-NaN rows.

    The span is the *evaluation* dataset's, so it comes from ``dataset_attrs``; the input
    window is the model's, so it comes from the forecast store.
    """
    end = pd.Timestamp(dataset_attrs["end_date"])
    first = pd.Timestamp(dataset_attrs["start_date"]) + _lookback(model)

    for keep, why in [
        ((predicted["valid_time"] <= end).all("fhr").compute(), f"verifying past {end:%Y-%m-%dT%H}"),
        (predicted["t0"] >= first, f"reaching back before {first:%Y-%m-%dT%H}"),
    ]:
        if not keep.all():
            dropped = predicted["t0"].values[~keep.values]
            print(f"  Dropping {len(dropped)} initial condition(s) {why}: "
                  f"{dropped[0]} .. {dropped[-1]}")
            predicted = predicted.sel(t0=keep)
    return predicted


def _lookback(model):
    """How far back of its own past the model was shown."""
    return (model["multistep_input"] - 1)*pd.Timedelta(model["timestep"])


def align(field, spec, run_dir=RUN_DIR, dataset=ANEMOI_DATASET):
    """The prediction, the observations verifying it, and where they were observed.

    Returns ``(prediction, values, observed, seen)``:

        prediction  (t0, fhr, cell), lazy
        values      (t0, fhr, group, cell), lazy
        observed    (t0, fhr, group, cell) bool, at the valid time, in memory
        seen        (t0, step, cell) bool, any live group across the input window

    The forecast store opens with nothing passed to it and describes itself, so the model
    metadata the alignment needs -- its channels' groups, its input window, its stepping --
    is read off ``forecast.attrs`` rather than out of the checkpoint. Its ``cell`` axis
    already holds parent-mesh HEALPix ids, so the prediction and the observations align on
    ``cell`` with neither being reindexed.

    Groups that observed nothing anywhere in the window are dropped here rather than
    after the reduction, so the conditioned accumulators below stay narrow. The two masks
    are computed eagerly because each of them is reused by every accumulator, and the
    vectorised selection behind them is the expensive part.
    """
    forecast = data.open_forecast(run_dir)
    ads = data.open_anemoi_dataset(dataset)
    groups = data.resolve_groups(spec, forecast.attrs)

    # fhr=0 carries the initial condition, where a diagnostic is all NaN.
    predicted = forecast[field]
    has_prediction = predicted.notnull().any(("t0", "cell")).compute()
    predicted = _verifiable(predicted.sel(fhr=has_prediction), ads.attrs, forecast.attrs)

    counts = _stack(ads, [spec["count"].format(group=g) for g in groups], groups)
    valid = np.unique(predicted["valid_time"].values)

    live = (counts.sel(time=valid) > 0).any(("time", "cell")).compute()
    groups = [group for group, ok in zip(groups, live.values) if ok]
    counts = counts.sel(group=groups)

    observed = (_at(counts, valid, predicted["valid_time"]) > 0).compute()
    values = _at(
        _stack(ads, [f"{spec['observed_as']}_{g}" for g in groups], groups),
        valid,
        predicted["valid_time"],
    )
    return predicted, values, observed, input_seen(counts, predicted["t0"], forecast.attrs)


def input_seen(counts, t0s, model):
    """Where any group observed, at each step of the model's input window.

    Dims ``(t0, step, cell)``, with ``step`` counting backwards from the initial time,
    so ``step=0`` is t0 itself. The window is the model's own ``multistep_input`` and
    ``timestep``, both recorded in the forecast store, so this follows a model that was
    trained differently.
    """
    steps = np.arange(model["multistep_input"])
    when = xr.DataArray(
        t0s.values[:, None] - steps[None, :]*pd.Timedelta(model["timestep"]).to_timedelta64(),
        dims=("t0", "step"),
        coords={"t0": t0s, "step": steps},
    )
    return _at((counts > 0).any("group"), np.unique(when.values), when).compute()


def region_masks(seen):
    """Where a cell sat relative to the input window: a partition, on dim ``region``.

    Mutually exclusive and exhaustive, so the accumulated sums split without loss. Any
    group counts as having seen a cell -- the question is whether the model had
    information there, not whether the same instrument came back around.
    """
    at_t0 = seen.sel(step=0, drop=True)
    return xr.concat(
        [at_t0, seen.isel(step=slice(1, None)).any("step") & ~at_t0, ~seen.any("step")],
        dim=pd.Index(list(REGIONS), name="region"),
    )


def overlap_masks(observed):
    """How many groups saw a cell at the valid time: a partition, on dim ``n_overlap``.

    Starts at 1: a group's own mask makes 0 unreachable, so that bin is always empty.
    """
    degree = observed.sum("group")
    counts = np.arange(1, observed.sizes["group"] + 1)
    return xr.concat([degree == k for k in counts], dim=pd.Index(counts, name="n_overlap"))


def accumulate(prediction, values, observed, where=None):
    """Sums over the observed cells, per (t0, fhr, group) and any dim ``where`` adds.

    Sums rather than scores, so that aggregating over t0 later stays exact for both
    combination rules. Bias is signed prediction minus observed.
    """
    mask = observed if where is None else observed & where
    difference = (prediction - values).where(mask)

    return xr.Dataset({
        "n_obs": mask.sum("cell"),
        "sse": (difference**2).sum("cell"),
        "sae": np.abs(difference).sum("cell"),
        "sd": difference.sum("cell"),
    }).compute()


def _scores(acc):
    """Turn sums into rmse/mae/bias, NaN where nothing was observed."""
    n = acc["n_obs"].where(acc["n_obs"] > 0)
    return acc["sse"]/n, acc["sae"]/n, acc["sd"]/n


def _write(xds, fname):
    """Write via a temporary path, so an open reader cannot block the write.

    These files get read from notebooks, and netCDF4 refuses to overwrite one another
    process holds open -- which fails *after* truncating it. Replacing the path instead
    leaves any existing reader on the old inode.
    """
    tmp = f"{fname}.tmp"
    xds.to_netcdf(tmp)
    os.replace(tmp, fname)


def _drop_t0_coords(xds):
    """Shed the t0-dimensioned coords left dangling once t0 has been reduced away."""
    return xds.drop_vars([name for name, c in xds.coords.items() if "t0" in c.dims])


def per_group(acc, aggregate_t0=AGGREGATE_T0):
    """Scores for each observing group separately, dims ``(t0, fhr, group, ...)``.

    :func:`align` has already dropped the groups that observed nothing in the window, so
    the surviving check only catches a group emptied by a conditioning mask.
    """
    observed = (acc["n_obs"] > 0).any([d for d in acc["n_obs"].dims if d != "group"])
    acc = acc.sel(group=observed)

    if aggregate_t0:
        acc = _drop_t0_coords(acc.sum("t0"))

    mse, mae, bias = _scores(acc)
    return xr.Dataset({
        "rmse": np.sqrt(mse),
        "mae": mae,
        "bias": bias,
        "n_obs": acc["n_obs"],
    })


def all_groups(acc, aggregate_t0=AGGREGATE_T0):
    """Scores aggregated across groups, dims ``(t0, fhr, ...)``.

    Two rules. The live-mission mean reproduces the training loss: dead groups are NaN
    at this point, so a skipna mean over ``group`` divides by the live ones only. The
    pooled variant weights each group by how many cells it observed.

    Under a conditioning mask both divisors follow the bin, which is what is wanted: a
    group that saw nothing in this region or at this overlap degree drops out of the
    mean there rather than being carried as a zero.

    They aggregate over t0 differently, and deliberately: pooling sums over t0 as it
    sums over groups, while the live-mission mean averages the per-sample scores, which
    is the quantity training optimises.
    """
    mse, mae, bias = _scores(acc)
    # Over groups first, so the divisor is the groups live in *that* bin, then over t0.
    reduce = lambda xda: xda.mean("group").mean("t0") if aggregate_t0 else xda.mean("group")
    live = xr.Dataset({
        "rmse": np.sqrt(reduce(mse)),
        "mae": reduce(mae),
        "bias": reduce(bias),
        "n_live_groups": (acc["n_obs"] > 0).sum("group"),
    })

    pooled = acc.sum("t0") if aggregate_t0 else acc
    pooled = pooled.sum("group")
    n = pooled["n_obs"].where(pooled["n_obs"] > 0)

    live["rmse_pooled"] = np.sqrt(pooled["sse"]/n)
    live["mae_pooled"] = pooled["sae"]/n
    live["bias_pooled"] = pooled["sd"]/n
    live["n_obs"] = pooled["n_obs"]
    if aggregate_t0:
        live["n_live_groups"] = live["n_live_groups"].mean("t0")
        live = _drop_t0_coords(live)
    return live


def main(
    observations=OBSERVATIONS,
    run_dir=RUN_DIR,
    dataset=ANEMOI_DATASET,
    output_dir=OUTPUT_DIR,
    aggregate_t0=AGGREGATE_T0,
):
    output_dir = output_dir if output_dir is not None else f"{run_dir}/metrics"
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for field, raw in observations.items():
        spec = data.normalise_spec(raw)
        print(f"Scoring '{field}' against {spec['observed_as']}_*")

        prediction, values, observed, seen = align(field, spec, run_dir, dataset)
        print(f"  {prediction.sizes['t0']} initial conditions, "
              f"fhr = {list(prediction['fhr'].values)}, "
              f"live groups: {', '.join(observed['group'].values)}")

        splits = {
            "": None,
            "by_region": region_masks(seen),
            "by_overlap": overlap_masks(observed),
        }

        results[field] = {}
        for split, where in splits.items():
            acc = accumulate(prediction, values, observed, where)
            for view, xds in [
                ("per_group", per_group(acc, aggregate_t0)),
                ("all_groups", all_groups(acc, aggregate_t0)),
            ]:
                key = f"{split}.{view}" if split else view
                results[field][key] = xds
                fname = f"{output_dir}/{field}.{key}.nc"
                _write(xds, fname)
                print(f"  Stored {fname}")

    return results


if __name__ == "__main__":
    main()
