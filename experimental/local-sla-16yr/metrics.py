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

All 13 missions are training loss targets, so none of them is held out and no group's
score is out of sample in that sense. The only held-out axis here is time. (The
``sla_filtered_jason_2`` -> ``sla`` alias in forecast.yaml does not make jason_2 special:
``sla`` is a diagnostic, so the aliased column is never an input, never a loss target,
and never in the statistics pool -- ``pool_statistics`` overwrites its mean and stdev
with the pooled ones.)

Note: may as well drop all NaN groups over the evaluation period.
"""

import os

import numpy as np
import xarray as xr
import pandas as pd

import data

FORECAST_CONFIG = "forecast.yaml"
ANEMOI_DATASET = "anemoi-hpx6-24h/dataset.zarr"
OUTPUT_DIR = "runs/v3/metrics"

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


def align(field, spec, config_file=FORECAST_CONFIG, dataset=ANEMOI_DATASET):
    """The prediction and the observations that verify it, on shared dims.

    Returns ``(prediction, values, counts)`` with dims ``(t0, fhr, cell)`` and
    ``(t0, fhr, group, cell)``. Both stay lazy.
    """
    ads = data.open_anemoi_dataset(dataset)
    groups = data.resolve_groups(spec, ads.attrs)

    forecast = data.open_forecasts([field], config_file, dataset)

    # fhr=0 carries the initial condition, where a diagnostic is all NaN.
    predicted = forecast[field]
    # TODO: This operation looks like it will not scale
    has_prediction = predicted.notnull().any(("t0", "cell")).compute()
    predicted = predicted.sel(fhr=has_prediction)

    # The dataset stops before the last forecast's valid time, so that one has nothing
    # to be scored against.
    end = pd.Timestamp(ads.attrs["end_date"])
    verifiable = (predicted["valid_time"] <= end).all("fhr").compute()
    if not verifiable.all():
        dropped = predicted["t0"].values[~verifiable.values]
        print(
            f"Dropping {len(dropped)} initial condition(s) verifying past the dataset's "
            f"{end:%Y-%m-%dT%H}: {dropped[0]} .. {dropped[-1]}"
        )
        predicted = predicted.sel(t0=verifiable)

    truth = ads.sel(time=np.unique(predicted["valid_time"].values))
    stack = lambda names: xr.concat(
        [truth[name] for name in names], dim=pd.Index(groups, name="group")
    ).sel(time=predicted["valid_time"]).drop_vars("time")  # duplicate of valid_time

    values = stack([f"{spec['observed_as']}_{group}" for group in groups])
    counts = stack([spec["count"].format(group=group) for group in groups])
    return predicted, values, counts


def accumulate(prediction, values, counts):
    """Sums over the observed cells, per (t0, fhr, group).

    Sums rather than scores, so that aggregating over t0 later stays exact for both
    combination rules. Bias is signed prediction minus observed.
    """
    mask = counts > 0
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
    """Scores for each observing group separately, dims ``(t0, fhr, group)``.

    Groups that observed nothing anywhere in the window are dropped: the file describes
    this evaluation, not the dataset's declared mission list. Dropping them is inert for
    :func:`all_groups`, which is why that keeps the full set -- a group with no
    observations is NaN under the live-mission mean and contributes nothing to the pool.
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
    """Scores aggregated across groups, dims ``(t0, fhr)``.

    Two rules. The live-mission mean reproduces the training loss: dead groups are NaN
    at this point, so a skipna mean over ``group`` divides by the live ones only. The
    pooled variant weights each group by how many cells it observed.

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
    config_file=FORECAST_CONFIG,
    dataset=ANEMOI_DATASET,
    output_dir=OUTPUT_DIR,
    aggregate_t0=AGGREGATE_T0,
):
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for field, raw in observations.items():
        spec = data.normalise_spec(raw)
        print(f"Scoring '{field}' against {spec['observed_as']}_*")

        prediction, values, counts = align(field, spec, config_file, dataset)
        print(f"  {prediction.sizes['t0']} initial conditions, "
              f"fhr = {list(prediction['fhr'].values)}, "
              f"{values.sizes['group']} groups")

        acc = accumulate(prediction, values, counts)
        results[field] = {
            "per_group": per_group(acc, aggregate_t0),
            "all_groups": all_groups(acc, aggregate_t0),
        }
        observed = list(results[field]["per_group"]["group"].values)
        print(f"  observed in this window: {len(observed)} of {values.sizes['group']} "
              f"-- {', '.join(observed)}")

        for name, xds in results[field].items():
            fname = f"{output_dir}/{field}.{name}.nc"
            _write(xds, fname)
            print(f"  Stored {fname}")

    return results


if __name__ == "__main__":
    main()
