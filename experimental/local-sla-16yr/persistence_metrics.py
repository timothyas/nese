"""Persistence: the observations at the initial time, carried forward and scored as a model.

The reference line for :mod:`metrics`. Everything about the scoring is that module's --
the masks, the sums, the two combination rules, the ``(t0, fhr)`` grid -- so the only
thing here is the prediction, which is not predicted at all: it is whatever a mission saw
at ``t0``, held constant to the valid time.

``nese-inference`` writes the initial conditions into the forecast store beside the
forecast, so this needs no second data source for the prediction side. ``fhr=0`` carries
every input channel in physical units and is bit-identical to the anemoi dataset at the
same timestamp; ``fhr=24`` carries only the ``sla`` diagnostic, the observation channels
there being all NaN.

Two group axes, and they are not the same axis:

    source   the mission whose t0 observations are being persisted, plus ``pooled``.
             One netcdf per source.
    group    the mission whose observations at the valid time are the truth. A dim
             inside each file, exactly as in :mod:`metrics`.

Each source is treated as a candidate dense field, the way the model's ``sla`` is: a
source's t0 field is scored against *every* live mission at the valid time, not only
against itself. Its own diagonal is still in there -- ``sla.cryosat_2.per_group.nc`` at
``group="cryosat_2"`` is same-mission persistence -- but it is one column of the row
rather than the whole story, and it is the sparsest one, since it needs the ground track
to come back around within a day.

``pooled`` is the obs-count weighted mean over the groups that observed a cell at t0,
which is the closest thing the observations alone offer to the dense field the model
emits. Weighting by count rather than uniformly mirrors ``data.pooled_statistics``.

Two caveats worth carrying into any comparison:

    coverage    Persistence exists only where something was observed at t0, which is
                about a fifth of the cells scored at a valid time. The matching model
                number is therefore ``{run_dir}/metrics/sla.by_region.*.nc`` at
                ``region="seen_at_t0"``, *not* the unconditioned file. For ``pooled`` the
                two cell sets are identical by construction, so their ``n_obs`` should
                agree exactly; for a single-mission source it is a subset of that.
    lead time   Only ``fhr`` the model actually predicts are scored, so persistence is
                penalised over the same horizon and no further.

``by_region`` is absent, and degenerate if it were added: ``region_masks``' ``seen_at_t0``
is ``seen.sel(step=0)``, which is precisely where pooled persistence is defined, so the
other two regions would be empty. Only the model, which fills cells it was shown nothing
about, has anything to say there. ``by_overlap`` is kept and means what it does in
:mod:`metrics`: whether error falls where missions corroborate one another.
"""

import os

import numpy as np
import xarray as xr
import pandas as pd

import data
import metrics

RUN_DIR = "runs/v4"
ANEMOI_DATASET = "anemoi-hpx6-24h/dataset.zarr"
OUTPUT_DIR = None  # defaults to {run_dir}/../persistence-metrics

POOLED = "pooled"

# Average over initial conditions instead of keeping one row per t0.
AGGREGATE_T0 = False

# Conditionings to split the sums by, alongside the unconditioned score. Dropping
# by_overlap is the cheap way out if the run is large: its mask is the expensive one.
SPLITS = ("by_overlap",)

OBSERVATIONS = {
    "sla": {
        "observed_as": "sla_filtered",
        "groups": "auto",
        "count": "obs_count_{group}",
    },
}


def initial_conditions(spec, groups, run_dir=RUN_DIR):
    """What each group observed at t0, from the forecast store's own ``fhr=0`` slice.

    Returns ``(values, counts)``, both ``(t0, group, cell)`` and lazy. This is the run's
    account of its own input, so it follows the model's cell selection and stepping
    without being told either; it happens to equal the anemoi dataset at those timestamps,
    but nothing here depends on that dataset being the one inference was pointed at.
    """
    ic = data.open_forecast(run_dir).sel(fhr=0, drop=True)
    values = metrics._stack(ic, [f"{spec['observed_as']}_{g}" for g in groups], groups)
    counts = metrics._stack(ic, [spec["count"].format(group=g) for g in groups], groups)
    return values, counts


def persistence_fields(values, counts, pooled=POOLED):
    """The candidate t0 field of each source: ``{source: (t0, cell)}``, NaN where undefined.

    One entry per group, holding that mission's own observations, plus ``pooled``, the
    obs-count weighted mean over whichever groups observed the cell. The weight is masked
    to cells a group actually reported, so a group's zero count never drags the mean and
    a cell nobody saw comes back NaN rather than 0/0.
    """
    weight = counts.where((counts > 0) & values.notnull())
    total = weight.sum("group")

    fields = {group: values.sel(group=group, drop=True) for group in values["group"].values}
    fields[pooled] = (weight*values).sum("group")/total.where(total > 0)
    return fields


def score(field, values, observed, available, where, aggregate_t0=AGGREGATE_T0):
    """One source's sums turned into both views, ``(per_group, all_groups)``.

    ``available`` is where this source has anything to persist; it composes with the
    conditioning mask exactly as ``observed`` does, so the bins still partition.
    """
    acc = metrics.accumulate(
        field, values, observed, available if where is None else where & available
    )
    return metrics.per_group(acc, aggregate_t0), metrics.all_groups(acc, aggregate_t0)


def main(
    observations=OBSERVATIONS,
    run_dir=RUN_DIR,
    dataset=ANEMOI_DATASET,
    output_dir=OUTPUT_DIR,
    aggregate_t0=AGGREGATE_T0,
    splits=SPLITS,
):
    output_dir = output_dir if output_dir is not None else f"{run_dir}/../persistence-metrics"
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for field, raw in observations.items():
        spec = data.normalise_spec(raw)
        print(f"Persisting {spec['observed_as']}_* at t0, scoring '{field}' against them")

        # The truth side is metrics' own, so persistence lands on the initial conditions
        # that survived verification, the lead times the model predicts, and the groups
        # that were live -- the same samples the model was scored on. Its prediction is
        # the one thing not wanted.
        _, values, observed, _ = metrics.align(field, spec, run_dir, dataset)
        groups = list(observed["group"].values)

        ic_values, ic_counts = initial_conditions(spec, groups, run_dir)
        fields = persistence_fields(ic_values.sel(t0=observed["t0"]),
                                    ic_counts.sel(t0=observed["t0"]))

        conditions = {"": None}
        if "by_overlap" in splits:
            conditions["by_overlap"] = metrics.overlap_masks(observed)

        print(f"  {observed.sizes['t0']} initial conditions, "
              f"fhr = {list(observed['fhr'].values)}, "
              f"sources: {', '.join(fields)}")

        results[field] = {}
        collected = {}
        for source, persisted in fields.items():
            available = persisted.notnull().compute()
            print(f"  {source}: {int(available.sum())} observed cell(s) to persist")

            for split, where in conditions.items():
                for view, xds in zip(
                    ("per_group", "all_groups"),
                    score(persisted, values, observed, available, where, aggregate_t0),
                ):
                    key = f"{split}.{view}" if split else view
                    results[field][f"{source}.{key}"] = xds
                    collected.setdefault(key, {})[source] = xds
                    fname = f"{output_dir}/{field}.{source}.{key}.nc"
                    metrics._write(xds, fname)
                    print(f"    Stored {fname}")

        # The same scores stacked on a `source` dim, which is the form worth plotting:
        # one open, every source against every other.
        for key, by_source in collected.items():
            xds = xr.concat(list(by_source.values()),
                            dim=pd.Index(list(by_source), name="source"))
            results[field][f"by_source.{key}"] = xds
            fname = f"{output_dir}/{field}.by_source.{key}.nc"
            metrics._write(xds, fname)
            print(f"  Stored {fname}")

    return results


if __name__ == "__main__":
    main()
