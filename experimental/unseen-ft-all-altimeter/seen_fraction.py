"""How much of the target is unseen in the input window, over random training samples.

A target observation is unseen when no group of its field observed that cell in any input bin
(nese-verify's ``seen_at_hour == +timestep``). The base loss puts about ``f_unseen`` of each
field's weight on those points, and base + lambda * unseen about
``(f_unseen + lambda) / (1 + lambda)``, since each term is a pooled mean.

    python seen_fraction.py training.yaml verify.validation.yaml
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from nese.verify.config import ObservationSpec
from nese.verify.data import open_anemoi_dataset
from nese.verify.metrics import input_times
from nese.verify.metrics import seen_at_hour_masks
from nese.verify.observations import resolve_groups
from nese.verify.observations import stack_observations


def sample_t0(times: xr.DataArray, n_input: int, n_samples: int, seed: int) -> xr.DataArray:
    """``n_samples`` initial times, each with its full input window and target inside ``times``."""
    candidates = times.isel(time=slice(n_input - 1, -1)).rename(time="sample")
    chosen = np.random.default_rng(seed).choice(candidates.sizes["sample"], n_samples, replace=False)
    return candidates.isel(sample=np.sort(chosen)).drop_vars(["sample", "time"], errors="ignore")


def target_counts(dataset: xr.Dataset, spec: ObservationSpec, t0: xr.DataArray, attrs: dict) -> xr.Dataset:
    """Per sample, the target (group, cell) observations and how many sit at unseen cells."""
    values, counts = stack_observations(dataset, spec, resolve_groups(spec, dataset.attrs))
    valid = values.notnull().all("component") if spec.is_vector else values.notnull()
    timestep = pd.Timedelta(attrs["timestep"])

    target = ((counts > 0) & valid).sel(time=t0 + timestep.to_timedelta64()).drop_vars("time")
    window = input_times(t0, attrs).stack(window=("sample", "step"))
    seen = (counts.sel(time=window).drop_vars("time") > 0).any("group").unstack("window")
    unseen = seen_at_hour_masks(seen, attrs["timestep"]).sel(
        seen_at_hour=int(timestep / pd.Timedelta(hours=1)), drop=True
    )
    return xr.Dataset(
        {"n_target": target.sum(["group", "cell"]), "n_unseen": (target & unseen).sum(["group", "cell"])}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("training", help="training.yaml, for the task and training period")
    parser.add_argument("verify", help="verify yaml, for the dataset and observation specs")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    args = parser.parse_args()

    with open(args.training) as f:
        training = yaml.safe_load(f)
    with open(args.verify) as f:
        verification = yaml.safe_load(f)["verification"]

    attrs = {"multistep_input": training["task"]["multistep_input"], "timestep": training["data"]["frequency"]}
    period = training["dataloader"]["training"]["datasets"]["ocean"]
    dataset = open_anemoi_dataset(verification["dataset"])
    times = dataset["time"].sel(time=slice(str(period["start"]), str(period["end"])))
    t0 = sample_t0(times, attrs["multistep_input"], args.samples, args.seed)

    counts = xr.concat(
        [
            target_counts(dataset, ObservationSpec(**spec), t0, attrs).expand_dims(field=[field])
            for field, spec in verification["observations"].items()
        ],
        dim="field",
    ).compute()

    print(
        f"{args.samples} samples, {attrs['multistep_input']} x {attrs['timestep']} inputs -> +{attrs['timestep']}, "
        f"{period['start']} to {period['end']}, seed {args.seed}\n"
    )
    pooled = counts.sum("sample")
    fraction = pooled["n_unseen"] / pooled["n_target"]
    per_sample = counts["n_unseen"] / counts["n_target"]
    print(f"{'field':8} {'unseen':>12} {'target':>12} {'raw fraction':>13} {'per-sample mean +- std [min, max]':>36}")
    for field in counts["field"].values:
        f, s = fraction.sel(field=field), per_sample.sel(field=field)
        print(
            f"{field:8} {int(pooled['n_unseen'].sel(field=field)):>12d} {int(pooled['n_target'].sel(field=field)):>12d} "
            f"{float(f):>13.3f}       {float(s.mean()):.3f} +- {float(s.std()):.3f} "
            f"[{float(s.min()):.3f}, {float(s.max()):.3f}]"
        )

    weights = xr.DataArray([0.0, *args.weights], dims="weight")
    share = (fraction + weights) / (1 + weights)
    print("\nshare of each field's loss on unseen points, base + weight * unseen")
    print(f"{'field':8} " + " ".join(f"{f'w={w:g}':>8}" for w in weights.values))
    for field in counts["field"].values:
        print(f"{field:8} " + " ".join(f"{float(v):>8.3f}" for v in share.sel(field=field).values))


if __name__ == "__main__":
    main()
