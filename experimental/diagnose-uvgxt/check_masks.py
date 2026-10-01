"""Where the velocity masks of the training stack disagree with the store's own NaNs.

Every (time, cell, mission) with obs_count > 0 is one of
    A  finite velocity, defined    -- encoded and scored, correctly
    B  NaN velocity, defined       -- leak: encoder and loss read the imputer's fill
    C  NaN velocity, undefined     -- correctly excluded
    D  finite velocity, undefined  -- real observation thrown away
`defined` is taken from the v2 inference output (finite diagnosed ugxt), so it is the
model's own mask. Usage: python check_masks.py [n_days]  (all days by default)
"""

import sys
from pathlib import Path

import dask
import dask.array as da
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr

STORE = "anemoi-hpx6-24h/dataset.zarr"
FORECAST = "runs/v2/inference-validation/forecast.zarr"
OUT = Path("runs/mask-check")
PERIODS = {"training": slice("1993-11-01", "2018-12-31"), "validation": slice("2019-01-01", "2020-12-31")}
MIN_NEIGHBOURS = 3
BAND = 5.0
# HEALPix nside=64 rings sit near 4.2 and 4.8 degrees, either side of the band edge
LATITUDE_BINS = [0, 4.5, 5, 6, 10, 30, 60, 90]


def open_store(n_days: int | None) -> tuple[xr.DataArray, xr.Dataset]:
    root = zarr.open(STORE, mode="r")
    variables = list(root.attrs["variables"])
    cell = np.asarray(root["healpix_cell_index"])
    data = xr.DataArray(
        da.from_zarr(root["data"])[:, :, 0, :],
        dims=("time", "variable", "cell"),
        coords={"time": np.asarray(root["dates"]), "variable": variables, "cell": cell},
    )
    static = xr.Dataset(
        {
            "latitude": ("cell", np.asarray(root["latitudes"])),
            "longitude": ("cell", np.asarray(root["longitudes"])),
            "mdt": data.sel(variable="mdt").isel(time=0).compute().drop_vars(["variable", "time"]),
        },
        coords={"cell": cell},
    )
    return (data if n_days is None else data.isel(time=slice(0, n_days))), static


def sea_neighbours(cell: np.ndarray) -> np.ndarray:
    neighbours = hp.get_all_neighbours(64, cell, nest=True)
    return np.isin(neighbours, cell).sum(axis=0)


def model_defined(static: xr.Dataset) -> xr.DataArray:
    defined = xr.open_zarr(FORECAST)["ugxt"].sel(fhr=24).isel(t0=0).notnull()
    assert not defined.where(np.abs(static["latitude"]) < BAND, False).any(), "defined inside the band"
    return defined.drop_vars([name for name in defined.coords if name != "cell"])


def classify(data: xr.DataArray, groups: list[str], defined: xr.DataArray) -> xr.Dataset:
    def pick(prefix: str) -> xr.DataArray:
        names = [f"{prefix}_{group}" for group in groups]
        return data.sel(variable=names).assign_coords(variable=groups).rename(variable="group")

    count, u, v = pick("obs_count"), pick("ugxt"), pick("vgxt")
    observed, finite = count > 0, u.notnull()
    classes = {
        "A": observed & finite & defined,
        "B": observed & ~finite & defined,
        "C": observed & ~finite & ~defined,
        "D": observed & finite & ~defined,
        "B_single_obs": observed & ~finite & defined & (count == 1),
        "u_v_nan_mismatch": u.isnull() != v.isnull(),
        "finite_without_count": ~observed & finite,
    }
    return xr.Dataset(
        {
            name: xr.concat([mask.sel(time=period).sum("time") for period in PERIODS.values()], dim="period")
            for name, mask in classes.items()
        }
    ).assign_coords(period=list(PERIODS))


def summarise(totals: xr.Dataset, causes: dict, latitude: xr.DataArray) -> str:
    lines = []
    for period in totals["period"].values:
        per_group = totals.sel(period=period).sum("cell").to_dataframe().drop(columns="period")
        per_group.loc["pooled"] = per_group.sum()
        per_group["leak_B/(A+B)"] = per_group["B"] / (per_group["A"] + per_group["B"])
        per_group["lost_D/(A+D)"] = per_group["D"] / (per_group["A"] + per_group["D"])
        per_group["B_single_obs/B"] = per_group["B_single_obs"] / per_group["B"]
        lines += [f"== {period} ==", per_group.to_string(float_format="{:.4f}".format), ""]

        by_cause = {
            name: {
                cause: int(totals[name].sel(period=period).sum("group").where(mask, 0).sum())
                for cause, mask in causes.items()
            }
            for name in ("B", "D")
        }
        lines += [f"-- {period}: observations by cell cause (causes can overlap) --", str(by_cause), ""]

    pooled = totals[["A", "B", "C", "D"]].sum(["period", "group"])
    by_latitude = pooled.groupby_bins(np.abs(latitude), LATITUDE_BINS).sum().to_dataframe()
    lines += ["-- both periods: observations by cell-centre |lat| --", by_latitude.to_string(), ""]
    top10 = pooled["B"].sortby(pooled["B"], ascending=False)[:10].sum() / pooled["B"].sum()
    lines += [f"B: {int((pooled['B'] > 0).sum())} cells, top-10 cell share {float(top10):.3f}"]
    return "\n".join(lines)


def plot_map(values: xr.DataArray, static: xr.Dataset, title: str, path: Path) -> None:
    hit = values > 0
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.scatter(static["longitude"], static["latitude"], s=0.3, c="0.85")
    points = ax.scatter(
        static["longitude"].where(hit), static["latitude"].where(hit), s=2, c=values.where(hit), norm="log", cmap="viridis"
    )
    fig.colorbar(points, ax=ax, label="observations (summed over time and missions)")
    ax.set(title=title, xlabel="longitude", ylabel="latitude", ylim=(-90, 90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else None
    data, static = open_store(n_days)
    groups = sorted(name.removeprefix("ugxt_") for name in data["variable"].values if name.startswith("ugxt_"))
    defined = model_defined(static)

    with dask.config.set(scheduler="threads"):
        totals = classify(data, groups, defined).compute()

    latitude = np.abs(static["latitude"])
    causes = {
        "band (|lat|<5)": latitude < BAND,
        "band edge (5<=|lat|<6)": (latitude >= BAND) & (latitude < BAND + 1),
        "coast (<3 sea neighbours)": xr.DataArray(sea_neighbours(static["cell"].values) < MIN_NEIGHBOURS, dims="cell"),
        "MDT missing": static["mdt"].isnull(),
    }
    causes["other"] = ~(causes["band (|lat|<5)"] | causes["band edge (5<=|lat|<6)"] | causes["coast (<3 sea neighbours)"] | causes["MDT missing"])

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if n_days is None else f".first{n_days}days"
    summary = summarise(totals, causes, static["latitude"])
    (OUT / f"summary{suffix}.txt").write_text(summary)
    print(summary)
    totals.to_netcdf(OUT / f"totals{suffix}.nc")
    for name, title in (("B", "Leak (B): count > 0, velocity NaN, model-defined"), ("D", "Lost (D): velocity finite, model-undefined")):
        plot_map(totals[name].sum(["period", "group"]), static, title, OUT / f"{name}_map{suffix}.png")


if __name__ == "__main__":
    main()
