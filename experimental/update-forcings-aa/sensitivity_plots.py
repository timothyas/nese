"""Loading and drawing for sensitivity_global.ipynb (``nese`` env), from ``../unseen-ft-all-altimeter``.

Developer notes:

* The stores index nodes by position. Data cells follow the dataset's grid, so their HEALPix ids
  are the dataset's ``cell`` coordinate; hidden node i is nested pixel i (``HEALPixNodes`` with
  ``nest_ordering: true``). :func:`load` attaches both and checks them against the stored
  coordinates, so a wrong ordering fails instead of plotting.
"""

from functools import cache
from pathlib import Path

import cartopy.crs as ccrs
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from nese.verify.data import open_anemoi_dataset
from nese.verify.maps import LAND
from nese.verify.maps import draw_cells

SENSITIVITY_ROOT = Path("sensitivity")
# The store the model was trained on, for the HEALPix ids of its cells
DATASET = "../all-altimeter/anemoi-hpx8-24h/dataset.zarr"
# graph.nodes.hidden in training.yaml: HEALPixNodes, resolution 6, nested
HIDDEN_MESH = {"healpix_nside": 2**6, "healpix_ordering": "nested"}
# Below this fraction of the maximum, sensitivities can be numerical; see
# ../unseen-ft-all-altimeter/README.md, "Sensitivity Noise Floor"
NOISE_FLOOR = 1e-6
# Lower end of the log10(sensitivity / max) colour scale
LOG_FLOOR = -4


@cache
def data_mesh() -> dict:
    attrs = open_anemoi_dataset(DATASET).attrs
    return {key: attrs[key] for key in ("healpix_nside", "healpix_ordering")}


def load(run: str, checkpoints: list[str]) -> xr.Dataset:
    """Every checkpoint's store under ``sensitivity/<run>/``, on dim ``checkpoint``, with HEALPix ids.

    Checkpoints with fewer input hours (v2p2 has no t0+24h step) are NaN at the others.
    """
    sens = xr.concat(
        [xr.open_zarr(SENSITIVITY_ROOT / run / f"{label}.zarr") for label in checkpoints],
        dim=pd.Index(checkpoints, name="checkpoint"),
        combine_attrs="drop_conflicts",
    ).load()
    sens = sens.assign_coords(cell=open_anemoi_dataset(DATASET)["cell"].values, hidden=np.arange(sens.sizes["hidden"]))
    check_ids(sens["cell"], data_mesh(), sens["latitude"], sens["longitude"])
    check_ids(sens["hidden"], HIDDEN_MESH, sens["hidden_latitude"], sens["hidden_longitude"])
    sens["input_total"] = sens["input_sensitivity"].sum(["kind", "input_hour"])
    # Observations are the same for every checkpoint; only the input hours they cover differ
    for name in ("sla_seen", "sla_target_observed"):
        sens[name] = sens[name].fillna(False).astype(bool).any("checkpoint")
    return sens


def check_ids(ids: xr.DataArray, mesh: dict, lat: xr.DataArray, lon: xr.DataArray) -> None:
    """The ids' cell centres must be the stored node coordinates."""
    nest = mesh["healpix_ordering"] == "nested"
    id_lon, id_lat = hp.pix2ang(mesh["healpix_nside"], ids.values, nest=nest, lonlat=True)
    offset = np.hypot(id_lat - lat.values, (id_lon - lon.values + 180) % 360 - 180)
    assert offset.max() < 1e-3, f"HEALPix ids do not match the stored coordinates (max offset {offset.max():.3g} deg)"


def normalised_log(values: xr.DataArray, dim) -> xr.DataArray:  # noqa: ANN001
    return np.log10(values / values.max(dim)).clip(min=LOG_FLOOR)


def map_axes(nrows: int, ncols: int, extent: list[float] | None = None) -> tuple:
    """Map axes, global (Robinson) or cropped to ``[lon0, lon1, lat0, lat1]`` (PlateCarree)."""
    projection = ccrs.Robinson(central_longitude=-60) if extent is None else ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, (2.8 if extent is None else 3.2) * nrows), squeeze=False,
        subplot_kw={"projection": projection}, constrained_layout=True,
    )
    for ax in axes.flat:
        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor(LAND)
        ax.coastlines(resolution="50m", lw=0.4)
    return fig, axes


def on_mesh(values: xr.DataArray) -> tuple[xr.DataArray, dict]:
    """``values`` with its node dim named ``cell``, and that mesh's HEALPix attributes."""
    if "hidden" in values.dims:
        return values.rename(hidden="cell"), HIDDEN_MESH
    return values, data_mesh()


def draw(ax, values: xr.DataArray, **kwargs):  # noqa: ANN001, ANN201
    """``values`` on its own mesh, data cells or hidden nodes, as true HEALPix quadrilaterals."""
    cells, mesh = on_mesh(values)
    kwargs = {"cmap": "viridis", "vmin": LOG_FLOOR, "vmax": 0, "rasterized": True, **kwargs}
    return draw_cells(ax, cells, mesh, **kwargs)


def draw_solid(ax, cells: xr.DataArray, color: str, **kwargs) -> None:  # noqa: ANN001
    """``cells`` filled with one ``color``."""
    if cells.size:
        cells, mesh = on_mesh(cells)
        draw_cells(ax, xr.ones_like(cells, dtype=float), mesh, cmap=ListedColormap([color]), rasterized=True, **kwargs)


def draw_seen(ax, seen: xr.DataArray) -> None:  # noqa: ANN001
    """Cells where sla was observed, as small grey dots."""
    seen = seen.where(seen, drop=True)
    ax.scatter(seen["longitude"], seen["latitude"], s=0.2, c="0.6", transform=ccrs.PlateCarree(), rasterized=True)


def normalised_share(sensitivity: xr.DataArray, group: str, size: xr.DataArray | None = None) -> xr.DataArray:
    """Each group's share of the total, per input hour, optionally divided by ``size`` and renormalised."""
    share = sensitivity.sum("cell") / sensitivity.sum(["cell", group, "input_hour"])
    if size is None:
        return share
    share = share / size
    return share / share.sum([group, "input_hour"])


def plot_shares(axes, sens: xr.Dataset) -> None:  # noqa: ANN001
    """Three stacked-bar panels of one scalar's input sensitivity, by kind and input hour: the raw
    share, the share per input variable, and the share per encoder-input channel."""
    panels = {
        "share of input sensitivity": normalised_share(sens["input_sensitivity"], "kind"),
        "share per input variable": normalised_share(sens["input_sensitivity"], "kind", sens["n_variables"]),
        "share per encoder-input channel": normalised_share(
            sens["encoder_input_sensitivity"], "encoder_group", sens["n_channels"]
        ),
    }
    for ax, (title, share) in zip(axes, panels.items(), strict=True):
        group = "kind" if "kind" in share.dims else "encoder_group"
        rows = share.stack(row=("checkpoint", "input_hour")).transpose("row", group).to_pandas()
        rows.dropna(how="all").plot.bar(ax=ax, stacked=True)
        ax.set(ylabel="share of total", title=title)
        ax.legend(fontsize=7, ncol=2)
