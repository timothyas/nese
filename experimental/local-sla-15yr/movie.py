import sys

import numpy as np
import xarray as xr
import pandas as pd
import zarr
import dask.array as dsa
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import cmocean  # noqa: F401  -- registers the "cmo.*" colormaps
import xmovie

sys.path.append(f"../..")
from regions import set_region, projection, draw_cells

plt.style.use(f"../../nice.mplstyle")

REGION = "global"
REGIONS_TO_PLOT = ("north_atlantic", "tropical_pacific")
DATE_FORMAT = "%Y-%m-%dT%H"

# The observations the model is given. The forecast netcdfs only carry the initial time,
# so the earlier input bins have to come from the source dataset.
INPUT_DATASET = "anemoi-hpx6-24h/dataset.zarr"
MULTISTEP = 3               # training.yaml: task.multistep_input
TIMESTEP = pd.Timedelta("24h")   # training.yaml: data.frequency

# Oldest to newest, so recency reads as increasing depth of colour. Index 0 is
# "unobserved" and is transparent; land is NaN and painted grey by set_bad.
INPUT_COLORS = ["#9ecae1", "#4292c6", "#08306b"]
INPUT_CMAP = ListedColormap([(1, 1, 1, 0)] + INPUT_COLORS)
INPUT_CMAP.set_bad("0.9")


def get_vmin_vmax_extend(xda, vmin, vmax):

    minval = xda.min().values
    maxval = xda.max().values

    if vmin is None and vmax is None:
        vmax = .7*np.abs(xda).max().values
        vmin = -vmax

    extend = "neither"
    if minval < vmin:
        extend = "min"
    if maxval > vmax:
        extend = "max" if extend == "neither" else "both"
    return extend, vmin, vmax


def forecast_paths(config_file="forecast.yaml"):
    """The netcdf written for each initial condition in the forecast config.

    Rebuilds each path the way nese-inference does: set ``date`` and let OmegaConf
    resolve the ``${date}`` in ``output.netcdf``.
    """
    config = OmegaConf.load(config_file)
    dates = pd.date_range(**OmegaConf.to_container(config.dates, resolve=True))

    paths = []
    for date in dates:
        config.date = date.strftime(config.get("date_format", DATE_FORMAT))
        paths.append(OmegaConf.to_container(config, resolve=True)["output"]["netcdf"])
    return paths


def input_observation_mask(valid_times, sea, dataset=INPUT_DATASET):
    """Which input bin, if any, observed each cell -- the mask from plot_forecast.ipynb.

    Returns 0 where no input bin saw the cell, then 1..MULTISTEP oldest to newest, and
    NaN over land. Where several bins saw a cell the most recent wins, so the colours
    read as "how stale is what the model knows here".

    The bins are the model's inputs for a forecast valid at ``valid_times``, i.e. the
    initial time and the MULTISTEP-1 steps before it.
    """
    group = zarr.open(dataset, mode="r")
    variables = list(group.attrs["variables"])
    missions = [name for name in variables if name.startswith("sla_filtered_")]

    observed = xr.DataArray(
        dsa.from_zarr(group["data"])[:, [variables.index(m) for m in missions], 0, :],
        dims=("time", "variable", "cell"),
        coords={"time": group["dates"][:], "variable": missions},
    ).notnull().any("variable")

    initial = pd.DatetimeIndex(valid_times) - TIMESTEP
    earliest = initial.min() - (MULTISTEP - 1)*TIMESTEP
    observed = observed.sel(time=slice(earliest, initial.max())).compute()

    mask = np.zeros((len(initial), observed.sizes["cell"]))
    # Oldest first, so a cell seen more than once ends up labelled by its newest bin.
    for step in reversed(range(MULTISTEP)):
        seen = observed.sel(time=initial - step*TIMESTEP).values
        mask[seen] = MULTISTEP - step

    mask[:, ~sea] = np.nan
    return xr.DataArray(mask, dims=("time", "cell"), coords={"time": valid_times})


def open_forecasts(config_file="forecast.yaml"):
    """One frame per forecast, as the single dataset xmovie wants.

    Each file splits its variables over two timestamps, and only the diagnostic ``sla``
    is finite at the forecast time, so each file contributes one frame stamped with its
    valid time. The observations that fed it are recovered separately, on their own
    input times, by :func:`input_observation_mask`.
    """
    xds = xr.open_mfdataset(
        forecast_paths(config_file),
        combine="nested",
        concat_dim="time",
        preprocess=lambda xds: xds[["sla"]].isel(time=1),
    ).load()

    # The sea mask the model was trained with: land is NaN in every output field.
    sea = np.isfinite(xds["sla"].isel(time=0).values)
    mask = input_observation_mask(xds["time"].values, sea)
    # onto the netcdf's own spatial dim, so the two fields share it rather than sitting
    # on parallel dims of equal length
    xds["inputs"] = mask.rename({"cell": xds["sla"].dims[-1]})
    return xds


def frame_size(region, panel=600, below=220):
    """Frame pixels for two side-by-side panels of ``region``, plus room underneath.

    The regions differ a lot in shape -- the tropical Pacific is nearly three times as
    wide as it is tall, the North Atlantic close to square -- so a fixed frame would
    letterbox one and crop the titles off the other. ``below`` is the strip the legend,
    colorbar and two-line titles need.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection(region))
    set_region(ax, region)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)

    width, height = 2*panel, int(panel*(y1 - y0)/(x1 - x0)) + below
    return width + width % 2, height + height % 2   # h264 wants even dimensions


def plot_single_timestamp(xds, fig, time, *args, **kwargs):

    region = kwargs.pop("region", REGION)
    extend = kwargs.pop("extend", "neither")
    cmap = plt.get_cmap(kwargs.pop("cmap", "cmo.balance")).copy()
    # NaN is land, masked out of the model's output by the sea mask
    cmap.set_bad("0.9")

    axs = [fig.add_subplot(1, 2, i+1, projection=projection(region)) for i in range(2)]
    for ax in axs:
        set_region(ax, region)

    valid = pd.Timestamp(xds["time"].isel(time=time).values)
    initial = valid - TIMESTEP

    # Left: where the model's input observations are, coloured by how recent they are.
    # vmin/vmax put each integer in the middle of its own colour band.
    draw_cells(axs[0], xds["inputs"].isel(time=time), cmap=INPUT_CMAP, vmin=-.5, vmax=MULTISTEP+.5)
    labels = [
        f"$t_0-{step if step > 1 else ''}\\Delta t$" if step else "$t_0$"
        for step in reversed(range(MULTISTEP))
    ]
    # A figure legend, dropped into the space the colorbar leaves free on the left. An
    # axes legend hung below the panel is invisible to constrained layout, which would
    # leave the two maps at different heights, and inside the axes it covers the map.
    fig.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(INPUT_COLORS, labels)],
        loc="lower center",
        bbox_to_anchor=(.27, .01),
        ncol=MULTISTEP,
        frameon=False,
    )
    axs[0].set(title=f"Input observations\n$t_0$ = {initial.strftime(DATE_FORMAT)}")

    # Right: what the model made of them.
    cc = draw_cells(axs[1], xds["sla"].isel(time=time), cmap=cmap, **kwargs)
    axs[1].set(title=f"Predicted SLA\nvalid {valid.strftime(DATE_FORMAT)}")
    # ax=axs so both columns give up the same height and the maps stay aligned; anchored
    # right so the bar still sits under the panel it describes.
    fig.colorbar(
        cc,
        ax=axs,
        orientation="horizontal",
        shrink=.4,
        anchor=(1., .5),
        pad=0.05,
        aspect=35,
        label="Sea Level Anomaly [m]",
        extend=extend,
    )
    # xmovie saves each frame at a fixed pixel size, with no bbox_inches="tight" to
    # rescue the titles, so the layout has to fit itself.
    fig.set_constrained_layout(True)
    return None, None


if __name__ == "__main__":

    xds = open_forecasts()
    print(f"Opened {xds.sizes['time']} forecasts")

    # Once, over the whole record: computed per frame, the scale would flicker. Shared
    # across regions too, so the movies are directly comparable.
    extend, vmin, vmax = get_vmin_vmax_extend(xds["sla"], vmin=None, vmax=None)
    print(f"Color limits: {vmin:.3f} to {vmax:.3f}, extend={extend}")

    for region in REGIONS_TO_PLOT:
        width, height = frame_size(region)
        print(f"{region}: {width} x {height}")

        mov = xmovie.Movie(
            xds,
            plot_single_timestamp,
            framedim="time",
            input_check=False,
            pixelwidth=width,
            pixelheight=height,
            dpi=100,
            region=region,
            extend=extend,
            vmin=vmin,
            vmax=vmax,
        )
        mov.save(
            f"sla.{region}.mp4",
            progress=True,
            overwrite_existing=True,
            remove_frames=True,
            framerate=10,
            # One string, substituted into the command whole, so it has to carry the
            # codec and pixel format too -- it replaces xmovie's defaults rather than
            # adding to them. Those default to -crf 10, archival grade: the recency
            # colours repaint most of the left panel every frame, leaving the encoder
            # little to reuse, so that setting gets expensive.
            ffmpeg_options="-c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p",
        )
