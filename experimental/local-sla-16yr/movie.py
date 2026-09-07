"""Animate a run's forecasts beside the observations that fed them.

Reads the `nese-zarr` store directly. The store is `(t0, fhr, cell)` with `valid_time`,
`lead_time` and the grid as coordinates, and it carries the model's own `multistep_input`
and `timestep`, so nothing here restates the input window or the stepping as a constant,
and there is no inference config to re-resolve to find the files.

The one thing still read from the anemoi dataset is the earlier input bins: the store holds
the initial time and the forecast, not the two bins before t0 that the model was also shown.
"""

import os
import sys

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Patch
import cmocean  # noqa: F401  -- registers the "cmo.*" colormaps
import xmovie

sys.path.append(f"../..")
from regions import set_region, projection, draw_cells
import data

plt.style.use(f"../../nice.mplstyle")

RUN_DIR = f"runs/v4"
REGION = "global"
REGIONS_TO_PLOT = ("tropical_pacific",)
DATE_FORMAT = "%Y-%m-%dT%H"

FIELD = "sla"
OBSERVED_AS = "sla_filtered"
LABEL = "Sea Level Anomaly [m]"

# Which forecast hour to animate. None takes the longest lead time in the store, which for
# a single-step model is the only one carrying a diagnostic at all.
FHR = None

# Oldest to newest, so recency reads as increasing depth of colour. ColorBrewer Blues
# 4/6/9, for the three input bins this model has; a model trained with a different
# multistep_input samples the same ramp instead.
INPUT_COLORS = ["#9ecae1", "#4292c6", "#08306b"]


def input_cmap(multistep):
    """Colours for the input-recency mask.

    Index 0 is "unobserved" and is transparent; land is NaN and painted grey by set_bad.
    """
    colors = INPUT_COLORS if multistep == len(INPUT_COLORS) else [
        to_hex(c) for c in plt.get_cmap("Blues")(np.linspace(.35, 1., multistep))
    ]
    cmap = ListedColormap([(1, 1, 1, 0)] + colors)
    cmap.set_bad("0.9")
    return cmap


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


def get_save_dir(run_dir=RUN_DIR):
    movie_dir = f"{run_dir}/movies"
    os.makedirs(movie_dir, exist_ok=True)
    return movie_dir


def input_observation_mask(t0s, attrs, dataset=data.ANEMOI_DATASET):
    """Which input bin, if any, observed each cell -- the mask from plot_forecast.ipynb.

    Returns 0 where no input bin saw the cell, then 1..multistep oldest to newest, on the
    forecast's own cells. Where several bins saw a cell the most recent wins, so the
    colours read as "how stale is what the model knows here".

    The bins are the model's inputs for a forecast started at each of ``t0s``: the initial
    time and the ``multistep_input - 1`` steps before it. Both the number of bins and their
    spacing come from ``attrs``, the forecast store's, so this follows a model that was
    trained differently rather than assuming this one.

    Any group counts as having seen a cell -- the question is whether the model had
    information there, not whether the same instrument came back around.
    """
    multistep = attrs["multistep_input"]
    timestep = pd.Timedelta(attrs["timestep"]).to_timedelta64()

    ads = data.open_anemoi_dataset(dataset)
    groups = data.resolve_groups(data.normalise_spec(OBSERVED_AS), attrs)
    counts = xr.concat([ads[f"obs_count_{group}"] for group in groups], dim="group")

    steps = np.arange(multistep)
    when = xr.DataArray(
        t0s.values[:, None] - steps[None, :]*timestep,
        dims=("t0", "step"),
        coords={"t0": t0s, "step": steps},
    )
    # Select the distinct timestamps first, so the vectorised indexing that follows runs on
    # a small subset rather than on the whole store.
    seen = counts.sel(time=np.unique(when.values)) > 0
    seen = seen.any("group").sel(time=when).drop_vars("time").compute()

    # ``step`` counts backwards from t0, so the smallest step that saw a cell is its most
    # recent bin. argmax on a boolean gives that first True; 0 is left for unobserved.
    return xr.where(seen.any("step"), multistep - seen.argmax("step"), 0)


def open_forecasts(run_dir=RUN_DIR, field=FIELD, fhr=FHR, dataset=data.ANEMOI_DATASET):
    """One frame per forecast, as the single dataset xmovie wants.

    Frames run along ``t0``, which carries ``valid_time`` with it, so a frame knows both
    its initial and its valid time without either being recomputed from a lead time.

    Both fields come back on the full mesh, land included and NaN, ready for ``draw_cells``.
    """
    xds = data.open_forecast(run_dir)
    fhr = int(xds["fhr"].values[-1]) if fhr is None else fhr

    on_mesh = lambda xda: data.scatter_to_mesh(xda, xds.attrs)
    return xr.Dataset(
        {
            field: on_mesh(xds[field].sel(fhr=fhr)),
            "inputs": on_mesh(input_observation_mask(xds["t0"], xds.attrs, dataset)),
        },
        attrs=dict(xds.attrs),
    )


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


def plot_single_timestamp(xds, fig, frame, *args, **kwargs):

    region = kwargs.pop("region", REGION)
    extend = kwargs.pop("extend", "neither")
    field = kwargs.pop("field", FIELD)
    cmap = plt.get_cmap(kwargs.pop("cmap", "cmo.balance")).copy()
    # NaN is land: the cells data.yaml pruned off the mesh
    cmap.set_bad("0.9")

    multistep = xds.attrs["multistep_input"]

    axs = [fig.add_subplot(1, 2, i+1, projection=projection(region)) for i in range(2)]
    for ax in axs:
        set_region(ax, region)

    initial = pd.Timestamp(xds["t0"].isel(t0=frame).values)
    valid = pd.Timestamp(xds["valid_time"].isel(t0=frame).values)

    # Left: where the model's input observations are, coloured by how recent they are.
    # vmin/vmax put each integer in the middle of its own colour band.
    draw_cells(axs[0], xds["inputs"].isel(t0=frame), cmap=input_cmap(multistep),
               vmin=-.5, vmax=multistep+.5)
    labels = [
        f"$t_0-{step if step > 1 else ''}\\Delta t$" if step else "$t_0$"
        for step in reversed(range(multistep))
    ]
    # A figure legend, dropped into the space the colorbar leaves free on the left. An
    # axes legend hung below the panel is invisible to constrained layout, which would
    # leave the two maps at different heights, and inside the axes it covers the map.
    fig.legend(
        handles=[Patch(facecolor=c, label=l)
                 for c, l in zip(input_cmap(multistep).colors[1:], labels)],
        loc="lower center",
        bbox_to_anchor=(.27, .01),
        ncol=multistep,
        frameon=False,
    )
    axs[0].set(title=f"Input observations\n$t_0$ = {initial.strftime(DATE_FORMAT)}")

    # Right: what the model made of them.
    cc = draw_cells(axs[1], xds[field].isel(t0=frame), cmap=cmap, **kwargs)
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
        label=LABEL,
        extend=extend,
    )
    # xmovie saves each frame at a fixed pixel size, with no bbox_inches="tight" to
    # rescue the titles, so the layout has to fit itself.
    fig.set_constrained_layout(True)
    return None, None


if __name__ == "__main__":

    xds = open_forecasts()
    print(f"Opened {xds.sizes['t0']} forecasts")

    # Once, over the whole record: computed per frame, the scale would flicker. Shared
    # across regions too, so the movies are directly comparable.
    extend, vmin, vmax = get_vmin_vmax_extend(xds[FIELD], vmin=None, vmax=None)
    print(f"Color limits: {vmin:.3f} to {vmax:.3f}, extend={extend}")

    save_dir = get_save_dir()

    for region in REGIONS_TO_PLOT:
        width, height = frame_size(region)
        print(f"{region}: {width} x {height}")

        mov = xmovie.Movie(
            xds,
            plot_single_timestamp,
            framedim="t0",
            input_check=False,
            pixelwidth=width,
            pixelheight=height,
            dpi=100,
            region=region,
            extend=extend,
            vmin=vmin,
            vmax=vmax,
        )
        fname = f"{save_dir}/{FIELD}.{region}.mp4"
        mov.save(
            fname,
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
