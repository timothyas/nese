"""Daily maxima and minima: predictions against observations, over the validation period.

Left panel max, right panel min. The wide gray violin is the observations: one sample per
(day, mission), the extreme over that mission's observed cells. The split violin is the
baseline and experiment predictions, at the same valid times. Under "All cells" each sample
is the extreme over every cell, and a dense field reaches further than one mission's tracks
for that reason alone. Under "Observed cells" each sample is the extreme over one mission's
observed cells that day, the same sampling as the observations.

uvgxt is shown as speed. The observed one is the cross-track speed, at most the full speed,
and the min panel for speed sits near zero by construction. Observations are read at
fhr=0, the only lead time forecast.zarr stores them at.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# --- things to update -------------------------------------------------------

FIELDS = ["sla", "uvgxt"]

# Compare to field encoder
BASELINE_DIRS = {
    "sla": "../field-encoder/runs/v2/inference-validation",
    "uvgxt": None,
}
BASELINE_LABEL = "field-encoder"

## Compare within diagnose-uvgxt versions
#BASELINE_DIRS = {
#    "sla": "runs/v0/inference-validation",
#    "uvgxt": "runs/v0/inference-validation",
#}
#BASELINE_LABEL = "diagnose-uvgxt-v0"

EXPERIMENT_DIR = "runs/v2/inference-validation"
EXPERIMENT_LABEL = "diagnose-uvgxt-v2"
OBSERVATIONS_LABEL = "observations"
CELLS = ["All cells", "Observed cells"]

STYLE = "../experiments.mplstyle"
FIG_DIR = f"{EXPERIMENT_DIR}/figures"
# ----------------------------------------------------------------------------

OBSERVED_AS = {"sla": ["sla_filtered"], "uvgxt": ["ugxt", "vgxt"]}
UNITS = {"sla": "m", "uvgxt": "m/s"}
ACRONYMS = {"rmse", "mae", "sla", "rmsd", "mse", "std"}
here = Path(__file__).parent


def prettify(name):
    """Underscores to spaces, words capitalized, acronyms upper, values left alone."""
    keep = lambda word: any(c.isdigit() or c == "=" for c in word)
    words = str(name).replace("_", " ").split()
    return " ".join(
        word if keep(word) else word.upper() if word.lower() in ACRONYMS else word.capitalize()
        for word in words
    )


def version_colors():
    """Baseline then experiment, in the style's color cycle, whether or not the baseline is drawn."""
    return dict(zip([prettify(BASELINE_LABEL), prettify(EXPERIMENT_LABEL)], sns.color_palette()))


def magnitude(components):
    return components[0] if len(components) == 1 else np.hypot(*components)


def predicted(run_dir, field):
    """The prediction indexed by the time it is valid at, so it lines up with the observations."""
    forecast = xr.open_zarr(here / run_dir / "forecast.zarr").sel(fhr=24)
    names = [field] if field == "sla" else ["ugxt", "vgxt"]
    values = magnitude([forecast[name] for name in names])
    return values.swap_dims(t0="valid_time").drop_vars("t0").rename(valid_time="time")


def observed(field):
    forecast = xr.open_zarr(here / EXPERIMENT_DIR / "forecast.zarr").sel(fhr=0)
    prefix = f"{OBSERVED_AS[field][0]}_"
    groups = [name.removeprefix(prefix) for name in forecast.data_vars if name.startswith(prefix)]
    group = pd.Index(groups, name="group")
    values = magnitude(
        [xr.concat([forecast[f"{name}_{g}"] for g in groups], dim=group) for name in OBSERVED_AS[field]]
    )
    return values.rename(t0="time")


def extremes(values):
    return xr.Dataset({"max": values.max("cell"), "min": values.min("cell")})


def tidy(ds, cells, version):
    df = ds.to_dataframe()[["max", "min"]].reset_index().dropna(subset=["max", "min"], how="all")
    return df.assign(cells=cells, version=prettify(version))


def samples(field):
    """One row per sample: the observations at both cell choices, and each run's prediction."""
    observations = observed(field)
    frames = [tidy(extremes(observations), cells, OBSERVATIONS_LABEL) for cells in CELLS]
    runs = {BASELINE_LABEL: BASELINE_DIRS[field], EXPERIMENT_LABEL: EXPERIMENT_DIR}
    for label, run_dir in runs.items():
        if run_dir is None:
            continue
        prediction, at_observations = xr.align(predicted(run_dir, field), observations, join="inner")
        frames.append(tidy(extremes(prediction), CELLS[0], label))
        frames.append(tidy(extremes(prediction.where(at_observations.notnull())), CELLS[1], label))
    return pd.concat(frames, ignore_index=True)


def draw_panel(ax, df, extreme, field):
    is_observation = df["version"] == prettify(OBSERVATIONS_LABEL)
    sns.violinplot(
        df[is_observation],
        x="cells",
        y=extreme,
        order=CELLS,
        color="lightgray",
        linecolor="0.4",
        inner=None,
        width=0.95,
        zorder=0,
        ax=ax,
    )
    versions = list(dict.fromkeys(df.loc[~is_observation, "version"]))
    sns.violinplot(
        df[~is_observation],
        x="cells",
        y=extreme,
        order=CELLS,
        hue="version",
        hue_order=versions,
        palette={version: version_colors()[version] for version in versions},
        split=len(versions) > 1,
        inner="quart",
        legend=True,
        ax=ax,
    )
    ax.set(xlabel="", ylabel=f"{prettify(field)} {extreme} ({UNITS[field]})")


def print_percentiles(df, field):
    print(f"\n{prettify(field)}: 5th / 50th / 95th percentiles ({UNITS[field]})")
    for extreme in ("max", "min"):
        table = df.groupby(["cells", "version"], sort=False)[extreme].quantile([0.05, 0.5, 0.95]).unstack()
        print(f"{extreme}:\n{table.to_string(float_format='{:8.3f}'.format)}")


def make_figure(field):
    df = samples(field)
    print_percentiles(df, field)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for extreme, ax in zip(("max", "min"), axs):
        draw_panel(ax, df, extreme, field)
    axs[0].get_legend().remove()
    sns.move_legend(axs[1], "center left", bbox_to_anchor=(1, 0.5), title="Version", frameon=False)

    path = here / FIG_DIR / f"extremes.{field}.jpeg"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    plt.style.use(here / STYLE)
    (here / FIG_DIR).mkdir(exist_ok=True)
    for field in FIELDS:
        make_figure(field)


if __name__ == "__main__":
    main()
