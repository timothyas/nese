"""Daily maxima and minima: predictions against observations, over the validation period.

Left panel max, right panel min. The wide gray violin is the observations: one sample per
(day, mission), the extreme over that mission's observed cells. The split violin is the
baseline and experiment predictions, at the same valid times. Under "All cells" each sample
is the extreme over every cell, and a dense field reaches further than one mission's tracks
for that reason alone. Under "Observed cells" each sample is the extreme over one mission's
observed cells that day, the same sampling as the observations.

uvgxt is shown as speed. Under "Observed cells" both the observations and the predictions are
the cross-track speed; under "All cells" the prediction is the full speed, at least the
cross-track one. The min panel for speed sits near zero by construction.

The extremes are read from each run's `nese-verify metrics` output, <run>/metrics/<field>.nc,
so every run drawn (the baseline included) must have been through it with aggregate_t0: false.
The observations are the experiment's.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import xarray as xr

# --- things to update -------------------------------------------------------

FIELDS = ["sla", "uvgxt", "swh"]
FHR = 24

# (label, inference dir) per field
BASELINES = {
    "sla":   ("update-forcings-aa-v0", "../update-forcings-aa/runs/v0/inference-validation"),
    "uvgxt": ("update-forcings-aa-v0", "../update-forcings-aa/runs/v0/inference-validation"),
    "swh":   ("update-forcings-aa-v0", "../update-forcings-aa/runs/v0/inference-validation"),
}

EXPERIMENT_DIR = "runs/v0/inference-validation"
EXPERIMENT_LABEL = "update-forcings-v0"
OBSERVATIONS_LABEL = "observations"
CELLS = ["All cells", "Observed cells"]

STYLE = "../experiments.mplstyle"
FIG_DIR = f"{EXPERIMENT_DIR}/figures"
# ----------------------------------------------------------------------------

UNITS = {"sla": "m", "uvgxt": "m/s", "swh": "m"}
ACRONYMS = {"rmse", "mae", "sla", "uvgxt", "swh", "rmsd", "mse", "std"}
here = Path(__file__).parent


def prettify(name):
    """Underscores to spaces, words capitalized, acronyms upper, values left alone."""
    keep = lambda word: any(c.isdigit() or c == "=" for c in word)
    words = str(name).replace("_", " ").split()
    return " ".join(
        word if keep(word) else word.upper() if word.lower() in ACRONYMS else word.capitalize()
        for word in words
    )


def place_legend(ax, reference_label):
    """The versions' legend, outside the axes, with the gray reference violin added."""
    legend = ax.get_legend()
    handles = [*legend.legend_handles, Patch(facecolor="lightgray", edgecolor="0.4")]
    labels = [*(text.get_text() for text in legend.get_texts()), prettify(reference_label)]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Version", frameon=False)


def version_colors(field):
    """Baseline then experiment, in the style's color cycle, whether or not the baseline is drawn."""
    return dict(zip([prettify(BASELINES[field][0]), prettify(EXPERIMENT_LABEL)], sns.color_palette()))


def read_extremes(run_dir, field):
    """One sample per (t0, group): pooled is left out, since each sample is one mission's."""
    path = here / run_dir / "metrics" / f"{field}.nc"
    return xr.open_dataset(path).sel(fhr=FHR).drop_sel(group="pooled").load()


def pick(ds, pattern):
    """The variables ``pattern`` names for max and min, renamed to just that."""
    names = {pattern.format(extreme): extreme for extreme in ("max", "min")}
    return ds[list(names)].rename(names)


def tidy(ds, cells, version):
    df = ds.to_dataframe()[["max", "min"]].reset_index().dropna(subset=["max", "min"], how="all")
    return df.assign(cells=cells, version=prettify(version))


def samples(field):
    """One row per sample: the observations at both cell choices, and each run's prediction."""
    observations = pick(read_extremes(EXPERIMENT_DIR, field), "observation_{}")
    frames = [tidy(observations, cells, OBSERVATIONS_LABEL) for cells in CELLS]
    runs = {**dict([BASELINES[field]]), EXPERIMENT_LABEL: EXPERIMENT_DIR}
    for label, run_dir in runs.items():
        if run_dir is None:
            continue
        extremes = read_extremes(run_dir, field)
        frames.append(tidy(pick(extremes, "prediction_{}_all_cells"), CELLS[0], label))
        frames.append(tidy(pick(extremes, "prediction_{}"), CELLS[1], label))
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
        palette={version: version_colors(field)[version] for version in versions},
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
    place_legend(axs[1], OBSERVATIONS_LABEL)

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
