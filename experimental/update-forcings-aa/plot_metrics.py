"""A/B comparison of verification metrics between a baseline and this experiment.

Run immediately after nese-verify. Each x category shows split (half) violins for the
two versions, drawn on top of a wide persistence violin for reference, split in half when
PERSISTENCE has two resolutions. A field whose BASELINES dir is None gets one full violin,
in the color it has beside a baseline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import xarray as xr

# --- things to update -------------------------------------------------------

FIELDS = ["sla", "uvgxt", "swh"]
FHR = 24

# (label, inference dir) per field
# Baseline = all-altimeter v2p2, the same model without the forcing changes
BASELINES = {
    "sla":   ("all-altimeter-v2p2", "../all-altimeter/runs/v2p2/inference-validation"),
    "uvgxt": ("all-altimeter-v2p2", "../all-altimeter/runs/v2p2/inference-validation"),
    "swh":   ("all-altimeter-v2p2", "../all-altimeter/runs/v2p2/inference-validation"),
}

EXPERIMENT_DIR = "runs/v0/inference-validation"
EXPERIMENT_LABEL = "update-forcings-v0"

# (inference dir, violin color) per persistence resolution
PERSISTENCE = {
    #"100km": ("runs/v1/inference-validation", "#cfe8dc"),
    "25km": ("../all-altimeter/runs/v2p0/inference-validation", "lightgray"),
}
PERSISTENCE_LABEL = "persistence"

METRICS = ["rmse"]#, "bias"]#, "mae", "bias"]
STYLE = "../experiments.mplstyle"
SEEN_AT_LABELS = {24: "Unseen"}  # hours outside the input window get a name, not "+24h"
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


def metric_label(metric):
    return metric.upper() if metric.lower() in ACRONYMS else metric.capitalize()


def read_metrics(run_dir, field, suffix=""):
    path = here / run_dir / "metrics" / f"{field}{suffix}.nc"
    return xr.open_dataset(path).sel(fhr=FHR, drop=True)[METRICS].load()


def read_persistence(field):
    def read(run_dir):
        path = here / run_dir / "persistence-metrics" / f"{field}.nc"
        return xr.open_dataset(path).sel(fhr=FHR, source="pooled", drop=True)[METRICS].load()

    return xr.concat(
        [read(run_dir).expand_dims({"resolution": [resolution]})
         for resolution, (run_dir, _) in PERSISTENCE.items()],
        dim="resolution",
    )


def tidy(ds, x_dim):
    df = ds.to_dataframe().reset_index()
    df[x_dim] = df[x_dim].map(prettify)
    if "version" in df:
        df["version"] = df["version"].map(prettify)
    return df


def place_legend(ax, reference_label):
    """The versions' legend, outside the axes, with the reference violins added."""
    legend = ax.get_legend()
    handles = [*legend.legend_handles,
               *(Patch(facecolor=color, edgecolor="0.4") for _, color in PERSISTENCE.values())]
    labels = [*(text.get_text() for text in legend.get_texts()),
              *(f"{prettify(reference_label)} {resolution}" if len(PERSISTENCE) > 1
                else prettify(reference_label) for resolution in PERSISTENCE)]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Version", frameon=False)


def version_colors(field):
    """Baseline then experiment, in the style's color cycle, whether or not the baseline is drawn."""
    return dict(zip([prettify(BASELINES[field][0]), prettify(EXPERIMENT_LABEL)], sns.color_palette()))


def draw_panel(ax, model, persistence, metric, x_dim, field):
    sns.violinplot(
        tidy(persistence[[metric]], x_dim),
        x=x_dim,
        y=metric,
        hue="resolution",
        hue_order=list(PERSISTENCE),
        palette={resolution: color for resolution, (_, color) in PERSISTENCE.items()},
        split=len(PERSISTENCE) > 1,
        linecolor="0.4",
        inner=None,
        width=0.95,
        legend=False,
        zorder=0,
        ax=ax,
    )
    versions = [prettify(v) for v in model["version"].values]
    sns.violinplot(
        tidy(model[[metric]], x_dim),
        x=x_dim,
        y=metric,
        hue="version",
        hue_order=versions,
        palette={version: version_colors(field)[version] for version in versions},
        split=len(versions) > 1,
        inner="quart",
        legend=True,
        ax=ax,
    )
    ax.set(xlabel="", ylabel=f"{prettify(field)} {metric_label(metric)} ({UNITS[field]})")
    if metric == "bias":
        ax.axhline(y=0, zorder=0, alpha=0.5, color="gray")


def print_differences(seen_at, field):
    """Percent change of each metric, experiment relative to baseline, per seen_at case."""
    scores = abs(seen_at).mean("t0")
    baseline_label = BASELINES[field][0]
    baseline = scores.sel(version=baseline_label, drop=True)
    experiment = scores.sel(version=EXPERIMENT_LABEL, drop=True)
    percent = 100 * (experiment - baseline) / abs(baseline)

    header = f"{prettify(field)}: {prettify(EXPERIMENT_LABEL)} relative to {prettify(baseline_label)}, pooled over groups"
    print(f"\n{header}\n{'-' * len(header)}")
    print("(time mean of |metric|; negative is an improvement)\n")
    width = max(len(str(label)) for label in seen_at["seen_at_hour"].values)
    print(" " * width + "".join(f"{metric_label(m):>26}" for m in METRICS))
    for label in seen_at["seen_at_hour"].values:
        cells = "".join(
            f"{baseline[m].sel(seen_at_hour=label).item():11.5f} ->"
            f"{experiment[m].sel(seen_at_hour=label).item():9.5f}"
            f" ({percent[m].sel(seen_at_hour=label).item():+6.1f}%)"
            for m in METRICS
        )
        print(f"{label:<{width}}{cells}")
    print()


def make_figure(model, persistence, x_dim, field, name, rotate_xticks):
    fig, axs = plt.subplots(len(METRICS), 1, figsize=(12, 5 * len(METRICS)), sharex=True, squeeze=False)
    axs = axs[:, 0]
    for metric, ax in zip(METRICS, axs):
        draw_panel(ax, model, persistence, metric, x_dim, field)
        if ax is not axs[0]:
            ax.get_legend().remove()
    place_legend(axs[0], PERSISTENCE_LABEL)
    if rotate_xticks:
        axs[-1].set_xticks(axs[-1].get_xticks())
        axs[-1].set_xticklabels(
            axs[-1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

    path = here / FIG_DIR / f"{name}.jpeg"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_field(field):
    runs = {**dict([BASELINES[field]]), EXPERIMENT_LABEL: EXPERIMENT_DIR}
    runs = {label: run_dir for label, run_dir in runs.items() if run_dir is not None}

    def by_version(suffix=""):
        return xr.concat(
            [read_metrics(run_dir, field, suffix).expand_dims({"version": [label]})
             for label, run_dir in runs.items()],
            dim="version",
        )

    by_group = by_version()
    seen_at = by_version(".seen_at_hour").sel(group="pooled", drop=True)
    seen_at = seen_at.assign_coords(
        seen_at_hour=[SEEN_AT_LABELS.get(hour, f"{hour:+d}h")
                      for hour in seen_at["seen_at_hour"].values]
    )

    persistence = read_persistence(field)
    pooled_persistence = persistence.sel(group="pooled", drop=True).expand_dims(
        {"seen_at_hour": seen_at["seen_at_hour"].values}
    )

    if BASELINES[field][1] is not None:
        print_differences(seen_at, field)

    make_figure(by_group, persistence, "group", field, f"metrics.{field}.by_group", rotate_xticks=True)
    make_figure(seen_at, pooled_persistence, "seen_at_hour", field, f"metrics.{field}.seen_at", rotate_xticks=False)


def main():
    plt.style.use(here / STYLE)
    (here / FIG_DIR).mkdir(exist_ok=True)
    for field in FIELDS:
        plot_field(field)


if __name__ == "__main__":
    main()
