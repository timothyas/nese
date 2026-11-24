"""A/B comparison of verification metrics between the update-forcings forecast and this state
estimate, plus the state estimate's error by fhr.

Run immediately after nese-verify. Builds on update-forcings-aa/plot_metrics.py: each x category
shows split (half) violins for the two versions, drawn on top of the baseline's persistence for
reference. Each version is read at its own FHR. The seen_at_hour slots are matched by their hour
relative to each run's t0; a slot only one version has (e.g. the forecast's -48h, the estimate's
+24h) gets only that version's half. Each run's "seen at none" bin, one timestep past its latest
input, is relabeled "Unseen" for both.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import xarray as xr

# --- things to update -------------------------------------------------------

FIELDS = ["sla", "uvgxt", "swh"]
FHR = {"baseline": 0, "experiment": 0}

BASELINE_DIR = "runs/v1.25/inference-validation"
BASELINE_LABEL = "State Estimation (multistep-v1.25)"

EXPERIMENT_DIR = "runs/v1.95/inference-validation-0.25"
EXPERIMENT_LABEL = "State Estimation (multistep-v1.95-i0.25)"

# (inference dir, violin color) per persistence resolution; {} for none. A guiding reference at
# PERSISTENCE_FHR: persistence is scored at forecast hours only, not at the runs' fhr 0
PERSISTENCE = {
    "25km": ("../all-altimeter/runs/v2p0/inference-validation", "lightgray"),
}
PERSISTENCE_FHR = 24
PERSISTENCE_LABEL = "persistence"

METRICS = ["rmse"]
STYLE = "../experiments.mplstyle"
UNSEEN = "Unseen"
FIG_DIR = f"{EXPERIMENT_DIR}/figures"
# ----------------------------------------------------------------------------

UNITS = {"sla": "m", "uvgxt": "m/s", "swh": "m"}
ACRONYMS = {"rmse", "mae", "sla", "uvgxt", "swh", "rmsd", "mse", "std"}
here = Path(__file__).parent
RUNS = {"baseline": (BASELINE_LABEL, BASELINE_DIR), "experiment": (EXPERIMENT_LABEL, EXPERIMENT_DIR)}


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


def hour_label(hour):
    return f"{hour:+d}h"


def open_metrics(run_dir, field, suffix=""):
    metrics = xr.open_dataset(here / run_dir / "metrics" / f"{field}{suffix}.nc")[METRICS]
    return metrics.reset_coords(drop=True).load()


def read_metrics(run, field, suffix=""):
    """One run's metrics at its own FHR, on a ``version`` dim."""
    label, run_dir = RUNS[run]
    metrics = open_metrics(run_dir, field, suffix).sel(fhr=FHR[run], drop=True)
    if suffix == ".seen_at_hour":
        hours = metrics["seen_at_hour"].values
        unseen = hours.max()  # nese-verify's "seen at none" bin is one step past the latest input
        metrics = metrics.assign_coords(
            seen_at_hour=[UNSEEN if hour == unseen else hour_label(hour) for hour in hours]
        )
    return metrics.expand_dims(version=[label])


def by_version(field, suffix=""):
    return xr.concat([read_metrics(run, field, suffix) for run in RUNS], dim="version", join="outer", coords="minimal")


def in_time_order(seen_at):
    """Input hours ascending, then Unseen."""
    hours = sorted((label for label in seen_at["seen_at_hour"].values if label != UNSEEN),
                   key=lambda label: int(label.rstrip("h")))
    present = [UNSEEN] if UNSEEN in seen_at["seen_at_hour"] else []
    return seen_at.sel(seen_at_hour=[*hours, *present])


def read_persistence(field):
    def read(run_dir):
        path = here / run_dir / "persistence-metrics" / f"{field}.nc"
        persistence = xr.open_dataset(path).sel(fhr=PERSISTENCE_FHR, source="pooled", drop=True)
        return persistence[METRICS].reset_coords(drop=True).load()

    return xr.concat(
        [read(run_dir).expand_dims({"resolution": [resolution]})
         for resolution, (run_dir, _) in PERSISTENCE.items()],
        dim="resolution",
    )


def tidy(ds, x_dim):
    """Long form, NaN rows dropped: a slot or t0 one version lacks is simply not drawn."""
    df = ds.to_dataframe().reset_index().dropna(subset=list(ds.data_vars))
    df[x_dim] = df[x_dim].map(prettify)
    if "version" in df:
        df["version"] = df["version"].map(prettify)
    return df


def version_labels():
    return [f"{prettify(label)} (fhr {FHR[run]})" for run, (label, _) in RUNS.items()]


def place_legend(ax):
    """The versions' legend, outside the axes, with the reference violins added."""
    legend = ax.get_legend()
    handles = [*legend.legend_handles,
               *(Patch(facecolor=color, edgecolor="0.4") for _, color in PERSISTENCE.values())]
    labels = [*version_labels(),
              *(f"{prettify(PERSISTENCE_LABEL)} {resolution} (fhr {PERSISTENCE_FHR})" if len(PERSISTENCE) > 1
                else f"{prettify(PERSISTENCE_LABEL)} (fhr {PERSISTENCE_FHR})" for resolution in PERSISTENCE)]
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Version", frameon=False)


def version_colors():
    return dict(zip([prettify(label) for label, _ in RUNS.values()], sns.color_palette()))


def draw_panel(ax, model, persistence, metric, x_dim, field):
    order = [prettify(x) for x in model[x_dim].values]
    if persistence is not None:
        sns.violinplot(
            tidy(persistence[[metric]], x_dim),
            x=x_dim,
            y=metric,
            order=order,
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
        order=order,
        hue="version",
        hue_order=versions,
        palette=version_colors(),
        split=True,
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
    (baseline_label, _), (experiment_label, _) = RUNS.values()
    baseline = scores.sel(version=baseline_label, drop=True)
    experiment = scores.sel(version=experiment_label, drop=True)
    percent = 100 * (experiment - baseline) / abs(baseline)

    header = (f"{prettify(field)}: {prettify(experiment_label)} (fhr {FHR['experiment']}) relative to "
              f"{prettify(baseline_label)} (fhr {FHR['baseline']}), pooled over groups")
    print(f"\n{header}\n{'-' * len(header)}")
    print("(time mean of |metric|; negative is an improvement; nan where only one version has the slot)\n")
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


def save(fig, name):
    path = here / FIG_DIR / f"{name}.jpeg"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_figure(model, persistence, x_dim, field, name, rotate_xticks):
    fig, axs = plt.subplots(len(METRICS), 1, figsize=(12, 5 * len(METRICS)), sharex=True, squeeze=False)
    axs = axs[:, 0]
    for metric, ax in zip(METRICS, axs):
        draw_panel(ax, model, persistence, metric, x_dim, field)
        if ax is not axs[0]:
            ax.get_legend().remove()
    place_legend(axs[0])
    if rotate_xticks:
        axs[-1].set_xticks(axs[-1].get_xticks())
        axs[-1].set_xticklabels(
            axs[-1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
    save(fig, name)


def make_fhr_figure(field):
    """The state estimate's pooled error at each fhr, i.e. each time in its window."""
    label, run_dir = RUNS["experiment"]
    metrics = open_metrics(run_dir, field).sel(group="pooled", drop=True)
    metrics = metrics.assign_coords(fhr=[hour_label(fhr) for fhr in metrics["fhr"].values])
    fig, axs = plt.subplots(len(METRICS), 1, figsize=(6, 5 * len(METRICS)), sharex=True, squeeze=False)
    for metric, ax in zip(METRICS, axs[:, 0]):
        sns.violinplot(
            tidy(metrics[[metric]], "fhr"),
            x="fhr",
            y=metric,
            color=version_colors()[prettify(label)],
            inner="quart",
            ax=ax,
        )
        ax.set(xlabel="", ylabel=f"{prettify(field)} {metric_label(metric)} ({UNITS[field]})")
        if metric == "bias":
            ax.axhline(y=0, zorder=0, alpha=0.5, color="gray")
    axs[0, 0].set_title(prettify(label))
    axs[-1, 0].set_xlabel("Hour from $t_0$")
    save(fig, f"metrics.{field}.by_fhr")


def plot_field(field):
    by_group = by_version(field)
    seen_at = in_time_order(by_version(field, ".seen_at_hour").sel(group="pooled", drop=True))

    persistence = pooled_persistence = None
    if PERSISTENCE:
        persistence = read_persistence(field)
        pooled_persistence = persistence.sel(group="pooled", drop=True).expand_dims(
            {"seen_at_hour": seen_at["seen_at_hour"].values}
        )

    print_differences(seen_at, field)
    make_figure(by_group, persistence, "group", field, f"metrics.{field}.by_group", rotate_xticks=True)
    make_figure(seen_at, pooled_persistence, "seen_at_hour", field, f"metrics.{field}.seen_at", rotate_xticks=False)
    make_fhr_figure(field)


def main():
    plt.style.use(here / STYLE)
    (here / FIG_DIR).mkdir(exist_ok=True)
    for field in FIELDS:
        plot_field(field)


if __name__ == "__main__":
    main()
