"""A/B comparison of verification metrics between a baseline and this experiment.

Run immediately after nese-verify. Each x category shows split (half) violins for the
two versions, drawn on top of a wide gray persistence violin for reference. With
BASELINE_DIR = None, the experiment gets one full violin, in the color it has beside a baseline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

# --- things to update -------------------------------------------------------

VARIABLE = "uvgxt"

BASELINE_DIRS = {
    "sla": "../field-encoder/runs/v2/inference-validation",
    "uvgxt": None,
}
BASELINE_DIR = BASELINE_DIRS[VARIABLE]
BASELINE_LABEL = "field-encoder-v2"

EXPERIMENT_DIR = "runs/v0/inference-validation"
EXPERIMENT_LABEL = "diagnose-uvgxt"
PERSISTENCE_DIR = EXPERIMENT_DIR
PERSISTENCE_LABEL = "persistence"

METRICS = ["rmse", "mae", "bias"]
STYLE = "../experiments.mplstyle"
SEEN_AT_LABELS = {24: "Unseen"}  # hours outside the input window get a name, not "+24h"
FIG_DIR = f"{EXPERIMENT_DIR}/figures"
# ----------------------------------------------------------------------------

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


def metric_label(metric):
    return metric.upper() if metric.lower() in ACRONYMS else metric.capitalize()


def read_metrics(run_dir, suffix=""):
    path = here / run_dir / "metrics" / f"{VARIABLE}{suffix}.nc"
    return xr.open_dataset(path).squeeze(drop=True)[METRICS].load()


def read_persistence():
    path = here / PERSISTENCE_DIR / "persistence-metrics" / f"{VARIABLE}.nc"
    ds = xr.open_dataset(path).squeeze(drop=True)[METRICS].load()
    return ds.sel(source="pooled", drop=True)


def tidy(ds, x_dim):
    df = ds.to_dataframe().reset_index()
    df[x_dim] = df[x_dim].map(prettify)
    if "version" in df:
        df["version"] = df["version"].map(prettify)
    return df


def version_colors():
    """Baseline then experiment, in the style's color cycle, whether or not the baseline is drawn."""
    return dict(zip([prettify(BASELINE_LABEL), prettify(EXPERIMENT_LABEL)], sns.color_palette()))


def draw_panel(ax, model, persistence, metric, x_dim):
    sns.violinplot(
        tidy(persistence[[metric]], x_dim),
        x=x_dim,
        y=metric,
        color="lightgray",
        linecolor="0.4",
        inner=None,
        width=0.95,
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
        palette={version: version_colors()[version] for version in versions},
        split=len(versions) > 1,
        inner="quart",
        legend=True,
        ax=ax,
    )
    ax.set(xlabel="", ylabel=metric_label(metric))
    if metric == "bias":
        ax.axhline(y=0, zorder=0, alpha=0.5, color="gray")


def print_differences(seen_at):
    """Percent change of each metric, experiment relative to baseline, per seen_at case."""
    scores = abs(seen_at).mean("t0")
    baseline = scores.sel(version=BASELINE_LABEL, drop=True)
    experiment = scores.sel(version=EXPERIMENT_LABEL, drop=True)
    percent = 100 * (experiment - baseline) / abs(baseline)

    header = f"{prettify(EXPERIMENT_LABEL)} relative to {prettify(BASELINE_LABEL)}, pooled over groups"
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


def make_figure(model, persistence, x_dim, name, rotate_xticks):
    fig, axs = plt.subplots(len(METRICS), 1, figsize=(12, 5 * len(METRICS)), sharex=True)
    for metric, ax in zip(METRICS, axs):
        draw_panel(ax, model, persistence, metric, x_dim)
        if ax is not axs[0]:
            ax.get_legend().remove()
    sns.move_legend(axs[0], "center left", bbox_to_anchor=(1, 0.5), title="Version", frameon=False)
    if rotate_xticks:
        axs[-1].set_xticks(axs[-1].get_xticks())
        axs[-1].set_xticklabels(
            axs[-1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

    path = here / FIG_DIR / f"{name}.jpeg"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    plt.style.use(here / STYLE)
    (here / FIG_DIR).mkdir(exist_ok=True)

    runs = {BASELINE_LABEL: BASELINE_DIR, EXPERIMENT_LABEL: EXPERIMENT_DIR}
    runs = {label: run_dir for label, run_dir in runs.items() if run_dir is not None}

    def by_version(suffix=""):
        return xr.concat(
            [read_metrics(run_dir, suffix).expand_dims({"version": [label]})
             for label, run_dir in runs.items()],
            dim="version",
        )

    by_group = by_version()
    seen_at = by_version(".seen_at_hour").sel(group="pooled", drop=True)
    seen_at = seen_at.assign_coords(
        seen_at_hour=[SEEN_AT_LABELS.get(hour, f"{hour:+d}h")
                      for hour in seen_at["seen_at_hour"].values]
    )

    persistence = read_persistence()
    pooled_persistence = persistence.sel(group="pooled", drop=True).expand_dims(
        {"seen_at_hour": seen_at["seen_at_hour"].values}
    )

    if BASELINE_DIR is not None:
        print_differences(seen_at)

    make_figure(by_group, persistence, "group", f"metrics.{VARIABLE}.by_group", rotate_xticks=True)
    make_figure(seen_at, pooled_persistence, "seen_at_hour", f"metrics.{VARIABLE}.seen_at", rotate_xticks=False)


if __name__ == "__main__":
    main()
