"""Plot training/validation loss curves for a run's offline mlflow log.

Always writes loss_curves.pdf for the total loss. When the run logged per-field losses
(nese_training.diagnostics.FieldLossLogger, for a loss scoring more than one field), also
writes loss_curves.<field>.pdf for each field.
"""

import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from mlflow.tracking import MlflowClient

plt.style.use("../../nice.mplstyle")

# FieldLossLogger's names, with the _step/_epoch suffix Lightning appends.
FIELD_LOSS = re.compile(r"^(?P<stage>train|val)_field_loss/(?P<dataset>[^/]+)/(?P<field>.+)_(?P<when>step|epoch)$")


def open_run(tracking_dir, experiment_name=None, run_id=None):
    """Return (client, run) for one offline mlflow run, defaulting to the most recent."""
    client = MlflowClient(tracking_uri=f"file://{os.path.abspath(tracking_dir)}")
    experiments = client.search_experiments()
    if experiment_name is not None:
        experiments = [e for e in experiments if e.name == experiment_name]
    runs = client.search_runs([e.experiment_id for e in experiments], order_by=["attributes.start_time DESC"])
    if run_id is not None:
        runs = [r for r in runs if r.info.run_id == run_id]
    return client, runs[0]


def read_metric(client, run, key):
    """One metric as a pandas Series indexed by training step."""
    history = client.get_metric_history(run.info.run_id, key)
    df = pd.DataFrame(
        {
            "step": [m.step for m in history],
            "value": [m.value for m in history],
        }
    ).drop_duplicates(subset="step", keep="last")
    return df.set_index("step")["value"].rename(key).sort_index()


def read_metrics(client, run, keys=None):
    """Several metrics joined on step, NaN where a metric was not logged."""
    keys = sorted(run.data.metrics) if keys is None else keys
    return pd.concat([read_metric(client, run, k) for k in keys], axis=1)


def read_val_step_loss(client, run, key, step_key):
    """Validation step-level loss remapped onto the trainer's global step.

    mlflow's "step" for on_step validation metrics is just a per-call
    counter, not trainer.global_step, so it can't be joined against the
    training curves directly. Interpolate it from `step_key`'s (timestamp,
    step) pairs instead, which are logged at the real global step.
    """
    history = client.get_metric_history(run.info.run_id, key)
    values = pd.Series(
        [m.value for m in history], index=[m.timestamp for m in history], name=key
    ).sort_index()

    ref = client.get_metric_history(run.info.run_id, step_key)
    ref_ts = np.array([m.timestamp for m in ref])
    ref_step = np.array([m.step for m in ref])
    order = np.argsort(ref_ts)

    global_step = np.interp(values.index, ref_ts[order], ref_step[order])
    return pd.Series(values.values, index=global_step, name=key).sort_index()


def field_losses(metric_names):
    """``{(dataset, field): {"train_step": key, ...}}`` for every per-field loss logged."""
    found = {}
    for name in metric_names:
        match = FIELD_LOSS.match(name)
        if match:
            keys = found.setdefault((match["dataset"], match["field"]), {})
            keys[f"{match['stage']}_{match['when']}"] = name
    return found


def plot_curves(client, run, metrics, keys, path, title=None):
    """Iteration and epoch curves for one loss, training and validation, on a log axis."""
    val_step = read_val_step_loss(client, run, keys["val_step"], keys["train_step"])

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics[keys["train_step"]].dropna().plot(ax=ax, lw=0.8, alpha=0.8, label="Training (iteration)")
    val_step.dropna().plot(ax=ax, lw=0.8, alpha=0.8, label="Validation (iteration)")
    metrics[keys["train_epoch"]].dropna().plot(ax=ax, label="Training (epoch)")
    metrics[keys["val_epoch"]].dropna().plot(ax=ax, label="Validation (epoch)")

    ax.set(xlabel="Training Iteration", ylabel="Loss", yscale="log", title=title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    logname = "v2"
    experiment = "v2"
    log_dir = f"runs/{logname}"
    run_dir = f"runs/{experiment}"
    client, run = open_run(f"{log_dir}/logs/mlflow", experiment_name=experiment)

    print(run.info.run_id, run.info.status)
    print(sorted(run.data.metrics))

    metrics = read_metrics(client, run)
    total = {
        "train_step": "train_multi_dataset_loss_step",
        "val_step": "val_multi_dataset_loss_step",
        "train_epoch": "train_multi_dataset_loss_epoch",
        "val_epoch": "val_multi_dataset_loss_epoch",
    }
    plot_curves(client, run, metrics, total, f"{run_dir}/loss_curves.pdf")

    fields = field_losses(run.data.metrics)
    datasets = {dataset for dataset, _ in fields}
    for (dataset, field), keys in fields.items():
        name = field if len(datasets) == 1 else f"{dataset}.{field}"
        plot_curves(client, run, metrics, keys, f"{run_dir}/loss_curves.{name}.pdf", title=field)
