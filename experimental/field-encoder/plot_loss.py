"""Plot training/validation loss curves for a run's offline mlflow log."""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from mlflow.tracking import MlflowClient

plt.style.use("../../nice.mplstyle")


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


if __name__ == "__main__":
    logname = "v1"
    experiment = "v1"
    log_dir = f"runs/{logname}"
    run_dir = f"runs/{experiment}"
    client, run = open_run(f"{log_dir}/logs/mlflow", experiment_name=experiment)

    print(run.info.run_id, run.info.status)
    print(sorted(run.data.metrics))

    metrics = read_metrics(client, run)
    val_step_loss = read_val_step_loss(
        client, run, "val_multi_dataset_loss_step", "train_multi_dataset_loss_step"
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    metrics["train_multi_dataset_loss_step"].dropna().plot(ax=ax, lw=0.8, alpha=0.8, label="Training (iteration)")
    val_step_loss.dropna().plot(ax=ax, lw=0.8, alpha=0.8, label="Validation (iteration)")
    metrics["train_multi_dataset_loss_epoch"].dropna().plot(ax=ax, label="Training (epoch)")
    metrics["val_multi_dataset_loss_epoch"].dropna().plot(ax=ax, label="Validation (epoch)")

    ax.set(
        xlabel="Training Iteration",
        ylabel="Loss",
        yscale="log",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig(f"{run_dir}/loss_curves.pdf", bbox_inches="tight")
    print(f"Saved {run_dir}/loss_curves.pdf")
