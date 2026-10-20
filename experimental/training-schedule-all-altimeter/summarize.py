"""Rank the sweep's runs by final validation loss, the checkpoint stage 2 would start from.

    python summarize.py
"""

from pathlib import Path

FIELDS = ("sla", "uvgxt", "swh")


def read_metric(path: Path) -> list[float]:
    """An mlflow file-store metric: one 'timestamp value step' line per log, in step order."""
    rows = sorted((int(step), float(value)) for _, value, step in map(str.split, path.open()))
    return [value for _, value in rows]


def summarize(run: Path) -> dict | None:
    metrics = next(run.glob("logs/mlflow/*/*/metrics"), None)
    if metrics is None or not (metrics / "val_multi_dataset_loss_epoch").is_file():
        return None
    total = read_metric(metrics / "val_multi_dataset_loss_epoch")
    best = min(range(len(total)), key=total.__getitem__)
    fields = {f: read_metric(metrics / f"val_field_loss/ocean/{f}_epoch") for f in FIELDS}
    return {
        "run": run.name,
        "epochs": len(total),
        "final": total[-1],
        "best": total[best],
        "best_epoch": best + 1,
        **{f"{f}_final": values[-1] for f, values in fields.items()},
        **{f"{f}_best": min(values) for f, values in fields.items()},
    }


def main() -> None:
    rows = [r for run in sorted(Path("runs").iterdir()) if (r := summarize(run))]
    rows.sort(key=lambda r: r["final"])
    header = f"{'run':28} {'ep':>3} {'final':>7} {'best':>7} {'@ep':>4}"
    header += "".join(f" {f + ' fin/best':>16}" for f in FIELDS)
    print(header)
    for r in rows:
        line = f"{r['run']:28} {r['epochs']:3d} {r['final']:7.4f} {r['best']:7.4f} {r['best_epoch']:4d}"
        line += "".join(f"  {r[f + '_final']:7.4f}/{r[f + '_best']:.4f}" for f in FIELDS)
        print(line)


if __name__ == "__main__":
    main()
