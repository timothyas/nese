"""Tabulate the throughput sweep: speed, dataloader wait and peak memory per run.

    python summarize.py
"""

import csv
import re
from pathlib import Path

LABEL = re.compile(r"bs(\d+)_w(\d+)_pf(\d+)(?:_blk(\d+))?(?:_\d+h)?")
DEFAULT_BLOCK_SIZE = 64  # model.processor.block_size in training.yaml


def read_speed(run: Path) -> dict[str, float]:
    """anemoi's speed_profiler.csv: one 'index,metric,value' row per metric."""
    path = next(run.glob("**/speed_profiler.csv"), None)
    if path is None:
        return {}
    return {row["metric"]: float(row["value"]) for row in csv.DictReader(path.open())}


def peak_gpu_gb(run: Path, stat: str) -> float:
    """Rank 0's peak from Lightning's DeviceStatsMonitor, e.g. stat='reserved_bytes.all.peak'."""
    values = [float(line.split()[1]) for path in run.glob(f"logs/mlflow/*/*/metrics/**/{stat}") for line in path.open()]
    return max(values, default=float("nan")) / 1024**3


def slurm_log(label: str) -> str:
    logs = sorted(Path("slurm").glob(f"{label}-*.out"))
    return logs[-1].read_text(errors="replace") if logs else ""


def peak_host_gb(log: str) -> float:
    """The training step's peak from the job epilogue's 'Memory statistics, in GiB' table."""
    match = re.search(r"^\S+\.apptainer\s+[\d.]+\s+([\d.]+)\s*$", log, re.MULTILINE)
    return float(match.group(1)) if match else float("nan")


def failure(log: str) -> str:
    if not log:
        return "no log"
    if "oom_kill" in log or "Out Of Memory" in log:
        return "host OOM"
    if "CUDA out of memory" in log or "OutOfMemoryError" in log:
        return "GPU OOM"
    if "DUE TO TIME LIMIT" in log:
        return "time limit"
    return "failed"


def summarize(run: Path) -> dict:
    batch_size, workers, prefetch, block_size = LABEL.fullmatch(run.name).groups(default=DEFAULT_BLOCK_SIZE)
    row = {"run": run.name, "bs": int(batch_size), "w": int(workers), "pf": int(prefetch), "blk": int(block_size)}
    speed = read_speed(run)
    log = slurm_log(run.name)
    row["host_gb"] = peak_host_gb(log)
    if "training_avg_throughput" not in speed:
        return row | {"status": failure(log)}
    steps_per_s = speed["training_avg_throughput"]
    return row | {
        "status": "ok",
        "samples_per_s": steps_per_s * row["bs"],
        "step_s": 1 / steps_per_s,
        "wait_s": 1 / speed["avg_training_dataloader_throughput"],
        "gpu_alloc_gb": peak_gpu_gb(run, "allocated_bytes.all.peak"),
        "gpu_reserved_gb": peak_gpu_gb(run, "reserved_bytes.all.peak"),
    }


def main() -> None:
    rows = [summarize(run) for run in sorted(Path("runs").iterdir()) if LABEL.fullmatch(run.name)]
    header = f"{'run':22} {'blk':>4} {'samples/s/GPU':>13} {'step s':>7} {'wait s':>7}"
    print(header + f" {'GPU alloc GB':>12} {'GPU rsvd GB':>11} {'host GB':>8}")
    for r in sorted(rows, key=lambda r: (r["bs"], r["w"], r["pf"], r["blk"])):
        if r["status"] != "ok":
            print(f"{r['run']:22} {r['blk']:4d} {r['status']:>13} {'':>7} {'':>7} {'':>12} {'':>11} {r['host_gb']:8.1f}")
            continue
        print(
            f"{r['run']:22} {r['blk']:4d} {r['samples_per_s']:13.2f} {r['step_s']:7.2f} {r['wait_s']:7.3f}"
            f" {r['gpu_alloc_gb']:12.1f} {r['gpu_reserved_gb']:11.1f} {r['host_gb']:8.1f}"
        )


if __name__ == "__main__":
    main()
