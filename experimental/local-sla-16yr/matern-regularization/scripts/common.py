"""Shared entry points for the table scripts."""

import argparse
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "anemoi-hpx6-24h" / "dataset.zarr"


def parse_args(description, **extra):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    for name, kwargs in extra.items():
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    return parser.parse_args()


def banner(title, table=None):
    print(f"\n{title}")
    if table:
        print(f"  -> main.tex, Table \\ref{{{table}}}")
    print("-" * max(len(title), 60))
