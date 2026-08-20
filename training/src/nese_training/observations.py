"""The nese-dataprep naming convention, and pooled statistics built on top of it.

Observations of a field are ``{observed_as}_{group}`` and their per-bin counts are
``count`` with ``{group}`` substituted. The loss and the preprocessors both have to
recover the same mission list from those names, so the spec lives here rather than in
either of them, and one config block can feed both.
"""

from typing import Any

import numpy as np

DEFAULT_COUNT = "obs_count_{group}"


def normalise_spec(spec: str | dict) -> dict:
    """Expand a bare ``observed_as`` string, or a partial mapping, into a full spec."""
    if isinstance(spec, str):
        spec = {"observed_as": spec}
    spec = dict(spec)
    if "observed_as" not in spec:
        msg = f"Observation spec {spec!r} is missing 'observed_as'."
        raise ValueError(msg)
    spec.setdefault("groups", "auto")
    spec.setdefault("count", DEFAULT_COUNT)
    return spec


def resolve_groups(spec: dict, available: dict[str, int], owner: str = "") -> list[str]:
    """The observing groups of a field, ordered by where their columns sit."""
    prefix = f"{spec['observed_as']}_"
    if spec["groups"] == "auto":
        found = [(pos, name[len(prefix) :]) for name, pos in available.items() if name.startswith(prefix)]
        if not found:
            msg = f"No variables matching '{prefix}*' are available to {owner or 'the caller'}."
            raise ValueError(msg)
        return [group for _, group in sorted(found)]
    return list(spec["groups"])


def pool_statistics(
    pool: dict[str, Any],
    name_to_index: dict[str, int],
    statistics: dict[str, np.ndarray],
    owner: str = "",
) -> dict[str, np.ndarray]:
    """Give a field and every mission observing it one shared mean, stdev, min and max.

    Per-mission statistics are not the neutral choice they look like. A mission's mean is
    the sea level of the years it happened to fly -- across this archive the per-mission
    means track observation epoch at r=0.99 and 2.6 mm/yr, which is the trend itself --
    so normalising each mission by its own mean subtracts that trend mission by mission.
    It also leaves the single predicted field scored, in normalised space, against
    targets whose zero points sit up to 0.44 sigma apart, an offset no prediction can
    satisfy. One affine map for the field and all its observations fixes both: the
    inter-mission offsets cancel and the trend survives as signal.

    Groups are weighted by their obs-count mean, which is proportional to the number of
    observations each contributed. anemoi hands a processor only mean/stdev/minimum/
    maximum, never the counts behind them, and this reproduces true count-weighted
    pooling to under 0.001 sigma here.

    Returns a new dict. The caller's is left alone: anemoi passes the same statistics
    object to every processor, so mutating it would reach across to the others.
    """
    stats = {key: np.array(values, copy=True) for key, values in statistics.items()}

    for field, raw in pool.items():
        spec = normalise_spec(raw)
        groups = resolve_groups(spec, name_to_index, owner)
        observed = [f"{spec['observed_as']}_{group}" for group in groups]
        counts = [spec["count"].format(group=group) for group in groups]

        missing = [name for name in observed + counts if name not in name_to_index]
        if missing:
            msg = f"Cannot pool statistics for {field!r}: {missing} not in the dataset."
            raise ValueError(msg)

        obs_idx = [name_to_index[name] for name in observed]
        weights = np.array([stats["mean"][name_to_index[name]] for name in counts], dtype="float64")
        if weights.sum() <= 0:
            msg = f"Cannot pool statistics for {field!r}: every observing group has a zero obs count."
            raise ValueError(msg)
        weights = weights / weights.sum()

        mean, stdev = stats["mean"][obs_idx], stats["stdev"][obs_idx]
        pooled_mean = float((weights * mean).sum())
        # Pool the second moment about zero, then shift: E[x^2] = var + mean^2.
        pooled_var = float((weights * (stdev**2 + mean**2)).sum() - pooled_mean**2)

        # The field itself (the diagnostic the model emits) shares the map when present,
        # so its prediction and its targets land in the same space.
        members = obs_idx + ([name_to_index[field]] if field in name_to_index else [])
        stats["mean"][members] = pooled_mean
        stats["stdev"][members] = np.sqrt(max(pooled_var, 0.0))
        stats["minimum"][members] = stats["minimum"][obs_idx].min()
        stats["maximum"][members] = stats["maximum"][obs_idx].max()

    return stats
