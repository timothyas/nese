"""The nese-dataprep naming convention, and pooled statistics built on top of it.

Observations of a field are ``{observed_as}_{group}`` and their per-bin counts are
``count`` with ``{group}`` substituted. The loss and the preprocessors both have to
recover the same mission list from those names, so the spec lives here rather than in
either of them, and one config block can feed both.

A vector field names its ``components``, each a model output observed as
``{component}_{group}`` (``observed_as`` defaults to the components). Its components share
one stdev, so a direction survives normalisation; see :func:`pool_statistics`.
"""

from typing import Any

import numpy as np

DEFAULT_COUNT = "obs_count_{group}"


def normalise_spec(spec: str | dict) -> dict:
    """Expand a bare ``observed_as`` string, or a partial mapping, into a full spec."""
    if isinstance(spec, str):
        spec = {"observed_as": spec}
    spec = dict(spec)
    if "components" in spec:
        spec["components"] = list(spec["components"])
        spec["observed_as"] = list(spec.get("observed_as", spec["components"]))
        if len(spec["observed_as"]) != len(spec["components"]):
            msg = f"Observation spec {spec!r} needs one 'observed_as' per component."
            raise ValueError(msg)
    if "observed_as" not in spec:
        msg = f"Observation spec {spec!r} is missing 'observed_as'."
        raise ValueError(msg)
    spec.setdefault("groups", "auto")
    spec.setdefault("count", DEFAULT_COUNT)
    return spec


def is_vector(spec: dict) -> bool:
    return "components" in spec


def field_members(field: str, spec: dict) -> list[tuple[str, str]]:
    """``(model output, observed_as)`` for each component, or for the field itself."""
    if is_vector(spec):
        return list(zip(spec["components"], spec["observed_as"], strict=True))
    return [(field, spec["observed_as"])]


def _groups_with_prefix(prefix: str, available: dict[str, int], owner: str) -> list[str]:
    found = [(pos, name[len(prefix) :]) for name, pos in available.items() if name.startswith(prefix)]
    if not found:
        msg = f"No variables matching '{prefix}*' are available to {owner or 'the caller'}."
        raise ValueError(msg)
    return [group for _, group in sorted(found)]


def resolve_groups(spec: dict, available: dict[str, int], owner: str = "") -> list[str]:
    """The observing groups of a field, ordered by where their columns sit.

    Every component of a vector field must be observed by the same groups.
    """
    if spec["groups"] != "auto":
        return list(spec["groups"])
    first, *others = [observed_as for _, observed_as in field_members("", spec)]
    groups = _groups_with_prefix(f"{first}_", available, owner)
    for observed_as in others:
        other = _groups_with_prefix(f"{observed_as}_", available, owner)
        if sorted(other) != sorted(groups):
            msg = f"'{observed_as}_*' and '{first}_*' are observed by different groups: {other} vs {groups}."
            raise ValueError(msg)
    return groups


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

    A vector field pools each component's mean separately but gives all components one
    stdev, the root of their mean pooled variance. A per-component stdev would stretch
    the vector anisotropically, and the loss could no longer recover the direction it
    projects onto.

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
        counts = [spec["count"].format(group=group) for group in groups]
        members = field_members(field, spec)
        observed = {output: [f"{observed_as}_{group}" for group in groups] for output, observed_as in members}

        missing = [name for name in sum(observed.values(), []) + counts if name not in name_to_index]
        if missing:
            msg = f"Cannot pool statistics for {field!r}: {missing} not in the dataset."
            raise ValueError(msg)

        weights = np.array([stats["mean"][name_to_index[name]] for name in counts], dtype="float64")
        if weights.sum() <= 0:
            msg = f"Cannot pool statistics for {field!r}: every observing group has a zero obs count."
            raise ValueError(msg)
        weights = weights / weights.sum()

        moments = {
            output: _pooled_moments(stats, [name_to_index[name] for name in names], weights)
            for output, names in observed.items()
        }
        pooled_stdev = np.sqrt(max(np.mean([var for _, var in moments.values()]), 0.0))

        for output, names in observed.items():
            obs_idx = [name_to_index[name] for name in names]
            # The output itself (the diagnostic the model emits) shares the map when present,
            # so its prediction and its targets land in the same space.
            columns = obs_idx + ([name_to_index[output]] if output in name_to_index else [])
            stats["mean"][columns] = moments[output][0]
            stats["stdev"][columns] = pooled_stdev
            stats["minimum"][columns] = stats["minimum"][obs_idx].min()
            stats["maximum"][columns] = stats["maximum"][obs_idx].max()

    return stats


def _pooled_moments(stats: dict[str, np.ndarray], obs_idx: list[int], weights: np.ndarray) -> tuple[float, float]:
    """Count-weighted mean and variance, pooling the second moment about zero then shifting."""
    mean, stdev = stats["mean"][obs_idx], stats["stdev"][obs_idx]
    pooled_mean = float((weights * mean).sum())
    return pooled_mean, float((weights * (stdev**2 + mean**2)).sum() - pooled_mean**2)
