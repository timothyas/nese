"""Pooled statistics across a field's observing groups.

The failure this guards against is silent: per-mission mean-std statistics look neutral,
but a mission's mean is the sea level of the years it flew, so per-mission centring
subtracts the trend and leaves the one predicted field scored against targets whose zero
points sit apart in normalised space. Nothing raises -- the run just learns the wrong
thing -- so the invariant worth asserting is that every member of a pool ends up with
exactly one shared map.
"""

import numpy as np
import pytest
from nese_training.observations import pool_statistics

GROUPS = ["cryosat_2", "ers_2", "topex_poseidon", "topex_poseidon_interleaved"]
SPEC = {"sla": {"observed_as": "sla_filtered", "groups": "auto", "count": "obs_count_{group}"}}


def make_stats(means, stdevs=None, counts=None, field=True):
    """A statistics dict shaped like the one anemoi hands a processor."""
    names = []
    if field:
        names.append("sla")
    names += [f"sla_filtered_{g}" for g in GROUPS] + [f"obs_count_{g}" for g in GROUPS]
    name_to_index = {n: i for i, n in enumerate(names)}

    stdevs = [0.087] * len(GROUPS) if stdevs is None else stdevs
    counts = [1.0] * len(GROUPS) if counts is None else counts
    n = len(names)
    stats = {k: np.zeros(n) for k in ("mean", "stdev", "minimum", "maximum")}
    if field:
        stats["mean"][name_to_index["sla"]] = means[0]
        stats["stdev"][name_to_index["sla"]] = stdevs[0]
    for g, mu, sd, ct in zip(GROUPS, means, stdevs, counts):
        i = name_to_index[f"sla_filtered_{g}"]
        stats["mean"][i], stats["stdev"][i] = mu, sd
        stats["minimum"][i], stats["maximum"][i] = mu - 1, mu + 1
        stats["mean"][name_to_index[f"obs_count_{g}"]] = ct
    return stats, name_to_index


def test_field_and_every_group_share_one_mean_and_stdev():
    means = [0.0212, 0.0489, 0.0537, 0.0602]
    stats, n2i = make_stats(means)
    out = pool_statistics(SPEC, n2i, stats)

    members = ["sla"] + [f"sla_filtered_{g}" for g in GROUPS]
    assert len({out["mean"][n2i[m]] for m in members}) == 1
    assert len({out["stdev"][n2i[m]] for m in members}) == 1
    # the pooled centre is inside the range it was pooled from
    assert min(means) < out["mean"][n2i["sla"]] < max(means)


def test_pooling_is_weighted_by_observation_count():
    """A mission contributing most of the observations should dominate the centre."""
    means = [0.00, 0.10, 0.10, 0.10]
    stats, n2i = make_stats(means, counts=[97.0, 1.0, 1.0, 1.0])
    out = pool_statistics(SPEC, n2i, stats)

    assert out["mean"][n2i["sla"]] == pytest.approx(0.003, abs=1e-9)  # 0.97*0 + 0.03*0.1

    unweighted, _ = make_stats(means)
    assert pool_statistics(SPEC, n2i, unweighted)["mean"][n2i["sla"]] == pytest.approx(0.075)


def test_pooled_variance_absorbs_the_spread_between_groups():
    """Pooling the second moment, not the stdevs: groups centred apart are jointly wider
    than any of them alone, which averaging the stdevs would miss."""
    stats, n2i = make_stats([-1.0, -1.0, 1.0, 1.0], stdevs=[0.5] * 4)
    out = pool_statistics(SPEC, n2i, stats)

    assert out["mean"][n2i["sla"]] == pytest.approx(0.0)
    assert out["stdev"][n2i["sla"]] == pytest.approx(np.sqrt(0.25 + 1.0))  # var + spread^2


def test_count_columns_are_left_alone():
    stats, n2i = make_stats([0.02, 0.04, 0.05, 0.06], counts=[1.0, 2.0, 3.0, 4.0])
    out = pool_statistics(SPEC, n2i, stats)
    for g, ct in zip(GROUPS, [1.0, 2.0, 3.0, 4.0]):
        assert out["mean"][n2i[f"obs_count_{g}"]] == ct


def test_the_callers_statistics_are_not_mutated():
    """anemoi hands the same statistics object to every processor, so pooling in place
    would silently reach across into the others."""
    stats, n2i = make_stats([0.02, 0.04, 0.05, 0.06])
    before = {k: v.copy() for k, v in stats.items()}
    pool_statistics(SPEC, n2i, stats)
    for k, v in before.items():
        assert np.array_equal(stats[k], v), k


def test_a_pool_without_the_field_column_still_pools_its_groups():
    """`sla` is a diagnostic and need not be present; the observations still share a map."""
    stats, n2i = make_stats([0.02, 0.04, 0.05, 0.06], field=False)
    out = pool_statistics(SPEC, n2i, stats)
    assert len({out["mean"][n2i[f"sla_filtered_{g}"]] for g in GROUPS}) == 1


def test_a_missing_count_column_is_rejected():
    stats, n2i = make_stats([0.02, 0.04, 0.05, 0.06])
    del n2i["obs_count_ers_2"]
    with pytest.raises(ValueError, match="obs_count_ers_2"):
        pool_statistics(SPEC, n2i, stats)


def test_groups_with_no_observations_at_all_are_rejected():
    """Weights come from the counts, so an all-zero archive would divide by zero rather
    than quietly produce NaN normalisation factors."""
    stats, n2i = make_stats([0.02, 0.04, 0.05, 0.06], counts=[0.0] * 4)
    with pytest.raises(ValueError, match="zero obs count"):
        pool_statistics(SPEC, n2i, stats)


def test_explicit_group_list_is_honoured():
    stats, n2i = make_stats([0.00, 0.10, 0.20, 0.30], counts=[1.0] * 4)
    spec = {"sla": {"observed_as": "sla_filtered", "groups": ["cryosat_2", "ers_2"]}}
    out = pool_statistics(spec, n2i, stats)

    assert out["mean"][n2i["sla"]] == pytest.approx(0.05)
    # a group left out of the spec keeps its own statistics
    assert out["mean"][n2i["sla_filtered_topex_poseidon"]] == pytest.approx(0.20)


VECTOR_SPEC = {"uvgxt": {"components": ["ugxt", "vgxt"]}}


def make_vector_stats(u_means, v_means, u_stdevs, v_stdevs):
    names = ["ugxt", "vgxt"] + [f"{c}_{g}" for c in ("ugxt", "vgxt") for g in GROUPS] + [f"obs_count_{g}" for g in GROUPS]
    n2i = {n: i for i, n in enumerate(names)}
    stats = {k: np.zeros(len(names)) for k in ("mean", "stdev", "minimum", "maximum")}
    for g, um, vm, us, vs in zip(GROUPS, u_means, v_means, u_stdevs, v_stdevs):
        for c, m, s in (("ugxt", um, us), ("vgxt", vm, vs)):
            i = n2i[f"{c}_{g}"]
            stats["mean"][i], stats["stdev"][i] = m, s
            stats["minimum"][i], stats["maximum"][i] = m - 1, m + 1
        stats["mean"][n2i[f"obs_count_{g}"]] = 1.0
    return stats, n2i


def test_vector_components_share_one_stdev_but_keep_their_means():
    stats, n2i = make_vector_stats([0.1] * 4, [-0.2] * 4, [0.3] * 4, [0.4] * 4)
    out = pool_statistics(VECTOR_SPEC, n2i, stats)

    u_members = ["ugxt"] + [f"ugxt_{g}" for g in GROUPS]
    v_members = ["vgxt"] + [f"vgxt_{g}" for g in GROUPS]
    assert [out["mean"][n2i[m]] for m in u_members] == pytest.approx([0.1] * len(u_members))
    assert [out["mean"][n2i[m]] for m in v_members] == pytest.approx([-0.2] * len(v_members))
    stdevs = {out["stdev"][n2i[m]] for m in u_members + v_members}
    assert len(stdevs) == 1
    assert stdevs.pop() == pytest.approx(np.sqrt((0.3**2 + 0.4**2) / 2))


def test_vector_components_must_share_groups():
    stats, n2i = make_vector_stats([0.1] * 4, [-0.2] * 4, [0.3] * 4, [0.4] * 4)
    del n2i["vgxt_ers_2"]
    with pytest.raises(ValueError, match="different groups"):
        pool_statistics(VECTOR_SPEC, n2i, stats)
