"""Name resolution and live-group averaging in ObservationMaskedMSE.

The loss replaces one CombinedLoss entry per mission with a naming convention, so the
two things that can fail silently are resolving the wrong set of columns and dividing by
the wrong number of groups. Both are checked here against the real nese-dataprep names.
"""

import pytest
import torch
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace
from nese_training.losses import CountMaskedAndWeightedMSE
from nese_training.losses import ObservationMaskedMSE

GRID = 64

# The groups the 2000-2001 store declares, including the two that a naive split("_")
# would break apart.
GROUPS = [
    "cryosat_2",
    "cryosat_2_interleaved",
    "envisat",
    "ers_2",
    "gfo",
    "jason_1",
    "topex_poseidon",
    "topex_poseidon_interleaved",
]


class FakeIndices:
    """The three name->position maps ObservationMaskedMSE reads off IndexCollection."""

    def __init__(self, groups: list[str], *, drop_counts: tuple[str, ...] = ()) -> None:
        names = ["sla"]
        for group in groups:
            names.append(f"sla_filtered_{group}")
            if group not in drop_counts:
                names.append(f"obs_count_{group}")

        self.data_full_name_to_position = {name: i for i, name in enumerate(names)}
        self.data = type("_D", (), {"output": type("_O", (), {"name_to_position": self.data_full_name_to_position})})
        self.model = type("_M", (), {"output": type("_O", (), {"name_to_position": {"sla": 0}})})

    @property
    def n_columns(self) -> int:
        return len(self.data_full_name_to_position)


def make_loss(
    groups: list[str] | str = "auto",
    indices: FakeIndices | None = None,
    *,
    node_weights: bool = True,
) -> ObservationMaskedMSE:
    indices = indices or FakeIndices(GROUPS)
    loss = ObservationMaskedMSE(
        observations={"sla": {"observed_as": "sla_filtered", "groups": groups}},
        data_indices=indices,
    )
    # unit-sum node weights, so BaseLoss.reduce's grid sum reads as a mean
    if node_weights:
        loss.add_scaler(dimension=TensorDim.GRID.value, scaler=torch.ones(GRID) / GRID, name="node_weights")
    return loss


def make_batch(indices: FakeIndices, observed: dict[str, torch.Tensor], value: float = 0.0) -> tuple:
    """(pred, target) in MODEL_OUTPUT / DATA_FULL layout."""
    shape = (1, 1, 1, GRID)
    pred = torch.full((*shape, 1), 0.5)
    target = torch.zeros(*shape, indices.n_columns)
    for group, mask in observed.items():
        target[..., indices.data_full_name_to_position[f"sla_filtered_{group}"]] = value
        target[..., indices.data_full_name_to_position[f"obs_count_{group}"]] = mask.float()
    return pred, target


def call(loss: ObservationMaskedMSE, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return loss(pred, target, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)


def all_observed(groups: list[str]) -> dict[str, torch.Tensor]:
    mask = torch.zeros(GRID, dtype=torch.bool)
    mask[:16] = True
    return {group: mask for group in groups}


def test_auto_resolves_every_group() -> None:
    loss = make_loss()
    assert loss.observations["sla"]["groups"] == "auto"
    # two columns (value, count) per group
    assert len(loss.target_variables) == 2 * len(GROUPS)
    assert loss.target_variables[:2] == ["sla_filtered_cryosat_2", "obs_count_cryosat_2"]


def test_multi_underscore_groups_survive_resolution() -> None:
    """Stripping the declared prefix, rather than splitting on '_', is what keeps
    topex_poseidon_interleaved one group instead of three."""
    loss = make_loss()
    assert "sla_filtered_topex_poseidon_interleaved" in loss.target_variables
    assert "obs_count_topex_poseidon_interleaved" in loss.target_variables


def test_predicted_variable_is_repeated_per_group() -> None:
    loss = make_loss()
    assert loss.predicted_variables == ["sla"] * 2 * len(GROUPS)
    assert loss.predicted_indices_by_layout[IndexSpace.MODEL_OUTPUT] == [0] * 2 * len(GROUPS)


def test_value_column_without_a_count_raises() -> None:
    indices = FakeIndices(GROUPS, drop_counts=("gfo",))
    with pytest.raises(ValueError, match="obs_count_gfo"):
        make_loss(indices=indices)


def test_explicit_group_list_pins_the_set() -> None:
    loss = make_loss(groups=["ers_2", "gfo"])
    assert loss.target_variables == [
        "sla_filtered_ers_2",
        "obs_count_ers_2",
        "sla_filtered_gfo",
        "obs_count_gfo",
    ]


def test_explicit_group_missing_from_store_raises() -> None:
    with pytest.raises(ValueError, match="sla_filtered_sentinel_3a"):
        make_loss(groups=["ers_2", "sentinel_3a"])


def test_string_shorthand_is_accepted() -> None:
    loss = ObservationMaskedMSE(observations={"sla": "sla_filtered"}, data_indices=FakeIndices(GROUPS))
    assert len(loss.target_variables) == 2 * len(GROUPS)


def test_matches_mean_of_separate_per_mission_terms() -> None:
    """Regression guard: with every group live this must reproduce exactly what the
    3-term CombinedLoss config computed, up to CombinedLoss summing where this averages."""
    indices = FakeIndices(GROUPS)
    generator = torch.Generator().manual_seed(0)
    observed = {
        group: torch.rand(GRID, generator=generator) < 0.3 for group in GROUPS
    }
    pred, target = make_batch(indices, observed)
    for group in GROUPS:
        column = indices.data_full_name_to_position[f"sla_filtered_{group}"]
        target[..., column] = torch.randn(GRID, generator=generator) * 0.1

    combined = call(make_loss(), pred, target)

    separate = []
    for group in GROUPS:
        single = CountMaskedAndWeightedMSE()
        single.add_scaler(dimension=TensorDim.GRID.value, scaler=torch.ones(GRID) / GRID, name="node_weights")
        columns = [
            indices.data_full_name_to_position[f"sla_filtered_{group}"],
            indices.data_full_name_to_position[f"obs_count_{group}"],
        ]
        separate.append(single(pred.repeat(1, 1, 1, 1, 2), target[..., columns]))

    assert torch.allclose(combined, sum(separate) / len(separate), atol=1e-6)


def test_dead_groups_do_not_dilute_the_loss() -> None:
    """Only 3-6 of the record's 14 groups fly at once. Declaring the other 8 must not
    weaken the gradient, or training strength would track satellite availability."""
    live_groups = GROUPS[:3]

    few = make_loss(groups=live_groups)
    many = make_loss(groups=GROUPS)

    indices = FakeIndices(GROUPS)
    pred, target = make_batch(indices, all_observed(live_groups), value=0.0)

    assert torch.allclose(call(few, pred, target), call(many, pred, target), atol=1e-6)


def test_declaring_more_groups_would_dilute_without_the_correction() -> None:
    """The failure mode the live divisor exists to prevent, measured against the
    uncorrected loss: 3 live of 8 declared reads 3/8 of its true value."""
    live_groups = GROUPS[:3]
    indices = FakeIndices(GROUPS)
    pred, target = make_batch(indices, all_observed(live_groups))

    uncorrected = CountMaskedAndWeightedMSE()
    uncorrected.add_scaler(dimension=TensorDim.GRID.value, scaler=torch.ones(GRID) / GRID, name="node_weights")
    columns = []
    for group in GROUPS:
        columns += [
            indices.data_full_name_to_position[f"sla_filtered_{group}"],
            indices.data_full_name_to_position[f"obs_count_{group}"],
        ]
    diluted = uncorrected(pred.repeat(1, 1, 1, 1, 2 * len(GROUPS)), target[..., columns])
    corrected = call(make_loss(groups=GROUPS), pred, target)

    assert torch.allclose(corrected * len(live_groups) / len(GROUPS), diluted, atol=1e-6)


def test_unobserved_cells_do_not_affect_the_loss() -> None:
    indices = FakeIndices(GROUPS)
    observed = all_observed(GROUPS)
    loss = make_loss()
    pred, target = make_batch(indices, observed)

    before = call(loss, pred, target)
    for group in GROUPS:
        column = indices.data_full_name_to_position[f"sla_filtered_{group}"]
        target[..., ~observed[group], column] = 1e6
    assert torch.allclose(before, call(loss, pred, target))


def test_observed_cells_do_affect_the_loss() -> None:
    indices = FakeIndices(GROUPS)
    observed = all_observed(GROUPS)
    loss = make_loss()
    pred, target = make_batch(indices, observed)

    before = call(loss, pred, target)
    column = indices.data_full_name_to_position["sla_filtered_ers_2"]
    target[..., observed["ers_2"], column] = 3.0
    assert not torch.allclose(before, call(loss, pred, target))


@pytest.mark.parametrize("n_observed", [4, 16, 32, 64])
def test_loss_is_invariant_to_coverage(n_observed: int) -> None:
    """Per-group coverage normalisation is inherited unchanged: a constant per-cell error
    gives a constant loss however much of the mesh was seen."""
    mask = torch.zeros(GRID, dtype=torch.bool)
    mask[:n_observed] = True

    indices = FakeIndices(GROUPS)
    pred, target = make_batch(indices, {group: mask for group in GROUPS})

    assert torch.allclose(call(make_loss(), pred, target), torch.tensor(0.25), atol=1e-5)


@pytest.mark.parametrize("n_observed", [4, 16, 32, 64])
def test_loss_is_invariant_to_coverage_with_no_grid_scaler(n_observed: int) -> None:
    """The same invariance without a node_weights scaler, which is how a HEALPix run is
    configured: equal-area cells make the weight uniform, so the loss supplies the 1/N
    itself by dividing the mask by its own sum."""
    mask = torch.zeros(GRID, dtype=torch.bool)
    mask[:n_observed] = True

    indices = FakeIndices(GROUPS)
    pred, target = make_batch(indices, {group: mask for group in GROUPS})

    unscaled = call(make_loss(node_weights=False), pred, target)
    assert torch.allclose(unscaled, torch.tensor(0.25), atol=1e-5)
    assert torch.allclose(unscaled, call(make_loss(), pred, target), atol=1e-5)


def test_layout_is_required() -> None:
    indices = FakeIndices(GROUPS)
    pred, target = make_batch(indices, all_observed(GROUPS))
    with pytest.raises(ValueError, match="pred_layout"):
        make_loss()(pred, target)
