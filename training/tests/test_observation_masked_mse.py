"""Name resolution and observation-count pooling in ObservationMaskedMSE.

The loss replaces one CombinedLoss entry per mission with a naming convention, so the
two things that can fail silently are resolving the wrong set of columns and weighting
the groups wrongly. Both are checked here against the real nese-dataprep names.
"""

import pytest
import torch
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace
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

    def __init__(
        self,
        groups: list[str],
        *,
        drop_counts: tuple[str, ...] = (),
        fields: dict[str, str] | None = None,
    ) -> None:
        fields = fields or {"sla": "sla_filtered"}
        names = list(fields)
        for observed_as in fields.values():
            names += [f"{observed_as}_{group}" for group in groups]
        names += [f"obs_count_{group}" for group in groups if group not in drop_counts]

        self.data_full_name_to_position = {name: i for i, name in enumerate(names)}
        model_output = {name: i for i, name in enumerate(fields)}
        self.data = type("_D", (), {"output": type("_O", (), {"name_to_position": self.data_full_name_to_position})})
        self.model = type("_M", (), {"output": type("_O", (), {"name_to_position": model_output})})

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


def test_matches_pooled_mse_over_observed_cells() -> None:
    """Every observed (group, cell) pair counts once, as in nese-verify's pooled score."""
    indices = FakeIndices(GROUPS)
    generator = torch.Generator().manual_seed(0)
    observed = {group: torch.rand(GRID, generator=generator) < 0.3 for group in GROUPS}
    pred, target = make_batch(indices, observed)
    squared_errors, n_obs = 0.0, 0
    for group in GROUPS:
        column = indices.data_full_name_to_position[f"sla_filtered_{group}"]
        target[..., column] = torch.randn(GRID, generator=generator) * 0.1
        error = pred[0, 0, 0, :, 0] - target[0, 0, 0, :, column]
        squared_errors += (error[observed[group]] ** 2).sum()
        n_obs += observed[group].sum()

    assert torch.allclose(call(make_loss(), pred, target), squared_errors / n_obs, atol=1e-6)


def test_groups_weigh_by_observation_count() -> None:
    """A group seeing 48 cells outweighs one seeing 16 three to one, not one to one."""
    indices = FakeIndices(GROUPS)
    many, few = torch.zeros(GRID, dtype=torch.bool), torch.zeros(GRID, dtype=torch.bool)
    many[:48], few[48:] = True, True
    pred, target = make_batch(indices, {"ers_2": many, "gfo": few})
    target[..., indices.data_full_name_to_position["sla_filtered_gfo"]] = 2.5  # error 2.0

    expected = (48 * 0.5**2 + 16 * 2.0**2) / 64
    assert torch.allclose(call(make_loss(), pred, target), torch.tensor(expected), atol=1e-5)


def test_dead_groups_do_not_dilute_the_loss() -> None:
    """Only 3-6 of the record's 14 groups fly at once. Declaring the other 8 must not
    weaken the gradient, or training strength would track satellite availability."""
    live_groups = GROUPS[:3]

    few = make_loss(groups=live_groups)
    many = make_loss(groups=GROUPS)

    indices = FakeIndices(GROUPS)
    pred, target = make_batch(indices, all_observed(live_groups), value=0.0)

    assert torch.allclose(call(few, pred, target), call(many, pred, target), atol=1e-6)


def make_two_field_batch(swh_target: float) -> tuple:
    """sla predicted 0.5 and swh 2.0 against targets of 0 and ``swh_target``, seen together."""
    fields = {"sla": "sla_filtered", "swh": "swh_denoised"}
    indices = FakeIndices(GROUPS, fields=fields)
    loss = ObservationMaskedMSE(
        observations={name: {"observed_as": observed_as, "groups": "auto"} for name, observed_as in fields.items()},
        data_indices=indices,
    )
    pred = torch.full((1, 1, 1, GRID, 2), 0.5)
    pred[..., 1] = 2.0
    target = torch.zeros(1, 1, 1, GRID, indices.n_columns)
    for group, mask in all_observed(GROUPS[:3]).items():
        target[..., indices.data_full_name_to_position[f"obs_count_{group}"]] = mask.float()
    for group in GROUPS:
        target[..., indices.data_full_name_to_position[f"swh_denoised_{group}"]] = swh_target
    return loss, pred, target


def test_fields_are_weighted_equally() -> None:
    loss, pred, target = make_two_field_batch(swh_target=0.0)
    assert torch.allclose(call(loss, pred, target), torch.tensor((0.5**2 + 2.0**2) / 2), atol=1e-5)


def test_a_field_without_error_still_counts_as_live() -> None:
    """Liveness comes from the counts, not the error: a perfect swh halves the sla loss."""
    loss, pred, target = make_two_field_batch(swh_target=2.0)
    assert torch.allclose(call(loss, pred, target), torch.tensor(0.5**2 / 2), atol=1e-5)


def test_an_unobserved_field_does_not_dilute_the_loss() -> None:
    fields = {"sla": "sla_filtered", "swh": "swh_denoised"}
    indices = FakeIndices(GROUPS, fields=fields)
    loss = ObservationMaskedMSE(
        observations={
            "sla": {"observed_as": "sla_filtered", "groups": "auto"},
            "swh": {"observed_as": "swh_denoised", "groups": ["gfo"]},
        },
        data_indices=indices,
    )
    pred = torch.full((1, 1, 1, GRID, 2), 0.5)
    target = torch.zeros(1, 1, 1, GRID, indices.n_columns)
    for group, mask in all_observed(GROUPS[:3]).items():  # gfo is not among them
        target[..., indices.data_full_name_to_position[f"obs_count_{group}"]] = mask.float()

    assert torch.allclose(call(loss, pred, target), torch.tensor(0.25), atol=1e-5)


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
    """A constant per-cell error gives a constant loss however much of the mesh was seen."""
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
