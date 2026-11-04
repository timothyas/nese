"""UnseenObservationMaskedMSE scores exactly the target observations unseen in the input window.

The reference is ObservationMaskedMSE with the target counts zeroed wherever the field was seen,
i.e. the pooled score over nese-verify's ``seen_at_hour == +timestep`` cells.
"""

import pytest
import torch
from anemoi.training.utils.enums import TensorDim
from nese_training.losses import ObservationMaskedMSE
from nese_training.losses import UnseenObservationMaskedMSE
from nese_training.methods import UnseenAwareTraining
from test_observation_masked_mse import GRID
from test_observation_masked_mse import GROUPS
from test_observation_masked_mse import FakeIndices
from test_observation_masked_mse import all_observed
from test_observation_masked_mse import call
from test_observation_masked_mse import cross_track_batch
from test_observation_masked_mse import make_vector_loss
from test_observation_masked_mse import mesh

N_INPUT = 3
SLA = {"sla": {"observed_as": "sla_filtered", "groups": "auto"}}
SEEN = torch.arange(GRID) < 8


def make_pair(indices: FakeIndices, observations: dict = SLA) -> tuple:
    pair = ObservationMaskedMSE(observations, data_indices=indices), UnseenObservationMaskedMSE(
        observations, data_indices=indices
    )
    for loss in pair:
        loss.add_scaler(dimension=TensorDim.GRID.value, scaler=torch.ones(GRID) / GRID, name="node_weights")
    return pair


def make_inputs(indices, seen: dict[tuple[int, str], torch.Tensor], grid: int = GRID) -> torch.Tensor:
    """Model input ``(batch, step, ensemble, grid, variable)`` with ``seen[(step, count)]`` set."""
    data_input = indices.data.input.name_to_position
    x = torch.zeros(1, N_INPUT, 1, grid, len(data_input))
    for (step, name), mask in seen.items():
        x[:, step, :, :, data_input[name]] = mask.float()
    return x


def observed_batch(indices: FakeIndices, groups: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets observed on the first 16 cells by ``groups``, against a random prediction."""
    torch.manual_seed(0)
    target = torch.zeros(1, 1, 1, GRID, indices.n_columns)
    for group, mask in all_observed(groups).items():
        target[..., indices.data_full_name_to_position[f"obs_count_{group}"]] = mask.float()
        target[..., indices.data_full_name_to_position[f"sla_filtered_{group}"]] = torch.randn(GRID)
    return torch.randn(1, 1, 1, GRID, 1, requires_grad=True), target


def zero_counts_where(target: torch.Tensor, columns: list[int], seen: torch.Tensor) -> torch.Tensor:
    target = target.clone()
    for column in columns:
        target[..., seen, column] = 0.0
    return target


def test_nothing_seen_matches_the_base_loss() -> None:
    indices = FakeIndices(GROUPS)
    base, unseen = make_pair(indices)
    pred, target = observed_batch(indices, GROUPS[:2])
    unseen.set_inputs(make_inputs(indices, {}))
    torch.testing.assert_close(call(unseen, pred, target), call(base, pred, target))


@pytest.mark.parametrize(("step", "group"), [(0, GROUPS[0]), (N_INPUT - 1, GROUPS[5])])
def test_matches_the_base_loss_over_unseen_observations(step: int, group: str) -> None:
    """Seen at any step, by any of the field's groups, including one not in the target."""
    indices = FakeIndices(GROUPS)
    base, unseen = make_pair(indices)
    pred, target = observed_batch(indices, GROUPS[:2])
    unseen.set_inputs(make_inputs(indices, {(step, f"obs_count_{group}"): SEEN}))

    counts = [indices.data_full_name_to_position[f"obs_count_{g}"] for g in GROUPS]
    expected = call(base, pred, zero_counts_where(target, counts, SEEN))
    torch.testing.assert_close(call(unseen, pred, target), expected)
    assert not torch.isclose(expected, call(base, pred, target))


def test_only_unseen_observed_cells_carry_gradient() -> None:
    indices = FakeIndices(GROUPS)
    _, unseen = make_pair(indices)
    pred, target = observed_batch(indices, GROUPS[:2])
    unseen.set_inputs(make_inputs(indices, {(1, f"obs_count_{GROUPS[0]}"): SEEN}))
    call(unseen, pred, target).backward()

    grad = pred.grad[0, 0, 0, :, 0]
    observed = torch.arange(GRID) < 16
    assert (grad[SEEN] == 0).all()
    assert (grad[observed & ~SEEN] != 0).all()
    assert (grad[~observed] == 0).all()


def test_each_field_has_its_own_seen_mask() -> None:
    fields = {"sla": "sla_filtered", "swh": "swh_denoised"}
    indices = FakeIndices(GROUPS, fields=fields)
    observations = {
        "sla": {"observed_as": "sla_filtered", "groups": GROUPS[:2]},
        "swh": {"observed_as": "swh_denoised", "groups": GROUPS[2:4]},
    }
    base, unseen = make_pair(indices, observations)
    torch.manual_seed(0)
    pred = torch.randn(1, 1, 1, GRID, 2)
    target = torch.zeros(1, 1, 1, GRID, indices.n_columns)
    for group, mask in all_observed(GROUPS[:4]).items():
        target[..., indices.data_full_name_to_position[f"obs_count_{group}"]] = mask.float()
        for observed_as in fields.values():
            target[..., indices.data_full_name_to_position[f"{observed_as}_{group}"]] = torch.randn(GRID)

    unseen.set_inputs(make_inputs(indices, {(0, f"obs_count_{GROUPS[0]}"): SEEN}))
    call(base, pred, target), call(unseen, pred, target)

    torch.testing.assert_close(unseen.field_losses["swh"], base.field_losses["swh"])
    assert not torch.isclose(unseen.field_losses["sla"], base.field_losses["sla"])


def test_cross_track_term_is_masked_where_seen(monkeypatch) -> None:
    base, indices = make_vector_loss(monkeypatch, *mesh())
    unseen, _ = make_vector_loss(monkeypatch, *mesh(), cls=UnseenObservationMaskedMSE)
    grid = len(mesh()[0])
    pred, target = cross_track_batch(indices, normal_error=0.05, along_track=0.0)
    pred[..., : grid // 2, 1] += 1.0  # a large error confined to the seen half
    seen = torch.arange(grid) < grid // 2
    unseen.set_inputs(make_inputs(indices, {(2, "obs_count_topex_poseidon"): seen}, grid=grid))

    counts = [indices.column(f"obs_count_{g}") for g in ("jason_2", "topex_poseidon")]
    expected = call(base, pred, zero_counts_where(target, counts, seen))
    torch.testing.assert_close(call(unseen, pred, target), expected)


def test_everything_seen_scores_zero_with_finite_gradient() -> None:
    indices = FakeIndices(GROUPS)
    _, unseen = make_pair(indices)
    pred, target = observed_batch(indices, GROUPS[:2])
    unseen.set_inputs(make_inputs(indices, {(0, f"obs_count_{GROUPS[0]}"): torch.ones(GRID, dtype=torch.bool)}))
    loss = call(unseen, pred, target)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.isfinite(pred.grad).all()


def test_inputs_are_required() -> None:
    indices = FakeIndices(GROUPS)
    _, unseen = make_pair(indices)
    pred, target = observed_batch(indices, GROUPS[:2])
    with pytest.raises(RuntimeError, match="set_inputs"):
        call(unseen, pred, target)


def test_a_seen_mask_on_another_grid_raises() -> None:
    indices = FakeIndices(GROUPS)
    _, unseen = make_pair(indices)
    pred, target = observed_batch(indices, GROUPS[:2])
    unseen.set_inputs(make_inputs(indices, {}, grid=GRID // 2))
    with pytest.raises(ValueError, match="cells"):
        call(unseen, pred, target)


def test_training_method_hands_every_unseen_leaf_the_input() -> None:
    indices = FakeIndices(GROUPS)
    base, unseen = make_pair(indices)
    x = make_inputs(indices, {(1, f"obs_count_{GROUPS[3]}"): SEEN})

    class FakeTask:
        def get_inputs(self, batch, data_indices):
            assert data_indices == "indices"
            return {"ocean": batch["ocean"]}

    method = type(
        "FakeMethod",
        (),
        {
            "task": FakeTask(),
            "data_indices": "indices",
            "loss": torch.nn.ModuleDict({"ocean": torch.nn.ModuleList([base, unseen])}),
        },
    )()
    UnseenAwareTraining.set_loss_inputs(method, {"ocean": x})

    assert unseen.seen.shape == (1, 1, GRID, 1)
    torch.testing.assert_close(unseen.seen[0, 0, :, 0], SEEN)
