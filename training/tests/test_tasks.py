"""KnownForcingForecaster appends the target times to the input window; MaskedStateEstimator hides observations."""

import datetime

import pytest
import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.methods import SingleTraining
from omegaconf import OmegaConf

from nese_training.methods import MaskedInputTraining
from nese_training.tasks import KnownForcingForecaster
from nese_training.tasks import MaskedStateEstimator
from nese_training.tasks import hide_observed

DAY = datetime.timedelta(days=1)


def task(**rollout) -> KnownForcingForecaster:
    return KnownForcingForecaster(
        multistep_input=3, multistep_output=1, timestep="24h", rollout={"start": 1, "maximum": 1} | rollout
    )


def test_input_window_ends_at_the_target_time():
    forecaster = task()
    assert forecaster.get_input_offsets() == [-2 * DAY, -DAY, 0 * DAY, DAY]
    assert forecaster.num_input_timesteps == 4


def test_target_is_unchanged():
    forecaster = task()
    assert forecaster.get_output_offsets() == [DAY]
    assert forecaster.get_batch_output_indices(rollout_step=0) == [3]
    assert forecaster.get_offsets("training") == [-2 * DAY, -DAY, 0 * DAY, DAY]


def test_plots_start_from_t0():
    forecaster = task()
    adapter = forecaster._plot_adapter
    assert forecaster.get_input_offsets()[adapter.get_init_step()] == 0 * DAY
    assert adapter.get_loss_plot_batch_start(0) == 3


def test_rollout_is_refused():
    with pytest.raises(ValueError, match="rollout"):
        task(maximum=2)


GROUPS = ["jason_2", "saral"]
SPEC = {"sla": {"observed_as": "sla_filtered", "count": "obs_count_{group}"}}
VARIABLES = [f"sla_filtered_{g}" for g in GROUPS] + [f"obs_count_{g}" for g in GROUPS] + ["cos_julian_day", "sla"]
COUNTS = [2, 3]
N_GRID = 20_000


@pytest.fixture
def indices() -> dict[str, IndexCollection]:
    config = OmegaConf.create({"forcing": VARIABLES[:-1], "diagnostic": ["sla"]})
    return {"ocean": IndexCollection(config, {name: i for i, name in enumerate(VARIABLES)})}


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    data = torch.randn(2, 3, 1, N_GRID, len(VARIABLES))
    data[..., COUNTS] = (torch.rand(2, 3, 1, N_GRID, len(COUNTS)) < 0.3).float()
    return {"ocean": data}


def estimator(**kwargs) -> MaskedStateEstimator:
    return MaskedStateEstimator(timestep="24h", fields=SPEC, **kwargs)


def observed(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return (batch["ocean"][..., COUNTS] > 0).any(-1)


def hidden(x: dict[str, torch.Tensor]) -> torch.Tensor:
    return (x["ocean"][..., COUNTS] == 0).all(-1)


def test_estimates_the_input_times():
    task = estimator()
    assert task.get_input_offsets() == task.get_output_offsets() == [-DAY, 0 * DAY, DAY]
    assert task.get_batch_input_indices() == task.get_batch_output_indices() == [0, 1, 2]
    assert task._get_timestep_for_metadata() == "1d"


def test_hides_exactly_the_mask_fraction_of_observed_cells_every_group_together(batch, indices):
    seen = observed(batch)
    x = estimator(mask_fraction=0.25).get_inputs(batch, indices)
    newly_hidden = seen & hidden(x)
    assert torch.equal((x["ocean"][..., COUNTS] > 0).any(-1), seen & ~newly_hidden)
    torch.testing.assert_close(newly_hidden.sum(-1), torch.round(0.25 * seen.sum(-1)).long())


def test_hide_observed_never_hides_an_unobserved_cell():
    observed_cells = torch.tensor([[True, False, True, True], [False, False, False, False]])
    hidden_cells = hide_observed(observed_cells, 0.5, torch.Generator().manual_seed(0))
    assert not (hidden_cells & ~observed_cells).any()
    assert hidden_cells.sum(-1).tolist() == [2, 0]


def test_only_counts_change_and_targets_keep_every_observation(batch, indices):
    original = batch["ocean"].clone()
    task = estimator()
    x = task.get_inputs(batch, indices)
    others = [i for i in range(len(VARIABLES) - 1) if i not in COUNTS]
    torch.testing.assert_close(x["ocean"][..., others], original[..., others])
    torch.testing.assert_close(task.get_targets(batch)["ocean"], original)


def test_validation_mask_repeats_each_epoch(batch, indices):
    task = estimator(validation_seed=3)
    task.validation_mode = True
    first = hidden(task.get_inputs(batch, indices))
    assert not torch.equal(first, hidden(task.get_inputs(batch, indices)))
    task.reset_validation_mask()
    assert torch.equal(first, hidden(task.get_inputs(batch, indices)))


@pytest.mark.parametrize(("validation_mode", "validation_seed"), [(False, 3), (True, None)])
def test_unseeded_masks_differ_each_epoch(batch, indices, validation_mode, validation_seed):
    task = estimator(validation_seed=validation_seed)
    task.validation_mode = validation_mode
    first = hidden(task.get_inputs(batch, indices))
    task.reset_validation_mask()
    assert not torch.equal(first, hidden(task.get_inputs(batch, indices)))


def test_training_method_tells_the_task_its_mode(monkeypatch):
    monkeypatch.setattr(SingleTraining, "_step", lambda _self, _batch, validation_mode=False: validation_mode)
    task = estimator()
    task._validation_generator = torch.Generator()
    method = MaskedInputTraining.__new__(MaskedInputTraining)
    method.__dict__["task"] = task
    assert method._step({}, validation_mode=True)
    assert task.validation_mode
    method.on_validation_epoch_start()
    assert task._validation_generator is None
    method._step({})
    assert not task.validation_mode
