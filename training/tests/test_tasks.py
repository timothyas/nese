"""KnownForcingForecaster appends the target times to the input window, and nothing else."""

import datetime

import pytest

from nese_training.tasks import KnownForcingForecaster

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
