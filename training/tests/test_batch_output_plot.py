"""Narrowing the truth tensor to the columns the plot's variable index refers to.

The callback indexes truth and prediction with one map built from model-output positions,
but hands the plot function a truth tensor in data-output space. Those spaces differ as
soon as `target` variables are configured, and nothing downstream notices -- the figure
just shows a different variable under the right title. So the mapping is worth pinning
down here rather than by eye on a plot.
"""

import numpy as np
import pytest
from anemoi.training.diagnostics.callbacks.plot import BatchOutputPlot
from nese_training.diagnostics import ObservationBatchOutputPlot

N_TARGETS = 26  # 13 sla_filtered_* + 13 obs_count_*, as the ocean config has them
N_OUTPUT = N_TARGETS + 1  # ... plus the one diagnostic


class FakeModule:
    """A ``pl_module`` holding only what the override reads."""

    def __init__(self, positions):
        indices = type("_I", (), {"model_output_positions_in_data_output": positions})
        self.data_indices = {"ocean": indices}


@pytest.fixture
def tensors():
    """A truth tensor whose every column is distinguishable, and a one-column output."""
    data = np.arange(2 * 4 * N_OUTPUT, dtype=float).reshape(2, 4, N_OUTPUT)
    return data, np.zeros((2, 4, 1))


@pytest.fixture
def callback(monkeypatch, tensors):
    """The subclass over a stubbed base, built without touching the real constructor."""
    monkeypatch.setattr(BatchOutputPlot, "process", lambda self, *args, **kwargs: tensors)
    return object.__new__(ObservationBatchOutputPlot)


def test_truth_narrows_to_the_diagnostic(callback, tensors):
    """`sla` sorts last of the 27 output columns, and that is the column that survives."""
    data, output = tensors
    narrowed, passed_through = callback.process(FakeModule([N_TARGETS]), "ocean")

    assert narrowed.shape == (2, 4, 1)
    np.testing.assert_array_equal(narrowed[..., 0], data[..., N_TARGETS])
    assert passed_through is output


def test_prediction_is_untouched(callback, tensors):
    """Only the truth is in the wrong space; the prediction already indexes correctly."""
    _, output = tensors
    assert callback.process(FakeModule([0]), "ocean")[1] is output


def test_narrowing_is_identity_without_targets(callback, tensors):
    """A stock anemoi setup has data.output == model.output, and must come back unchanged."""
    data, _ = tensors
    narrowed, _ = callback.process(FakeModule(list(range(N_OUTPUT))), "ocean")

    np.testing.assert_array_equal(narrowed, data)


def test_column_order_follows_the_positions(callback, tensors):
    """The positions are taken in order, not sorted, so the plot's map stays aligned."""
    data, _ = tensors
    narrowed, _ = callback.process(FakeModule([N_TARGETS, 0]), "ocean")

    np.testing.assert_array_equal(narrowed[..., 0], data[..., N_TARGETS])
    np.testing.assert_array_equal(narrowed[..., 1], data[..., 0])
