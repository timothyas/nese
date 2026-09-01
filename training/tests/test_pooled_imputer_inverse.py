"""The imputer declining to undo its own fill on a ``data.output``-shaped tensor.

The base class restores gaps from a mask captured once per forward pass, at the oldest
input bin, and stamps it over every timestep. The tensors that reach the plots span the
target bin too, where the tracks have moved, so most of what a mission saw there is
overwritten -- and nothing raises, the truth panel just fills with the imputer's fill
value. ObsCountNaNRestorer rebuilds the same gaps exactly from the per-bin counts, so the
invariant here is simply that the wide tensor comes back untouched.
"""

import numpy as np
import pytest
import torch
from anemoi.models.preprocessing.imputer import InputImputer
from nese_training.processors import PooledObservationImputer

DATA_OUTPUT_WIDTH = 5
MODEL_OUTPUT_WIDTH = 2


class FakeIndices:
    """Only ``data.output.full``, whose length is what the override branches on."""

    def __init__(self, width=DATA_OUTPUT_WIDTH):
        output = type("_O", (), {"full": np.arange(width)})
        self.data = type("_D", (), {"output": output})


@pytest.fixture
def imputer():
    """The subclass without its constructor: the branch reads only ``data_indices``."""
    obj = object.__new__(PooledObservationImputer)
    obj.data_indices = FakeIndices()
    return obj


@pytest.fixture
def spy(monkeypatch):
    """Records whether the base class's restoration was reached."""
    calls = []

    def record(self, x, in_place=True, **kwargs):
        calls.append(x.shape[-1])
        return x

    monkeypatch.setattr(InputImputer, "inverse_transform", record)
    return calls


def test_data_output_tensor_passes_through(imputer, spy):
    x = torch.arange(2 * 3 * DATA_OUTPUT_WIDTH, dtype=torch.float32).reshape(2, 3, DATA_OUTPUT_WIDTH)

    out = imputer.inverse_transform(x.clone())

    torch.testing.assert_close(out, x)
    assert spy == []


def test_pass_through_respects_in_place(imputer, spy):
    """``in_place=False`` has to hand back a copy, or a caller's tensor gets aliased."""
    x = torch.zeros(1, 3, DATA_OUTPUT_WIDTH)

    assert imputer.inverse_transform(x, in_place=True) is x
    assert imputer.inverse_transform(x, in_place=False) is not x


def test_other_widths_still_reach_the_base_class(imputer, spy):
    """A model-output tensor is the imputer's own business and is left to it."""
    x = torch.zeros(1, 3, MODEL_OUTPUT_WIDTH)

    imputer.inverse_transform(x)

    assert spy == [MODEL_OUTPUT_WIDTH]
