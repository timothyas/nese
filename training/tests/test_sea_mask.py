"""Marking the model's output invalid where the graph gave it nothing to work from.

Land nodes have zero in-degree, so the decoder still emits a number for them but that
number carries no information. Nothing downstream can tell it apart from a prediction, so
these tests pin the two properties that make the marking trustworthy: it never touches the
forward pass, and it survives serialisation -- which is the only reason it reaches
inference rather than just the training plots.
"""

import numpy as np
import pytest
import torch
from nese_training import processors
from nese_training.processors import SeaMaskPostprocessor

GRID = 8
N_PREDICTED = 1
SEA = np.array([True, True, True, True, True, False, False, False])


class _Output:
    def __init__(self, n_predicted: int):
        self.full = list(range(n_predicted))


class _Model:
    def __init__(self, n_predicted: int):
        self.output = _Output(n_predicted)


class FakeIndices:
    """Just the `model.output.full` width the processor checks its tensors against.

    Written as real classes rather than `type(...)` on the fly so it pickles -- the
    processor keeps a reference to `data_indices`, and the whole model is pickled into the
    inference checkpoint, so anything hanging off it has to survive that.
    """

    def __init__(self, n_predicted: int = N_PREDICTED):
        self.model = _Model(n_predicted)


@pytest.fixture
def restorer(monkeypatch):
    monkeypatch.setattr(processors, "variable_below", lambda *_a, **_k: SEA.copy())
    return SeaMaskPostprocessor(
        dataset="ignored", variable="land_fraction", threshold=0.1, data_indices=FakeIndices()
    )


def prediction(value: float = 0.5, n_predicted: int = N_PREDICTED) -> torch.Tensor:
    return torch.full((1, 1, GRID, n_predicted), value)


def test_forward_pass_is_a_no_op(restorer):
    """Training must never see a NaN: preprocessing asserts against it, and one in the
    forward graph would poison every gradient it reached."""
    x = prediction()
    assert torch.equal(restorer(x.clone(), inverse=False), x)


def test_output_is_nan_off_sea_and_untouched_on_sea(restorer):
    out = restorer(prediction(), inverse=True).reshape(-1)
    assert torch.isnan(out).tolist() == (~SEA).tolist()
    assert torch.equal(out[torch.from_numpy(SEA)], torch.full((int(SEA.sum()),), 0.5))


def test_wider_tensors_pass_through(restorer):
    """The truth tensor is wider and is deliberately left alone -- its land cells are
    either already NaN or a real coastal observation, and blanking those loses data."""
    truth = prediction(n_predicted=27)
    assert torch.equal(restorer(truth.clone(), inverse=True), truth)


def test_a_mismatched_grid_passes_through(restorer):
    """Guards against silently masking a tensor on some other grid."""
    other = torch.full((1, 1, GRID + 1, N_PREDICTED), 0.5)
    assert torch.equal(restorer(other.clone(), inverse=True), other)


def test_in_place_false_leaves_the_caller_tensor_alone(restorer):
    x = prediction()
    out = restorer(x, in_place=False, inverse=True)
    assert not torch.isnan(x).any()
    assert torch.isnan(out).any()


def test_the_mask_survives_serialisation(restorer, tmp_path):
    """The whole reason this is a processor rather than a training-side output mask: it is
    pickled with the model into the inference checkpoint, so a registered buffer is what
    carries the mask to inference."""
    assert "sea" in restorer.state_dict()

    path = tmp_path / "processor.pt"
    torch.save(restorer, path)
    reloaded = torch.load(path, weights_only=False)

    assert torch.equal(reloaded.sea, restorer.sea)
    out = reloaded(prediction(), inverse=True).reshape(-1)
    assert torch.isnan(out).tolist() == (~SEA).tolist()
