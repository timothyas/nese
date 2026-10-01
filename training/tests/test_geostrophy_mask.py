"""NaN on the way out wherever the diagnosed geostrophic velocity is undefined.

The loss already ignores those cells, so nothing fails if this is wrong -- the plots and
inference output would just show a mean flow, a zero or an imputed fill where the velocity
means nothing. The invariants: training never sees a NaN, exactly the undefined cells of
exactly the velocity columns are blanked, in both the prediction and the truth, and the
mask travels with the checkpoint.
"""

import numpy as np
import pytest
import torch
from nese_training import processors
from nese_training.processors import GeostrophyMaskPostprocessor

DEFINED = np.array([True, True, False, True, False, True])
GRID = DEFINED.size
MODEL_OUTPUT = ["sla", "ugxt", "vgxt"]
DATA_OUTPUT = ["obs_count_jason_3", "sla", "sla_filtered_jason_3", "ugxt", "ugxt_jason_3", "vgxt", "vgxt_jason_3"]


class _Index:
    def __init__(self, names: list[str], name_to_index: dict[str, int]):
        self.name_to_index = name_to_index
        self.full = torch.tensor([name_to_index[name] for name in names])


class _Indices:
    def __init__(self, names: list[str], name_to_index: dict[str, int]):
        self.output = _Index(names, name_to_index)


class FakeIndices:
    """Real classes, not ``type(...)``, so the processor pickles with them."""

    def __init__(self):
        data_full = {name: i for i, name in enumerate(["elevation", *DATA_OUTPUT])}
        self.model = _Indices(MODEL_OUTPUT, {name: i for i, name in enumerate(MODEL_OUTPUT)})
        self.data = _Indices(DATA_OUTPUT, data_full)


@pytest.fixture
def mask(monkeypatch):
    monkeypatch.setattr(processors, "store_geostrophy_defined", lambda *_a, **_k: DEFINED.copy())
    return GeostrophyMaskPostprocessor(dataset="ignored", data_indices=FakeIndices())


def tensor(width: int) -> torch.Tensor:
    return torch.full((1, 1, GRID, width), 0.5)


def test_forward_pass_is_a_no_op(mask):
    x = tensor(len(MODEL_OUTPUT))
    assert torch.equal(mask(x.clone(), inverse=False), x)


@pytest.mark.parametrize("names", [MODEL_OUTPUT, DATA_OUTPUT], ids=["prediction", "truth"])
def test_only_velocity_columns_are_blanked_at_undefined_cells(mask, names):
    out = mask(tensor(len(names)), inverse=True)[0, 0]
    for col, name in enumerate(names):
        expected = ~DEFINED if name in ("ugxt", "vgxt") else np.zeros(GRID, dtype=bool)
        assert torch.isnan(out[:, col]).tolist() == expected.tolist(), name


def test_other_shapes_pass_through(mask):
    for x in (tensor(4), torch.full((1, 1, GRID + 1, len(MODEL_OUTPUT)), 0.5)):
        assert torch.equal(mask(x.clone(), inverse=True), x)


def test_in_place_false_leaves_the_caller_tensor_alone(mask):
    x = tensor(len(MODEL_OUTPUT))
    out = mask(x, in_place=False, inverse=True)
    assert not torch.isnan(x).any()
    assert torch.isnan(out).any()


def test_the_mask_survives_serialisation(mask, tmp_path):
    path = tmp_path / "processor.pt"
    torch.save(mask, path)
    reloaded = torch.load(path, weights_only=False)
    assert torch.equal(reloaded.defined, mask.defined)
