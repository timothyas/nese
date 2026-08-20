"""The masking claim in CountMaskedMSE, and the coverage claim that separates it from
CountMaskedAndWeightedMSE.

The loss is what keeps the fill value the imputer writes into unobserved cells out of
the gradient. If that is wrong it fails silently -- training still runs, it just
regresses toward the climatological mean everywhere the satellites did not look.

The two classes differ in exactly one line -- the weighted one divides the mask by its
own grid mean -- so the tests pin that difference from both sides rather than assuming
either behaviour.
"""

import pytest
import torch
from anemoi.training.utils.enums import TensorDim
from nese_training.losses import CountMaskedAndWeightedMSE
from nese_training.losses import CountMaskedMSE

GRID = 64


def make_loss(cls: type = CountMaskedMSE) -> CountMaskedMSE:
    loss = cls()
    # Matches the node_weights scaler in config/training/scalers/nese.yaml, which uses
    # norm: unit-sum. BaseLoss.reduce sums over the grid, so the weights must sum to 1
    # for the reduction to read as a mean.
    loss.add_scaler(dimension=TensorDim.GRID.value, scaler=torch.ones(GRID) / GRID, name="node_weights")
    return loss


def make_batch(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(pred, target) with target laid out as paired (value, count) columns."""
    shape = (1, 1, 1, GRID, 1)
    values = torch.zeros(shape)
    counts = observed.to(torch.float32).reshape(shape)
    pred = torch.full(shape, 0.5).repeat(1, 1, 1, 1, 2)
    target = torch.cat([values, counts], dim=-1)
    return pred, target


def test_unobserved_cells_do_not_affect_the_loss() -> None:
    observed = torch.zeros(GRID, dtype=torch.bool)
    observed[:16] = True

    loss = make_loss()
    pred, target = make_batch(observed)
    before = loss(pred, target)

    # Whatever the imputer wrote into the unobserved cells must not matter.
    target[..., ~observed, 0] = 1e6
    after = loss(pred, target)

    assert torch.allclose(before, after), f"{before.item()} != {after.item()}"


def test_observed_cells_do_affect_the_loss() -> None:
    observed = torch.zeros(GRID, dtype=torch.bool)
    observed[:16] = True

    loss = make_loss()
    pred, target = make_batch(observed)
    before = loss(pred, target)

    target[..., observed, 0] = 3.0
    after = loss(pred, target)

    assert not torch.allclose(before, after)


@pytest.mark.parametrize("n_observed", [4, 16, 32, 64])
def test_weighted_loss_is_invariant_to_coverage(n_observed: int) -> None:
    """Coverage swings by an order of magnitude across the altimetry record, so a
    constant per-cell error must give a constant loss regardless of how many cells
    were seen -- otherwise gradients are biased toward well-observed epochs.

    This is the whole reason CountMaskedAndWeightedMSE exists; the plain variant below
    is the behaviour it corrects."""
    observed = torch.zeros(GRID, dtype=torch.bool)
    observed[:n_observed] = True

    loss = make_loss(CountMaskedAndWeightedMSE)
    pred, target = make_batch(observed)

    assert torch.allclose(loss(pred, target), torch.tensor(0.25), atol=1e-5)


@pytest.mark.parametrize("n_observed", [4, 16, 32, 64])
def test_unweighted_loss_scales_with_coverage(n_observed: int) -> None:
    """The counterpart: BaseLoss.reduce sums over the grid with no denominator, so
    without the mask.mean() division the magnitude tracks how much of the mesh was seen.

    Pinned rather than merely noted, because it is what makes CountMaskedMSE the wrong
    choice for a multi-mission run and the two easy to confuse."""
    observed = torch.zeros(GRID, dtype=torch.bool)
    observed[:n_observed] = True

    loss = make_loss(CountMaskedMSE)
    pred, target = make_batch(observed)

    expected = 0.25 * n_observed / GRID
    assert torch.allclose(loss(pred, target), torch.tensor(expected), atol=1e-5)


def test_odd_column_count_is_rejected() -> None:
    loss = make_loss()
    pred = torch.zeros(1, 1, 1, GRID, 1)
    target = torch.zeros(1, 1, 1, GRID, 1)

    with pytest.raises(ValueError, match="paired"):
        loss(pred, target)
