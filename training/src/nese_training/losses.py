"""Losses for observation-supervised state estimation."""

import torch
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim


class CountMaskedMSE(FunctionalLoss):
    """MSE scored only where a mission actually observed, at the target time.

    Assumptions:
        * Data on HEALPix so area weighting is free with summation

    The model emits one dense ``sla`` field but the only truth available is sparse
    per-mission observations, so every loss term needs its own coverage mask. The
    existing machinery cannot supply one: ``NaNMaskScaler`` reads
    ``loss_mask_training`` off the imputer, which builds it from ``nan_locations[:, 0]``
    -- timestep 0, i.e. the oldest input bin rather than the target bin -- and the
    scaler is keyed on model-output variables, of which there is only one. Scaler
    callbacks never receive the batch either, so the mask cannot be recovered there.

    Instead the mask rides in on the target tensor. The loss is configured with a
    paired variable list so that each column of ``target`` arrives alongside its
    observation count::

        predicted_variables: [sla, sla]
        target_variables:    [sla_filtered_ers_2, obs_count_ers_2]

    Even columns carry values, odd columns carry counts. Counts are ``max``-normalised
    (``x / maximum``, no additive shift), so zero survives normalisation and
    ``count > 0`` remains exactly "this mission observed this cell in this bin".
    """

    name: str = "count_masked_mse"

    def __init__(self, *args, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] % 2 != 0:
            msg = (
                f"{type(self).__name__} expects target variables paired as "
                f"(value, count); got {target.shape[-1]} columns."
            )
            raise ValueError(msg)

        values, counts = target[..., 0::2], target[..., 1::2]
        predicted = pred[..., 0::2]

        mask = (counts > 0).to(pred.dtype)

        return torch.square(predicted - values) * mask


class CountMaskedAndWeightedMSE(FunctionalLoss):
    """Same as CountMaskedMSE, except weighted by nonzero obs-count mean

    Assumptions:
        * Data on HEALPix so area weighting is free with summation
        * Weight by nonzero obs-count mean (i.e., mask.mean() where mask = obs_count>0)
          In order to de-bias well observed time periods

    Coverage varies strongly over the record -- one mission in the early 1990s versus
    ten today, and roughly 19% of the mesh per bin here. Since ``BaseLoss.reduce``
    sums over the grid with no denominator correction, an unnormalised mask would make
    the loss magnitude scale with coverage and bias gradients toward well-observed
    epochs. Dividing the mask by its own grid mean fixes the effective weight sum at
    one. HEALPix cells are equal-area, so the node weights are uniform and this
    normalisation is exact rather than approximate.
    """

    name: str = "count_masked_and_weighted_mse"

    def __init__(self, *args, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] % 2 != 0:
            msg = (
                f"{type(self).__name__} expects target variables paired as "
                f"(value, count); got {target.shape[-1]} columns."
            )
            raise ValueError(msg)

        values, counts = target[..., 0::2], target[..., 1::2]
        predicted = pred[..., 0::2]

        mask = (counts > 0).to(pred.dtype)
        mask = mask / mask.mean(dim=TensorDim.GRID, keepdim=True).clamp_min(self.eps)

        return torch.square(predicted - values) * mask
