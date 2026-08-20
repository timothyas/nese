"""Losses for observation-supervised state estimation."""

import logging
from typing import Any

import torch
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

from nese_training.observations import normalise_spec
from nese_training.observations import resolve_groups

LOGGER = logging.getLogger(__name__)


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


class ObservationMaskedMSE(FunctionalLoss):
    """``CountMaskedAndWeightedMSE`` over every (variable, group) pair, resolved by name.

    Assumptions:
        * Data on HEALPix so area weighting is free with summation
        * ``nese-dataprep`` naming: observations are ``{observed_as}_{group}`` and their
          counts are ``obs_count_{group}`` (default, also changeable)

    This loss function avoids creating a ``CombinedLoss`` entry for every variable and
    every observing platform. Instead, this loss assumes a naming convention to
    figure them out::

        observations:
          sla:
            observed_as: sla_filtered
            groups: auto
            count: obs_count_sla_{group} # <- this resets the default obs_count_{group} spec

    ``groups: auto`` recovers the set from the variables the model was given. For example,
    a dataset with ``sla_filtered_topex_poseidon``, ``sla_filtered_jason_2`` will recognize
    ``topex_poseidon`` and ``jason_2`` as the observing groups for the field ``sla``, observed as
    ``sla_filtered``.

    The per-group loss is unchanged from ``CountMaskedAndWeightedMSE``, but when aggregated
    per-variable (e.g., when all groups are aggregated to get the SLA loss) only currently
    live missions are taken into account for the given time. On the other hand,
    ``CountMaskedAndWeightedMSE`` averages over the total number of observing missions
    no matter how many were live for the current loss sample
    (via BaseLoss.reduce, over the entire ``variable`` dimension of the array).
    """

    name: str = "observation_masked_mse"
    factory_context_keys = frozenset({LossFactoryContextKey.DATA_INDICES})

    def __init__(
        self,
        observations: dict[str, Any],
        *args,
        data_indices: Any = None,  # noqa: ANN401
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.observations = {name: self._normalise_spec(spec) for name, spec in observations.items()}
        self.predicted_variables: list[str] = []
        self.target_variables: list[str] = []
        self.predicted_indices_by_layout: dict[IndexSpace, list[int]] = {}
        self.target_indices_by_layout: dict[IndexSpace, list[int]] = {}
        if data_indices is not None:
            self.set_data_indices(data_indices)

    _normalise_spec = staticmethod(normalise_spec)

    def _resolve_groups(self, spec: dict, available: dict[str, int]) -> list[str]:
        return resolve_groups(spec, available, type(self).__name__)

    def set_data_indices(self, data_indices: Any) -> "ObservationMaskedMSE":  # noqa: ANN401
        """Resolve every (variable, group) pair into paired column indices."""
        data_full = data_indices.data_full_name_to_position
        data_output = data_indices.data.output.name_to_position
        model_output = data_indices.model.output.name_to_position

        predicted, targets = [], []
        for predicted_name, spec in self.observations.items():
            if predicted_name not in model_output:
                msg = f"Predicted variable '{predicted_name}' is not a model output. Available: {list(model_output)}"
                raise ValueError(msg)

            groups = self._resolve_groups(spec, data_full)
            for group in groups:
                value_name = f"{spec['observed_as']}_{group}"
                count_name = spec["count"].format(group=group)
                missing = [name for name in (value_name, count_name) if name not in data_full]
                if missing:
                    msg = (
                        f"Cannot score '{predicted_name}' against group '{group}': {missing} not in the dataset. "
                        f"Every observation column needs its paired count column."
                    )
                    raise ValueError(msg)
                predicted += [predicted_name, predicted_name]
                targets += [value_name, count_name]

            LOGGER.info(
                "%s: '%s' scored against %d group(s): %s",
                type(self).__name__,
                predicted_name,
                len(groups),
                ", ".join(groups),
            )

        self.predicted_variables, self.target_variables = predicted, targets
        self.predicted_indices_by_layout = self._indices_by_layout(
            predicted,
            {IndexSpace.MODEL_OUTPUT: model_output, IndexSpace.DATA_OUTPUT: data_output, IndexSpace.DATA_FULL: data_full},
        )
        self.target_indices_by_layout = self._indices_by_layout(
            targets,
            {IndexSpace.DATA_OUTPUT: data_output, IndexSpace.DATA_FULL: data_full},
        )
        return self

    @staticmethod
    def _indices_by_layout(
        names: list[str],
        lookups: dict[IndexSpace, dict[str, int]],
    ) -> dict[IndexSpace, list[int]]:
        resolved = {}
        for layout, lookup in lookups.items():
            if all(name in lookup for name in names):
                resolved[layout] = [lookup[name] for name in names]
        return resolved

    @staticmethod
    def _select(
        tensor: torch.Tensor,
        layout: IndexSpace | str | None,
        indices_by_layout: dict[IndexSpace, list[int]],
        role: str,
    ) -> torch.Tensor:
        if layout is None:
            msg = f"ObservationMaskedMSE requires a '{role}_layout' kwarg."
            raise ValueError(msg)
        layout = IndexSpace(layout)
        if layout not in indices_by_layout:
            msg = (
                f"{role}_layout '{layout.value}' cannot address the configured variables. "
                f"Available: {[space.value for space in indices_by_layout]}"
            )
            raise ValueError(msg)
        return tensor[..., indices_by_layout[layout]].contiguous()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        pred_layout: IndexSpace | str | None = None,
        target_layout: IndexSpace | str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.predicted_indices_by_layout:
            msg = f"{type(self).__name__} must be given data_indices before use."
            raise RuntimeError(msg)

        pred = self._select(pred, pred_layout, self.predicted_indices_by_layout, "pred")
        target = self._select(target, target_layout, self.target_indices_by_layout, "target")
        return super().forward(pred, target, squash, **kwargs)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values, counts = target[..., 0::2], target[..., 1::2]
        predicted = pred[..., 0::2]

        mask = (counts > 0).to(pred.dtype)
        mask = mask / mask.mean(dim=TensorDim.GRID, keepdim=True).clamp_min(self.eps)

        # reduce() averages the variable dimension over every declared group; rescale so
        # the divisor counts only the groups that observed something in this sample.
        live = (counts > 0).any(dim=TensorDim.GRID, keepdim=True).to(pred.dtype)
        mask = mask * counts.shape[-1] / live.sum(dim=-1, keepdim=True).clamp_min(1.0)

        return torch.square(predicted - values) * mask
