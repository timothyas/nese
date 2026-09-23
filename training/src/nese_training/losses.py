"""Losses for observation-supervised state estimation."""

import logging
from typing import Any

import numpy as np
import torch
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

from nese_training.matern import HEALPixMaternOperator
from nese_training.matern import delta_hat
from nese_training.observations import normalise_spec
from nese_training.observations import resolve_groups

LOGGER = logging.getLogger(__name__)


def _coverage_divisor(loss: FunctionalLoss, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """The grid-weighted coverage of ``mask``, i.e. what ``reduce`` will actually sum to.

    ``BaseLoss.reduce`` computes ``sum_i(d_i * w_i)`` over the grid, so a constant per-cell
    error ``e`` scored through ``d_i = e**2 * m_i / s`` gives ``e**2 * sum(m_i w_i) / s``.
    Coverage invariance therefore needs ``s = sum_i(m_i * w_i)``, not ``mean(m)``.

    The two agree exactly while ``w`` is uniform over every cell, which is why ``mean(m)``
    was right until the output mask arrived: HEALPix cells are equal-area, so unit-sum
    ``area_weight`` gives ``w_i = 1/N``. Masking land renormalises ``w`` over the sea cells
    alone, and ``mean(m)`` then both mis-scales the loss and reintroduces the coverage
    dependence this division exists to remove -- it counts observations on land that carry
    zero weight, and that fraction moves from sample to sample.

    With no grid scaler attached the divisor is ``sum(m)``: there are no weights to sum,
    and ``reduce``'s bare grid sum contributes no ``1/N`` of its own, so
    ``sum(e**2 m) / sum(m)`` is again the mean over observed cells -- the same number a
    unit-sum ``node_weights`` scaler gives. That is the configuration a HEALPix run wants,
    since equal-area cells make the node weight a vector of ones and the graph attribute
    exists only to be summed and divided back out. The equivalence is exactly what equal
    areas buy: a mesh with unequal cells needs a real grid scaler, not this branch.
    """
    weights = loss.scaler.subset_by_dim(TensorDim.GRID.value).tensors
    if not weights:
        return mask.sum(dim=TensorDim.GRID, keepdim=True).clamp_min(eps)

    grid_weight = None
    for _, tensor in weights.values():
        grid_weight = tensor if grid_weight is None else grid_weight * tensor

    grid_weight = grid_weight.to(device=mask.device, dtype=mask.dtype).reshape(-1, 1)
    return (mask * grid_weight).sum(dim=TensorDim.GRID, keepdim=True).clamp_min(eps)


def _indices_by_layout(
    names: list[str],
    lookups: dict[IndexSpace, dict[str, int]],
) -> dict[IndexSpace, list[int]]:
    """Resolve ``names`` into column indices, keeping only layouts that address them all."""
    resolved = {}
    for layout, lookup in lookups.items():
        if all(name in lookup for name in names):
            resolved[layout] = [lookup[name] for name in names]
    return resolved


def _select(
    tensor: torch.Tensor,
    layout: IndexSpace | str | None,
    indices_by_layout: dict[IndexSpace, list[int]],
    role: str,
    owner: str,
) -> torch.Tensor:
    """Index ``tensor``'s variable axis for the given layout."""
    if layout is None:
        msg = f"{owner} requires a '{role}_layout' kwarg."
        raise ValueError(msg)
    layout = IndexSpace(layout)
    if layout not in indices_by_layout:
        msg = (
            f"{role}_layout '{layout.value}' cannot address the configured variables. "
            f"Available: {[space.value for space in indices_by_layout]}"
        )
        raise ValueError(msg)
    return tensor[..., indices_by_layout[layout]].contiguous()


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
        mask = mask / _coverage_divisor(self, mask, self.eps)

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

    Each field's loss is pooled over its groups, weighted by observation count: every
    observed (group, cell) pair counts once, i.e. ``sum(err**2) / sum(count > 0)`` over all
    of the field's groups. This is the ``pooled`` score in ``nese-verify``'s metrics, and
    groups that observed nothing in a sample drop out without diluting it. Fields are then
    weighted equally, averaging over those observed in the sample.
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
        fields = list(self.observations)
        field_of_column = torch.tensor([fields.index(name) for name in predicted[0::2]])
        self.register_buffer(
            "field_membership",
            torch.nn.functional.one_hot(field_of_column, len(fields)).float(),
            persistent=False,
        )
        self.predicted_indices_by_layout = _indices_by_layout(
            predicted,
            {IndexSpace.MODEL_OUTPUT: model_output, IndexSpace.DATA_OUTPUT: data_output, IndexSpace.DATA_FULL: data_full},
        )
        self.target_indices_by_layout = _indices_by_layout(
            targets,
            {IndexSpace.DATA_OUTPUT: data_output, IndexSpace.DATA_FULL: data_full},
        )
        return self

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

        pred = _select(pred, pred_layout, self.predicted_indices_by_layout, "pred", type(self).__name__)
        target = _select(
            target, target_layout, self.target_indices_by_layout, "target", type(self).__name__
        )
        return super().forward(pred, target, squash, **kwargs)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values, counts = target[..., 0::2], target[..., 1::2]
        predicted = pred[..., 0::2]

        mask = (counts > 0).to(pred.dtype)
        membership = self.field_membership.to(dtype=mask.dtype)
        field_coverage = _coverage_divisor(self, mask, 0.0) @ membership
        live_fields = (field_coverage > 0).sum(dim=-1, keepdim=True).clamp_min(1)

        # reduce() averages over every (field, group) column; rescale so each field's columns
        # sum to its pooled mean and the average runs over the live fields only.
        column_coverage = (field_coverage @ membership.T).clamp_min(self.eps)
        mask = mask / column_coverage * mask.shape[-1] / live_fields

        return torch.square(predicted - values) * mask


class MaternPenalty(BaseLoss):
    """Matérn-class smoothness prior on the full predicted field.

    The background term of a variational cost function, ``x^T C^-1 x`` with ``x_b = 0``,
    which supplies a gradient at the cells no mission observed -- where the data term is
    exactly zero. Derived and validated in
    ``experimental/local-sla-16yr/matern-regularization``::

        L_M = || A_hat^M Sigma_hat x ||^2 / N

    Note ``Sigma_hat`` acts on ``x`` *before* the operator, not after: with
    ``C^{1/2} = Lambda_hat A_hat^-M`` the inverse square root is ``A_hat^M Sigma_hat``.

    ``x`` is the model output as the loss receives it, i.e. in data-normalised space.
    Nothing here de-normalises, and nothing upstream does either: ``post_processors`` run
    only in ``_compute_metrics`` for validation metrics, never in ``_compute_loss``.

    Overall strength comes from ``CombinedLoss.loss_weights``, not from a parameter here.
    The ``1/2`` of the variational form is likewise omitted, since the paired data term is
    a plain MSE without one; fold it into the weight if wanted.
    """

    name: str = "matern_penalty"
    needs_graph_data: bool = True
    factory_context_keys = frozenset({LossFactoryContextKey.DATA_INDICES})

    def __init__(
        self,
        *args: Any,
        order: int = 1,
        rho_hat: float = 4.0,
        min_neighbours: int = 3,
        variables: dict[str, Any] | None = None,
        n_randomisation: int = 1000,
        seed: int = 0,
        graph_data: Any = None,  # noqa: ANN401
        data_node_name: str | None = None,
        data_indices: Any = None,  # noqa: ANN401
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.order, self.rho_hat, self.min_neighbours = order, rho_hat, min_neighbours
        self.variables = dict(variables or {})
        self.n_randomisation, self.seed = n_randomisation, seed
        self.supports_sharding = False

        if graph_data is None or data_node_name is None:
            msg = f"{type(self).__name__} needs graph_data to build its stencil."
            raise ValueError(msg)

        coords = np.degrees(np.asarray(graph_data[data_node_name]["x"].cpu()))
        self.operator = HEALPixMaternOperator(
            coords[:, 0], coords[:, 1], min_neighbours=min_neighbours
        )
        self.penalised_variables: list[str] = []
        self.indices_by_layout: dict[IndexSpace, list[int]] = {}
        if data_indices is not None:
            self.set_data_indices(data_indices)

    def set_data_indices(self, data_indices: Any) -> "MaternPenalty":  # noqa: ANN401
        """Resolve which model outputs to penalise, and normalise the operator for each."""
        model_output = data_indices.model.output.name_to_position
        unknown = set(self.variables) - set(model_output)
        if unknown:
            msg = f"Not model outputs: {sorted(unknown)}. Available: {list(model_output)}"
            raise ValueError(msg)

        names = list(self.variables) if self.variables else list(model_output)
        specs = [self.variables.get(name, {}) for name in names]
        orders = [int(spec.get("order", self.order)) for spec in specs]
        rho_hats = [float(spec.get("rho_hat", self.rho_hat)) for spec in specs]

        self.penalised_variables = names
        self.indices_by_layout = _indices_by_layout(
            names,
            {
                IndexSpace.MODEL_OUTPUT: model_output,
                IndexSpace.DATA_OUTPUT: data_indices.data.output.name_to_position,
                IndexSpace.DATA_FULL: data_indices.data_full_name_to_position,
            },
        )
        self.register_buffer(
            "delta",
            torch.tensor([delta_hat(r, m) for r, m in zip(rho_hats, orders, strict=True)]),
        )
        self.register_buffer("order_per_variable", torch.tensor(orders))
        self.operator.estimate_normalisation(
            rho_hats, orders, n_samples=self.n_randomisation, seed=self.seed
        )
        LOGGER.info(
            "%s: penalising %s with rho_hat=%s, order=%s, min_neighbours=%d",
            type(self).__name__,
            ", ".join(names),
            rho_hats,
            orders,
            self.min_neighbours,
        )
        return self

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,  # noqa: ARG002 -- a prior has no target
        squash: bool = True,
        *,
        pred_layout: IndexSpace | str | None = None,
        group: Any = None,  # noqa: ANN401
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.indices_by_layout:
            msg = f"{type(self).__name__} must be given data_indices before use."
            raise RuntimeError(msg)

        x = _select(pred, pred_layout, self.indices_by_layout, "pred", type(self).__name__)
        scaled = self.operator.sigma * x
        penalty = self.operator(scaled, self.delta, self.order_per_variable)

        active = self.operator.active.unsqueeze(-1)
        # reduce() *sums* over GRID, so divide by the cell count to leave a per-cell mean;
        # without this the weight would carry a hidden factor of N.
        out = torch.square(penalty) * active / active.sum().clamp_min(1)
        return self.reduce(out, squash, group=group)
