"""Losses for observation-supervised state estimation."""

import functools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.base import LossFactoryContextKey
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

from nese_training.matern import HEALPixMaternOperator
from nese_training.matern import delta_hat
from nese_training.observations import field_pairs
from nese_training.observations import is_vector
from nese_training.observations import normalise_spec
from nese_training.observations import pool_statistics
from nese_training.observations import resolve_groups
from nese_training.physics import EQUATORIAL_BAND
from nese_training.physics import MIN_NEIGHBOURS
from nese_training.physics import GeostrophicVelocity
from nese_training.physics import geostrophy_defined
from nese_training.physics import read_static

LOGGER = logging.getLogger(__name__)
MAX_NON_FINITE_REPORTS = 3


def _in_float32(calculate_difference):  # noqa: ANN001, ANN202
    """Run ``calculate_difference`` in float32 with autocast off.

    Under 16-mixed precision autocast runs matmuls in float16 whatever their inputs, and at
    hpx8/24h a field's observed (group, cell) count passes float16's 65504 (~85k for sla in
    2019). The inf then meets ``field_membership``'s zeros and NaNs the whole loss.
    """

    @functools.wraps(calculate_difference)
    def wrapper(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return calculate_difference(self, pred.float(), target.float())

    return wrapper


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

    @_in_float32
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


@dataclass(frozen=True)
class _Term:
    """One (field, group) pair and the columns it is scored on.

    ``targets`` is ``(value, count)`` for a scalar field and ``(u, v, count)`` for a vector
    one; ``predicted`` lines up with it column for column.
    """

    field: str
    predicted: list[str]
    targets: list[str]
    vector: bool


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
          uvgxt:
            components: [ugxt, vgxt]     # a vector field, scored cross-track
            weight: 0.9                  # scales this field's term, default 1

    ``groups: auto`` recovers the set from the variables the model was given. For example,
    a dataset with ``sla_filtered_topex_poseidon``, ``sla_filtered_jason_2`` will recognize
    ``topex_poseidon`` and ``jason_2`` as the observing groups for the field ``sla``, observed as
    ``sla_filtered``.

    Each field's loss is pooled over its groups, weighted by observation count: every
    observed (group, cell) pair counts once, i.e. ``sum(err**2) / sum(count > 0)`` over all
    of the field's groups. This is the ``pooled`` score in ``nese-verify``'s metrics, and
    groups that observed nothing in a sample drop out without diluting it. Each field's
    pooled loss is then multiplied by its ``weight`` and averaged over the fields observed in
    the sample, i.e. ``sum(weight * loss) / n_live``. ``field_losses`` records them unweighted.

    A vector field is observed only across the satellite track, so the full predicted
    vector is scored by its component along the observed one, ``((p - o) . n)**2`` with
    ``n = o / |o|``. The sign of ``n`` cancels. The direction has to come from the raw
    observation, so the loss re-derives the pooled statistics from ``dataset`` exactly as
    the normaliser does (:func:`nese_training.observations.pool_statistics`). The shared
    stdev keeps the projection valid in normalised space, and the shared mean cancels in
    ``p - o``. Only cells where the diagnosed velocity is defined are scored
    (:func:`nese_training.physics.geostrophy_defined`: off the equator, where the
    observation is NaN and was imputed, with ``min_neighbours`` for the gradient, and with
    a finite MDT; keep both settings at the physics block's values), and only where the observed speed exceeds
    ``min_speed``, since a still observation has no direction.
    """

    name: str = "observation_masked_mse"
    # Prepended to each field in ``FieldLossLogger``'s keys, so two leaves cannot collide
    field_loss_prefix: str = ""
    factory_context_keys = frozenset({LossFactoryContextKey.DATA_INDICES})
    needs_graph_data: bool = True

    def __init__(
        self,
        observations: dict[str, Any],
        *args,
        data_indices: Any = None,  # noqa: ANN401
        dataset: str | dict | None = None,
        graph_data: Any = None,  # noqa: ANN401
        data_node_name: str | None = None,
        equatorial_band: float = EQUATORIAL_BAND,
        min_neighbours: int = MIN_NEIGHBOURS,
        min_speed: float = 1e-6,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps, self.min_speed = eps, min_speed
        self.observations = {name: self._normalise_spec(spec) for name, spec in observations.items()}
        self.vector_fields = {name: spec for name, spec in self.observations.items() if is_vector(spec)}
        self.predicted_variables: list[str] = []
        self.target_variables: list[str] = []
        self.predicted_indices_by_layout: dict[IndexSpace, list[int]] = {}
        self.target_indices_by_layout: dict[IndexSpace, list[int]] = {}
        # Each field's own pooled loss from the latest call, for logging; see forward.
        self.field_losses: dict[str, torch.Tensor] = {}
        self._only_field: int | None = None
        self.non_finite_reports = 0  # see _report_non_finite
        if self.vector_fields:
            if dataset is None or graph_data is None or data_node_name is None:
                msg = f"{type(self).__name__}: vector fields need `dataset` and the graph for their statistics and latitudes."
                raise ValueError(msg)
            self.dataset = dataset
            lat, lon = np.degrees(np.asarray(graph_data[data_node_name]["x"].cpu(), dtype=np.float64)).T
            static = read_static(dataset, GeostrophicVelocity.static)
            defined = geostrophy_defined(
                lat, lon, min_neighbours=min_neighbours, equatorial_band=equatorial_band, static=static
            )
            self.register_buffer("defined", torch.from_numpy(defined).unsqueeze(-1), persistent=False)
            self.supports_sharding = False
        if data_indices is not None:
            self.set_data_indices(data_indices)

    _normalise_spec = staticmethod(normalise_spec)

    def _resolve_groups(self, spec: dict, available: dict[str, int]) -> list[str]:
        return resolve_groups(spec, available, type(self).__name__)

    def set_data_indices(self, data_indices: Any) -> "ObservationMaskedMSE":  # noqa: ANN401
        """Lay out the columns the loss gathers from ``pred`` and ``target``.

        Every (field, group) pair is one :class:`_Term`. Scalar terms come first, then
        vector terms, so ``calculate_difference`` can split the gathered columns at
        ``n_scalar_columns`` and read them in chunks of 2 and 3.
        """
        data_full = data_indices.data_full_name_to_position
        data_output = data_indices.data.output.name_to_position
        model_output = data_indices.model.output.name_to_position

        terms = [
            term
            for field, spec in self.observations.items()
            for term in self._field_terms(field, spec, data_full, model_output)
        ]
        scalar = [term for term in terms if not term.vector]
        vector = [term for term in terms if term.vector]
        ordered = scalar + vector

        self.n_scalar_columns = sum(len(term.targets) for term in scalar)
        self.predicted_variables = [name for term in ordered for name in term.predicted]
        self.target_variables = [name for term in ordered for name in term.targets]
        self.term_names = [f"{term.field}:{term.targets[0]}" for term in ordered]

        field_names = list(self.observations)
        field_of_term = torch.tensor([field_names.index(term.field) for term in ordered])
        self.register_buffer(
            "field_membership",
            torch.nn.functional.one_hot(field_of_term, len(field_names)).float(),
            persistent=False,
        )
        weights = [float(self.observations[field].get("weight", 1.0)) for field in field_names]
        self.register_buffer("field_weight", torch.tensor(weights), persistent=False)
        if vector:
            self._register_vector_statistics([(term.targets[0], term.targets[1]) for term in vector])

        self.predicted_indices_by_layout = _indices_by_layout(
            self.predicted_variables,
            {IndexSpace.MODEL_OUTPUT: model_output, IndexSpace.DATA_OUTPUT: data_output, IndexSpace.DATA_FULL: data_full},
        )
        self.target_indices_by_layout = _indices_by_layout(
            self.target_variables,
            {IndexSpace.DATA_OUTPUT: data_output, IndexSpace.DATA_FULL: data_full},
        )
        return self

    def _field_terms(
        self, field: str, spec: dict, data_full: dict[str, int], model_output: dict[str, int]
    ) -> list["_Term"]:
        """One term per group observing ``field``, checked against the dataset.

        "term" is basically the loss-side terminology for groups, since we could have a common
        group for different fields, so "term" is associated with each field and group.
        """
        pairs = field_pairs(field, spec)
        outputs = [output for output, _ in pairs]
        unknown = [output for output in outputs if output not in model_output]
        if unknown:
            msg = f"Predicted variables {unknown} are not model outputs. Available: {list(model_output)}"
            raise ValueError(msg)

        groups = self._resolve_groups(spec, data_full)
        terms = []
        for group in groups:
            targets = [f"{observed_as}_{group}" for _, observed_as in pairs] + [spec["count"].format(group=group)]
            missing = [name for name in targets if name not in data_full]
            if missing:
                msg = (
                    f"Cannot score '{field}' against group '{group}': {missing} not in the dataset. "
                    f"Every observation column needs its paired count column."
                )
                raise ValueError(msg)
            # The count column has no prediction; pad so pred and target gather the same width.
            terms.append(_Term(field, [*outputs, outputs[0]], targets, is_vector(spec)))

        LOGGER.info(
            "%s: '%s' scored against %d group(s): %s",
            type(self).__name__,
            field,
            len(groups),
            ", ".join(groups),
        )
        return terms

    def _register_vector_statistics(self, observed: list[tuple[str, str]]) -> None:
        """The pooled (mean_u, mean_v) and shared stdev behind each vector term's (u, v) columns."""
        store = open_dataset(self.dataset)
        name_to_index = store.name_to_index
        stats = pool_statistics(self.vector_fields, name_to_index, store.statistics, type(self).__name__)
        mean = [[stats["mean"][name_to_index[u]], stats["mean"][name_to_index[v]]] for u, v in observed]

        # At this point, pool_statistics has already computed a common stdev for both components
        # σ = √((σ_u² + σ_v²)/2)
        stdev = [stats["stdev"][name_to_index[u]] for u, _ in observed]
        self.register_buffer("vector_mean", torch.tensor(mean, dtype=torch.float32).T, persistent=False)
        self.register_buffer("vector_stdev", torch.tensor(stdev, dtype=torch.float32), persistent=False)

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
        self._difference_finite = True
        loss = super().forward(pred, target, squash, **kwargs)
        if self._difference_finite and not torch.isfinite(loss).all() and self.non_finite_reports < MAX_NON_FINITE_REPORTS:
            self.non_finite_reports += 1
            LOGGER.warning("%s: difference finite but loss %s after anemoi's scale/reduce", type(self).__name__, loss)
        if squash and len(self.observations) > 1:
            self._record_field_losses(pred, target, **kwargs)
        return loss

    @torch.no_grad()
    def _record_field_losses(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        """Each field's pooled loss on its own, for :class:`nese_training.diagnostics.FieldLossLogger`.

        The same reduction as the total, with every other field's columns masked out, so a
        field's value is exactly what it contributes before the average over live fields. A
        field nobody observed in the batch comes out zero. Recomputed without gradients, so
        training is unaffected.
        """
        try:
            for index, field in enumerate(self.observations):
                self._only_field = index
                self.field_losses[field] = super().forward(pred, target, True, **kwargs).detach()
        finally:
            self._only_field = None

    def _scalar_terms(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, counts = target[..., 0::2], target[..., 1::2]
        return torch.square(pred[..., 0::2] - values), (counts > 0).to(pred.dtype)

    def _vector_terms(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed_u, observed_v, counts = target[..., 0::3], target[..., 1::3], target[..., 2::3]
        raw_u = observed_u * self.vector_stdev + self.vector_mean[0]
        raw_v = observed_v * self.vector_stdev + self.vector_mean[1]
        speed = torch.hypot(raw_u, raw_v)
        normal_u, normal_v = raw_u / speed.clamp_min(self.eps), raw_v / speed.clamp_min(self.eps)

        error = (pred[..., 0::3] - observed_u) * normal_u + (pred[..., 1::3] - observed_v) * normal_v
        mask = (counts > 0) & (speed > self.min_speed) & self.defined
        return torch.square(error), mask.to(pred.dtype)

    def _masked_terms(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each term's squared error and the mask of cells it is scored on, before pooling."""
        split = self.n_scalar_columns
        terms = [self._scalar_terms(pred[..., :split], target[..., :split])]
        if self.vector_fields:
            terms.append(self._vector_terms(pred[..., split:], target[..., split:]))
        return torch.cat([error for error, _ in terms], dim=-1), torch.cat([mask for _, mask in terms], dim=-1)

    @_in_float32
    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared_error, mask = self._masked_terms(pred, target)

        # membership.shape = (n_terms, n_fields)
        membership = self.field_membership.to(dtype=mask.dtype)
        if self._only_field is not None:
            mask = mask * membership[:, self._only_field]
        field_coverage = _coverage_divisor(self, mask, 0.0) @ membership
        live_fields = (field_coverage > 0).sum(dim=-1, keepdim=True).clamp_min(1)

        # reduce() averages over every (field, group) column; rescale so each field's columns
        # sum to its pooled mean and the average runs over the live fields only.
        column_coverage = (field_coverage @ membership.T).clamp_min(self.eps)
        mask = mask / column_coverage * mask.shape[-1] / live_fields
        # The per-field values recorded for logging stay unweighted
        if self._only_field is None:
            mask = mask * (membership @ self.field_weight.to(dtype=mask.dtype))

        difference = squared_error * mask
        if not torch.isfinite(difference).all():
            self._difference_finite = False
            self._report_non_finite(pred, target, squared_error, mask, field_coverage, column_coverage)
        return difference

    @torch.no_grad()
    def _report_non_finite(self, pred, target, squared_error, mask, field_coverage, column_coverage) -> None:  # noqa: ANN001, PLR0913
        """Log which intermediate of :meth:`calculate_difference` first went non-finite, by term.

        Costs one ``isfinite`` reduction per call while everything is finite; reports at most
        ``MAX_NON_FINITE_REPORTS`` times per process.
        """
        if self.non_finite_reports >= MAX_NON_FINITE_REPORTS:
            return
        self.non_finite_reports += 1

        def bad_terms(tensor: torch.Tensor) -> dict[str, int]:
            counts = (~torch.isfinite(tensor)).flatten(end_dim=-2).sum(dim=0).tolist()
            return {self.term_names[i]: int(n) for i, n in enumerate(counts) if n}

        def bad_columns(tensor: torch.Tensor, names: list[str]) -> dict[str, int]:
            counts = (~torch.isfinite(tensor)).flatten(end_dim=-2).sum(dim=0).tolist()
            return {names[i]: int(n) for i, n in enumerate(counts) if n}

        fields = list(self.observations)
        lines = [
            f"{type(self).__name__}: non-finite difference (only_field={self._only_field})",
            f"  pred {pred.dtype} non-finite {bad_columns(pred, self.predicted_variables) or 'none'}, max |pred| {float(pred.nan_to_num().abs().max())}",
            f"  target {target.dtype} non-finite {bad_columns(target, self.target_variables) or 'none'}, max |target| {float(target.nan_to_num().abs().max())}",
            f"  squared_error non-finite {bad_terms(squared_error) or 'none'}",
            f"  mask non-finite {bad_terms(mask) or 'none'}",
            f"  field coverage {dict(zip(fields, field_coverage.flatten(end_dim=-2).tolist()[0]))}",
            f"  column coverage non-finite {bad_terms(column_coverage) or 'none'}, min {float(column_coverage.min())}",
        ]
        LOGGER.warning("\n".join(lines))


class UnseenObservationMaskedMSE(ObservationMaskedMSE):
    """``ObservationMaskedMSE`` scored only on target observations at cells unseen in the input.

    A cell is seen for a field when any of that field's groups observed it at any input step,
    i.e. the complement of nese-verify's ``seen_at_hour == +timestep`` split. Each field gets
    its own seen mask, read from its ``count`` columns. The loss is called with the target
    time only, so the training method has to hand it the inputs first with :meth:`set_inputs`;
    see :class:`nese_training.methods.UnseenAwareTraining`.
    """

    name: str = "unseen_observation_masked_mse"
    field_loss_prefix: str = "unseen_"
    seen: torch.Tensor | None = None

    def set_data_indices(self, data_indices: Any) -> "UnseenObservationMaskedMSE":  # noqa: ANN401
        super().set_data_indices(data_indices)
        data_input = data_indices.data.input.name_to_position
        columns, field_of_column = [], []
        for index, (field, spec) in enumerate(self.observations.items()):
            groups = self._resolve_groups(spec, data_indices.data_full_name_to_position)
            counts = [spec["count"].format(group=group) for group in groups]
            missing = [name for name in counts if name not in data_input]
            if missing:
                msg = f"{type(self).__name__}: '{field}' needs its counts {missing} as model inputs."
                raise ValueError(msg)
            columns += [data_input[name] for name in counts]
            field_of_column += [index] * len(counts)

        self.register_buffer("input_count_columns", torch.tensor(columns), persistent=False)
        self.register_buffer(
            "input_count_membership",
            torch.nn.functional.one_hot(torch.tensor(field_of_column), len(self.observations)).float(),
            persistent=False,
        )
        return self

    @torch.no_grad()
    def set_inputs(self, x: torch.Tensor) -> None:
        """Record, per field, where any group observed at any input step.

        ``x`` is the model input, ``(batch, time, [ensemble,] grid, variable)``. Counts are
        max-normalised and never imputed, so ``> 0`` still means observed.
        """
        observed = (x[..., self.input_count_columns] > 0).any(dim=1).float()
        seen = observed @ self.input_count_membership > 0
        self.seen = seen if seen.ndim == 4 else seen.unsqueeze(1)

    def _masked_terms(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.seen is None:
            msg = f"{type(self).__name__} needs set_inputs before use; train with nese_training.methods.UnseenAwareTraining."
            raise RuntimeError(msg)
        if self.seen.shape[-2] != pred.shape[-2]:
            msg = f"{type(self).__name__}: seen mask has {self.seen.shape[-2]} cells, prediction {pred.shape[-2]}."
            raise ValueError(msg)

        squared_error, mask = super()._masked_terms(pred, target)
        seen_by_term = self.seen.to(mask.dtype) @ self.field_membership.to(mask.dtype).T
        seen_by_term = seen_by_term.reshape(seen_by_term.shape[0], *[1] * (mask.ndim - 4), *seen_by_term.shape[1:])
        return squared_error, mask * (1 - seen_by_term)


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
