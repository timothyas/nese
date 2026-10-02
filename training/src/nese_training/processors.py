"""Preprocessors for observation-supervised state estimation."""

import torch
from anemoi.models.preprocessing.imputer import InputImputer
from anemoi.models.preprocessing.normalizer import InputNormalizer
from torch import nn

from nese_training.masks import variable_below
from nese_training.physics import EQUATORIAL_BAND
from nese_training.physics import MIN_NEIGHBOURS
from nese_training.physics import store_geostrophy_defined
from nese_training.observations import field_pairs
from nese_training.observations import normalise_spec
from nese_training.observations import pool_statistics
from nese_training.observations import resolve_groups


class ObsCountNaNRestorer(nn.Module):
    """Restore observation gaps on the inverse pass, using the paired obs-count channel.

    ``InputImputer`` fills unobserved cells with the field mean so the encoder never sees
    a NaN, and undoes that on the inverse by writing NaN back at ``nan_locations``. That
    undo never runs for ``sla``: it is a diagnostic, so it is absent from ``model.input``,
    its ``index_inference_input`` entry is None, and ``BaseImputer.inverse_transform``
    skips any pair with a None index. anemoi says as much at startup --
    "If placement of NaNs for diagnostic variables in inference output is desired, this
    needs to be handled by postprocessors: sla" -- and this is that postprocessor.

    The imputer's own mask would be the wrong one to reuse anyway: it is captured from
    ``nan_locations[:, 0]``, the oldest *input* bin, whereas the tensors being plotted
    span the target bin, and satellite tracks move between bins. The obs counts are
    per-bin and travel inside the tensor itself, so they mask each timestep correctly.
    A value's count above zero is exactly equivalent to "value is not NaN" (verified for
    ``sla_filtered``: 0 disagreements in 491520 cells; the velocity has a count of its own
    for the same reason), which makes the count an exact stand-in for the missing NaN mask.

    Only tensors shaped like ``data.output.full`` are touched, because only those carry
    the count columns. Model-output tensors pass through untouched, so predictions stay
    dense and only the truth regains its gaps -- which is the comparison the plots want.

    ``composed`` additionally rebuilds a predicted field's truth column as the count-
    weighted mean of every mission that observed the bin; see :meth:`_compose`. Without
    it that column holds a single mission, which for this archive means a blank truth
    panel on 56% of dates and a sixth of the coverage the loss actually scores against.
    Both settings take the same spec as the loss's ``observations``, so one config block
    feeds the loss, the pooled statistics and this.
    """

    def __init__(
        self,
        masked_by: dict[str, str],
        data_indices=None,  # noqa: ANN001
        statistics: dict | None = None,
        composed: dict | None = None,
    ) -> None:
        super().__init__()
        self.masked_by = dict(masked_by)
        self.composed = dict(composed or {})
        self.data_indices = data_indices

    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **_kwargs) -> torch.Tensor:
        if not inverse:
            return x if in_place else x.clone()
        return self.inverse_transform(x, in_place=in_place)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **_kwargs) -> torch.Tensor:
        output_full = self.data_indices.data.output.full.tolist()
        if x.shape[-1] != len(output_full):
            return x if in_place else x.clone()

        if not in_place:
            x = x.clone()

        column = {data_index: col for col, data_index in enumerate(output_full)}
        name_to_index = self.data_indices.data.output.name_to_index

        for value_name, count_name in self.masked_by.items():
            value_col = column.get(name_to_index.get(value_name))
            count_col = column.get(name_to_index.get(count_name))
            if value_col is None or count_col is None:
                continue
            observed = x[..., count_col] > 0
            x[..., value_col] = torch.where(observed, x[..., value_col], torch.nan)

        return self._compose(x, column, name_to_index)

    def _compose(self, x: torch.Tensor, column: dict, name_to_index: dict) -> torch.Tensor:
        """Fill a predicted field's truth column from every mission that observed the bin.

        The column arrives holding one mission's observations, which is what the dataset
        can offer -- anemoi needs a diagnostic to exist on disk, and merging the missions
        there would pre-empt the thing the model is being trained to do. So the merge
        happens here, on the way out, where it reaches the plots and nothing else.

        Weighting is by obs count, and the missions being combined are simultaneous, so
        they are measuring one sea state rather than different epochs of a rising one
        (co-located same-day pairs differ by 0.21 sigma, against 0.44 sigma between the
        record's extreme eras). Cells nobody saw stay NaN, so the truth keeps its gaps
        and only the prediction is dense.
        """
        for field, raw in self.composed.items():
            spec = normalise_spec(raw)
            groups = resolve_groups(spec, name_to_index, type(self).__name__)
            for output, observed_as in field_pairs(field, spec):
                field_col = column.get(name_to_index.get(output))
                if field_col is None:
                    continue
                observations = [(f"{observed_as}_{group}", spec["count"].format(group=group)) for group in groups]
                x[..., field_col] = self._count_weighted_mean(x, column, name_to_index, observations)

        return x

    @staticmethod
    def _count_weighted_mean(x: torch.Tensor, column: dict, name_to_index: dict, observations: list) -> torch.Tensor:
        """Count-weighted mean of the ``(value, count)`` columns, NaN where nobody observed."""
        weighted = torch.zeros_like(x[..., 0])
        weights = torch.zeros_like(weighted)
        for value_name, count_name in observations:
            value_col = column.get(name_to_index.get(value_name))
            count_col = column.get(name_to_index.get(count_name))
            if value_col is None or count_col is None:
                continue
            count = x[..., count_col].clamp_min(0)
            observed = count > 0
            # The gaps are NaN by now, and NaN*0 is NaN rather than 0, so they have to
            # be cleared before they enter the sum rather than masked after.
            value = torch.nan_to_num(x[..., value_col], nan=0.0)
            weighted = weighted + torch.where(observed, value * count, torch.zeros_like(count))
            weights = weights + torch.where(observed, count, torch.zeros_like(count))

        seen = weights > 0
        return torch.where(
            seen,
            weighted / torch.where(seen, weights, torch.ones_like(weights)),
            torch.full_like(weighted, torch.nan),
        )


class SeaMaskPostprocessor(nn.Module):
    """NaN the model's output wherever the graph gave it nothing to predict from.

    Land nodes are excluded from both edge sets (``source_mask_attr_name`` on the encoder,
    ``target_mask_attr_name`` on the decoder), so they have zero in-degree and no
    information reaches them. They are still nodes, though, and the decoder still maps
    their trainable embedding to a number -- a confident-looking one: in an earlier
    forecast the land mean was 0.0398 m against a sea mean of 0.0176 m. Anything reading
    the output has no way to tell that apart from a real prediction.

    This runs on the inverse pass only. The forward pass must stay NaN-free: preprocessing
    asserts as much (``preprocessing/__init__.py:205``) and a NaN in the forward graph
    would poison gradients. Being a processor rather than a training-side mask is what
    makes it travel -- ``post_processors`` are part of the model, pickled into the
    inference checkpoint and applied inside ``predict_step``, so the mask holds wherever
    the model is used rather than only in the training plots.

    The mask comes from :func:`nese_training.masks.variable_below`, the same function the
    graph attribute uses, so the two cannot disagree about what sea is.
    """

    def __init__(
        self,
        dataset: str | dict,
        variable: str,
        threshold: float,
        data_indices=None,  # noqa: ANN001
        statistics: dict | None = None,
    ) -> None:
        super().__init__()
        sea = torch.from_numpy(variable_below(dataset, variable, threshold))
        # A buffer, not a plain attribute: this is what carries the mask into the
        # inference checkpoint and onto the accelerator with the rest of the model.
        self.register_buffer("sea", sea)
        self.data_indices = data_indices

    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **_kwargs) -> torch.Tensor:
        if not inverse:
            return x if in_place else x.clone()
        return self.inverse_transform(x, in_place=in_place)

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **_kwargs) -> torch.Tensor:
        # Only the model's own output. The truth tensor is wider and is left alone: its
        # land cells are either already NaN or a real coastal observation, and hiding
        # those would be losing data rather than marking the absence of a prediction.
        n_predicted = len(self.data_indices.model.output.full)
        if x.shape[-1] != n_predicted or x.shape[-2] != self.sea.numel():
            return x if in_place else x.clone()

        if not in_place:
            x = x.clone()

        return torch.where(self.sea.unsqueeze(-1), x, torch.nan)


class GeostrophyMaskPostprocessor(nn.Module):
    """NaN the diagnosed geostrophic velocity wherever it is undefined, on the way out.

    ``GeostrophicVelocity`` must emit a finite number everywhere, and the loss already
    ignores the cells where that number means nothing: the equatorial band, cells with too
    few neighbours for a gradient, and cells with no MDT. This marks the same cells --
    through the same :func:`nese_training.physics.geostrophy_defined` -- in the prediction
    and in the truth's composed columns, so plots and inference show a gap there rather
    than a mean flow, a zero, or an imputed fill.

    Inverse pass only, like :class:`SeaMaskPostprocessor`. List it before
    ``ObsCountNaNRestorer`` so that, with the inverse chain reversed, it runs after the
    truth has been composed. Keep ``min_neighbours`` and ``equatorial_band`` at the
    physics block's values.
    """

    def __init__(
        self,
        dataset: str | dict,
        variables: tuple[str, ...] = ("ugxt", "vgxt"),
        min_neighbours: int = MIN_NEIGHBOURS,
        equatorial_band: float = EQUATORIAL_BAND,
        data_indices=None,  # noqa: ANN001
        statistics: dict | None = None,
    ) -> None:
        super().__init__()
        defined = store_geostrophy_defined(dataset, min_neighbours=min_neighbours, equatorial_band=equatorial_band)
        self.register_buffer("defined", torch.from_numpy(defined))
        self.variables = list(variables)
        self.data_indices = data_indices

    def forward(self, x: torch.Tensor, in_place: bool = True, inverse: bool = False, **_kwargs) -> torch.Tensor:
        if not inverse:
            return x if in_place else x.clone()
        return self.inverse_transform(x, in_place=in_place)

    def _columns(self, width: int) -> list[int]:
        """The masked variables' columns in a model-output or data-output tensor."""
        model_output = self.data_indices.model.output
        if width == len(model_output.full):
            return [model_output.name_to_index[name] for name in self.variables if name in model_output.name_to_index]
        output_full = self.data_indices.data.output.full.tolist()
        if width == len(output_full):
            column = {data_index: col for col, data_index in enumerate(output_full)}
            name_to_index = self.data_indices.data.output.name_to_index
            return [column[name_to_index[name]] for name in self.variables if name_to_index.get(name) in column]
        return []

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **_kwargs) -> torch.Tensor:
        columns = self._columns(x.shape[-1]) if x.shape[-2] == self.defined.numel() else []
        if not in_place:
            x = x.clone()
        if columns:
            x[..., columns] = x[..., columns].masked_fill(~self.defined.unsqueeze(-1), torch.nan)
        return x


class _PooledStatistics:
    """Constructor mixin for processors that share one map across a field's groups.

    ``pool`` takes the same spec as the loss's ``observations``, so both recover the
    mission list from the same names and cannot drift apart. Omitting it leaves the
    processor behaving exactly as its anemoi base class.
    """

    def __init__(self, config=None, data_indices=None, statistics=None, pool=None) -> None:
        if pool:
            statistics = pool_statistics(
                pool, data_indices.data.input.name_to_index, statistics, type(self).__name__
            )
        super().__init__(config, data_indices, statistics)


class PooledObservationNormalizer(_PooledStatistics, InputNormalizer):
    """``InputNormalizer`` giving a field and its observing groups one mean and stdev.

    See :func:`nese_training.observations.pool_statistics` for why the per-mission
    default is wrong here -- in short, a mission's mean is the sea level of the years it
    flew, so per-mission centring subtracts the very trend the model should learn.
    """


class PooledObservationImputer(_PooledStatistics, InputImputer):
    """``InputImputer`` filling from the same pooled statistics as the normaliser.

    Needed for the pair to agree. The imputer fills an unobserved cell in raw space and
    the normaliser then centres it, so if the imputer used the mission's own mean while
    the normaliser used the pooled one, the fill would land up to 0.25 sigma off zero --
    varying by mission, and therefore by era, across the ~77% of cells that are
    unobserved. Sharing the pooled mean puts every filled cell at exactly zero.

    It also declines to undo its own fill on the way back out; see
    :meth:`inverse_transform`.
    """

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        """Leave the gaps to :class:`ObsCountNaNRestorer` on a ``data.output``-shaped tensor.

        The base class restores NaN from ``nan_locations``, which is captured once per
        forward pass as ``nan_locations[:, 0]`` (imputer.py:210) -- the *oldest* input bin
        -- and then stamped over every timestep of whatever it is handed. The tensors that
        reach the plots span the target bin as well, and satellite tracks move between
        bins, so most of what a mission saw in the target bin is NaN in the oldest one and
        is overwritten. Measured on one validation window: 1506 of the store's 14537
        observations survived, and the composed truth came out 88% exact zeros, because
        the restorer masks on counts (intact) while the values behind them had already
        been destroyed.

        Skipping it loses nothing. Each value is masked by its own count (``sla_count_*`` for
        ``sla_filtered_*``, ``uvgxt_count_*`` for ``ugxt_*``/``vgxt_*``), and a count above
        zero is exactly equivalent to "value is not NaN", so the restorer reproduces every
        gap from the counts, per bin and without reference to any other one -- the
        de-imputation is redundant here as well as destructive. With it skipped the composed
        truth matches a count-weighted mean taken straight from the store in every cell.

        Only the wide tensor is skipped. Model-output-shaped tensors are the imputer's own
        business and go to the base class untouched.
        """
        if x.shape[-1] == len(self.data_indices.data.output.full):
            return x if in_place else x.clone()
        return super().inverse_transform(x, in_place=in_place, **kwargs)
