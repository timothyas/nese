"""Preprocessors for observation-supervised state estimation."""

import torch
from torch import nn


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
    ``obs_count > 0`` is exactly equivalent to "value is not NaN" in this dataset
    (verified: 0 disagreements in 491520 cells), which makes the count an exact stand-in
    for the missing NaN mask.

    Only tensors shaped like ``data.output.full`` are touched, because only those carry
    the count columns. Model-output tensors pass through untouched, so predictions stay
    dense and only the truth regains its gaps -- which is the comparison the plots want.
    """

    def __init__(
        self,
        masked_by: dict[str, str],
        data_indices=None,  # noqa: ANN001
        statistics: dict | None = None,
    ) -> None:
        super().__init__()
        self.masked_by = dict(masked_by)
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

        return x
