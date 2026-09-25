"""Encode observations per field rather than per observing group.

``AnemoiModelEncProcDec._assemble_input`` concatenates every input column, so each
mission gets its own weights in the encoder's input Linear. Here each field's group
columns go through one shared MLP and a masked mean over the groups that observed
the cell, so the weights do not depend on which or how many groups exist.

The field encoding happens in ``_assemble_input`` rather than in a mapper because the
parent ``forward`` hands the assembled vector to the decoder too (as ``x_dst``), and
no per-group column should reach the network by either path. ``obs_count_*`` columns
act only as the observed mask and are not passed through, so coverage is not an input.
"""

from dataclasses import dataclass

import einops
import torch
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.models import AnemoiModelEncProcDec
from torch import nn

from nese_training.observations import normalise_spec
from nese_training.observations import resolve_groups


@dataclass
class FieldColumns:
    """Model-input positions of each field's (value, count) columns, and the rest."""

    fields: dict[str, tuple[list[int], list[int]]]
    passthrough: list[int]


def resolve_field_columns(fields: dict, name_to_index: dict[str, int]) -> FieldColumns:
    encoded, consumed = {}, set()
    for field, raw in fields.items():
        spec = normalise_spec(raw)
        groups = resolve_groups(spec, name_to_index, FieldEncoder.__name__)
        values = [f"{spec['observed_as']}_{group}" for group in groups]
        counts = [spec["count"].format(group=group) for group in groups]
        missing = [name for name in values + counts if name not in name_to_index]
        if missing:
            msg = f"Cannot encode field {field!r}: {missing} are not model inputs."
            raise ValueError(msg)
        encoded[field] = ([name_to_index[name] for name in values], [name_to_index[name] for name in counts])
        consumed.update(values + counts)

    passthrough = sorted(index for name, index in name_to_index.items() if name not in consumed)
    return FieldColumns(encoded, passthrough)


class FieldEncoder(nn.Module):
    """One MLP per field, averaged over the groups observing each cell."""

    def __init__(self, fields: list[str], channels: int, layer_kernels: dict | None = None) -> None:
        super().__init__()
        factory = load_layer_kernels(layer_kernels)
        self.mlps = nn.ModuleDict(
            {
                field: nn.Sequential(factory.Linear(1, channels), factory.Activation(), factory.Linear(channels, channels))
                for field in fields
            }
        )

    def encode(self, field: str, values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """``(..., groups)`` values and counts -> ``(..., channels)``, zero where unobserved."""
        embedded = self.mlps[field](values.unsqueeze(-1))
        observed = (counts > 0).to(embedded.dtype).unsqueeze(-1)
        return (embedded * observed).sum(dim=-2) / observed.sum(dim=-2).clamp_min(1)

    def forward(self, x: torch.Tensor, columns: dict[str, tuple[list[int], list[int]]]) -> torch.Tensor:
        return torch.cat(
            [self.encode(field, x[..., values], x[..., counts]) for field, (values, counts) in columns.items()],
            dim=-1,
        )


class FieldEncoderEncProcDec(AnemoiModelEncProcDec):
    """``AnemoiModelEncProcDec`` whose input is ``[field latents || other inputs || node attributes]``.

    Configured under ``model.model``::

        fields: ${training.training_loss.datasets.ocean.observations}
        field_channels: 32
    """

    def __init__(self, *, model_config, **kwargs) -> None:  # noqa: ANN001
        # Needed by _calculate_input_dim, which the parent calls before _build_networks.
        self.fields = dict(model_config.model.model.fields)
        self.field_channels = model_config.model.model.field_channels
        super().__init__(model_config=model_config, **kwargs)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.field_columns = {
            name: resolve_field_columns(self.fields, indices.model.input.name_to_index)
            for name, indices in data_indices.items()
        }
        super()._calculate_shapes_and_indices(data_indices)

    def _calculate_input_dim(self, dataset_name: str) -> int:
        per_step = len(self.fields) * self.field_channels + len(self.field_columns[dataset_name].passthrough)
        return self.n_step_input * per_step + self.node_attributes.attr_ndims[dataset_name]

    def _build_networks(self, model_config) -> None:  # noqa: ANN001
        super()._build_networks(model_config)
        self.field_encoder = FieldEncoder(list(self.fields), self.field_channels, model_config.model.layer_kernels)

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_sizes=None,  # noqa: ANN001
        model_comm_group=None,  # noqa: ANN001
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        # this is only used in _assemble_output, and only passes prognostic variables
        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        columns = self.field_columns[dataset_name]
        x_encoded = torch.cat((self.field_encoder(x, columns.fields), x[..., columns.passthrough]), dim=-1)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x_encoded, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )
        return x_data_latent, x_skip, grid_shard_sizes
