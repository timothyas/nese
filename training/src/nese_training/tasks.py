"""Tasks whose input window also holds the target times.

``KnownForcingForecaster`` is a forecaster for forcings known in advance. The batch already
carries the target times, so appending the output offsets to the input offsets is enough for
stock ``get_inputs`` to give ``x`` one more slot per output step. That slot also holds the
target time's observations, so the model must read only its known forcings:
``FieldEncoderEncProcDec`` does, with ``future_forcing_steps`` equal to ``multistep_output``.
Rolling out would roll that slot into the past, so it is refused.

``MaskedStateEstimator`` estimates the states at ``[-dt, 0, +dt]`` from the observations at
those same times. ``FieldEncoder`` treats a cell as observed exactly where its count is above
zero, so zeroing every field's counts at a random fraction of ``(cell, time)`` hides all
missions there. ``get_inputs`` indexes a copy, so the targets keep every observation.
``get_inputs`` has no mode argument: ``nese_training.methods.MaskedInputTraining`` sets
``validation_mode`` and resets the seeded validation mask each validation epoch.
"""

import torch
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import TemporalDownscalerPlotAdapter
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from nese_training.field_encoder import resolve_field_columns


class KnownForcingForecaster(Forecaster):
    name: str = "known_forcing_forecaster"

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        if max(self.rollout.start, self.rollout.maximum) > 1 or (self.validation_rollout or 1) > 1:
            msg = f"{type(self).__name__} supports a single forecast step only, not a rollout."
            raise ValueError(msg)
        self._input_offsets = sorted(self._input_offsets + self._output_offsets)
        self._plot_adapter = KnownForcingPlotAdapter(self)


class KnownForcingPlotAdapter(ForecasterPlotAdapter):
    """Plots the last observed input (t0) rather than the forcing-only target slot."""

    def get_init_step(self) -> int:
        return -1 - self._task.num_output_timesteps

    def get_loss_plot_batch_start(self, rollout_step: int) -> int:
        return super().get_loss_plot_batch_start(rollout_step) - self._task.num_output_timesteps


class MaskedStateEstimator(BaseSingleStepTask):
    name: str = "masked_state_estimator"

    def __init__(
        self,
        timestep: str,
        fields: dict,
        mask_fraction: float = 0.25,
        validation_seed: int | None = 0,
        **_kwargs,  # noqa: ANN003
    ) -> None:
        self.timestep = frequency_to_timedelta(timestep)
        offsets = [-self.timestep, 0 * self.timestep, self.timestep]
        super().__init__(input_offsets=offsets, output_offsets=offsets)
        self.fields = fields
        self.mask_fraction = mask_fraction
        self.validation_seed = validation_seed
        self.validation_mode = False
        self._validation_generator = None
        self._plot_adapter = TemporalDownscalerPlotAdapter(self)

    def _get_timestep_for_metadata(self) -> str:
        return frequency_to_string(self.timestep)

    def reset_validation_mask(self) -> None:
        self._validation_generator = None

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if not self.validation_mode or self.validation_seed is None:
            return None
        if self._validation_generator is None:
            self._validation_generator = torch.Generator(device=device).manual_seed(self.validation_seed)
        return self._validation_generator

    def count_columns(self, name_to_index: dict[str, int]) -> list[int]:
        columns = resolve_field_columns(self.fields, name_to_index)
        return sorted({index for _, counts in columns.fields.values() for index in counts})

    def get_inputs(self, batch: dict[str, torch.Tensor], data_indices: dict, **kwargs) -> dict[str, torch.Tensor]:
        x = super().get_inputs(batch, data_indices, **kwargs)
        for dataset_name, inputs in x.items():
            counts = self.count_columns(data_indices[dataset_name].model.input.name_to_index)
            draw = torch.rand(inputs.shape[:-1], generator=self._generator(inputs.device), device=inputs.device)
            hidden = (draw < self.mask_fraction).unsqueeze(-1)
            inputs[..., counts] = inputs[..., counts].masked_fill(hidden, 0)
        return x
