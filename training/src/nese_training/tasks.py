"""A forecaster whose input window also holds the target times, for forcings known in advance.

The batch already carries the target times, so appending the output offsets to the input
offsets is enough for stock ``get_inputs`` to give ``x`` one more slot per output step. That
slot also holds the target time's observations, so the model must read only its known
forcings: ``FieldEncoderEncProcDec`` does, with ``future_forcing_steps`` equal to
``multistep_output``. Rolling out would roll that slot into the past, so it is refused.
"""

from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.tasks import Forecaster


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
