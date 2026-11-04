"""Training methods that hand losses more than ``(pred, target)``.

anemoi calls a loss with the target time only, so a loss that needs the model input (e.g.
:class:`nese_training.losses.UnseenObservationMaskedMSE`) exposes ``set_inputs(x)`` and the
method calls it on every such leaf before the stock step runs.
"""

import torch
from anemoi.training.train.methods import SingleTraining
from anemoi.training.train.step_output import TrainingStepOutput


class UnseenAwareTraining(SingleTraining):
    """``SingleTraining`` that shows each loss leaf with ``set_inputs`` the step's input."""

    def _step(self, batch: dict[str, torch.Tensor], validation_mode: bool = False) -> TrainingStepOutput:
        self.set_loss_inputs(batch)
        return super()._step(batch, validation_mode=validation_mode)

    def set_loss_inputs(self, batch: dict[str, torch.Tensor]) -> None:
        x = self.task.get_inputs(batch, data_indices=self.data_indices)
        for dataset_name, loss in self.loss.items():
            for leaf in loss.modules():
                if hasattr(leaf, "set_inputs"):
                    leaf.set_inputs(x[dataset_name])
