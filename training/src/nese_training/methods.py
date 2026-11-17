"""Training methods that tell the task which mode it runs in; see ``nese_training.tasks``."""

import torch
from anemoi.training.train.methods import SingleTraining
from anemoi.training.train.step_output import TrainingStepOutput


class MaskedInputTraining(SingleTraining):
    def _step(self, batch: dict[str, torch.Tensor], validation_mode: bool = False) -> TrainingStepOutput:
        self.task.validation_mode = validation_mode
        return super()._step(batch, validation_mode=validation_mode)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.task.reset_validation_mask()
