"""FieldLossLogger: each field's loss reaches the logger under a name plot_loss.py can find."""

import torch

from nese_training.diagnostics import FieldLossLogger
from nese_training.losses import ObservationMaskedMSE


class FakeModule:
    """The two things the callback reads off the LightningModule, and a recording ``log``."""

    def __init__(self, loss):
        self.loss = torch.nn.ModuleDict({"ocean": loss})
        self.logger_enabled = True
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = (value, kwargs)


def recorded_loss(field_losses):
    loss = ObservationMaskedMSE.__new__(ObservationMaskedMSE)
    torch.nn.Module.__init__(loss)
    loss.field_losses = field_losses
    return loss


BATCH = {"ocean": torch.zeros(4, 3, 1, 10, 2)}


def test_each_field_is_logged_per_stage_step_and_epoch():
    module = FakeModule(recorded_loss({"sla": torch.tensor(0.2), "uvgxt": torch.tensor(0.4)}))
    FieldLossLogger().on_train_batch_end(None, module, None, BATCH, 0)
    FieldLossLogger().on_validation_batch_end(None, module, None, BATCH, 0)

    assert set(module.logged) == {
        "train_field_loss/ocean/sla",
        "train_field_loss/ocean/uvgxt",
        "val_field_loss/ocean/sla",
        "val_field_loss/ocean/uvgxt",
    }
    value, kwargs = module.logged["val_field_loss/ocean/uvgxt"]
    assert float(value) == float(torch.tensor(0.4))
    assert kwargs["on_step"] and kwargs["on_epoch"] and kwargs["sync_dist"]
    assert kwargs["batch_size"] == 4


def test_a_loss_without_fields_logs_nothing():
    module = FakeModule(recorded_loss({}))
    FieldLossLogger().on_train_batch_end(None, module, None, BATCH, 0)
    assert module.logged == {}
