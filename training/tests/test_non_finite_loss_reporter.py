"""NonFiniteLossReporter: silent on finite steps, and names what is wrong on a NaN step."""

import logging
from types import SimpleNamespace

import torch

from nese_training.diagnostics import NonFiniteLossReporter
from test_field_loss_logger import recorded_loss

NAMES = ["sla_filtered_ers_1", "sla_filtered_count_ers_1", "sla_filtered_count_gfo"]


def module(field_losses):
    indices = SimpleNamespace(name_to_index={name: i for i, name in enumerate(NAMES)})
    return SimpleNamespace(loss=torch.nn.ModuleDict({"ocean": recorded_loss(field_losses)}), data_indices={"ocean": indices})


def batch():
    x = torch.zeros(1, 2, 1, 4, len(NAMES))
    x[0, 1, 0, 2, 0] = torch.nan
    x[0, 1, 0, :, 1] = 3.0  # ers_1 observed in the last step, gfo not
    return {"ocean": x}


TRAINER = SimpleNamespace(global_rank=0, global_step=7)


def test_finite_steps_log_nothing(caplog):
    reporter = NonFiniteLossReporter()
    with caplog.at_level(logging.WARNING):
        reporter.on_train_batch_end(TRAINER, module({"sla": torch.tensor(0.5)}), torch.tensor(0.5), batch(), 0)
    assert not caplog.records and reporter.reports == 0


def test_a_nan_field_loss_reports_the_bad_variable_and_live_counts(caplog):
    reporter = NonFiniteLossReporter()
    with caplog.at_level(logging.WARNING):
        reporter.on_train_batch_end(TRAINER, module({"sla": torch.tensor(torch.nan)}), {"loss": torch.tensor(0.5)}, batch(), 3)
    text = caplog.text
    assert "step 7, batch 3" in text
    assert "time 0: non-finite none" in text
    assert "time 1: non-finite {'sla_filtered_ers_1': 1}" in text
    assert "live counts in the last step: ['sla_filtered_count_ers_1']" in text


def test_validation_reports_the_prediction_and_stops_after_max_reports(caplog):
    reporter = NonFiniteLossReporter(max_reports=1)
    outputs = SimpleNamespace(loss=torch.tensor(torch.nan), predictions=[{"ocean": torch.tensor([1.0, torch.nan])}])
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            reporter.on_validation_batch_end(TRAINER, module({}), outputs, batch(), 0)
    assert caplog.text.count("non-finite val loss") == 1
    assert "prediction: 1 non-finite of 2" in caplog.text
