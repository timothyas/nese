"""Callbacks for observation-supervised state estimation."""

import logging
import math
from typing import Any

import torch

from anemoi.training.diagnostics.callbacks.plot import BatchOutputPlot
from pytorch_lightning import Callback

from nese_training.losses import ObservationMaskedMSE

LOGGER = logging.getLogger(__name__)


class ObservationBatchOutputPlot(BatchOutputPlot):
    """``BatchOutputPlot`` that indexes the truth tensor in the space the plots assume.

    The callback hands its plot function two tensors in different index spaces. ``data``,
    which becomes ``x`` and ``y_true``, is sliced to ``data.output.full`` (plot.py:872);
    the predictions come out of the task in model-output space (single.py:52, via
    ``pred_layout=IndexSpace.MODEL_OUTPUT``). The ``parameters`` map that the plot
    functions index *both* with is built from ``model.output.name_to_index``
    (model_introspection.py:110), so it is only ever right for the predictions.

    The two spaces coincide unless ``target`` variables are configured, which is why stock
    anemoi never trips over this. Here they are: ``data.output`` is 27 columns -- ``sla``
    plus the 13 ``sla_filtered_*`` and 13 ``obs_count_*`` targets -- against one column of
    model output, so ``y_true[..., 0]`` picks out whichever target sorts first rather than
    ``sla``. With the missions and their counts in one store the names sort alphabetically,
    which puts ``obs_count_cryosat_2`` there: a field of mostly zeros reaching 44, plotted
    as the truth for a quantity bounded to a metre or so.

    Narrowing the truth to the model's own columns puts all three tensors in one space and
    makes the existing map correct, which is a smaller change than threading a second map
    through every plot function. ``model_output_positions_in_data_output`` is anemoi's own
    (collection.py:98), so the mapping cannot drift from the one the task uses to reduce
    its targets. It is the identity without ``target`` variables, leaving a stock setup
    untouched.
    """

    def process(self, pl_module, dataset_name: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
        # numpy by this point, so this is a fancy index on the last axis. Narrowing before
        # the caller's focus mask is safe: that one only ever touches the grid dimension.
        data, output_tensor = super().process(pl_module, dataset_name, *args, **kwargs)
        positions = pl_module.data_indices[dataset_name].model_output_positions_in_data_output
        return data[..., positions], output_tensor


class FieldLossLogger(Callback):
    """Log each field's loss beside the total, for training and validation.

    ``ObservationMaskedMSE`` averages its fields into one number before anemoi sees it. It
    also records each field's own pooled loss (``field_losses``), and this logs those as
    ``{stage}_field_loss/{dataset}/{field}``, per step and per epoch like anemoi's total.
    Nothing is logged for a loss with a single field, whose value is the total.

    Configured under ``diagnostics.callbacks``::

        callbacks:
          - _target_: nese_training.diagnostics.FieldLossLogger
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001, ARG002
        self._log(pl_module, "train", batch)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0  # noqa: ANN001, ARG002
    ) -> None:
        self._log(pl_module, "val", batch)

    @staticmethod
    def _log(pl_module: Any, stage: str, batch: dict) -> None:  # noqa: ANN401
        batch_size = next(iter(batch.values())).shape[0]
        for dataset_name, loss in pl_module.loss.items():
            for leaf in loss.modules():
                if not isinstance(leaf, ObservationMaskedMSE):
                    continue
                for field, value in leaf.field_losses.items():
                    pl_module.log(
                        f"{stage}_field_loss/{dataset_name}/{field}",
                        value,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=pl_module.logger_enabled,
                        batch_size=batch_size,
                        sync_dist=True,
                    )


class NonFiniteLossReporter(Callback):
    """Log what a batch looked like whenever its loss comes out NaN or inf.

    anemoi checks the preprocessed batch for NaN on the first batch only, and the mixed
    precision grad scaler silently skips steps whose gradients are not finite, so a run can
    lose most of its updates with nothing but ``nan`` in the progress bar. For the first
    ``max_reports`` such batches per rank this logs each field's loss, the variables that
    are not finite in the preprocessed batch (per time step), the largest magnitudes, and
    which count columns are live in the target step. In validation it also says whether the
    prediction itself is finite. Debugging only: it reads the whole batch on every step.
    """

    def __init__(self, max_reports: int = 3, top: int = 5) -> None:
        self.max_reports, self.top = max_reports, top
        self.reports = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self._check(trainer, pl_module, "train", batch, batch_idx, loss, None)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0  # noqa: ANN001, ARG002
    ) -> None:
        self._check(trainer, pl_module, "val", batch, batch_idx, outputs.loss, outputs.predictions)

    def _check(self, trainer, pl_module, stage, batch, batch_idx, loss, predictions) -> None:  # noqa: ANN001, PLR0913
        field_losses = {
            f"{name}/{field}": float(value)
            for name, dataset_loss in pl_module.loss.items()
            for leaf in dataset_loss.modules()
            if isinstance(leaf, ObservationMaskedMSE)
            for field, value in leaf.field_losses.items()
        }
        if all(map(math.isfinite, [float(loss), *field_losses.values()])) or self.reports >= self.max_reports:
            return
        self.reports += 1
        lines = [f"non-finite {stage} loss, rank {trainer.global_rank}, step {trainer.global_step}, batch {batch_idx}"]
        lines.append(f"  loss {float(loss)}, field losses {field_losses}")
        for dataset_name, tensor in batch.items():
            names = list(pl_module.data_indices[dataset_name].name_to_index)
            lines += [f"  [{dataset_name}] {line}" for line in self.describe(tensor, names, self.top)]
        if predictions is not None:
            for step in predictions:
                for dataset_name, prediction in (step.items() if isinstance(step, dict) else [("", step)]):
                    bad = int((~torch.isfinite(prediction)).sum())
                    lines.append(f"  [{dataset_name}] prediction: {bad} non-finite of {prediction.numel()}")
        LOGGER.warning("\n".join(lines))

    @staticmethod
    def describe(tensor: torch.Tensor, names: list[str], top: int) -> list[str]:
        """Per time step of a ``(batch, time, ensemble, grid, variable)`` tensor."""
        values = tensor.detach().float()
        not_finite = (~torch.isfinite(values)).sum(dim=(0, 2, 3))
        magnitude = values.nan_to_num(nan=0.0, posinf=float("inf"), neginf=float("inf")).abs().amax(dim=(0, 2, 3))
        live = [i for i, name in enumerate(names) if "_count_" in name]
        lines = []
        for step in range(values.shape[1]):
            bad = {names[i]: int(n) for i, n in enumerate(not_finite[step].tolist()) if n}
            largest = torch.topk(magnitude[step], min(top, len(names)))
            lines.append(f"time {step}: non-finite {bad or 'none'}")
            lines.append(f"time {step}: largest |x| {[(names[i], round(float(v), 1)) for v, i in zip(*largest)]}")
        observed = (values[:, -1, ..., live] > 0).any(dim=(0, 1, 2))
        lines.append(f"live counts in the last step: {[names[i] for i, on in zip(live, observed.tolist()) if on]}")
        return lines
