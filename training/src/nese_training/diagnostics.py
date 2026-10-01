"""Callbacks for observation-supervised state estimation."""

from typing import Any

from anemoi.training.diagnostics.callbacks.plot import BatchOutputPlot
from pytorch_lightning import Callback

from nese_training.losses import ObservationMaskedMSE


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
