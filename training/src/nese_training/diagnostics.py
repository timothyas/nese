"""Plot callbacks for observation-supervised state estimation."""

from anemoi.training.diagnostics.callbacks.plot import BatchOutputPlot


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
