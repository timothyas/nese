"""Graph node attributes for observation-supervised state estimation."""

import torch
from anemoi.datasets import open_dataset
from anemoi.graphs.nodes.attributes.base_attributes import BooleanBaseNodeAttribute
from torch_geometric.data.storage import NodeStorage


class DatasetVariableBelow(BooleanBaseNodeAttribute):
    """Boolean node mask, True where an Anemoi dataset variable is below a threshold.

    Anemoi ships ``NonzeroAnemoiDatasetVariable`` and ``NonmissingAnemoiDatasetVariable``
    for the same job, but both assert the nodes were built by ``AnemoiDatasetNodes`` and
    read the store off ``nodes["_dataset"]``. Here the nodes come from ``HEALPixNodes``,
    so the store is named explicitly and node ``i`` is taken to be dataset column ``i``.
    That correspondence holds because ``HEALPixNodes(resolution=6)`` emits
    ``hp.pix2ang(64, ..., nest=True)`` and nese-dataprep writes the same ordering -- an
    exact match on both latitude and longitude -- but nothing in the graph enforces it,
    so the node count is checked.

    Only the first timestep is read; the intended variables are static.
    """

    def __init__(self, dataset: str | dict, variable: str, threshold: float) -> None:
        super().__init__()
        self.dataset = dataset
        self.variable = variable
        self.threshold = threshold

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        values = open_dataset(self.dataset, select=self.variable)[0].squeeze()

        if values.shape[0] != nodes.x.shape[0]:
            msg = (
                f"{type(self).__name__}: '{self.variable}' has {values.shape[0]} points "
                f"but the node set has {nodes.x.shape[0]}."
            )
            raise ValueError(msg)

        return torch.from_numpy(values < self.threshold)
