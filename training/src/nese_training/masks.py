"""The sea mask, defined once.

The graph uses it to decide which nodes get edges, and the postprocessor uses it to decide
where the model's output means anything. Those two have to agree exactly -- a node the
graph left unconnected is precisely a node the model cannot predict -- so the threshold
test lives here rather than in either of them.
"""

import numpy as np
from anemoi.datasets import open_dataset


def variable_below(dataset: str | dict, variable: str, threshold: float) -> np.ndarray:
    """``True`` where an Anemoi dataset variable is below ``threshold``, at the first step.

    The intended variables are static, so only that first step is read.
    """
    values = open_dataset(dataset, select=variable)[0].squeeze()
    return values < threshold
