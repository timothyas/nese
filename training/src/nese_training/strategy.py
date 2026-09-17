"""Single-device strategy, so local runs can use the Mac GPU.

anemoi's DDPGroupStrategy is the only strategy shipped, and Lightning refuses any
DDP-family strategy on the MPS accelerator. Nothing about this model needs DDP on a
single device: model_comm_group is threaded through as None everywhere sharding is
optional, and BaseGraphModel._assert_valid_sharding only objects to a null comm group
when the input is actually sharded.

train.py:687 instantiates the strategy with a `static_graph` kwarg that Lightning's own
strategies do not accept, so the extra kwargs from the anemoi config are swallowed here.
"""

import logging

import torch
from pytorch_lightning.strategies import SingleDeviceStrategy

LOGGER = logging.getLogger(__name__)


class SingleDeviceGroupStrategy(SingleDeviceStrategy):
    """SingleDeviceStrategy that satisfies anemoi's comm-group contract."""

    def __init__(self, device: str = "mps", **kwargs) -> None:
        # static_graph / num_gpus_per_model / read_group_size / use_local_synchronization
        # are all DDP concerns.
        ignored = {k: v for k, v in kwargs.items() if k not in {"accelerator", "checkpoint_io", "precision_plugin"}}
        if ignored:
            LOGGER.info("Ignoring distributed-only strategy options on a single device: %s", sorted(ignored))

        super().__init__(
            device=torch.device(device),
            accelerator=kwargs.get("accelerator"),
            checkpoint_io=kwargs.get("checkpoint_io"),
            precision_plugin=kwargs.get("precision_plugin"),
        )

    def setup(self, trainer) -> None:  # noqa: ANN001
        super().setup(trainer)

        # One group of one, with no process group behind it.
        self.lightning_module.set_model_comm_group(None, 0, 0, 1, 1)
        self.lightning_module.set_reader_groups([None], 0, 0, 1)
