"""Globe-wide loss sensitivities at the processor input and the model input, for one date.

Adapted from ``../unseen-ft-all-altimeter/sensitivity.py``, keeping only its globe-wide ``base``
scalar: ObservationMaskedMSE over every target observation. Each checkpoint is built from its own
training yaml, since v0's task and input layout differ from v2p2's. Runs in nese-anemoi on one GPU
(submit_sensitivity.sh) and writes ``sensitivity/global/<t0>/<label>.zarr`` for
sensitivity_global.ipynb.

The gradients, as norms over channels at every hidden node:

    latent     |dL/dz|, where z is the processor's input. With latent_skip the decoder sees
               processor(z) + z, so this is J_processor^T g + g, with g = dL/d(processor(z) + z).
    decoder    |g|, at the decoder's hidden-node input: the skip path alone.
    processor  |dL/dz - g| = |J_processor^T g|: what reached z through the processor alone.

and at the model input:

    input      |dL/dx| summed over the variables of each kind, at every input step and data cell.
               x is the normalised, imputed model input. Counts enter only through the observed
               mask, so their gradient is zero, as is every observation's at a forcing-only step.
    encoder    |dL/de| at the encoder's input e: each field's FieldEncoder latent summed over its
               channels (zero at forcing-only steps, where there is none), and the passthrough inputs.

The passes run in 16-mixed precision, as in training, with the loss scaled up before backward so
float16 does not flush small gradients to zero; the scale halves until every gradient is finite.

    python sensitivity.py --run v2p2 <training.yaml> <last.ckpt> --run v0 <training.yaml> <last.ckpt> \
        --date 2020-09-13T12
"""

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from anemoi.datasets import open_dataset
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.index_space import IndexSpace
from hydra import compose
from hydra import initialize_config_dir
from nese_training.losses import ObservationMaskedMSE

LOGGER = logging.getLogger("sensitivity")
DATASET = "ocean"
# First match wins, so the count prefixes come before the observation prefixes they extend
KINDS = {
    "sla_count": r"^sla_filtered_count_",
    "uvgxt_count": r"^uvgxt_count_",
    "swh_count": r"^swh_denoised_count_",
    "sla": r"^sla_filtered_",
    "ugxt": r"^ugxt_",
    "vgxt": r"^vgxt_",
    "swh": r"^swh_denoised_",
    "time_forcing": r"^((cos|sin)_julian_day|insolation)$",
}
STATIC = "static"  # everything else: elevation, land fraction, lat/lon encodings, MDT and its gradients

# Training compiles flex_attention with max-autotune, whose kernel pick varies between runs and
# sometimes crashed (jobs 2716267, 2721681). The default kernel needs no autotuning and computes
# the same attention up to rounding.
PROCESSOR_KERNEL = ["model.processor.block_size=128", "model.processor.compile_mode=null"]


def load_module(config_path: Path, checkpoint: str) -> torch.nn.Module:
    """The training module from the yaml at ``config_path``, with ``checkpoint``'s weights, in eval mode."""
    os.environ.setdefault("SLURM_GPUS_PER_NODE", "1")
    os.environ.setdefault("SLURM_NNODES", "1")
    overrides = [
        "diagnostics.log.mlflow.enabled=false",
        *PROCESSOR_KERNEL,
        "training.run_id=null",
        "training.load_weights_only=true",
        f"system.input.warm_start={Path(checkpoint).resolve()}",
    ]
    with initialize_config_dir(config_dir=str(config_path.parent.resolve()), version_base=None):
        config = compose(config_name=config_path.stem, overrides=overrides)
    trainer = AnemoiTrainer(config)
    module = trainer.model.to("cuda").eval()
    module.trainer_ = trainer  # keeps the graph and task reachable
    return module


def make_batch(module: torch.nn.Module, t0: pd.Timestamp) -> tuple[torch.Tensor, list[pd.Timestamp]]:
    """The raw batch for ``t0``, ``(1, time, ensemble, grid, variable)``, over every task offset."""
    store = open_dataset(module.config.dataloader.dataset)
    dates = [t0 + pd.Timedelta(offset) for offset in module.task.get_offsets()]
    index = {pd.Timestamp(date): i for i, date in enumerate(store.dates)}
    fields = np.stack([store[index[date]] for date in dates])  # (time, variable, ensemble, grid)
    return torch.from_numpy(fields).permute(0, 2, 3, 1).unsqueeze(0).float(), dates


class GradientCapture:
    """Keeps the hidden-node tensor ``pick`` finds in ``module``'s inputs, with its grad retained."""

    def __init__(self, module: torch.nn.Module, pick) -> None:  # noqa: ANN001
        self.tensor, self.pick = None, pick
        module.register_forward_pre_hook(self, with_kwargs=True)

    def __call__(self, _module, args, kwargs):  # noqa: ANN001, ANN204
        self.tensor = self.pick(args, kwargs)
        self.tensor.retain_grad()

    def grad(self, n_hidden: int) -> torch.Tensor:
        """``(hidden, channel)`` for the first sample, in float32."""
        return self.tensor.grad.float().reshape(-1, n_hidden, self.tensor.shape[-1])[0]


class OutputCapture:
    """Keeps ``module``'s output from the latest forward, with its grad retained."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.tensor = None
        module.register_forward_hook(self)

    def __call__(self, _module, _args, output):  # noqa: ANN001, ANN204
        output.retain_grad()
        self.tensor = output

    def grad(self, _n_hidden: int) -> torch.Tensor:
        """``(time, grid, channel)`` for the first sample and ensemble member, in float32."""
        return self.tensor.grad.float()[0, :, 0]


def captures(module: torch.nn.Module) -> dict:
    """The processor's input z, the decoder's hidden-node input processor(z) + z, and the field latents."""
    network = module.model.model
    return {
        "processor_input": GradientCapture(network.processor, lambda args, kwargs: kwargs.get("x", args[0] if args else None)),
        "decoder_input": GradientCapture(network.decoder[DATASET], lambda args, kwargs: args[0][0]),
        "field_latents": OutputCapture(network.field_encoder),
    }


def scaled_backward(module, batch, y, hooks, scale: float) -> tuple:
    """One forward and backward of the loss, scaled by ``scale``, halved until finite.

    Returns ``(loss, scale, x_grad, {hook: grad})``, with every gradient divided back by the scale.
    """
    loss_fn = next(m for m in module.loss[DATASET].modules() if isinstance(m, ObservationMaskedMSE))
    n_hidden = module.trainer_.graph_data["hidden"].num_nodes
    while True:
        x = module.task.get_inputs({DATASET: batch}, data_indices=module.data_indices)[DATASET]
        x = x.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y_pred = module({DATASET: x})[DATASET]
            loss = loss_fn(y_pred, y, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)
        (loss * scale).backward()
        grads = {name: hook.grad(n_hidden) / scale for name, hook in hooks.items()}
        x_grad = x.grad[0, :, 0].float() / scale  # (step, grid, variable)
        module.zero_grad(set_to_none=True)
        if all(torch.isfinite(g).all() for g in (*grads.values(), x_grad)) or scale <= 1:
            return float(loss.detach()), scale, x_grad, grads
        LOGGER.info("non-finite gradient at loss scale %g, halving", scale)
        scale /= 2


def kind_of(name: str) -> str:
    return next((kind for kind, pattern in KINDS.items() if re.match(pattern, name)), STATIC)


def to_dataset(module, loss, scale, x_grad, grads, batch, dates, label: str) -> xr.Dataset:
    network = module.model.model
    n_observed = network.n_observed_steps
    input_index = module.task.get_batch_input_indices()
    target_index = module.task.get_batch_output_indices()[0]

    input_names = list(module.data_indices[DATASET].data.input.name_to_position)
    kinds = [*KINDS, STATIC]
    membership = np.array([[kind_of(name) == kind for kind in kinds] for name in input_names], dtype=np.float64)
    inputs = x_grad.abs().cpu().numpy() @ membership  # (step, grid, kind)

    fields = list(network.field_columns[DATASET].fields)
    latents = grads["field_latents"].abs().reshape(*grads["field_latents"].shape[:-1], len(fields), -1).sum(-1)
    latents = np.concatenate([latents.cpu().numpy(), np.zeros((len(input_index) - n_observed, *latents.shape[1:]))])
    passthrough = [kinds.index("time_forcing"), kinds.index(STATIC)]
    encoder = np.concatenate([latents, inputs[..., passthrough]], axis=-1)
    n_channels = [network.field_channels] * len(fields) + [int(membership[:, i].sum()) for i in passthrough]

    total, skip = grads["processor_input"], grads["decoder_input"]
    names = module.data_indices[DATASET].data_full_name_to_position
    sla_counts = [position for name, position in names.items() if name.startswith("sla_filtered_count_")]
    raw = batch[0, :, 0]  # (time, grid, variable), raw counts
    seen = np.zeros((len(input_index), raw.shape[1]), dtype=bool)  # nothing is observed at a forcing-only step
    seen[:n_observed] = (raw[input_index[:n_observed]][..., sla_counts] > 0).any(-1).cpu().numpy()
    target_observed = (raw[target_index][..., sla_counts] > 0).any(-1).cpu().numpy()

    graph = module.trainer_.graph_data
    data_lat, data_lon = np.degrees(graph[DATASET].x.cpu().double().numpy().T)
    hidden_lat, hidden_lon = np.degrees(graph["hidden"].x.cpu().double().numpy().T)
    input_hours = [int(pd.Timedelta(o) / pd.Timedelta(hours=1)) for o in module.task.get_input_offsets()]

    def per_scalar(values) -> np.ndarray:  # noqa: ANN001
        return np.asarray(values)[np.newaxis]

    return xr.Dataset(
        {
            "loss": ("scalar", per_scalar(loss)),
            "loss_scale": ("scalar", per_scalar(scale)),
            "latent_sensitivity": (("scalar", "hidden"), per_scalar(total.norm(dim=-1).cpu().numpy())),
            "decoder_sensitivity": (("scalar", "hidden"), per_scalar(skip.norm(dim=-1).cpu().numpy())),
            "processor_sensitivity": (("scalar", "hidden"), per_scalar((total - skip).norm(dim=-1).cpu().numpy())),
            "input_sensitivity": (("scalar", "input_hour", "cell", "kind"), per_scalar(inputs)),
            "encoder_input_sensitivity": (("scalar", "input_hour", "cell", "encoder_group"), per_scalar(encoder)),
            "sla_seen": (("input_hour", "cell"), seen),
            "sla_target_observed": ("cell", target_observed),
        },
        coords={
            "scalar": ["base"],
            "input_hour": input_hours,
            "kind": kinds,
            "n_variables": ("kind", membership.sum(axis=0).astype(int)),
            "encoder_group": [*fields, "time_forcing", STATIC],
            "n_channels": ("encoder_group", n_channels),
            "latitude": ("cell", data_lat),
            "longitude": ("cell", data_lon),
            "hidden_latitude": ("hidden", hidden_lat),
            "hidden_longitude": ("hidden", hidden_lon),
        },
        attrs={
            "checkpoint_label": label,
            "t0": str(dates[input_index[n_observed - 1]]),
            "target_time": str(dates[target_index]),
            "processor_kernel": " ".join(PROCESSOR_KERNEL),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", nargs=3, action="append", required=True, metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        help="label, training yaml and training (not inference) checkpoint",
    )
    parser.add_argument("--date", required=True, help="t0, the last observed input time")
    parser.add_argument("--loss-scale", type=float, default=2.0**16, help="starting loss scale")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "sensitivity")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("highest")

    t0 = pd.Timestamp(args.date)
    output = args.output / "global" / f"{t0:%Y-%m-%dT%H}"
    output.mkdir(parents=True, exist_ok=True)
    for label, config, checkpoint in args.run:
        module = load_module(Path(config), checkpoint)
        hooks = captures(module)
        batch, dates = make_batch(module, t0)
        normalised = module.on_after_batch_transfer({DATASET: batch.to("cuda")}, 0)[DATASET]
        target = module.task.get_targets({DATASET: normalised})[DATASET]
        loss, scale, x_grad, grads = scaled_backward(module, normalised, target, hooks, args.loss_scale)
        LOGGER.info("%s, t0 %s: loss %.4g at loss scale %g", label, t0, loss, scale)
        dataset = to_dataset(module, loss, scale, x_grad, grads, batch, dates, label)
        dataset.to_zarr(output / f"{label}.zarr", mode="w")
        del module, hooks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
