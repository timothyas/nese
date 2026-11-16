"""Loss sensitivities at the processor input and the model input, for one date.

Merges ``../update-forcings-aa/sensitivity.py`` (one training yaml per checkpoint, the
KnownForcingForecaster's forcing-only input step) with the single-cell scalar of
``../unseen-ft-all-altimeter/sensitivity.py``. Runs in nese-anemoi on one GPU
(submit_sensitivity.sh), for sensitivity_global.ipynb and sensitivity_cell.ipynb.

    --scalars global   base    ObservationMaskedMSE over every target observation,
                               written to <output>/global/<t0>/<label>.zarr
    --scalars cell     cell    ObservationMaskedMSE on sla at one target cell that no observed
                               input bin saw, written to <output>/<case>/<label>.zarr. The cell is
                               picked with the first --run and reused for the others.

The gradients, as norms over channels at every hidden node:

    latent     |dL/dz|, where z is the processor's input.
    decoder    |g|, with g the gradient at the decoder's hidden-node input. With latent_skip, the
               decoder sees processor(z) + z, so g is the skip path alone and latent = J^T g + g.
               Without it, the decoder sees processor(z), and g is at the processor's output.
    processor  |J_processor^T g|, what reached z through the processor: latent - g with
               latent_skip, and latent itself without.

and at the model input:

    input      |dL/dx| summed over the variables of each kind, at every input step and data cell.
               x is the normalised, imputed model input. Counts enter only through the observed
               mask, so their gradient is zero, as is every observation's at a forcing-only step.
    encoder    |dL/de| at the encoder's input e: each field's FieldEncoder latent summed over its
               channels (zero at forcing-only steps, where there is none), and the passthrough inputs.

The passes run in 16-mixed precision, as in training, with the loss scaled up before backward so
float16 does not flush small gradients to zero; the scale halves until every gradient is finite.

    python sensitivity.py --run skip <training.yaml> <last.ckpt> --run no_skip <training.yaml> <last.ckpt> \
        --date 2020-09-13T12 --scalars global
    python sensitivity.py --run ... --date 2020-09-13T12 --scalars cell --case gulf_stream --near 38.5 -63
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
EARTH_RADIUS_KM = 6371.0
SLA_COUNT = "sla_filtered_count_"
# First match wins, so the count prefixes come before the observation prefixes they extend
KINDS = {
    "sla_count": rf"^{SLA_COUNT}",
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
    """The processor's input z, the decoder's hidden-node input, and the field latents."""
    network = module.model.model
    return {
        "processor_input": GradientCapture(network.processor, lambda args, kwargs: kwargs.get("x", args[0] if args else None)),
        "decoder_input": GradientCapture(network.decoder[DATASET], lambda args, kwargs: args[0][0]),
        "field_latents": OutputCapture(network.field_encoder),
    }


def sla_counts(module: torch.nn.Module) -> list[int]:
    names = module.data_indices[DATASET].data_full_name_to_position
    return [position for name, position in names.items() if name.startswith(SLA_COUNT)]


def sla_seen(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """``(observed step, grid)``: where sla was observed in each observed input bin of the raw batch.

    The batch also holds the data at a forcing-only step, the target time, so that step is left out.
    """
    observed = module.task.get_batch_input_indices()[: module.model.model.n_observed_steps]
    return (batch[0, observed, 0][..., sla_counts(module)] > 0).any(-1)


def sla_target_observed(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """``(grid,)``: where sla was observed at the target time, in the raw batch."""
    return (batch[0, module.task.get_batch_output_indices()[0], 0][..., sla_counts(module)] > 0).any(-1)


def choose_target_cell(module, batch, near: tuple[float, float] | None, seed: int, max_distance_km: float) -> int:
    """A cell where sla was observed at the target time but in no observed input bin of the raw
    ``batch``: random, or the nearest to ``near``, which must lie within ``max_distance_km`` of it."""
    observed = sla_target_observed(module, batch)
    candidates = torch.nonzero(observed & ~sla_seen(module, batch).any(0)).flatten().cpu().numpy()
    if near is None:
        return int(np.random.default_rng(seed).choice(candidates))
    distance = distance_km(module, near)[candidates]
    if distance.min() > max_distance_km:
        msg = (
            f"No unseen sla target within {max_distance_km} km of {near}; the nearest is "
            f"{distance.min():.0f} km away. Try another --date, or a larger --max-distance-km."
        )
        raise ValueError(msg)
    return int(candidates[np.argmin(distance)])


def distance_km(module, point: tuple[float, float]) -> np.ndarray:
    """Great-circle distance from ``(lat, lon)`` in degrees to every data cell."""
    lat, lon = (np.radians(v) for v in point)
    cell_lat, cell_lon = module.trainer_.graph_data[DATASET].x.cpu().double().numpy().T
    cosine = np.sin(lat) * np.sin(cell_lat) + np.cos(lat) * np.cos(cell_lat) * np.cos(cell_lon - lon)
    return EARTH_RADIUS_KM * np.arccos(np.clip(cosine, -1, 1))


def only_sla_at(module, target: torch.Tensor, cell: int) -> torch.Tensor:
    """The target with every count zeroed except sla's at ``cell``."""
    names = module.data_indices[DATASET].data_full_name_to_position
    counts = [position for name, position in names.items() if "_count_" in name]
    masked = target.clone()
    masked[..., counts] = 0.0
    masked[..., cell, sla_counts(module)] = target[..., cell, sla_counts(module)]
    return masked


def scaled_backward(module, batch, y, hooks, scale: float) -> tuple:
    """One forward and backward of the loss against ``y``, scaled by ``scale``, halved until finite.

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


def to_dataset(module, scalar, loss, scale, x_grad, grads, batch, dates, label: str) -> xr.Dataset:
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

    total, decoder = grads["processor_input"], grads["decoder_input"]
    processor = total - decoder if network.latent_skip else total
    seen = np.zeros((len(input_index), batch.shape[3]), dtype=bool)  # nothing is observed at a forcing-only step
    seen[:n_observed] = sla_seen(module, batch).numpy()
    target_observed = sla_target_observed(module, batch).numpy()

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
            "decoder_sensitivity": (("scalar", "hidden"), per_scalar(decoder.norm(dim=-1).cpu().numpy())),
            "processor_sensitivity": (("scalar", "hidden"), per_scalar(processor.norm(dim=-1).cpu().numpy())),
            "input_sensitivity": (("scalar", "input_hour", "cell", "kind"), per_scalar(inputs)),
            "encoder_input_sensitivity": (("scalar", "input_hour", "cell", "encoder_group"), per_scalar(encoder)),
            "sla_seen": (("input_hour", "cell"), seen),
            "sla_target_observed": ("cell", target_observed),
        },
        coords={
            "scalar": [scalar],
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
            "latent_skip": int(network.latent_skip),
            "t0": str(dates[input_index[n_observed - 1]]),
            "target_time": str(dates[target_index]),
            "processor_kernel": " ".join(PROCESSOR_KERNEL),
        },
    )


def cell_attrs(module, cell: int, args: argparse.Namespace) -> dict:
    lat, lon = np.degrees(module.trainer_.graph_data[DATASET].x[cell].cpu().double().numpy())
    return {
        "case": args.case,
        "requested_point": "random" if args.near is None else f"{args.near[0]}, {args.near[1]}",
        "target_cell": cell,
        "target_latitude": float(lat),
        "target_longitude": float(lon),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", nargs=3, action="append", required=True, metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        help="label, training yaml and training (not inference) checkpoint",
    )
    parser.add_argument("--date", required=True, help="t0, the last observed input time")
    parser.add_argument("--scalars", choices=["global", "cell"], required=True)
    parser.add_argument("--case", help="with --scalars cell: name for this date and target")
    parser.add_argument("--near", type=float, nargs=2, metavar=("LAT", "LON"), help="target the unseen cell nearest this")
    parser.add_argument("--max-distance-km", type=float, default=250.0, help="how far from --near the target may be")
    parser.add_argument("--seed", type=int, default=0, help="picks the target cell when --near is not given")
    parser.add_argument("--loss-scale", type=float, default=2.0**16, help="starting loss scale")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "sensitivity")
    args = parser.parse_args()
    if (args.scalars == "cell") != (args.case is not None):
        parser.error("--case is required with --scalars cell, and only then")
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("highest")

    t0 = pd.Timestamp(args.date)
    output = args.output / (args.case if args.case else f"global/{t0:%Y-%m-%dT%H}")
    output.mkdir(parents=True, exist_ok=True)
    cell = None
    for label, config, checkpoint in args.run:
        module = load_module(Path(config), checkpoint)
        hooks = captures(module)
        batch, dates = make_batch(module, t0)
        normalised = module.on_after_batch_transfer({DATASET: batch.to("cuda")}, 0)[DATASET]
        target = module.task.get_targets({DATASET: normalised})[DATASET]
        if args.scalars == "cell":
            if cell is None:
                cell = choose_target_cell(module, batch, args.near, args.seed, args.max_distance_km)
                if args.near is not None:
                    LOGGER.info("target cell %d is %.0f km from %s", cell, distance_km(module, args.near)[cell], args.near)
            target = only_sla_at(module, target, cell)
        loss, scale, x_grad, grads = scaled_backward(module, normalised, target, hooks, args.loss_scale)
        scalar = "cell" if args.scalars == "cell" else "base"
        LOGGER.info("%s %s, t0 %s: %s loss %.4g at loss scale %g", output.name, label, t0, scalar, loss, scale)
        dataset = to_dataset(module, scalar, loss, scale, x_grad, grads, batch, dates, label)
        if cell is not None:
            dataset.attrs.update(cell_attrs(module, cell, args))
        dataset.to_zarr(output / f"{label}.zarr", mode="w")
        del module, hooks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
