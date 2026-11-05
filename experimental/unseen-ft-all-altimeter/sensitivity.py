"""Loss sensitivities at the processor input and the model input, for one validation date.

Runs in nese-anemoi on one GPU (submit_sensitivity.sh). The model, loss and data come from this
experiment's training.yaml. Only the weights change between checkpoints, so every checkpoint is
scored against the same date, the same target cell and the same loss definitions. The
sensitivities are written to ``sensitivity/<case>/<label>.zarr``, and sensitivity.ipynb plots them in
the ``nese`` environment.

Two modes, each differentiating its scalars in their own forward and backward passes:

    --scalars cell     cell    UnseenObservationMaskedMSE on sla at one unseen target cell only,
                               written to <output>/<case>/
    --scalars global   unseen  UnseenObservationMaskedMSE over the globe
                       base    ObservationMaskedMSE over the globe,
                               written to <output>/global/<t0>/, since they depend only on the date

--variant appends one more directory, for reruns with other numerics.

For each scalar, these gradients are kept, as norms over channels at every hidden node:

    latent     |dL/dz|, where z is the processor's input. With latent_skip, the decoder sees
               processor(z) + z, so this is J_processor^T g + g, with g = dL/d(processor(z) + z).
    decoder    |g|, captured at the decoder's hidden-node input: the skip path alone. For the
               single-cell scalar it is nonzero only at the target's 3 decoder neighbours.
    processor  |dL/dz - g| = |J_processor^T g|: what reached z through the processor alone.

and at the model input:

    input      |dL/dx| summed over the variables of each kind (sla obs, sla counts, ...), at
               every input step and data cell. x is the normalised, imputed model input.
               Counts enter only through the observed mask, so their gradient is zero.
    encoder    |dL/de| at the encoder's input e: each field's FieldEncoder latent summed over
               its channels, and the passthrough inputs (time-varying forcing, static) as above.
               The latents exist at every cell, observed or not.

By default the passes run in 16-mixed precision, as in training, and the loss is scaled up
before backward so float16 does not flush the small far-field gradients to zero. The scale
halves until every gradient is finite. Plain float32 crashed the compiled flex_attention
kernel ("illegal memory access", job 2716267), so --precision 32 is for debugging only.

    python sensitivity.py --scalars cell --case gulf_stream --checkpoint v2p2=<path> \
        --checkpoint v0=<path> [--date 2019-07-14T12] [--near 38.5 -63]
    python sensitivity.py --scalars global --checkpoint v2p2=<path> --date 2019-07-14T12
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
from nese_training.losses import UnseenObservationMaskedMSE

LOGGER = logging.getLogger("sensitivity")
EARTH_RADIUS_KM = 6371.0
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


def load_module(experiment: Path, checkpoint: str) -> torch.nn.Module:
    """The training module from ``training.yaml`` with ``checkpoint``'s weights, in eval mode."""
    os.environ.setdefault("SLURM_GPUS_PER_NODE", "1")
    os.environ.setdefault("SLURM_NNODES", "1")
    overrides = [
        "diagnostics.log.mlflow.enabled=false",
        *PROCESSOR_KERNEL,
        "training.run_id=null",
        "training.load_weights_only=true",
        f"system.input.warm_start={checkpoint}",
    ]
    with initialize_config_dir(config_dir=str(experiment.resolve()), version_base=None):
        config = compose(config_name="training", overrides=overrides)
    trainer = AnemoiTrainer(config)
    module = trainer.model.to("cuda").eval()
    module.trainer_ = trainer  # keeps the graph and task reachable
    return module


def make_batch(module: torch.nn.Module, t0: pd.Timestamp) -> tuple[torch.Tensor, list[pd.Timestamp]]:
    """The raw batch for ``t0``, ``(1, time, ensemble, grid, variable)``, over every task offset."""
    config = module.config
    store = open_dataset(config.dataloader.dataset)
    offsets = module.task.get_offsets()
    dates = [t0 + pd.Timedelta(offset) for offset in offsets]
    index = {pd.Timestamp(date): i for i, date in enumerate(store.dates)}
    fields = np.stack([store[index[date]] for date in dates])  # (time, variable, ensemble, grid)
    batch = torch.from_numpy(fields).permute(0, 2, 3, 1).unsqueeze(0).float()
    return batch, dates


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


# Training compiles flex_attention with max-autotune, which times ~13 kernel variants per process and
# keeps the fastest. The pick varies between runs, and so did the sensitivities, now and then with an
# illegal memory access (jobs 2716267, 2721681). The default kernel at flex_attention's default block
# size needs no autotuning; the block size only sets which tiles are computed, while the mask still
# applies to every node pair, so the attention is the same up to rounding.
PROCESSOR_KERNEL = ["model.processor.block_size=128", "model.processor.compile_mode=null"]

PRECISIONS = {"16-mixed": torch.float16, "bf16-mixed": torch.bfloat16, "32": None}


def leaves(module: torch.nn.Module) -> tuple[ObservationMaskedMSE, UnseenObservationMaskedMSE]:
    found = list(module.loss[DATASET].modules())
    base = next(m for m in found if type(m) is ObservationMaskedMSE)
    unseen = next(m for m in found if isinstance(m, UnseenObservationMaskedMSE))
    return base, unseen


def choose_target_cell(
    module, batch, unseen, seed: int, near: tuple[float, float] | None, max_distance_km: float
) -> int:
    """A cell where sla was observed at the target time but not in any input bin: random, or the
    nearest to ``near``, which must lie within ``max_distance_km`` of it."""
    names = module.data_indices[DATASET].data_full_name_to_position
    target = module.task.get_targets(batch)[DATASET][0, 0, 0]
    sla_counts = [position for name, position in names.items() if name.startswith("sla_filtered_count_")]
    observed = (target[:, sla_counts] > 0).any(-1)
    sla_field = list(unseen.observations).index("sla")
    candidates = torch.nonzero(observed & ~unseen.seen[0, 0, :, sla_field]).flatten().cpu().numpy()
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
    cell_lat, cell_lon = cell_coordinates(module, DATASET)
    cosine = np.sin(lat) * np.sin(cell_lat) + np.cos(lat) * np.cos(cell_lat) * np.cos(cell_lon - lon)
    return EARTH_RADIUS_KM * np.arccos(np.clip(cosine, -1, 1))


def cell_coordinates(module, node: str) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = module.trainer_.graph_data[node].x.cpu().double().numpy().T
    return lat, lon


def only_sla_at(module, target: torch.Tensor, cell: int) -> torch.Tensor:
    """The target with every count zeroed except sla's at ``cell``."""
    names = module.data_indices[DATASET].data_full_name_to_position
    counts = [position for name, position in names.items() if "_count_" in name]
    sla_counts = [position for name, position in names.items() if name.startswith("sla_filtered_count_")]
    masked = target.clone()
    masked[..., counts] = 0.0
    masked[..., cell, sla_counts] = target[..., cell, sla_counts]
    return masked


def scaled_backward(module, batch, leaf, y, hooks, precision: str, scale: float) -> tuple:
    """One forward and backward of ``leaf``, with the loss scaled by ``scale``, halved until finite.

    Returns ``(loss, scale, x, {hook: grad})``, with every gradient divided back by the scale.
    """
    n_hidden = module.trainer_.graph_data["hidden"].num_nodes
    dtype = PRECISIONS[precision]
    while True:
        x = module.task.get_inputs({DATASET: batch}, data_indices=module.data_indices)[DATASET]
        x = x.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            y_pred = module({DATASET: x})[DATASET]
            loss = leaf(y_pred, y, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)
        (loss * scale).backward()
        grads = {name: hook.grad(n_hidden) / scale for name, hook in hooks.items()}
        x_grad = x.grad[0, :, 0].float() / scale  # (step, grid, variable)
        module.zero_grad(set_to_none=True)
        if all(torch.isfinite(g).all() for g in (*grads.values(), x_grad)) or scale <= 1:
            return float(loss.detach()), scale, x_grad, grads
        LOGGER.info("non-finite gradient at loss scale %g, halving", scale)
        scale /= 2


def sensitivities(module, batch: torch.Tensor, cell: int | None, hooks: dict, precision: str, scale: float) -> dict:
    """``{scalar: {loss, loss_scale, latent, ...}}`` for one cell, or for the globe when ``cell`` is
    None; see the module docstring."""
    base, unseen = leaves(module)
    target = module.task.get_targets({DATASET: batch})[DATASET]
    if cell is None:
        scalars = {"unseen": (unseen, target), "base": (base, target)}
    else:
        scalars = {"cell": (unseen, only_sla_at(module, target, cell))}
    results = {}
    for name, (leaf, y) in scalars.items():
        loss, used_scale, x_grad, grads = scaled_backward(module, batch, leaf, y, hooks, precision, scale)
        total, skip = grads["processor_input"], grads["decoder_input"]
        network = module.model.model
        n_fields = len(network.field_columns[DATASET].fields)
        latents = grads["field_latents"].abs().reshape(*grads["field_latents"].shape[:-1], n_fields, -1)
        results[name] = {
            "loss": loss,
            "loss_scale": used_scale,
            "latent": total.norm(dim=-1).cpu().numpy(),
            "decoder": skip.norm(dim=-1).cpu().numpy(),
            "processor": (total - skip).norm(dim=-1).cpu().numpy(),
            "input": x_grad.abs().cpu().numpy(),
            "field_latents": latents.sum(-1).cpu().numpy(),  # (step, grid, field)
        }
        LOGGER.info("%s: loss %.4g at loss scale %g", name, loss, used_scale)
    return results


def kind_of(name: str) -> str:
    return next((kind for kind, pattern in KINDS.items() if re.match(pattern, name)), STATIC)


def to_dataset(module, results: dict, batch, dates, cell: int, label: str, args: argparse.Namespace) -> xr.Dataset:
    input_names = list(module.data_indices[DATASET].data.input.name_to_position)
    kinds = [*KINDS, STATIC]
    membership = np.array([[kind_of(name) == kind for kind in kinds] for name in input_names], dtype=np.float64)
    input_hours = [int(pd.Timedelta(o) / pd.Timedelta(hours=1)) for o in module.task.get_input_offsets()]
    data_lat, data_lon = cell_coordinates(module, DATASET)
    hidden_lat, hidden_lon = cell_coordinates(module, "hidden")

    scalars = list(results)
    def stacked(key: str) -> np.ndarray:
        return np.stack([results[s][key] for s in scalars])

    inputs = np.stack([results[s]["input"] @ membership for s in scalars])  # (scalar, step, grid, kind)
    network = module.model.model
    fields = list(network.field_columns[DATASET].fields)
    passthrough = [kinds.index("time_forcing"), kinds.index(STATIC)]
    encoder = np.concatenate([stacked("field_latents"), inputs[..., passthrough]], axis=-1)
    n_channels = [network.field_channels] * len(fields) + [int(membership[:, i].sum()) for i in passthrough]

    names = module.data_indices[DATASET].data_full_name_to_position
    sla_counts = [position for name, position in names.items() if name.startswith("sla_filtered_count_")]
    raw = batch[0, :, 0]  # (time, grid, variable), raw counts
    input_index = module.task.get_batch_input_indices()
    seen = (raw[input_index][..., sla_counts] > 0).any(-1).cpu().numpy()
    target_index = module.task.get_batch_output_indices()[0]
    target_observed = (raw[target_index][..., sla_counts] > 0).any(-1).cpu().numpy()

    degrees = np.degrees
    return xr.Dataset(
        {
            "loss": ("scalar", stacked("loss")),
            "loss_scale": ("scalar", stacked("loss_scale")),
            "latent_sensitivity": (("scalar", "hidden"), stacked("latent")),
            "decoder_sensitivity": (("scalar", "hidden"), stacked("decoder")),
            "processor_sensitivity": (("scalar", "hidden"), stacked("processor")),
            "input_sensitivity": (("scalar", "input_hour", "cell", "kind"), inputs),
            "encoder_input_sensitivity": (("scalar", "input_hour", "cell", "encoder_group"), encoder),
            "sla_seen": (("input_hour", "cell"), seen),
            "sla_target_observed": ("cell", target_observed),
        },
        coords={
            "scalar": scalars,
            "input_hour": input_hours,
            "kind": kinds,
            "n_variables": ("kind", membership.sum(axis=0).astype(int)),
            "encoder_group": [*fields, "time_forcing", STATIC],
            "n_channels": ("encoder_group", n_channels),
            "latitude": ("cell", degrees(data_lat)),
            "longitude": ("cell", degrees(data_lon)),
            "hidden_latitude": ("hidden", degrees(hidden_lat)),
            "hidden_longitude": ("hidden", degrees(hidden_lon)),
        },
        attrs={
            "checkpoint_label": label,
            "t0": str(dates[input_index[-1]]),
            "target_time": str(dates[target_index]),
            "precision": args.precision,
            "processor_kernel": " ".join(PROCESSOR_KERNEL),
            "variant": args.variant or "",
            "description": __doc__.split("\n\n")[1].replace("\n", " "),
            **target_attrs(cell, data_lat, data_lon, args),
        },
    )


def target_attrs(cell: int | None, lat: np.ndarray, lon: np.ndarray, args: argparse.Namespace) -> dict:
    if cell is None:
        return {}
    return {
        "case": args.case,
        "requested_point": "random" if args.near is None else f"{args.near[0]}, {args.near[1]}",
        "target_cell": cell,
        "target_latitude": float(np.degrees(lat[cell])),
        "target_longitude": float(np.degrees(lon[cell])),
    }


def random_t0(module, seed: int) -> pd.Timestamp:
    period = module.config.dataloader.validation.datasets[DATASET]
    store = open_dataset(module.config.dataloader.dataset)
    dates = pd.DatetimeIndex(store.dates)
    first, last = min(module.task.get_offsets()), max(module.task.get_offsets())
    valid = dates[
        (dates + pd.Timedelta(first) >= pd.Timestamp(period.start))
        & (dates + pd.Timedelta(last) <= pd.Timestamp(period.end) + pd.Timedelta(days=1))
    ]
    return valid[np.random.default_rng(seed).integers(len(valid))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="LABEL=PATH", help="training (not inference) checkpoint"
    )
    parser.add_argument("--date", help="t0, the last input time; default a random validation date")
    parser.add_argument("--scalars", choices=["cell", "global"], required=True)
    parser.add_argument("--case", help="with --scalars cell: name for this date and target")
    parser.add_argument("--variant", help="extra output directory, for reruns with other numerics")
    parser.add_argument("--near", type=float, nargs=2, metavar=("LAT", "LON"), help="target the unseen cell nearest this")
    parser.add_argument("--max-distance-km", type=float, default=250.0, help="how far from --near the target may be")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=list(PRECISIONS), default="16-mixed")
    parser.add_argument("--loss-scale", type=float, default=2.0**16, help="starting scale; 1 for --precision 32")
    parser.add_argument("--experiment", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "sensitivity")
    args = parser.parse_args()
    if (args.scalars == "cell") != (args.case is not None):
        parser.error("--case is required with --scalars cell, and only then")
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("highest")
    scale = 1.0 if args.precision == "32" else args.loss_scale

    cell, t0 = None, pd.Timestamp(args.date) if args.date else None
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        module = load_module(args.experiment, path)
        hooks = captures(module)
        t0 = t0 if t0 is not None else random_t0(module, args.seed)
        batch, dates = make_batch(module, t0)
        normalised = module.on_after_batch_transfer({DATASET: batch.to("cuda")}, 0)[DATASET]
        module.set_loss_inputs({DATASET: normalised})
        if args.scalars == "cell" and cell is None:
            cell = choose_target_cell(
                module, {DATASET: normalised}, leaves(module)[1], args.seed, args.near, args.max_distance_km
            )
            if args.near is not None:
                LOGGER.info("target is %.0f km from %s", distance_km(module, args.near)[cell], args.near)
        run = args.case if args.scalars == "cell" else f"global/{t0:%Y-%m-%dT%H}"
        LOGGER.info("%s %s: t0 %s, target cell %s", run, label, t0, cell)

        results = sensitivities(module, normalised, cell, hooks, args.precision, scale)
        output = args.output / run / (args.variant or "")
        output.mkdir(parents=True, exist_ok=True)
        to_dataset(module, results, batch, dates, cell, label, args).to_zarr(output / f"{label}.zarr", mode="w")
        del module, hooks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
