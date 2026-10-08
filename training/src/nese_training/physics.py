"""Physics diagnostics: model outputs computed from other model outputs, not by the network.

Configured on the model, by name, with optional keyword overrides::

    model:
      physics_diagnostics: [uvgxt]              # or [{uvgxt: {min_neighbours: 4}}]

The decoder emits only the network's own outputs; :class:`PhysicsDiagnosticsMixin` widens
its result to the full model output and fills each diagnostic's columns after the
residual and boundings, so a diagnosed field is never also a trainable decoder channel.

Model outputs arrive normalised. Each block is handed the exact affine maps of the
dataset's own normalizer (built from the same config, data indices and statistics as the
training one), so it can do its physics in raw units and hand back normalised values.
Static inputs are read in raw units straight from ``system.input.dataset``, so they need
not be model inputs, and they travel in the checkpoint as buffers.
"""

import logging

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.models.preprocessing.normalizer import InputNormalizer
from hydra.utils import get_class
from hydra.utils import instantiate
from torch import nn

from nese_training.matern import healpix_neighbours
from nese_training.matern import infer_nside

LOGGER = logging.getLogger(__name__)

# As in nese-dataprep's cmems_l3_sla, so the diagnosed and observed velocities agree.
GRAVITY = 9.81
OMEGA = 7.2921e-5
EARTH_RADIUS = 6371.0e3
EQUATORIAL_BAND = 5.0
MIN_NEIGHBOURS = 3


def geostrophy_defined(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    min_neighbours: int = MIN_NEIGHBOURS,
    equatorial_band: float = EQUATORIAL_BAND,
    static: dict[str, np.ndarray] | None = None,
    nside: int | None = None,
) -> np.ndarray:
    """Where the diagnosed geostrophic velocity is defined: off the equator, where the
    gradient has at least ``min_neighbours`` retained neighbours to fit to, and where every
    ``static`` field is finite (CNES-CLS22 has no MDT in the Caspian, Azov or at the pole).

    The one definition shared by :class:`GeostrophicVelocity` and the loss, so a cell is
    either diagnosed and scored or neither.
    """
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    local, _ = healpix_neighbours(lat, lon, infer_nside(lat, lon) if nside is None else nside)
    defined = ((local >= 0).sum(1) >= min_neighbours) & (np.abs(lat) >= equatorial_band)
    for values in (static or {}).values():
        defined &= np.isfinite(values)
    return defined


class HEALPixGradient(nn.Module):
    """Least-squares eastward and northward derivatives (per metre) on HEALPix cells.

    Each cell fits a plane to the differences to its retained neighbours, in the local
    east/north tangent plane. As in :class:`nese_training.matern.HEALPixMaternOperator`,
    neighbours on land are dropped, and a cell keeping fewer than ``min_neighbours`` is
    inactive: its gradient is zero.
    """

    def __init__(
        self, lat: np.ndarray, lon: np.ndarray, *, min_neighbours: int = MIN_NEIGHBOURS, nside: int | None = None
    ) -> None:
        super().__init__()
        lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
        nside = infer_nside(lat, lon) if nside is None else nside
        local, vec = healpix_neighbours(lat, lon, nside)
        valid = local >= 0
        active = valid.sum(1) >= min_neighbours

        phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[:, None]
        east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], -1)
        north = np.stack([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)], -1)
        offset = (vec[np.where(valid, local, 0)] - vec[:, None, :]) * EARTH_RADIUS
        design = np.stack([(offset * east).sum(-1), (offset * north).sum(-1)], -1) * valid[..., None]
        weights = np.linalg.pinv(design) * active[:, None, None]  # (n_cells, 2, 8)

        self.register_buffer("neighbour", torch.from_numpy(np.where(valid, local, 0)).long())
        self.register_buffer("weights", torch.from_numpy(weights).float())
        self.register_buffer("active", torch.from_numpy(active))
        LOGGER.info(
            "%s: nside=%d, %d cells, %d inactive with min_neighbours=%d",
            type(self).__name__,
            nside,
            lat.size,
            int((~active).sum()),
            min_neighbours,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` of shape ``(..., n_cells)`` -> ``(d/dx, d/dy)``, each ``(..., n_cells)``."""
        difference = x[..., self.neighbour] - x.unsqueeze(-1)
        gradient = torch.einsum("...nk,nck->...nc", difference, self.weights)
        return gradient[..., 0], gradient[..., 1]


class GeostrophicVelocity(nn.Module):
    """Full-field surface geostrophic velocity, ``ugxt`` and ``vgxt``, from ``sla``.

    ``u = mdt_u - (g/f) d(sla)/dy`` and ``v = mdt_v + (g/f) d(sla)/dx``: the geostrophic
    velocity of ``adt = sla + mdt``, with the mean part taken from the gridded CNES-CLS22
    ``mdt_u``/``mdt_v``. Where :func:`geostrophy_defined` is false -- near the equator,
    where f -> 0 and the observations are NaN, where the gradient has too few neighbours,
    or where the MDT is missing -- the velocity is the mean flow, or zero without one, and
    the loss does not score the cell.
    """

    requires = ("sla",)
    produces = ("ugxt", "vgxt")
    static = ("mdt_u", "mdt_v")

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        static: dict[str, np.ndarray],
        scales: dict[str, tuple[float, float]],
        *,
        min_neighbours: int = MIN_NEIGHBOURS,
        equatorial_band: float = EQUATORIAL_BAND,
    ) -> None:
        super().__init__()
        self.n_cells = lat.size
        self.gradient = HEALPixGradient(lat, lon, min_neighbours=min_neighbours)
        defined = geostrophy_defined(
            lat, lon, min_neighbours=min_neighbours, equatorial_band=equatorial_band, static=static
        )
        with np.errstate(divide="ignore"):
            g_over_f = np.where(defined, GRAVITY / (2 * OMEGA * np.sin(np.deg2rad(lat))), 0.0)

        self.register_buffer("g_over_f", torch.from_numpy(g_over_f).float())
        for name in self.static:
            values = np.nan_to_num(static[name], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            self.register_buffer(name, torch.from_numpy(values))
        for name, (mul, add) in scales.items():
            self.register_buffer(f"{name}_scale", torch.tensor([mul, add], dtype=torch.float32))

    def forward(self, sla: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised ``sla`` of shape ``(..., n_cells)`` -> normalised ``(ugxt, vgxt)``."""
        with torch.autocast(device_type=sla.device.type, enabled=False):
            # The normaliser's offset has no gradient, so only its scale needs undoing.
            dx, dy = self.gradient(sla.float() / self.sla_scale[0])
            u = self.mdt_u - self.g_over_f * dy
            v = self.mdt_v + self.g_over_f * dx
            return u * self.ugxt_scale[0] + self.ugxt_scale[1], v * self.vgxt_scale[0] + self.vgxt_scale[1]


PHYSICS_DIAGNOSTICS = {"uvgxt": GeostrophicVelocity}


def parse_physics_config(config) -> dict[str, dict]:  # noqa: ANN001
    """``[name, {name: {option: value}}, ...]`` -> ``{name: options}``, checking names early."""
    parsed = {}
    for entry in config or []:
        items = {entry: {}} if isinstance(entry, str) else {name: dict(options or {}) for name, options in entry.items()}
        parsed.update(items)
    unknown = sorted(set(parsed) - set(PHYSICS_DIAGNOSTICS))
    if unknown:
        msg = f"Unknown physics_diagnostics {unknown}. Available: {sorted(PHYSICS_DIAGNOSTICS)}"
        raise ValueError(msg)
    return parsed


def read_static(dataset: str | dict, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Static fields in raw units, from the first step of an Anemoi dataset, gaps kept as NaN."""
    available = open_dataset(dataset).variables
    missing = [name for name in names if name not in available]
    if missing:
        msg = f"Physics diagnostics need {missing} in the dataset {dataset!r}."
        raise ValueError(msg)
    static = {}
    for name in names:
        values = np.asarray(open_dataset(dataset, select=name)[0], dtype=np.float64).squeeze()
        n_missing = int((~np.isfinite(values)).sum())
        if n_missing:
            LOGGER.info("%s: %d non-finite cells, where the diagnosed velocity is undefined.", name, n_missing)
        static[name] = values
    return static


def store_geostrophy_defined(dataset: str | dict, **options) -> np.ndarray:
    """:func:`geostrophy_defined` on the cells of an Anemoi dataset, with its own MDT."""
    store = open_dataset(dataset)
    static = read_static(dataset, GeostrophicVelocity.static)
    return geostrophy_defined(np.asarray(store.latitudes), np.asarray(store.longitudes), static=static, **options)


def normalisation_scales(model_config, dataset_name: str, data_indices, statistics: dict) -> dict:  # noqa: ANN001
    """``{variable: (mul, add)}`` as the dataset's own normalizer maps raw -> normalised."""
    processors = model_config.data.datasets[dataset_name].get("processors", {}) or {}
    configs = [cfg for cfg in processors.values() if issubclass(get_class(cfg["_target_"]), InputNormalizer)]
    if len(configs) != 1:
        msg = f"Physics diagnostics need exactly one normalizer for dataset '{dataset_name}', found {len(configs)}."
        raise ValueError(msg)
    normalizer = instantiate(configs[0], data_indices=data_indices, statistics=statistics)
    return {
        name: (float(normalizer._norm_mul[index]), float(normalizer._norm_add[index]))
        for name, index in data_indices.name_to_index.items()
    }


class PhysicsDiagnosticsMixin:
    """Adds ``model.physics_diagnostics`` to an ``AnemoiModelEncProcDec`` subclass.

    List it before the anemoi base class. With no diagnostics configured it changes nothing.
    """

    def __init__(self, *, model_config, **kwargs) -> None:  # noqa: ANN001
        self.physics_config = parse_physics_config(model_config.model.get("physics_diagnostics"))
        super().__init__(model_config=model_config, **kwargs)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self._physics_columns, self._network_output_idx = {}, {}
        for dataset_name, indices in data_indices.items():
            model_input = indices.model.input.name_to_index
            model_output = indices.model.output.name_to_index
            columns, diagnosed = {}, set()
            for name in self.physics_config:
                block = PHYSICS_DIAGNOSTICS[name]
                missing = [var for var in block.requires + block.produces if var not in model_output]
                if missing:
                    msg = f"Physics diagnostic '{name}' needs model outputs {missing} in dataset '{dataset_name}'."
                    raise ValueError(msg)
                inputs = [var for var in block.produces if var in model_input]
                if inputs:
                    msg = (
                        f"Physics diagnostic '{name}' produces {inputs}, which are model inputs in "
                        f"'{dataset_name}'. List them as `diagnostic`, so the network never predicts them."
                    )
                    raise ValueError(msg)
                columns[name] = (
                    [model_output[var] for var in block.requires],
                    [model_output[var] for var in block.produces],
                )
                diagnosed.update(block.produces)
            self._physics_columns[dataset_name] = columns
            self._network_output_idx[dataset_name] = sorted(
                index for var, index in model_output.items() if var not in diagnosed
            )
        super()._calculate_shapes_and_indices(data_indices)

    def _calculate_output_dim(self, dataset_name: str) -> int:
        return self.n_step_output * len(self._network_output_idx[dataset_name])

    def _build_networks(self, model_config) -> None:  # noqa: ANN001
        super()._build_networks(model_config)
        self.physics = nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if not self.physics_config:
                continue
            lat, lon = np.degrees(np.asarray(self._graph_data[dataset_name]["x"].cpu(), dtype=np.float64)).T
            indices = self.data_indices[dataset_name]
            scales = normalisation_scales(model_config, dataset_name, indices, self.statistics[dataset_name])
            blocks = nn.ModuleDict()
            for name, options in self.physics_config.items():
                block = PHYSICS_DIAGNOSTICS[name]
                static = read_static(model_config.system.input.dataset, block.static)
                if any(values.shape != lat.shape for values in static.values()):
                    msg = f"Physics diagnostic '{name}': static fields do not match the {lat.size} graph nodes."
                    raise ValueError(msg)
                used = {var: scales[var] for var in block.requires + block.produces}
                blocks[name] = block(lat, lon, static, used, **options)
                LOGGER.info("Physics diagnostic '%s' on '%s': %s -> %s", name, dataset_name, block.requires, block.produces)
            self.physics[dataset_name] = blocks

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype, dataset_name):  # noqa: ANN001
        network_idx = self._network_output_idx[dataset_name]
        n_output = self.num_output_channels[dataset_name]
        if len(network_idx) != n_output:
            per_step = x_out.unflatten(-1, (self.n_step_output, len(network_idx)))
            full = per_step.new_zeros(*per_step.shape[:-1], n_output)
            full[..., network_idx] = per_step
            x_out = full.flatten(-2)

        y = super()._assemble_output(x_out, x_skip, batch_size, ensemble_size, dtype, dataset_name)

        for name, (requires, produces) in self._physics_columns[dataset_name].items():
            block = self.physics[dataset_name][name]
            if y.shape[-2] != block.n_cells:
                msg = f"Physics diagnostic '{name}' needs all {block.n_cells} cells, got {y.shape[-2]}: no grid sharding."
                raise RuntimeError(msg)
            # Cloned, since writing the outputs into y below would otherwise alter views autograd saved.
            outputs = block(*(y[..., index].clone() for index in requires))
            for index, values in zip(produces, outputs, strict=True):
                y[..., index] = values.to(y.dtype)
        return y
