"""The physics block: a HEALPix gradient, geostrophy, and the model mixin that runs them.

What can fail silently here is a wrong sign or scale in the derivatives, a normalisation
that does not match the training normaliser, or the network quietly still predicting a
diagnosed field. Each is checked against something computed independently.
"""

import healpy as hp
import numpy as np
import pytest
import torch
from anemoi.models.data_indices.collection import IndexCollection
from omegaconf import OmegaConf
from test_field_encoder import toy_graph

from nese_training import physics
from nese_training.field_encoder import FieldEncoderEncProcDec
from nese_training.physics import EARTH_RADIUS
from nese_training.physics import GRAVITY
from nese_training.physics import OMEGA
from nese_training.physics import GeostrophicVelocity
from nese_training.physics import HEALPixGradient
from nese_training.physics import geostrophy_defined
from nese_training.physics import parse_physics_config

NSIDE = 32


def sphere(nside: int = NSIDE) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    return lat, lon


def interior(lat: np.ndarray) -> np.ndarray:
    return np.abs(lat) < 70


def test_gradient_of_a_meridional_field():
    """eta = sin(phi): d/dy = cos(phi) / R and d/dx = 0."""
    lat, lon = sphere()
    gradient = HEALPixGradient(lat, lon)
    dx, dy = gradient(torch.from_numpy(np.sin(np.deg2rad(lat))).float())

    expected = np.cos(np.deg2rad(lat)) / EARTH_RADIUS
    keep = interior(lat)
    error = np.abs(dy.numpy()[keep] / expected[keep] - 1)
    # Near exact where the stencil is symmetric; first order along the HEALPix base-pixel
    # boundaries, where it is not (3% of cells off by >1%, at most ~6%, at nside 32).
    assert np.median(error) < 1e-3
    assert (error > 0.01).mean() < 0.05
    assert error.max() < 0.1
    assert np.abs(dx.numpy()[keep]).max() < 0.05 * expected.max()


def test_gradient_of_a_zonal_field():
    """eta = cos(phi) sin(lambda): d/dx = cos(lambda) / R, d/dy = -sin(phi) sin(lambda) / R."""
    lat, lon = sphere()
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    gradient = HEALPixGradient(lat, lon)
    dx, dy = gradient(torch.from_numpy(np.cos(phi) * np.sin(lam)).float())

    keep = interior(lat)
    scale = 1 / EARTH_RADIUS
    np.testing.assert_allclose(dx.numpy()[keep], (np.cos(lam) / EARTH_RADIUS)[keep], atol=0.05 * scale)
    np.testing.assert_allclose(dy.numpy()[keep], (-np.sin(phi) * np.sin(lam) / EARTH_RADIUS)[keep], atol=0.05 * scale)


def test_gradient_broadcasts_over_leading_dims():
    lat, lon = sphere(8)
    gradient = HEALPixGradient(lat, lon)
    x = torch.randn(2, 3, lat.size)
    dx, _ = gradient(x)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx[1, 2], gradient(x[1, 2])[0])


def retained_island(nside: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A mesh with one cell cut off from all but two of its neighbours."""
    lat, lon = sphere(nside)
    centre = hp.ang2pix(nside, 30.0, 20.0, nest=True, lonlat=True)
    neighbours = hp.get_all_neighbours(nside, centre, nest=True)
    drop = np.zeros(lat.size, dtype=bool)
    drop[neighbours[neighbours >= 0][2:]] = True
    return lat[~drop], lon[~drop]


@pytest.mark.parametrize("min_neighbours", [2, 3])
def test_min_neighbours_sets_which_cells_are_active(min_neighbours):
    lat, lon = retained_island()
    gradient = HEALPixGradient(lat, lon, min_neighbours=min_neighbours, nside=8)
    island = np.argmin(np.hypot(lat - 20.0, lon - 30.0))
    assert bool(gradient.active[island]) == (min_neighbours <= 2)


def test_inactive_cells_have_zero_gradient():
    lat, lon = retained_island()
    gradient = HEALPixGradient(lat, lon, min_neighbours=3, nside=8)
    dx, dy = gradient(torch.randn(lat.size))
    inactive = ~gradient.active
    assert inactive.any()
    assert torch.all(dx[inactive] == 0) and torch.all(dy[inactive] == 0)


SCALES = {"sla": (1 / 0.1, -0.02 / 0.1), "ugxt": (1 / 0.2, -0.01 / 0.2), "vgxt": (1 / 0.2, 0.03 / 0.2)}


def geostrophic(lat, lon, **options) -> GeostrophicVelocity:
    static = {"mdt_u": 0.1 * np.cos(np.deg2rad(lat)), "mdt_v": np.full(lat.size, -0.05)}
    return GeostrophicVelocity(lat, lon, static, SCALES, **options)


def test_geostrophic_velocity_matches_hand_computation():
    """sla = a sin(phi) gives u = mdt_u - (g/f) a cos(phi) / R and v = mdt_v."""
    lat, lon = sphere()
    block = geostrophic(lat, lon)
    amplitude = 0.3
    sla = amplitude * np.sin(np.deg2rad(lat))
    mul, add = SCALES["sla"]
    u_n, v_n = block(torch.from_numpy(sla * mul + add).float())

    u = (u_n.numpy() - SCALES["ugxt"][1]) / SCALES["ugxt"][0]
    v = (v_n.numpy() - SCALES["vgxt"][1]) / SCALES["vgxt"][0]
    phi = np.deg2rad(lat)
    expected_u = 0.1 * np.cos(phi) - GRAVITY / (2 * OMEGA * np.sin(phi)) * amplitude * np.cos(phi) / EARTH_RADIUS

    keep = interior(lat) & (np.abs(lat) >= 5)
    np.testing.assert_allclose(u[keep], expected_u[keep], rtol=0.05, atol=1e-4)
    np.testing.assert_allclose(v[keep], -0.05, atol=0.05 * np.abs(expected_u[keep]).max())


def test_equatorial_band_keeps_only_the_mean_flow():
    lat, lon = sphere()
    block = geostrophic(lat, lon, equatorial_band=5.0)
    u_n, v_n = block(torch.randn(lat.size))
    band = np.abs(lat) < 5
    u = (u_n.numpy() - SCALES["ugxt"][1]) / SCALES["ugxt"][0]
    v = (v_n.numpy() - SCALES["vgxt"][1]) / SCALES["vgxt"][0]
    np.testing.assert_allclose(u[band], 0.1 * np.cos(np.deg2rad(lat[band])), atol=1e-6)
    np.testing.assert_allclose(v[band], -0.05, atol=1e-6)


def test_cells_without_a_gradient_keep_only_the_mean_flow():
    lat, lon = retained_island()
    block = geostrophic(lat, lon)
    defined = geostrophy_defined(lat, lon)
    island = np.argmin(np.hypot(lat - 20.0, lon - 30.0))
    assert not defined[island]

    u_n, _ = block(torch.randn(lat.size))
    u = (u_n.numpy() - SCALES["ugxt"][1]) / SCALES["ugxt"][0]
    assert u[island] == pytest.approx(0.1 * np.cos(np.deg2rad(lat[island])), abs=1e-6)


def test_cells_without_an_mdt_are_undefined_and_finite():
    lat, lon = sphere(8)
    cell = int(np.argmin(np.abs(lat - 45.0)))
    static = {"mdt_u": np.zeros(lat.size), "mdt_v": np.zeros(lat.size)}
    static["mdt_u"][cell] = np.nan
    assert not geostrophy_defined(lat, lon, static=static)[cell]

    u_n, v_n = GeostrophicVelocity(lat, lon, static, SCALES)(torch.randn(lat.size))
    assert torch.isfinite(u_n).all() and torch.isfinite(v_n).all()


def test_defined_excludes_the_equator_and_sparse_cells():
    lat, lon = sphere(8)
    assert not geostrophy_defined(lat, lon)[np.abs(lat) < 5].any()
    assert geostrophy_defined(lat, lon)[np.abs(lat) >= 5].all()
    assert not geostrophy_defined(lat, lon, min_neighbours=9).any()


def test_sla_offset_does_not_change_the_velocity():
    lat, lon = sphere(8)
    block = geostrophic(lat, lon)
    sla = torch.randn(lat.size)
    for first, second in zip(block(sla), block(sla + 3.0), strict=True):
        torch.testing.assert_close(first, second)


def test_parse_physics_config():
    assert parse_physics_config(["uvgxt"]) == {"uvgxt": {}}
    assert parse_physics_config([{"uvgxt": {"min_neighbours": 4}}]) == {"uvgxt": {"min_neighbours": 4}}
    assert parse_physics_config(None) == {}
    with pytest.raises(ValueError, match="Unknown physics_diagnostics"):
        parse_physics_config(["vorticity"])


FORCING = ("sla_filtered_jason_2", "obs_count_jason_2", "elevation", "cos_latitude")
VARIABLES = [*FORCING, "sla", "ugxt", "vgxt"]
VELOCITY_FORCING = (*FORCING, "ugxt_jason_2", "vgxt_jason_2")
VELOCITY_FIELDS = {"sla": "sla_filtered", "uvgxt": {"components": ["ugxt", "vgxt"], "defined_by": "uvgxt"}}


def toy_model(
    physics_diagnostics=("uvgxt",), forcing=FORCING, diagnostic=("sla", "ugxt", "vgxt"), fields=None
):
    graph = toy_graph()
    variables = list(dict.fromkeys([*forcing, *VARIABLES]))
    name_to_index = {name: i for i, name in enumerate(variables)}
    indices = IndexCollection(
        OmegaConf.create({"forcing": list(forcing), "diagnostic": list(diagnostic)}), name_to_index
    )
    n = len(variables)
    statistics = {
        "mean": np.linspace(-0.1, 0.1, n),
        "stdev": np.linspace(0.1, 0.3, n),
        "minimum": np.full(n, -1.0),
        "maximum": np.full(n, 1.0),
    }
    mapper = {
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "sub_graph_edge_attributes": ["edge_length"],
        "trainable_size": 0,
        "graph_attention_backend": "pyg",
        "layer_kernels": {},
    }
    config = OmegaConf.create(
        {
            "system": {"input": {"dataset": "unused.zarr"}},
            "data": {
                "datasets": {
                    "data": {
                        "processors": {
                            "normalizer": {
                                "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                                "config": {"default": "mean-std"},
                            }
                        }
                    }
                }
            },
            "model": {
                "num_channels": 16,
                "physics_diagnostics": list(physics_diagnostics),
                "model": {
                    "_target_": "nese_training.field_encoder.FieldEncoderEncProcDec",
                    "hidden_nodes_name": "hidden",
                    "latent_skip": True,
                    "fields": fields or {"sla": "sla_filtered"},
                    "field_channels": 4,
                },
                "trainable_parameters": {"data": 0, "hidden": 0, "data2hidden": 0, "hidden2data": 0, "hidden2hidden": 0},
                "layer_kernels": {},
                "encoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper", **mapper},
                "decoder": {"_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper", **mapper},
                "processor": {
                    "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                    "num_layers": 1,
                    **mapper,
                },
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1},
                "bounding": [],
            },
        }
    )
    model = FieldEncoderEncProcDec(
        model_config=config,
        data_indices={"data": indices},
        statistics={"data": statistics},
        n_step_input=2,
        n_step_output=1,
        graph_data=graph,
    )
    return model, indices, statistics


@pytest.fixture(autouse=True)
def static_fields(monkeypatch):
    """No store in the unit tests: the static fields come from the toy graph's size."""

    def read(dataset, names):
        n_cells = toy_graph()["data"].x.shape[0]
        return {name: np.full(n_cells, 0.05 * (i + 1)) for i, name in enumerate(names)}

    monkeypatch.setattr(physics, "read_static", read)


def test_decoder_emits_only_the_network_outputs():
    model, indices, _ = toy_model()
    assert model.output_dim["data"] == 1
    assert len(indices.model.output) == 3


def test_diagnosed_columns_come_from_the_physics_block():
    model, indices, statistics = toy_model()
    grid = model._graph_data["data"].x.shape[0]
    y = model({"data": torch.randn(1, 2, 1, grid, len(indices.model.input))})["data"]
    assert y.shape == (1, 1, 1, grid, 3)

    output = indices.model.output.name_to_index
    expected_u, expected_v = model.physics["data"]["uvgxt"](y[..., output["sla"]])
    torch.testing.assert_close(y[..., output["ugxt"]], expected_u)
    torch.testing.assert_close(y[..., output["vgxt"]], expected_v)


def test_scales_match_the_normalizer():
    model, indices, statistics = toy_model()
    block = model.physics["data"]["uvgxt"]
    for name in ("sla", "ugxt", "vgxt"):
        index = indices.name_to_index[name]
        mul, add = getattr(block, f"{name}_scale").tolist()
        assert mul == pytest.approx(1 / statistics["stdev"][index])
        assert add == pytest.approx(-statistics["mean"][index] / statistics["stdev"][index])


def test_diagnosed_velocity_trains_the_network():
    model, indices, _ = toy_model()
    grid = model._graph_data["data"].x.shape[0]
    y = model({"data": torch.randn(1, 2, 1, grid, len(indices.model.input))})["data"]
    y[..., indices.model.output.name_to_index["ugxt"]].square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())


def test_without_physics_the_model_is_unchanged():
    model, indices, _ = toy_model(physics_diagnostics=())
    assert model.output_dim["data"] == 3
    assert len(model.physics) == 0


def test_missing_output_is_rejected_early():
    with pytest.raises(ValueError, match="needs model outputs"):
        toy_model(diagnostic=("sla",), forcing=(*FORCING, "ugxt", "vgxt"))


def test_diagnosed_field_as_input_is_rejected_early():
    """Prognostic ugxt would be both a network input and output."""
    with pytest.raises(ValueError, match="model inputs"):
        toy_model(diagnostic=("sla",))


def velocity_inputs(model, indices, defined_value: float, undefined_value: float) -> torch.Tensor:
    grid = model._graph_data["data"].x.shape[0]
    torch.manual_seed(0)
    x = torch.randn(1, 2, 1, grid, len(indices.model.input))
    x[..., indices.model.input.name_to_index["obs_count_jason_2"]] = 1.0
    defined = model.physics["data"]["uvgxt"].defined
    for name in ("ugxt_jason_2", "vgxt_jason_2"):
        column = indices.model.input.name_to_index[name]
        x[..., column] = torch.where(defined, defined_value, undefined_value)
    return x


def test_velocity_observations_are_encoded_as_one_vector_field():
    model, indices, _ = toy_model(forcing=VELOCITY_FORCING, fields=VELOCITY_FIELDS)
    passthrough = len(model.field_columns["data"].passthrough)
    assert passthrough == 2
    assert model.input_dim["data"] == 2 * (2 * 4 + passthrough) + model.node_attributes.attr_ndims["data"]
    assert model.field_encoder.mlps["uvgxt"][0].in_features == 2


def test_velocity_observations_are_ignored_where_geostrophy_is_undefined():
    model, indices, _ = toy_model(forcing=VELOCITY_FORCING, fields=VELOCITY_FIELDS)
    assert not model.physics["data"]["uvgxt"].defined.all()
    model.eval()
    with torch.no_grad():
        y = model({"data": velocity_inputs(model, indices, 0.5, 0.0)})["data"]
        undefined_changed = model({"data": velocity_inputs(model, indices, 0.5, 100.0)})["data"]
        defined_changed = model({"data": velocity_inputs(model, indices, -0.5, 0.0)})["data"]
    torch.testing.assert_close(undefined_changed, y)
    assert not torch.allclose(defined_changed, y)


def test_defined_by_must_name_a_physics_diagnostic():
    with pytest.raises(ValueError, match="defined_by"):
        toy_model(physics_diagnostics=(), forcing=VELOCITY_FORCING, fields=VELOCITY_FIELDS)
