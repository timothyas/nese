"""Tests for the Matérn regularization operator.

The spec is ``experimental/local-sla-16yr/matern-regularization/main.tex``, whose tables
are produced by ``matern-regularization/scripts/``. ``test_matches_reference_implementation``
is what makes those tables binding on this code: the reference is the NumPy implementation
the document's numbers were measured with.
"""

import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pytest
import scipy.sparse.linalg as spl
import torch

from nese_training.matern import HEALPixMaternOperator
from nese_training.matern import delta_hat
from nese_training.matern import infer_nside
from nese_training.matern import smoothness

REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "experimental"
    / "local-sla-16yr"
    / "matern-regularization"
    / "scripts"
)
DATASET = REFERENCE.parents[1] / "anemoi-hpx6-24h" / "dataset.zarr"


def full_sphere(nside: int, min_neighbours: int = 3) -> HEALPixMaternOperator:
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    return HEALPixMaternOperator(lat, lon, min_neighbours=min_neighbours)


@pytest.fixture(scope="module")
def reference():
    sys.path.insert(0, str(REFERENCE))
    import matern as reference_module

    return reference_module


def test_infer_nside_roundtrips():
    for nside in (4, 16, 64):
        lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
        assert infer_nside(lat, lon) == nside


def test_infer_nside_rejects_non_healpix():
    rng = np.random.default_rng(0)
    lat = rng.uniform(-80, 80, 500)
    lon = rng.uniform(0, 360, 500)
    with pytest.raises(ValueError, match="not HEALPix cell centres"):
        infer_nside(lat, lon)


def test_smoothness_and_delta():
    # Nd = 2, so nu = 2M - 1 and the M = 1/2 case is unavailable.
    assert smoothness(1) == pytest.approx(1.0)
    assert smoothness(2) == pytest.approx(3.0)
    assert delta_hat(4.0, 1) == pytest.approx(8.0 * 1.0 / 16.0)
    assert delta_hat(8.0, 2) == pytest.approx(8.0 * 3.0 / 64.0)
    with pytest.raises(ValueError, match="not defined"):
        delta_hat(4.0, 0)


def test_matches_reference_implementation(reference):
    """The torch operator must reproduce the NumPy one the write-up was measured with."""
    nside = 16
    mesh = reference.Mesh(nside)
    operator = full_sphere(nside)

    np.testing.assert_array_equal(operator.neighbour.numpy(), mesh.neighbour)
    np.testing.assert_allclose(operator.weight.numpy(), mesh.weight, rtol=1e-6)

    rho_hat, order = 4.0, 1
    rng = np.random.default_rng(0)
    x = rng.standard_normal(mesh.n_cells)
    expected = reference.apply_operator(mesh.operator(rho_hat, order), x, order)

    got = operator(
        torch.from_numpy(x).float().unsqueeze(-1),
        torch.tensor([delta_hat(rho_hat, order)]),
        torch.tensor([order]),
    ).squeeze(-1)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4, atol=1e-5)


def test_constant_field_is_in_the_laplacian_null_space():
    """L @ 1 = 0, so A_hat @ 1 = delta_hat * 1. Catches sign and row-sum errors."""
    operator = full_sphere(16)
    delta = delta_hat(4.0, 1)
    ones = torch.ones(operator.n_cells, 1)
    got = operator(ones, torch.tensor([delta]), torch.tensor([1]))
    np.testing.assert_allclose(got.squeeze(-1).numpy(), delta, rtol=1e-5)


def test_stencil_is_second_order_consistent():
    """0.5 sum_j w_ij d_ij d_ij^T = I, the condition that L -> grid_scale^2 grad^2."""
    nside = 32
    operator = full_sphere(nside)
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    vec = np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True), 1)
    offset = (vec[operator.neighbour.numpy()] - vec[:, None, :]) / operator.grid_scale

    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, vec)
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(vec, east)
    p = np.stack([(offset * east[:, None, :]).sum(-1), (offset * north[:, None, :]).sum(-1)], -1)
    tensor = 0.5 * np.einsum("nk,nki,nkj->nij", operator.weight.numpy(), p, p)
    trace = np.trace(tensor, axis1=1, axis2=2) / 2

    # Exact wherever all 8 neighbours are present, at every latitude.
    interior = operator.n_valid == 8
    np.testing.assert_allclose(trace[interior], 1.0, atol=1e-3)

    # HEALPix has 24 pixels with only 7 neighbours at every resolution, and they
    # under-weight the Laplacian by exactly that missing eighth -- the boundary
    # inconsistency of Section 3.3 in miniature, on a mesh with no boundary.
    assert int((~interior).sum()) == 24
    assert set(operator.n_valid[~interior].tolist()) == {7}
    np.testing.assert_allclose(trace[~interior], 7 / 8, atol=1e-3)


@pytest.mark.parametrize(("order", "rho_hat"), [(1, 8.0), (2, 8.0)])
def test_reproduces_analytic_matern_correlation(reference, order, rho_hat):
    """Cov = A_hat^-2M exactly, so a normalised column is the discrete correlation."""
    nside = 32
    mesh = reference.Mesh(nside)
    source = hp.ang2pix(nside, 0.0, 0.0, nest=True, lonlat=True)
    empirical = reference.correlation_column(mesh.operator(rho_hat, order), order, source)

    distance = mesh.distance_in_cells(source)
    keep = distance < 3 * rho_hat
    idx = np.argsort(distance[keep])
    d, r = distance[keep][idx], empirical[keep][idx]
    analytic = reference.matern_correlation(d, rho_hat, order)

    assert np.abs(r - analytic).max() < 0.2
    level = reference.matern_correlation(np.array([rho_hat]), rho_hat, order)[0]
    assert level == pytest.approx(0.14, abs=0.01), "r(rho_hat) ~ 0.14 pins delta_hat = 8 nu / rho^2"
    effective = np.interp(-level, -r, d) / rho_hat
    assert 0.8 < effective < 1.2, f"effective range off by more than 20%: {effective:.3f}"


def test_normalisation_gives_unit_marginal_variance(reference):
    """Lambda_hat A_hat^-M w has unit variance per cell, and E[x' C^-1 x] = N.

    This is the test that pins which side Sigma_hat acts on: the alternative reading
    || A_hat^M x / sigma ||^2 fails the second assertion by more than an order of magnitude.
    """
    nside = 16
    mesh = reference.Mesh(nside)
    rho_hat, order, n_samples = 4.0, 1, 400
    operator = mesh.operator(rho_hat, order)
    lu = spl.splu(operator)

    rng = np.random.default_rng(0)
    draws = np.array([lu.solve(rng.standard_normal(mesh.n_cells)) for _ in range(n_samples)])
    sigma = np.sqrt((draws**2).mean(0))
    x = draws / sigma  # ~ N(0, C)

    np.testing.assert_allclose((x**2).mean(0), 1.0, atol=0.15)
    quadratic = (reference.apply_operator(operator, sigma * x, order) ** 2).sum(1)
    assert quadratic.mean() == pytest.approx(mesh.n_cells, rel=0.05)

    wrong_side = (reference.apply_operator(operator, x, order) / sigma) ** 2
    assert wrong_side.sum(1).mean() > 3 * mesh.n_cells, "wrong side should fail loudly"


def test_min_neighbours_masks_and_counts():
    operator = full_sphere(16, min_neighbours=3)
    assert operator.active.all(), "a full sphere has no low-neighbour cells"

    keep = np.arange(hp.nside2npix(16)) % 3 != 0  # a synthetic ragged mask
    lon, lat = hp.pix2ang(16, np.flatnonzero(keep), nest=True, lonlat=True)
    for threshold in (0, 3, 8):
        masked = HEALPixMaternOperator(lat, lon, min_neighbours=threshold)
        expected = (masked.n_valid >= threshold).sum()
        assert int(masked.active.sum()) == expected
    assert HEALPixMaternOperator(lat, lon, min_neighbours=8).active.sum() < masked.n_cells


def test_gradients_are_correct():
    operator = full_sphere(4).double()
    x = torch.randn(operator.n_cells, 1, dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([delta_hat(3.0, 1)], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda t: operator(t, delta, torch.tensor([1])).square().sum(), (x,)
    )


def test_per_variable_parameters_are_independent():
    """Two variables with different rho_hat must match two single-variable applications."""
    operator = full_sphere(8)
    x = torch.randn(operator.n_cells, 2)
    deltas = torch.tensor([delta_hat(3.0, 1), delta_hat(9.0, 2)])
    orders = torch.tensor([1, 2])

    both = operator(x, deltas, orders)
    for column, (delta, order) in enumerate(zip(deltas, orders, strict=True)):
        alone = operator(
            x[:, column : column + 1], delta.unsqueeze(0), order.unsqueeze(0)
        ).squeeze(-1)
        torch.testing.assert_close(both[:, column], alone)


@pytest.mark.skipif(not DATASET.exists(), reason="needs the local hpx6 dataset")
def test_real_mesh_matches_documented_cell_counts():
    """The counts quoted throughout the write-up's Section 3.3."""
    import zarr

    store = zarr.open(str(DATASET), "r")
    lat = np.asarray(store["latitudes"])
    lon = np.asarray(store["longitudes"])
    operator = HEALPixMaternOperator(lat, lon, min_neighbours=3)

    assert operator.nside == 64
    assert operator.n_cells == 33469
    counts = np.bincount(operator.n_valid, minlength=9)
    assert counts[8] == 30064
    assert counts[3:8].sum() == 3295
    assert counts[1:3].sum() == 98
    assert counts[0] == 12
    assert int((~operator.active).sum()) == 110  # 0-2 neighbours, dropped at threshold 3


# --- the loss ---------------------------------------------------------------------


class _Indices:
    """Minimal stand-in for anemoi's IndexCollection.

    ``model_output`` is the narrow set the network emits; ``full`` is every column on
    disk. Keeping them distinct matters: the penalty defaults to every model output, and
    conflating the two would have it try to penalise the observation and count columns.
    """

    def __init__(self, model_output, full=None):
        full = model_output if full is None else full
        self.model = type("M", (), {"output": type("O", (), {"name_to_position": model_output})()})()
        self.data = type("D", (), {"output": type("O", (), {"name_to_position": full})()})()
        self.data_full_name_to_position = full


def build_penalty(nside=8, **kwargs):
    from anemoi.training.utils.index_space import IndexSpace

    from nese_training.losses import MaternPenalty

    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    x = torch.from_numpy(np.radians(np.stack([lat, lon], 1))).float()
    graph = {"ocean": {"x": x}}
    loss = MaternPenalty(
        graph_data=graph,
        data_node_name="ocean",
        data_indices=_Indices({"sla": 0}),
        n_randomisation=kwargs.pop("n_randomisation", 25),
        **kwargs,
    )
    return loss, IndexSpace


def test_penalty_runs_and_is_positive():
    loss, IndexSpace = build_penalty()
    pred = torch.randn(2, 1, 1, loss.operator.n_cells, 1)
    value = loss(pred, pred, pred_layout=IndexSpace.MODEL_OUTPUT)
    assert value.ndim == 0 and value.item() > 0


def test_penalty_acts_on_the_tensor_it_is_given():
    """No hidden de-normalisation: the value is a pure function of ``pred``.

    ``post_processors`` run only in ``_compute_metrics``, so the training loss sees
    normalised values. Rescaling ``pred`` must scale the penalty by exactly the square,
    which it cannot do if anything inside were de-normalising against fixed statistics.
    """
    loss, IndexSpace = build_penalty()
    pred = torch.randn(1, 1, 1, loss.operator.n_cells, 1)
    base = loss(pred, pred, pred_layout=IndexSpace.MODEL_OUTPUT)
    scaled = loss(3.0 * pred, pred, pred_layout=IndexSpace.MODEL_OUTPUT)
    torch.testing.assert_close(scaled, 9.0 * base, rtol=1e-4, atol=1e-6)

    shifted = loss(pred + 5.0, pred, pred_layout=IndexSpace.MODEL_OUTPUT)
    assert shifted.item() != pytest.approx(base.item()), "delta_hat penalises the mean too"


def test_penalty_matches_an_explicit_computation():
    loss, IndexSpace = build_penalty()
    pred = torch.randn(1, 1, 1, loss.operator.n_cells, 1)
    got = loss(pred, pred, pred_layout=IndexSpace.MODEL_OUTPUT)

    x = pred[0, 0, 0]
    applied = loss.operator(loss.operator.sigma * x, loss.delta, loss.order_per_variable)
    active = loss.operator.active.unsqueeze(-1)
    expected = (applied.square() * active).sum() / active.sum()
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-6)


def test_penalty_ignores_masked_cells():
    keep = np.arange(hp.nside2npix(8)) % 4 != 0
    lon, lat = hp.pix2ang(8, np.flatnonzero(keep), nest=True, lonlat=True)
    coords = torch.from_numpy(np.radians(np.stack([lat, lon], 1))).float()

    from anemoi.training.utils.index_space import IndexSpace

    from nese_training.losses import MaternPenalty

    loss = MaternPenalty(
        graph_data={"ocean": {"x": coords}},
        data_node_name="ocean",
        data_indices=_Indices({"sla": 0}),
        min_neighbours=8,
        n_randomisation=25,
    )
    dropped = ~loss.operator.active
    assert dropped.any(), "the ragged mask should strand some cells"

    pred = torch.zeros(1, 1, 1, loss.operator.n_cells, 1)
    quiet = loss(pred, pred, pred_layout=IndexSpace.MODEL_OUTPUT)
    pred[0, 0, 0, dropped, 0] = 1e3
    assert loss(pred, pred, pred_layout=IndexSpace.MODEL_OUTPUT).item() == pytest.approx(
        quiet.item(), abs=1e-6
    ), "cells below min_neighbours must not reach the loss"


def test_penalty_rejects_a_non_model_output():
    from nese_training.losses import MaternPenalty

    lon, lat = hp.pix2ang(8, np.arange(hp.nside2npix(8)), nest=True, lonlat=True)
    coords = torch.from_numpy(np.radians(np.stack([lat, lon], 1))).float()
    with pytest.raises(ValueError, match="Not model outputs"):
        MaternPenalty(
            graph_data={"ocean": {"x": coords}},
            data_node_name="ocean",
            data_indices=_Indices({"sla": 0}),
            variables={"sst": {}},
            n_randomisation=5,
        )


def test_zero_weight_combination_is_exactly_the_data_loss():
    """CombinedLoss([obs, matern], [1.0, 0.0]) must equal the bare observation loss.

    This is the equivalence the config change rests on: wrapping the existing loss in a
    CombinedLoss and shipping the penalty at weight 0 changes nothing. Asserted exactly
    here rather than eyeballed off two noisy training curves.
    """
    from anemoi.training.losses.combined import CombinedLoss
    from anemoi.training.utils.index_space import IndexSpace

    from nese_training.losses import MaternPenalty, ObservationMaskedMSE

    nside = 8
    lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
    coords = torch.from_numpy(np.radians(np.stack([lat, lon], 1))).float()
    indices = _Indices({"sla": 0}, {"sla": 0, "sla_filtered_a": 1, "obs_count_a": 2})

    observation = ObservationMaskedMSE(
        observations={"sla": {"observed_as": "sla_filtered", "groups": ["a"]}},
        data_indices=indices,
    )
    penalty = MaternPenalty(
        graph_data={"ocean": {"x": coords}},
        data_node_name="ocean",
        data_indices=indices,
        n_randomisation=25,
    )
    combined = CombinedLoss(losses=[observation, penalty], loss_weights=[1.0, 0.0])

    n_cells = penalty.operator.n_cells
    pred = torch.randn(2, 1, 1, n_cells, 1)
    target = torch.randn(2, 1, 1, n_cells, 3)
    target[..., 2] = (torch.rand(2, 1, 1, n_cells) > 0.5).float()  # obs_count
    kwargs = {
        "pred_layout": IndexSpace.MODEL_OUTPUT,
        "target_layout": IndexSpace.DATA_FULL,
    }

    alone = observation(pred, target, **kwargs)
    together = combined(pred, target, **kwargs)
    torch.testing.assert_close(together, alone, rtol=0, atol=0)

    # ... and a non-zero weight must actually change it.
    weighted = CombinedLoss(losses=[observation, penalty], loss_weights=[1.0, 0.01])
    assert weighted(pred, target, **kwargs).item() != pytest.approx(alone.item())
