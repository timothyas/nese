"""Matérn-class correlation operator on a land-masked HEALPix mesh.

Implements the operator derived in ``experimental/local-sla-16yr/matern-regularization``,
which is the spec for this module and carries the validation behind every constant here.
Notation follows Smith (2022): hatted quantities live in the grid-scale normalized space.

    A_hat     = delta_hat * I - L,   L ~ grid_scale^2 * grad^2
    delta_hat = 8 nu / rho_hat^2,    nu = 2M - n_dim/2

with ``rho_hat`` the correlation range in grid cells. Only forward applications of
``A_hat`` are needed to evaluate the penalty; the sparse solve appears once at setup, to
estimate the normalization ``sigma_hat``.
"""

import logging

import healpy as hp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

N_NEIGHBOURS = 8
MAX_HEALPIX_ORDER = 13


def smoothness(order: int, n_dim: int = 2) -> float:
    """nu = 2M - n_dim/2, as in Lindgren et al., 2012, Smith (2022) Eq. 10 rearranged."""
    return 2.0 * order - n_dim / 2.0


def delta_hat(rho_hat: float, order: int, n_dim: int = 2) -> float:
    """delta_hat = 8 nu / rho_hat^2, with rho_hat the range in grid cells."""
    nu = smoothness(order, n_dim)
    if nu <= 0:
        msg = (
            f"order={order} gives nu={nu} in {n_dim}D; the Matern field is not defined. "
            f"Use order >= {int(np.ceil((n_dim / 2 + 1e-9) / 2))}."
        )
        raise ValueError(msg)
    return 8.0 * nu / rho_hat**2


def infer_nside(lat: np.ndarray, lon: np.ndarray, rtol: float = 0.01) -> int:
    """Recover the HEALPix nside of a set of cell centres, in NESTED ordering.

    Raises if no resolution reproduces the coordinates, which is the guard against
    silently treating a non-HEALPix mesh as one.

    The match is judged by angular separation from the pixel centre relative to the pixel
    size, rather than by an absolute tolerance on the coordinates: graph node positions
    are stored as float32, whose ~1e-5 degree resolution would fail any absolute tolerance
    tight enough to be meaningful at fine nside.
    """
    given = np.atleast_2d(hp.ang2vec(lon, lat, lonlat=True)).T
    for order in range(MAX_HEALPIX_ORDER + 1):
        nside = 2**order
        pix = hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)
        centre = np.stack(hp.pix2vec(nside, pix, nest=True), 0)
        separation = np.arccos(np.clip((given * centre).sum(0), -1.0, 1.0))
        if separation.max() < rtol * hp.nside2resol(nside):
            return nside
    msg = (
        "Node coordinates are not HEALPix cell centres at any order up to "
        f"{MAX_HEALPIX_ORDER}. MaternPenalty needs a HEALPix mesh."
    )
    raise ValueError(msg)


class HEALPixMaternOperator(nn.Module):
    """A_hat on the retained cells of a HEALPix mesh.

    Neighbours outside the retained set are dropped, which is the zero-flux Neumann
    condition. Invalid slots are parked on row 0 with zero weight so the gather stays
    dense and branch-free.
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        min_neighbours: int = 3,
        nside: int | None = None,
    ) -> None:
        super().__init__()
        lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
        self.nside = infer_nside(lat, lon) if nside is None else nside
        self.n_pix = hp.nside2npix(self.nside)
        self.n_cells = lat.size
        self.grid_scale = float(np.sqrt(4.0 * np.pi / self.n_pix))

        cell_index = hp.ang2pix(self.nside, lon, lat, nest=True, lonlat=True)
        neighbours = hp.get_all_neighbours(self.nside, cell_index, nest=True).T
        lookup = np.full(self.n_pix, -1, np.int64)
        lookup[cell_index] = np.arange(self.n_cells)
        local = np.where(neighbours < 0, -1, lookup[np.clip(neighbours, 0, None)])
        valid = local >= 0
        n_valid = valid.sum(1)

        vec = np.stack(hp.pix2vec(self.nside, cell_index, nest=True), 1)
        offset = (vec[np.where(valid, local, 0)] - vec[:, None, :]) / self.grid_scale
        distance = np.linalg.norm(offset, axis=-1)

        # w_ij = 4 / (8 dhat^2) makes the graph Laplacian converge to grid_scale^2 grad^2;
        # the nominal count stays 8 at boundary cells, so dropping a face drops its flux.
        weight = np.where(
            valid, 4.0 / (N_NEIGHBOURS * np.maximum(distance, 1e-12) ** 2), 0.0
        )

        self.register_buffer("cell_index", torch.from_numpy(cell_index).long())
        self.register_buffer("neighbour", torch.from_numpy(np.where(valid, local, 0)).long())
        self.register_buffer("weight", torch.from_numpy(weight).float())
        self.register_buffer("row_sum", torch.from_numpy(weight.sum(1)).float())
        self.register_buffer("active", torch.from_numpy(n_valid >= min_neighbours))
        self.register_buffer("sigma", torch.ones(self.n_cells, 1))

        self._weight_np = weight
        self._local = np.where(valid, local, 0)
        self._valid = valid
        self.n_valid = n_valid
        LOGGER.info(
            "%s: nside=%d, %d cells, %d dropped by min_neighbours=%d",
            type(self).__name__,
            self.nside,
            self.n_cells,
            int((n_valid < min_neighbours).sum()),
            min_neighbours,
        )

    def sparse_operator(self, rho_hat: float, order: int) -> sp.csc_matrix:
        """A_hat as a scipy sparse matrix, for the setup-time solves."""
        flat = self._valid.ravel()
        rows = np.repeat(np.arange(self.n_cells), N_NEIGHBOURS)[flat]
        cols = self._local.ravel()[flat]
        lap = sp.coo_matrix(
            (self._weight_np.ravel()[flat], (rows, cols)),
            shape=(self.n_cells, self.n_cells),
        ).tocsr()
        lap = lap - sp.diags(np.asarray(lap.sum(1)).ravel())
        return (delta_hat(rho_hat, order) * sp.eye(self.n_cells) - lap).tocsc()

    def estimate_normalisation(
        self,
        rho_hat: float | list[float],
        order: int | list[int],
        n_samples: int = 1000,
        seed: int = 0,
    ) -> None:
        """Set ``sigma`` to sqrt((A_hat^-2M)_nn) by randomisation (Weaver et al. 2021).

        Lambda_hat = diag{1/sigma_hat} makes C a correlation matrix, and the powers of
        det(Phi) then cancel out of the penalty entirely. One column per variable, so
        variables with different (rho_hat, order) each get their own normalization.
        """
        rho_hats = np.atleast_1d(rho_hat).tolist()
        orders = np.atleast_1d(order).tolist()
        columns = []
        for rho, m in zip(rho_hats, orders, strict=True):
            lu = spl.splu(self.sparse_operator(rho, int(m)))
            rng = np.random.default_rng(seed)
            total = np.zeros(self.n_cells)
            for _ in range(n_samples):
                y = rng.standard_normal(self.n_cells)
                for _ in range(int(m)):
                    y = lu.solve(y)
                total += y**2
            columns.append(np.sqrt(total / n_samples))
        sigma = np.stack(columns, axis=1)
        self.sigma = torch.from_numpy(sigma).float().to(self.sigma.device)

    def apply_once(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """One application of A_hat to ``x`` of shape (..., n_cells, n_vars)."""
        gathered = x[..., self.neighbour, :]  # (..., n_cells, 8, n_vars)
        neighbour_sum = (gathered * self.weight.unsqueeze(-1)).sum(dim=-2)
        return (delta + self.row_sum.unsqueeze(-1)) * x - neighbour_sum

    def forward(self, x: torch.Tensor, delta: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """A_hat^M applied per variable, with M given by ``order`` per column."""
        max_order = int(order.max().item())
        out = x
        result = torch.zeros_like(x)
        for step in range(1, max_order + 1):
            out = self.apply_once(out, delta)
            result = torch.where(order == step, out, result)
        return result
