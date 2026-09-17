"""Discrete Matern-class correlation operator on a (possibly land-masked) HEALPix mesh.

Notation follows Smith (2022):

    C^{1/2}  = Lambda A^{-M} det(Phi)^{-1/2}
    A_hat    = det(Phi) A = delta_hat - grid_scale^2 grad^2,   delta_hat = 8 nu / rho_hat^2
    nu       = 2M - n_dim/2

Following Smith (2022), hatted quantities live in the grid-scale normalized space.
The operator built here is A_hat, and `normalisation` returns its sigma_hat.

The field is written x, since theta is the network parameters here.
"""

import numpy as np
import healpy as hp
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.special import kv, gamma

R_EARTH = 6371.0
N_NEIGHBOURS = 8


class Mesh:
    """A HEALPix mesh, optionally pruned to a subset of cells.

    Neighbours falling outside the retained set are dropped, which is the zero-flux
    Neumann condition. Invalid slots are parked on row 0 with zero weight so the
    gather stays dense.
    """

    def __init__(self, nside, cell_index=None, nest=True):
        self.nside = nside
        self.nest = nest
        self.n_pix = hp.nside2npix(nside)
        self.cell_index = np.arange(self.n_pix) if cell_index is None else np.asarray(cell_index)
        self.n_cells = self.cell_index.size
        self.grid_scale = np.sqrt(4.0 * np.pi / self.n_pix)

        neighbours = hp.get_all_neighbours(nside, self.cell_index, nest=nest).T
        lookup = np.full(self.n_pix, -1, np.int64)
        lookup[self.cell_index] = np.arange(self.n_cells)
        local = np.where(neighbours < 0, -1, lookup[np.clip(neighbours, 0, None)])

        self.valid = local >= 0
        self.neighbour = np.where(self.valid, local, 0)
        self.n_valid = self.valid.sum(1)

        self.vec = np.stack(hp.pix2vec(nside, self.cell_index, nest=nest), 1)
        self.lat = hp.pix2ang(nside, self.cell_index, nest=nest, lonlat=True)[1]

        # chord offsets to each neighbour, in units of the grid scale
        self.offset = (self.vec[self.neighbour] - self.vec[:, None, :]) / self.grid_scale
        self.offset[~self.valid] = 0.0
        self.distance = np.linalg.norm(self.offset, axis=-1)

        # w_ij = 4 / (n_nominal * dhat^2) makes the graph Laplacian converge to
        # grid_scale^2 * grad^2; the nominal count stays 8 at boundary cells.
        with np.errstate(divide="ignore"):
            self.weight = np.where(
                self.valid, 4.0 / (N_NEIGHBOURS * np.maximum(self.distance, 1e-12) ** 2), 0.0
            )

    @classmethod
    def from_anemoi_dataset(cls, path):
        import zarr

        store = zarr.open(str(path), "r")
        cell_index = np.asarray(store["healpix_cell_index"])
        nside = hp.npix2nside(int(dict(store.attrs)["healpix_cells_total"]))
        return cls(nside, cell_index=cell_index)

    def group_masks(self):
        """Cells grouped by how many neighbours survived the land mask."""
        return [
            ("8 (interior)", self.n_valid == 8),
            ("3-7", (self.n_valid >= 3) & (self.n_valid < 8)),
            ("1-2", (self.n_valid > 0) & (self.n_valid < 3)),
            ("0 (isolated)", self.n_valid == 0),
        ]

    def tangent_offsets(self):
        """Neighbour offsets projected onto the local (east, north) tangent plane."""
        pole = np.array([0.0, 0.0, 1.0])
        east = np.cross(pole, self.vec)
        east /= np.linalg.norm(east, axis=1, keepdims=True)
        north = np.cross(self.vec, east)
        return np.stack(
            [
                (self.offset * east[:, None, :]).sum(-1),
                (self.offset * north[:, None, :]).sum(-1),
            ],
            -1,
        )

    def second_moment(self):
        """T_i = 0.5 sum_j w_ij d_ij d_ij^T. T = I is the consistency condition."""
        p = self.tangent_offsets()
        return 0.5 * np.einsum("nk,nki,nkj->nij", self.weight, p, p)

    def gradient_response(self):
        """|sum_j w_ij d_ij|, which must vanish for the stencil to annihilate a linear field."""
        p = self.tangent_offsets()
        return np.linalg.norm(np.einsum("nk,nki->ni", self.weight, p), axis=1)

    def pixel_sides(self):
        """The four side lengths of each pixel, in units of the grid scale.

        HEALPix pixels are exactly equal-area but not square, so these are not all
        equal to 1 -- see table_pixel_shape.py.
        """
        corners = np.moveaxis(
            hp.boundaries(self.nside, self.cell_index, step=1, nest=self.nest), 1, 2
        )
        return np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=-1) / self.grid_scale

    def face_weights(self):
        """Two-point flux weights w_ij = s_ij / d_ij over the four shared faces.

        The finite-volume discretization. Only consistent where the line joining cell
        centres is orthogonal to the shared face, which HEALPix violates towards the
        poles; check_weights.py measures the cost.
        """
        corners = np.moveaxis(
            hp.boundaries(self.nside, self.cell_index, step=1, nest=self.nest), 1, 2
        )
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=-1) / self.grid_scale
        midpoint = (corners + np.roll(corners, -1, axis=1)) / 2
        face_dir = midpoint - self.vec[:, None, :]
        face_dir /= np.linalg.norm(face_dir, axis=-1, keepdims=True)
        nbr_dir = self.offset / np.maximum(self.distance, 1e-12)[..., None]
        slot = np.einsum("nei,nki->nek", face_dir, nbr_dir).argmax(-1)

        weight = np.zeros((self.n_cells, N_NEIGHBOURS))
        rows = np.repeat(np.arange(self.n_cells), 4)
        distance = np.maximum(np.take_along_axis(self.distance, slot, 1), 1e-12)
        weight[rows, slot.ravel()] = (sides / distance).ravel()
        return np.where(self.valid, weight, 0.0)

    def laplacian(self, weight=None):
        weight = self.weight if weight is None else weight
        flat = self.valid.ravel()
        rows = np.repeat(np.arange(self.n_cells), N_NEIGHBOURS)[flat]
        cols = self.neighbour.ravel()[flat]
        lap = sp.coo_matrix(
            (weight.ravel()[flat], (rows, cols)), shape=(self.n_cells, self.n_cells)
        ).tocsr()
        return lap - sp.diags(np.asarray(lap.sum(1)).ravel())

    def operator(self, rho_hat, order, weight=None):
        """A_hat = delta_hat I - L, the grid-scale normalized elliptic operator."""
        return (
            delta_hat(rho_hat, order) * sp.eye(self.n_cells) - self.laplacian(weight)
        ).tocsc()

    def distance_in_cells(self, source):
        """Great-circle distance from one cell to every other, in grid cells."""
        cos = np.clip(self.vec @ self.vec[source], -1.0, 1.0)
        return np.arccos(cos) / self.grid_scale


def smoothness(order, n_dim=2):
    """nu = 2M - n_dim/2 (Smith 2022 Eq. 10, rearranged)."""
    return 2.0 * order - n_dim / 2.0


def delta_hat(rho_hat, order, n_dim=2):
    """delta_hat = 8 nu / rho_hat^2, with rho_hat the range in grid cells."""
    return 8.0 * smoothness(order, n_dim) / rho_hat**2


def analytic_variance(rho_hat, order, n_dim=2):
    """Marginal variance of a stationary, unbounded Matern field, Smith (2022) Eq. 8.

    A scalar: it depends only on delta_hat, nu and n_dim, and carries no geometry. The
    land mask is what forces the per-cell estimate in `normalisation` instead.
    """
    nu = smoothness(order, n_dim)
    return gamma(nu) / (
        gamma(nu + n_dim / 2) * (4 * np.pi) ** (n_dim / 2) * delta_hat(rho_hat, order, n_dim) ** nu
    )


def matern_correlation(distance, rho_hat, order, n_dim=2):
    """Analytical Matern correlation, Smith (2022) Eq. 12."""
    nu = smoothness(order, n_dim)
    z = np.sqrt(8.0 * nu) * np.maximum(distance, 1e-12) / rho_hat
    return 2.0 ** (1.0 - nu) / gamma(nu) * z**nu * kv(nu, z)


def normalisation(operator, order, n_samples=1000, seed=0):
    """sigma_hat_i = sqrt((A_hat^-2M)_ii) by randomisation (Weaver et al. 2021).

    Lambda = det(Phi)^{1/2-M} / sigma_hat, so that C has unit diagonal. The powers of
    det(Phi) then cancel out of the penalty, leaving || A_hat^M (sigma_hat * x) ||^2.
    """
    lu = spl.splu(operator)
    n_cells = operator.shape[0]
    rng = np.random.default_rng(seed)
    total = np.zeros(n_cells)
    for _ in range(n_samples):
        y = rng.standard_normal(n_cells)
        for _ in range(order):
            y = lu.solve(y)
        total += y**2
    return np.sqrt(total / n_samples)


def apply_operator(operator, x, order):
    """A_hat^M x, for x of shape (..., n_cells)."""
    for _ in range(order):
        x = (operator @ x.T).T
    return x


def correlation_column(operator, order, source):
    """Column `source` of A^{-2M}, normalised by its diagonal.

    Cov = A^{-M} A^{-M,T} = A^{-2M} exactly for symmetric A, so this is the discrete
    correlation with the source cell -- no sampling required.
    """
    lu = spl.splu(operator)
    y = np.zeros(operator.shape[0])
    y[source] = 1.0
    for _ in range(2 * order):
        y = lu.solve(y)
    return y / y[source]


def smooth_fields(mesh, n_fields=50, seed=0, slope=3.0):
    """Generic smooth fields whose normal gradient at the coast is not zero.

    Drawn from a red spectrum so they are smooth but have no reason to satisfy the
    Neumann condition the operator assumes.
    """
    np.random.seed(seed)
    ell = np.arange(3 * mesh.nside)
    spectrum = np.zeros_like(ell, dtype=float)
    spectrum[2:] = 1.0 / ell[2:] ** slope
    fields = np.array(
        [
            hp.reorder(hp.synfast(spectrum, mesh.nside, new=True), r2n=True)[mesh.cell_index]
            for _ in range(n_fields)
        ]
    )
    return fields / fields.std(1, keepdims=True)


def latex_row(*cells):
    return "  " + " & ".join(str(c) for c in cells) + r" \\"
