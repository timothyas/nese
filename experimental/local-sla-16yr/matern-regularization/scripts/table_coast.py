"""Table 3 (tab:coast): per-cell penalty relative to the interior, for a smooth field.

A field drawn from the prior contributes 1.0 at every cell (check_normalisation.py).
A *generic* smooth field does not satisfy the Neumann condition the operator assumes,
and the first-order inconsistency of the boundary stencil then shows up as extra
penalty at coastal cells. Normalising and not normalising err in opposite directions.
"""

import numpy as np

from matern import Mesh, apply_operator, normalisation, smooth_fields, latex_row
from common import banner, parse_args


def main():
    args = parse_args(__doc__, order={"type": int, "default": 1},
                      rho_hat={"type": float, "default": 4.0},
                      n_fields={"type": int, "default": 50})
    mesh = Mesh.from_anemoi_dataset(args.dataset)
    operator = mesh.operator(args.rho_hat, args.order)
    sigma = normalisation(operator, args.order, args.n_samples, args.seed)
    interior = mesh.n_valid == 8

    x = smooth_fields(mesh, args.n_fields, args.seed)
    normalised = apply_operator(operator, sigma * x, args.order)
    constant = apply_operator(operator, x, args.order) * np.median(sigma[interior])

    banner(f"Table 3: penalty vs interior, M={args.order}, rho_hat={args.rho_hat}", "tab:coast")
    groups = [g for g in mesh.group_masks() if g[0] != "8 (interior)"]
    for name, field in [(r"normalised, $\Sigma$ from \eqref{eq:lambda}", normalised),
                        (r"unnormalised, $\Sigma$ constant ($\Lambda = I$)", constant)]:
        base = (field[:, interior] ** 2).mean()
        print(latex_row(name, *(f"${(field[:, m] ** 2).mean() / base:.2f}\\times$"
                                for _, m in groups)))

    banner("Section 3.3: cell counts and the isolated cells")
    for label, m in mesh.group_masks():
        print(f"  {label:14s} n={int(m.sum()):5d}")
    isolated = np.where(mesh.n_valid == 0)[0]
    import healpy as hp
    lon, lat = hp.pix2ang(mesh.nside, mesh.cell_index[isolated], nest=True, lonlat=True)
    print("  isolated cell locations (lat, lon):")
    for a, b in zip(lat, lon):
        print(f"     {a:7.2f}  {b:7.2f}")


if __name__ == "__main__":
    main()
