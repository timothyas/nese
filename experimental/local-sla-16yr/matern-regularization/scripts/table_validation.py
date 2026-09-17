"""Table 4 (tab:validation): discrete vs analytical Matern correlation.

Cov = A^-M A^-M,T = A^-2M exactly for symmetric A, so one column of A^-2M normalised
by its diagonal is the discrete correlation -- no sampling needed. Run on a full
sphere so stencil quality is isolated from the boundary effects of Section 3.3, from
three source latitudes since HEALPix pixel shape varies with latitude.
"""

import numpy as np
import healpy as hp

from matern import Mesh, correlation_column, matern_correlation, smoothness, latex_row
from common import banner, parse_args

CASES = [(1, 4.0), (1, 8.0), (2, 4.0), (2, 8.0)]


def main():
    args = parse_args(__doc__, nside={"type": int, "default": 32})
    mesh = Mesh(args.nside)
    sources = {
        "equator": hp.ang2pix(mesh.nside, 0.0, 0.0, nest=True, lonlat=True),
        "mid": hp.ang2pix(mesh.nside, 0.0, 41.8, nest=True, lonlat=True),
        "pole": hp.ang2pix(mesh.nside, 0.0, 88.0, nest=True, lonlat=True),
    }

    banner(f"Table 4: discrete vs analytical correlation, nside={args.nside}, full sphere",
           "tab:validation")
    print(r"$M$ & $\nu$ & $\rhohat$ & $r(\rhohat)$ & $\max|\text{err}|$"
          r" & $\rho_{\text{emp}}/\rhohat$ & directional spread \\")
    for order, rho_hat in CASES:
        nu = smoothness(order)
        level = matern_correlation(np.array([rho_hat]), rho_hat, order)[0]
        operator = mesh.operator(rho_hat, order)
        err, eff, spread = [], [], []
        for source in sources.values():
            emp = correlation_column(operator, order, source)
            dist = mesh.distance_in_cells(source)
            keep = dist < 3 * rho_hat
            order_idx = np.argsort(dist[keep])
            d, r = dist[keep][order_idx], emp[keep][order_idx]
            err.append(np.abs(r - matern_correlation(d, rho_hat, order)).max())
            eff.append(np.interp(-level, -r, d) / rho_hat)
            band = (d > 0.8 * rho_hat) & (d < 1.2 * rho_hat)
            spread.append(r[band].std())
        print(latex_row(
            order, f"{nu:.0f}", f"{rho_hat:g}", f"{level:.3f}",
            " / ".join(f"{v:.2f}" for v in err),
            " / ".join(f"{v:.2f}" for v in eff),
            " / ".join(f"{v:.3f}" for v in spread),
        ))


if __name__ == "__main__":
    main()
