"""Section 3.1: is the HEALPix pixel side length sqrt(4 pi / n_pix)?

No. That quantity is the square root of the pixel *area*; healpy's own nside2resol
docstring calls it "a gross approximation given the different pixel shapes". Pixel
area is exact, so det(Phi) = L_x L_y is exact, but the isotropic split
L_x = L_y = grid_scale is an approximation, good in the equatorial zone and poor
towards the poles.
"""

import numpy as np
import healpy as hp

from matern import Mesh, latex_row
from common import banner, parse_args

BANDS = [(0, 15), (15, 30), (30, 41.8), (41.8, 60), (60, 75), (75, 90)]


def main():
    args = parse_args(__doc__, nside={"type": int, "default": 64})
    mesh = Mesh(args.nside)
    sides = mesh.pixel_sides()
    aspect = sides.max(1) / sides.min(1)

    banner("pixel area is exact")
    print(f"  hp.nside2pixarea = {hp.nside2pixarea(args.nside):.8e} sr")
    print(f"  4 pi / n_pix     = {4 * np.pi / mesh.n_pix:.8e} sr   -> identical for every pixel")

    banner(f"Table: pixel side lengths at nside={args.nside}, in units of "
           f"grid_scale = sqrt(4 pi / n_pix)", "tab:sides")
    print(latex_row(r"$|\mathrm{lat}|$", "min", "median", "max", "aspect within a pixel"))
    for lo, hi in BANDS:
        m = (np.abs(mesh.lat) >= lo) & (np.abs(mesh.lat) < hi)
        s = sides[m]
        print(latex_row(f"${lo}$--${hi}$", f"{s.min():.3f}", f"{np.median(s):.3f}",
                        f"{s.max():.3f}", f"{np.median(aspect[m]):.3f}"))

    banner("whole mesh")
    print(f"  side / grid_scale : min {sides.min():.3f}  median {np.median(sides):.3f}  "
          f"max {sides.max():.3f}")
    print(f"  equatorial zone (|lat| < 41.8): median {np.median(sides[np.abs(mesh.lat) < 41.8]):.3f}"
          f"  aspect {np.median(aspect[np.abs(mesh.lat) < 41.8]):.3f}")
    print(f"  polar zone      (|lat| > 41.8): median {np.median(sides[np.abs(mesh.lat) >= 41.8]):.3f}"
          f"  aspect {np.median(aspect[np.abs(mesh.lat) >= 41.8]):.3f}")


if __name__ == "__main__":
    main()
