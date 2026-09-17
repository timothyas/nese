"""Table 1 (tab:phi) and Section 3.1: HEALPix pixel geometry and stencil consistency.

Shows that Phi != grid_scale * I -- pixel shape varies with latitude -- while
det(Phi) = grid_scale^2 holds exactly, and that the isotropic part of the discrete
operator is exact anyway because w ~ 1/d^2 depends only on neighbour directions.
"""

import numpy as np
import healpy as hp

from matern import Mesh, R_EARTH, latex_row
from common import banner, parse_args

BANDS = [(0, 15), (15, 30), (30, 41.8), (41.8, 60), (60, 75), (75, 90)]
EDGE_SLOTS = [0, 2, 4, 6]  # SW NW NE SE -- HEALPix's true grid axes
FAR_EW, FAR_NS = [1, 5], [3, 7]  # W E  /  N S


def main():
    args = parse_args(__doc__)

    banner("grid scale  l = sqrt(4 pi / n_pix)")
    for nside in (32, 64):
        l = np.sqrt(4.0 * np.pi / hp.nside2npix(nside))
        print(
            f"  nside={nside:3d}  n_pix={hp.nside2npix(nside):6d}  "
            f"l = {l:.6f} rad = {np.degrees(l):.4f} deg = {R_EARTH * l:.2f} km   "
            f"(healpy nside2resol {np.degrees(hp.nside2resol(nside)):.4f} deg)"
        )

    mesh = Mesh(64)  # full sphere: geometry only, no land mask
    d = mesh.distance
    tensor = mesh.second_moment()
    eig = np.linalg.eigvalsh(tensor)
    ratio = eig[:, 1] / eig[:, 0]
    trace = np.trace(tensor, axis1=1, axis2=2) / 2

    banner("Table 1: pixel geometry at nside=64, by latitude band", "tab:phi")
    print(r"$|\mathrm{lat}|$ & edge nbr & far nbr (E--W) & far nbr (N--S)"
          r" & $L_x/L_y$ & $\lambda_{\max}/\lambda_{\min}$ \\")
    for lo, hi in BANDS:
        m = (np.abs(mesh.lat) >= lo) & (np.abs(mesh.lat) < hi)
        ew, ns = d[m][:, FAR_EW].mean(1), d[m][:, FAR_NS].mean(1)
        print(latex_row(
            f"${lo}$--${hi}$",
            f"{np.median(d[m][:, EDGE_SLOTS]):.3f}",
            f"{np.median(ew):.3f}",
            f"{np.median(ns):.3f}",
            f"{np.median(ew / ns):.3f}",
            f"{np.median(ratio[m]):.3f}",
        ))

    banner("Section 3.1: consistency of the isotropic part")
    print(f"  trace(T)/2 over the whole mesh: median {np.median(trace):.4f}  "
          f"[{np.percentile(trace, 5):.4f}, {np.percentile(trace, 95):.4f}]")
    print("  by band: " + "  ".join(
        f"{lo}-{hi}: {np.median(trace[(np.abs(mesh.lat) >= lo) & (np.abs(mesh.lat) < hi)]):.4f}"
        for lo, hi in BANDS))

    banner("Section 3.2: residual anisotropy (the traceless part)")
    print(f"  lambda_max/lambda_min: median {np.median(ratio):.3f}  "
          f"[{np.percentile(ratio, 5):.3f}, {np.percentile(ratio, 95):.3f}]")

    banner("Section 3.1: det(Phi) = l^2 exactly")
    area = hp.nside2pixarea(64)
    print(f"  every pixel has area {area:.6e} sr;  l^2 = 4 pi / n_pix = "
          f"{mesh.grid_scale ** 2:.6e} sr   -> equal, so det(Phi)^(1/2) = l exactly")


if __name__ == "__main__":
    main()
