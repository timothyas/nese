"""Section 3.1: why w_ij = 4/(8 dhat^2) over eight neighbours rather than the
two-point flux w_ij = s_ij/d_ij over the four shared faces.

Consistency requires, from the Taylor expansion of sum_j w_ij (x_j - x_i),

    sum_j w_ij d_ij            = 0        (annihilate a linear field)
    0.5 sum_j w_ij d_ij d_ij^T = I        (reproduce grid_scale^2 * grad^2)

Two-point flux is only consistent where the line joining cell centres is orthogonal
to the shared face. HEALPix violates that towards the poles, and the cost is measured
here.
"""

import numpy as np

from matern import Mesh
from common import banner, parse_args

ZONES = [("equatorial (|lat| < 41.8)", lambda lat: np.abs(lat) < 41.8),
         ("polar      (|lat| > 41.8)", lambda lat: np.abs(lat) >= 41.8)]


def report(mesh, weight, label):
    original = mesh.weight
    mesh.weight = weight
    tensor = mesh.second_moment()
    trace = np.trace(tensor, axis1=1, axis2=2) / 2
    eig = np.linalg.eigvalsh(tensor)
    ratio = eig[:, 1] / eig[:, 0]
    gradient = mesh.gradient_response()
    mesh.weight = original

    print(f"  {label}")
    print(f"     trace(T)/2      median {np.median(trace):.4f}"
          f"  [{np.percentile(trace, 5):.4f}, {np.percentile(trace, 95):.4f}]")
    print(f"     lam_max/lam_min median {np.median(ratio):.3f}"
          f"  95th pct {np.percentile(ratio, 95):.3f}")
    print(f"     |sum w d|       median {np.median(gradient):.4f}")
    for name, test in ZONES:
        m = test(mesh.lat)
        print(f"       {name}: trace/2 {np.median(trace[m]):.4f}"
              f"   ratio {np.median(ratio[m]):.3f}")


def main():
    args = parse_args(__doc__, nside={"type": int, "default": 64})
    mesh = Mesh(args.nside)

    banner(f"consistency of the discrete operator, nside={args.nside}, full sphere")
    report(mesh, mesh.face_weights(), "two-point flux   w = s/d, 4 shared faces")
    report(mesh, mesh.weight, "implemented      w = 4/(8 dhat^2), 8 neighbours")

    slot = np.zeros(8, int)
    corners = mesh.face_weights() > 0
    for j in range(8):
        slot[j] = corners[:, j].sum()
    banner("which neighbour slots carry the four shared faces")
    print("  slot:      SW     W    NW     N    NE     E    SE     S")
    print("  count: " + " ".join(f"{v:5d}" for v in slot))
    print("  -> the shared faces are the SW/NW/NE/SE slots, so HEALPix's true grid")
    print("     axes lie on the diagonals of healpy's compass labelling.")


if __name__ == "__main__":
    main()
