"""Table 2 (tab:sigma): sigma / sigma_interior by neighbour count and correlation range.

Reproduces Fig. 4 of Smith (2022) on this mesh: dropping land fluxes weakens A locally,
inflating the implied prior variance near the coast, and the inflation grows with the
correlation length.
"""

import numpy as np

from matern import Mesh, R_EARTH, normalisation, latex_row
from common import banner, parse_args

RANGES = [2.0, 4.0, 8.0, 16.0]


def main():
    args = parse_args(__doc__, order={"type": int, "default": 1})
    mesh = Mesh.from_anemoi_dataset(args.dataset)
    groups = [g for g in mesh.group_masks() if g[0] != "8 (interior)"]

    ratios = {}
    for rho_hat in RANGES:
        sigma = normalisation(
            mesh.operator(rho_hat, args.order), args.order, args.n_samples, args.seed
        )
        interior = np.median(sigma[mesh.n_valid == 8])
        ratios[rho_hat] = [np.median(sigma[m]) / interior for _, m in groups]

    banner(f"Table 2: sigma/sigma_interior at M={args.order}", "tab:sigma")
    print(latex_row(r"$\rhohat$ (cells)", *(f"{r:g}" for r in RANGES)))
    print(latex_row(r"range (\si{\kilo\metre})",
                    *(f"{r * mesh.grid_scale * R_EARTH:.0f}" for r in RANGES)))
    for i, (label, _) in enumerate(groups):
        name = {"3-7": "3--7 neighbours", "1-2": "1--2 neighbours",
                "0 (isolated)": "isolated"}[label]
        print(latex_row(name, *(f"{ratios[r][i]:.2f}" for r in RANGES)))


if __name__ == "__main__":
    main()
