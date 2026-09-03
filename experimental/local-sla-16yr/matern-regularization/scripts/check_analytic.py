"""Section 2.1: the estimated sigma_hat_n against the analytic scalar (Smith 2022 Eq. 8).

On a full sphere the operator is stationary, so sigma_hat_n should be spatially constant
and should match the scalar up to discretization. It does, which is why nothing is lost
by estimating it per cell -- the land mask is the only thing that makes it a field.
"""

import numpy as np

from matern import Mesh, analytic_variance, delta_hat, normalisation, latex_row
from common import banner, parse_args

CASES = [(1, 4.0), (1, 8.0), (1, 16.0), (2, 4.0), (2, 8.0), (2, 16.0)]


def main():
    args = parse_args(__doc__, nside={"type": int, "default": 32})
    mesh = Mesh(args.nside)  # full sphere, no land mask

    banner(f"Table: estimated vs analytic marginal std, nside={args.nside}, full sphere",
           "tab:analytic")
    noise = 0.5 * 2 * 1.645 * np.sqrt(2.0 / args.n_samples)  # sigma = sqrt(sigma^2)
    print(f"  estimator noise at n_samples={args.n_samples} accounts for ~{noise:.1%} of the spread")
    print(latex_row("$M$", r"$\rhohat$", r"$\dhat$", r"estimated $\shat_n$", "spread",
                    "estimated / analytic"))
    for order, rho_hat in CASES:
        sigma = normalisation(mesh.operator(rho_hat, order), order, args.n_samples, args.seed)
        analytic = np.sqrt(analytic_variance(rho_hat, order))
        spread = (np.percentile(sigma, 95) - np.percentile(sigma, 5)) / np.median(sigma)
        print(latex_row(order, f"{rho_hat:g}", f"{delta_hat(rho_hat, order):.4f}",
                        f"{np.median(sigma):.5f}", f"{spread:.1%}".replace("%", r"\%"),
                        f"{np.median(sigma) / analytic:.3f}"))


if __name__ == "__main__":
    main()
