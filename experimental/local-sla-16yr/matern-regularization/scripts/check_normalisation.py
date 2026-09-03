"""Section 2: verify the normalisation matrix and which side sigma acts on.

  C^{1/2} = Lambda A^-M det(Phi)^{-1/2},  Lambda = det(Phi)^{1/2-M}/sigma_hat
  => C^{-1} = Lambda^{-1} A^{2M} Lambda^{-1}
  => L_M    = (lambda_M/2) || A_hat^M (sigma_hat * x) ||^2   -- sigma_hat multiplies x BEFORE A

Checks: (a) x ~ N(0,C) has unit marginal variance at every cell, including the
isolated ones; (b) E[x^T C^-1 x] = n_cells; (c) the per-cell contributions are 1
everywhere, so the normalisation does not favour any part of the mesh.
The alternative reading || A_hat^M x / sigma_hat ||^2 fails (b) badly, which is the point.
"""

import numpy as np
import scipy.sparse.linalg as spl

from matern import Mesh, apply_operator, delta_hat, normalisation
from common import banner, parse_args


def main():
    args = parse_args(__doc__, order={"type": int, "default": 1},
                      rho_hat={"type": float, "default": 4.0})
    mesh = Mesh.from_anemoi_dataset(args.dataset)
    operator = mesh.operator(args.rho_hat, args.order)

    banner(f"M={args.order}  rho_hat={args.rho_hat} cells  "
           f"delta_hat={delta_hat(args.rho_hat, args.order):.3f}  n_cells={mesh.n_cells}")

    lu = spl.splu(operator)
    rng = np.random.default_rng(args.seed)
    total, draws = np.zeros(mesh.n_cells), []
    for k in range(args.n_samples):
        y = rng.standard_normal(mesh.n_cells)
        for _ in range(args.order):
            y = lu.solve(y)
        total += y**2
        if k < 300:
            draws.append(y)
    sigma = np.sqrt(total / args.n_samples)

    x = np.array(draws) / sigma                      # = Lambda A^-M w ~ N(0, C)
    v = apply_operator(operator, sigma * x, args.order)

    print("  (a) marginal variance of x ~ N(0,C)      target 1.0")
    for label, m in mesh.group_masks():
        print(f"        {label:14s} n={int(m.sum()):5d}  {(x[:, m] ** 2).mean(0).mean():.3f}")

    print(f"\n  (b) E[x' C^-1 x] = {(v ** 2).sum(1).mean():8.1f}   target n_cells = {mesh.n_cells}")
    wrong = apply_operator(operator, x, args.order) / sigma
    print(f"      same with || A^M x / sigma ||^2 = {(wrong ** 2).sum(1).mean():8.1f}   (wrong side)")

    print("\n  (c) per-cell contribution v_i^2          target 1.0 everywhere")
    for label, m in mesh.group_masks():
        print(f"        {label:14s} {(v[:, m] ** 2).mean():.3f}")

    print("\n  implied prior variance without Lambda, relative to interior:")
    var = total / args.n_samples
    interior = np.median(var[mesh.n_valid == 8])
    for label, m in mesh.group_masks():
        print(f"        {label:14s} {np.median(var[m]) / interior:6.2f}")


if __name__ == "__main__":
    main()
