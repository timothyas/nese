"""Section 3.3: the two rejected fixes for the boundary stencil.

1. Gradient-consistency projection -- project w onto sum_j w_ij d_ij = 0 so the stencil
   annihilates a linear field, then rescale to restore trace(T)/2 = 1.
2. Trace renormalisation -- rescale boundary weights so trace(T)/2 = 1, keeping all
   weights positive. No longer strict zero-flux Neumann.
"""

import numpy as np

from matern import Mesh, apply_operator, normalisation, smooth_fields
from common import banner, parse_args

GROUPS = ["8 (interior)", "3-7", "1-2"]


def diagnostics(mesh, weight, label):
    original = mesh.weight
    mesh.weight = weight
    gradient, tensor = mesh.gradient_response(), mesh.second_moment()
    trace = np.trace(tensor, axis1=1, axis2=2) / 2
    print(f"  {label}")
    for name, m in mesh.group_masks():
        if name not in GROUPS:
            continue
        print(f"     {name:14s} |sum w d| median {np.median(gradient[m]):8.4f}"
              f"   trace/2 median {np.median(trace[m]):.4f}")
    mesh.weight = original


def project(mesh):
    """Least-squares projection onto gradient consistency, then trace rescale."""
    p, weight = mesh.tangent_offsets(), mesh.weight.copy()
    for i in range(mesh.n_cells):
        if mesh.n_valid[i] in (0, 8):
            continue
        basis = p[i][mesh.valid[i]]
        gram = basis.T @ basis
        if np.linalg.det(gram) < 1e-12:
            continue
        w = mesh.weight[i][mesh.valid[i]]
        w = w - basis @ np.linalg.solve(gram, basis.T @ w)
        trace = 0.5 * np.sum(w * (basis**2).sum(1)) / 2
        if trace > 1e-9:
            w = w / trace
        weight[i][mesh.valid[i]] = w
    return weight


def trace_renormalise(mesh):
    trace = 0.5 * np.sum(mesh.weight * mesh.distance**2, 1) / 2
    return np.where(trace[:, None] > 1e-9, mesh.weight / np.maximum(trace, 1e-9)[:, None],
                    mesh.weight)


def main():
    args = parse_args(__doc__, order={"type": int, "default": 1},
                      rho_hat={"type": float, "default": 4.0})
    mesh = Mesh.from_anemoi_dataset(args.dataset)

    banner("Section 3.3: consistency of the boundary stencil")
    diagnostics(mesh, mesh.weight, "as implemented (zero-flux, drop land fluxes)")
    projected = project(mesh)
    diagnostics(mesh, projected, "rejected fix 1: gradient-consistency projection")
    negative = (projected < -1e-9).sum()
    print(f"     -> {negative} negative weights of {int(mesh.valid.sum())} edges "
          f"(min {projected.min():.1f}); likely costs positive-definiteness")

    renormalised = trace_renormalise(mesh)
    diagnostics(mesh, renormalised, "rejected fix 2: trace renormalisation at the boundary")

    banner("effect of fix 2 on the coastal penalty")
    x = smooth_fields(mesh, 50, args.seed)
    interior = mesh.n_valid == 8
    for label, weight in [("zero-flux", mesh.weight), ("trace-renormalised", renormalised)]:
        operator = mesh.operator(args.rho_hat, args.order, weight=weight)
        sigma = normalisation(operator, args.order, args.n_samples, args.seed)
        v = apply_operator(operator, sigma * x, args.order)
        base = (v[:, interior] ** 2).mean()
        cells = "  ".join(
            f"{name} {np.median(sigma[m]) / np.median(sigma[interior]):.2f} / "
            f"{(v[:, m] ** 2).mean() / base:.2f}x"
            for name, m in mesh.group_masks() if name != "8 (interior)")
        print(f"  {label:20s} (sigma ratio / penalty)   {cells}")


if __name__ == "__main__":
    main()
