"""How the Matérn penalty responds to its two parameters, on one forecast.

`matern_panels.py` draws the term for a single setting. This sweeps the two parameters
that shape it and shows, for each, both the size of the penalty and which half of the
operator is producing it.

    A_hat = delta_hat I - L,    delta_hat = 8 nu / rho_hat^2,    nu = 2M - 1

so rho_hat moves the balance between the zeroth-order term, which penalises amplitude, and
the elliptic term, which penalises curvature, by a factor of ~1000 across the range swept
here. That balance is invisible in the total, which is why both are plotted together.

L_M is the loss term itself, ||A_hat^M Sigma_hat x||^2 / N over the active cells, exactly
as MaternPenalty.forward computes it -- so a value here is comparable to the one
matern_panels.py annotates, and to `loss_weights[1]` in training.yaml.

Usage: `python matern_parameters.py [YYYY-MM-DD]`, from this directory.
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

sys.path.append(f"../..")
sys.path.append(f"matern-regularization/scripts")
import data
import matern
import movie
from matern_panels import (
    DATE,
    FHR,
    MIN_NEIGHBOURS,
    cached_normalisation,
    normalised_field,
)

plt.style.use(f"../../nice.mplstyle")

RUN_DIR = f"runs/v3"
FIELD = "sla"

# Deliberately not read from matern_panels: those are that figure's settings, and this
# script sweeps rather than picks.
SWEEP_RHO_HAT = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SWEEP_RHO_HAT_ORDER = 1

SWEEP_ORDER = (1, 2, 4, 8)
SWEEP_ORDER_RHO_HAT = 8.0

LOSS_COLOR = "0.15"
DELTA_COLOR = "#d95f02"
LAPLACIAN_COLOR = "#1b9e77"


def decompose(mesh, x, rho_hat, order, active):
    """L_M at one (rho_hat, M), and which half of A_hat produced it.

    The split is of the *last* application of the operator: with y = sigma_hat x and
    z = A_hat^(M-1) y, the penalty field is exactly delta_hat*z - L z, for any M. At M = 1
    that is the plain delta_hat y versus L y; above it, everything before the final
    application is folded into z rather than expanded binomially, which would give M+1
    terms rather than two.

    The two terms do not oppose: -L is positive semi-definite, since
    <z, -Lz> = 0.5 sum_ij w_ij (z_i - z_j)^2 with every w_ij >= 0, so <delta_hat z, -Lz>
    is non-negative for any z whatsoever. Measured, the cosine runs 0.45 to 0.99. They
    reinforce, which is what makes A_hat positive definite and C a valid covariance, and
    it means their squares over-add rather than cancelling. The magnitudes are still the
    right basis for a relative contribution -- they are what each half puts into A_hat,
    and they sum to one -- but the reported fractions understate how much of ||.||^2 comes
    from the cross term. `cosine` is reported so that is visible rather than assumed.
    """
    operator = mesh.operator(rho_hat, order)
    delta = matern.delta_hat(rho_hat, order)
    laplacian = mesh.laplacian()

    y = cached_normalisation(mesh, rho_hat, order)*x
    z = matern.apply_operator(operator, y, order - 1)

    term_delta = delta*z
    term_laplacian = -(laplacian @ z)

    penalty = matern.apply_operator(operator, y, order)
    # The reported fractions have to describe the field whose norm is plotted, not a
    # neighbouring quantity, so this is asserted rather than trusted.
    assert np.allclose(term_delta + term_laplacian, penalty), (rho_hat, order)

    norm = lambda values: float(np.linalg.norm(values[active]))
    size_delta, size_laplacian = norm(term_delta), norm(term_laplacian)

    return {
        "rho_hat": rho_hat,
        "order": order,
        "delta_hat": delta,
        "loss": float((penalty[active]**2).sum())/int(active.sum()),
        "frac_delta": size_delta/(size_delta + size_laplacian),
        "frac_laplacian": size_laplacian/(size_delta + size_laplacian),
        "cosine": float(
            (term_delta[active]*term_laplacian[active]).sum()/(size_delta*size_laplacian)
        ),
    }


def sweep(mesh, x, active, rho_hats, orders):
    """`decompose` over paired parameter values, in order."""
    return [
        decompose(mesh, x, rho_hat, order, active)
        for rho_hat, order in zip(rho_hats, orders)
    ]


def draw_panel(ax, results, key, xlabel, title):
    """L_M on a log axis, with the two-term split on a twin linear axis."""
    xvalues = [result[key] for result in results]

    ax.plot(xvalues, [r["loss"] for r in results], "o-", color=LOSS_COLOR, lw=2, zorder=3)
    ax.set(xlabel=xlabel, title=title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xvalues, [f"{v:g}" for v in xvalues])
    ax.set_ylabel(r"$\mathcal{L}_M = \|\hat{A}^M\hat{\Sigma}x\|_2^2/N$", color=LOSS_COLOR)
    ax.tick_params(axis="y", colors=LOSS_COLOR)

    # A twin axis rather than a second panel: the crossover in the split is only legible
    # against the value of the loss at the same parameter.
    twin = ax.twinx()
    delta = [r["frac_delta"] for r in results]
    laplacian = [r["frac_laplacian"] for r in results]
    # No fill under these: on a log left axis a shaded region reads as area under the
    # loss curve, which is not what it is.
    twin.plot(xvalues, delta, "s--", color=DELTA_COLOR, lw=1.5,
              label=r"$\hat{\delta}$ term")
    twin.plot(xvalues, laplacian, "^--", color=LAPLACIAN_COLOR, lw=1.5,
              label=r"$\nabla\cdot K\nabla$ term")
    twin.axhline(.5, color="0.7", lw=.8, ls=":", zorder=0)
    twin.set_ylim(0, 1)
    twin.set_ylabel("relative contribution to $\\hat{A}$", color=DELTA_COLOR)
    twin.tick_params(axis="y", colors=DELTA_COLOR)
    return twin


def report(results, swept):
    """The table behind the figure, including the checks the plot cannot show."""
    print(f"\n{'rho_hat':>8} {'M':>3} {'delta_hat':>10} {'L_M':>12} "
          f"{'frac_delta':>11} {'frac_lap':>9} {'cosine':>8}")
    for r in results:
        print(f"{r['rho_hat']:8g} {r['order']:3d} {r['delta_hat']:10.4f} "
              f"{r['loss']:12.4g} {r['frac_delta']:11.3f} "
              f"{r['frac_laplacian']:9.3f} {r['cosine']:8.3f}")
    losses = [r["loss"] for r in results]
    print(f"  ({swept}: L_M spans {max(losses)/min(losses):.3g}x)")


def parameters(date=DATE, fhr=FHR, run_dir=RUN_DIR):
    """The figure, and the two sweeps behind it."""
    xds = data.open_forecast(run_dir)
    fhr = int(xds["fhr"].values[-1]) if fhr is None else fhr
    forecast = xds.sel(t0=date, method="nearest").sel(fhr=fhr)
    initial = pd.Timestamp(forecast["t0"].values)
    print(f"t0 = {initial}, fhr = {fhr}")

    x = normalised_field(forecast[FIELD], run_dir)
    mesh = matern.Mesh(xds.attrs["healpix_nside"], cell_index=xds["cell"].values)
    active = mesh.n_valid >= MIN_NEIGHBOURS

    by_rho = sweep(mesh, x, active, SWEEP_RHO_HAT,
                   [SWEEP_RHO_HAT_ORDER]*len(SWEEP_RHO_HAT))
    by_order = sweep(mesh, x, active, [SWEEP_ORDER_RHO_HAT]*len(SWEEP_ORDER),
                     SWEEP_ORDER)

    report(by_rho, f"M = {SWEEP_RHO_HAT_ORDER}")
    report(by_order, rf"rho_hat = {SWEEP_ORDER_RHO_HAT:g}")

    # The panels overlap at one setting, computed independently in each sweep.
    shared = [
        (a, b) for a in by_rho for b in by_order
        if a["rho_hat"] == b["rho_hat"] and a["order"] == b["order"]
    ]
    for a, b in shared:
        assert a["loss"] == b["loss"], (a, b)
        print(f"\nshared point rho_hat={a['rho_hat']:g}, M={a['order']}: "
              f"L_M = {a['loss']:.6g} in both panels")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    draw_panel(axs[0], by_rho, "rho_hat", r"$\hat{\rho}$  [grid cells]",
               rf"$M = {SWEEP_RHO_HAT_ORDER}$")
    twin = draw_panel(axs[1], by_order, "order", "$M$",
                      rf"$\hat{{\rho}} = {SWEEP_ORDER_RHO_HAT:g}$ cells")

    fig.legend(*twin.get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(.5, -.06))
    fig.suptitle(f"Matérn penalty sensitivity, $t_0$ = "
                 f"{initial.strftime(movie.DATE_FORMAT)}, "
                 f"min_neighbours = {MIN_NEIGHBOURS}")
    fig.set_constrained_layout(True)
    return fig, by_rho, by_order


if __name__ == "__main__":

    date = sys.argv[1] if len(sys.argv) > 1 else DATE
    fig, by_rho, by_order = parameters(date)

    save_dir = f"{RUN_DIR}/figures"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/matern-parameters.{date}.jpeg"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Wrote {fname}")
