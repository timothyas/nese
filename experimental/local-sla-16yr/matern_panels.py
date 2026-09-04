"""Four views of one forecast, to see what the Matérn penalty is actually penalising.

The regularization term of `nese_training.losses.MaternPenalty` is derived and validated
in `matern-regularization/`, and its value has so far only been inferred from the
difference between two training runs. This puts it on the map, beside the field it acts on
and the input coverage that field was diagnosed from.

    top left      which input bin, if any, observed each cell (as in movie.py)
    top right     the predicted SLA field, in metres (as in movie.py)
    bottom left   the squared gradient magnitude, ell^2 |grad x|^2
    bottom right  the operator the loss applies, A_hat^M Sigma_hat x

Both bottom panels are computed on the *normalised* field, because that is the space the
loss sees: `post_processors` run only for validation metrics, never in `_compute_loss`, so
`MaternPenalty` receives mean-std normalised model output. The forecast store has been
post-processed back to metres, so this undoes that with the same pooled statistics
training normalised with. The mean matters and not just the scale -- A_hat is affine in a
constant, so A_hat (x - mu)/sigma is not A_hat x / sigma.

The operator itself comes from `matern-regularization/scripts/matern.py` rather than from
`nese_training.matern`: the plotting environment is torch-free by design (see data.py),
and `training/tests/test_matern.py::test_matches_reference_implementation` binds the two
elementwise, so the figure cannot drift from the loss.

Usage: `python matern_panels.py [YYYY-MM-DD]`, from this directory.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cmocean  # noqa: F401  -- registers the "cmo.*" colormaps

sys.path.append(f"../..")
sys.path.append(f"matern-regularization/scripts")
from regions import set_region, projection, draw_cells
import data
import matern
import movie

plt.style.use(f"../../nice.mplstyle")

#REGION = "global"
#REGION = "tropical_pacific"
#REGION = "north_atlantic"
#REGION = "southern_ocean"
REGION = "south_atlandian"

# The Matérn operator, named and defaulted as in nese_training.losses.MaternPenalty so the
# figure and training.yaml are set from the same vocabulary.
ORDER = 8                 # M, the number of times the operator is applied
RHO_HAT = 8.0             # correlation range in grid cells
MIN_NEIGHBOURS = 3        # drop the term below this retained-neighbour count
N_RANDOMISATION = 1000    # randomisation samples behind sigma_hat
SEED = 0

# The gradient panel is |grad|^2, as asked for. Squaring makes it considerably more
# heavy-tailed than the Matérn panel beside it; True plots |grad| instead.
GRADIENT_SQRT = False

# Which forecast to draw. FHR None takes the longest lead in the store, which for a
# single-step model is the only one carrying a diagnostic at all.
DATE = "2015-07-15"
FHR = None

PERCENTILE = 98.0

# sigma_hat is a property of the mesh and the operator, not of the forecast, so it is
# estimated once per (mesh, rho_hat, M, samples, seed) and kept beside the run.
CACHE_DIR = f"{movie.RUN_DIR}/matern-normalisation"


def normalised_field(snapshot, run_dir=movie.RUN_DIR):
    """The predicted field in the space the loss sees it.

    Pooled over `sla` and every mission observing it, which is what
    `PooledObservationNormalizer` does in training.yaml -- per-mission statistics would
    carry each mission's epoch rather than the field.
    """
    stats = data.pooled_statistics(data.normalise_spec(movie.OBSERVED_AS), run_dir)
    x = (snapshot.values - float(stats["mean"]))/float(stats["stdev"])
    if not np.isfinite(x).all():
        raise ValueError(
            f"{int((~np.isfinite(x)).sum())} of {x.size} ocean cells are not finite. "
            "Filling them would be indistinguishable from a flat field in the operator."
        )
    return x


def tangent_gradient(mesh, x):
    """ell * grad(x) in the local (east, north) plane, by least squares on the stencil.

    Taylor expanding about cell i, sum_j w_ij (x_j - x_i) d_ij = 2 T_i grad x + O(d^3),
    with T_i the second moment of the same weights the operator uses. T_i is the identity
    at interior cells -- that is the consistency condition the weights were chosen to
    satisfy -- so this is a halving there, and the pseudo-inverse absorbs the rank
    deficiency at cells the land mask left with too few directions.

    Units are per grid cell, matching the operator, which is what makes the two bottom
    panels comparable.
    """
    offset = mesh.tangent_offsets()
    difference = np.where(mesh.valid, x[mesh.neighbour] - x[:, None], 0.0)
    weighted = np.einsum("nk,nk,nki->ni", mesh.weight, difference, offset)
    return np.einsum("nij,nj->ni", np.linalg.pinv(2*mesh.second_moment()), weighted)


def cached_normalisation(mesh, rho_hat=RHO_HAT, order=ORDER, cache=CACHE_DIR):
    """sigma_hat for this mesh and operator, estimated once and kept.

    A thousand sparse solves, ~1.6 s, and the result depends only on the cell set and the
    operator -- never on the date -- so every figure from a run would otherwise recompute
    the same field. The key carries everything sigma_hat varies with, so changing rho_hat,
    the order, the mask or the seed misses the cache rather than silently reusing a
    stale field. Delete the directory to force a re-estimate.
    """
    mesh_id = hashlib.sha1(mesh.cell_index.tobytes()).hexdigest()[:8]
    path = (f"{cache}/sigma.nside{mesh.nside}.{mesh_id}.rho{rho_hat:g}.M{order}"
            f".n{N_RANDOMISATION}.seed{SEED}.npy")
    if os.path.exists(path):
        print(f"sigma_hat from {path}")
        return np.load(path)

    sigma = matern.normalisation(mesh.operator(rho_hat, order), order, N_RANDOMISATION, SEED)
    os.makedirs(cache, exist_ok=True)
    np.save(path, sigma)
    print(f"sigma_hat estimated from {N_RANDOMISATION} samples, cached at {path}")
    return sigma


def matern_operator(mesh, x, rho_hat=RHO_HAT, order=ORDER):
    """A_hat^M Sigma_hat x, the quantity the penalty takes the norm of.

    Sigma_hat multiplies x *before* the operator is applied, not after: with
    C^{1/2} = Lambda_hat A_hat^-M the inverse square root is A_hat^M Sigma_hat. The other
    reading is a different quantity and fails the unit-variance check.
    """
    sigma = cached_normalisation(mesh, rho_hat, order)
    return matern.apply_operator(mesh.operator(rho_hat, order), sigma*x, order)


def symmetric_limits(values, percentile=PERCENTILE):
    """Limits for a diverging field, robust to the coastal tail.

    The stencil is first-order inconsistent where the land mask cut it, so boundary cells
    carry a few times the interior amplitude; taken from the maximum they would set the
    scale for the whole map.
    """
    level = np.nanpercentile(np.abs(values), percentile)
    return -level, level


def draw_inputs(fig, ax, inputs, multistep, initial):
    """The input-recency mask, with the legend movie.py hangs beneath it."""
    cmap = movie.input_cmap(multistep)
    draw_cells(ax, inputs, cmap=cmap, vmin=-.5, vmax=multistep + .5)
    labels = [
        f"$t_0-{step if step > 1 else ''}\\Delta t$" if step else "$t_0$"
        for step in reversed(range(multistep))
    ]
    # A figure legend rather than an axes one: an axes legend hung below the panel is
    # invisible to constrained layout, which would leave the panels at different heights.
    fig.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(cmap.colors[1:], labels)],
        loc="upper center",
        bbox_to_anchor=(.27, .575),
        ncol=multistep,
        frameon=False,
    )
    ax.set(title=f"Input observations\n$t_0$ = {initial.strftime(movie.DATE_FORMAT)}")


def draw_field(fig, ax, values, title, label, cmap, vmin, vmax, extend="neither"):
    """One map panel with its own colorbar. NaN is land, or a cell the penalty drops."""
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("0.9")
    collection = draw_cells(ax, values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(title=title)
    fig.colorbar(
        collection,
        ax=ax,
        orientation="horizontal",
        shrink=.7,
        pad=.05,
        aspect=30,
        label=label,
        extend=extend,
    )
    return collection


def panels(date=DATE, region=REGION, fhr=FHR, run_dir=movie.RUN_DIR):
    """The figure, and the loss value annotated on it."""
    xds = data.open_forecast(run_dir)
    fhr = int(xds["fhr"].values[-1]) if fhr is None else fhr
    forecast = xds.sel(t0=date, method="nearest").sel(fhr=fhr)

    initial = pd.Timestamp(forecast["t0"].values)
    valid = pd.Timestamp(forecast["valid_time"].values)
    multistep = xds.attrs["multistep_input"]
    print(f"t0 = {initial}, fhr = {fhr}, valid {valid}")

    x = normalised_field(forecast[movie.FIELD], run_dir)

    mesh = matern.Mesh(xds.attrs["healpix_nside"], cell_index=xds["cell"].values)
    # The same cells MaternPenalty drops: the Laplacian at a cell with no neighbours is
    # not a meaningful object, and the normalisation inflates by up to 20x there.
    active = mesh.n_valid >= MIN_NEIGHBOURS
    print(f"{mesh.n_cells} cells, {int((~active).sum())} dropped by "
          f"min_neighbours={MIN_NEIGHBOURS}")

    penalty = matern_operator(mesh, x)
    gradient = (tangent_gradient(mesh, x)**2).sum(-1)
    if GRADIENT_SQRT:
        gradient = np.sqrt(gradient)

    total = float((penalty[active]**2).sum())
    loss = total/int(active.sum())
    print(f"||A_hat^M Sigma_hat x||^2 = {total:.1f}, / N = {loss:.4f}")

    masked = lambda values: data.scatter_to_mesh(
        forecast[movie.FIELD].copy(data=np.where(active, values, np.nan)), xds.attrs
    )
    inputs = data.scatter_to_mesh(
        movie.input_observation_mask(xds["t0"].sel(t0=[initial]), xds.attrs).isel(t0=0),
        xds.attrs,
    )
    sla = data.scatter_to_mesh(forecast[movie.FIELD], xds.attrs)

    fig, axs = plt.subplots(
        2, 2, figsize=(13, 8), subplot_kw={"projection": projection(region)}
    )
    for ax in axs.ravel():
        set_region(ax, region)

    draw_inputs(fig, axs[0, 0], inputs, multistep, initial)

    extend, vmin, vmax = movie.get_vmin_vmax_extend(sla, vmin=None, vmax=None)
    draw_field(fig, axs[0, 1], sla, f"Predicted SLA\nvalid {valid.strftime(movie.DATE_FORMAT)}",
               movie.LABEL, "cmo.balance", vmin, vmax, extend)

    gradient_label = (
        r"$|\hat{\nabla} x|$" if GRADIENT_SQRT else r"$|\hat{\nabla} x|^2$"
    )
    draw_field(fig, axs[1, 0], masked(gradient),
               f"Squared gradient magnitude\n{gradient_label}, per grid cell",
               gradient_label + "  [normalised SLA]", "cmo.tempo",
               0.0, np.nanpercentile(gradient[active], PERCENTILE), "max")

    draw_field(fig, axs[1, 1], masked(penalty),
               "Matérn operator\n" + r"$\hat{A}^M\hat{\Sigma}x$",
               r"$\hat{A}^M\hat{\Sigma}x$  [normalised SLA]", "cmo.tarn_r",
               *symmetric_limits(penalty[active]), "both")

    fig.suptitle(
        rf"$\hat{{\rho}} = {RHO_HAT:g}$ cells "
        rf"(${RHO_HAT*mesh.grid_scale*matern.R_EARTH:.0f}$ km), $M = {ORDER}$, "
        rf"min_neighbours $= {MIN_NEIGHBOURS}$"
        "\n"
        rf"$\mathcal{{L}}_M = \|\hat{{A}}^M\hat{{\Sigma}}x\|_2^2/N = {loss:.4f}$"
        rf"    ($\|\cdot\|_2^2 = {total:.0f}$, $N = {int(active.sum())}$)"
    )
    fig.set_constrained_layout(True)
    return fig, loss


if __name__ == "__main__":

    date = sys.argv[1] if len(sys.argv) > 1 else DATE
    fig, loss = panels(date)

    save_dir = f"{movie.RUN_DIR}/figures-matern"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/{REGION}.{date}.M{ORDER}.rho{RHO_HAT}.jpeg"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Wrote {fname}")
