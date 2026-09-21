"""Visualize the radius-masked processor stencil on the HEALPix latent mesh.

Sibling of ../local-sla-02yr/graphs/plot_sliding_window.py (whose map helpers
and styling are reused). The stencil is the ``(hidden, to, hidden)`` edge set
of graph.pt plus self attention, exactly as RadiusTransformerProcessor builds
its mask; the N-layer reach is that mask propagated N times.

Also prints how many query-key pairs each attention layout computes, as a
first, timing-free look at efficiency:
  - sliding window: the current +/-W band in RING order (flash-attention)
  - exact disk: the mask nonzeros (a perfect sparse kernel)
  - block-sparse: nonzero BLOCK x BLOCK tiles of the mask (flex_attention),
    for the mesh in NEST vs RING order

Run in the `nese-anemoi` conda env:
    conda run -n nese-anemoi python plot_radius_footprint.py
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local-sla-02yr" / "graphs"))
from plot_sliding_window import FEATURES  # noqa: E402
from plot_sliding_window import QUERY_POINTS  # noqa: E402
from plot_sliding_window import add_cartopy_features  # noqa: E402
from plot_sliding_window import great_circle_km  # noqa: E402

GRAPH = "graph.1200.pt"
HIDDEN_NAME = "hidden"
N_PASSES = 4                # processor layers in training.yaml
N_PASSES_PRODUCTION = 16    # only reported, not plotted
WINDOW_SIZE = 512           # sliding-window baseline, +/-W tokens
BLOCK = 128                 # flex_attention default block size
OUT_DIR = "figures/radius-1200"

C_BG, C_REACH, C_WIN, C_QUERY = "darkgray", "dodgerblue", "navy", "goldenrod"
_ms_latent, _ms_reach, _ms_window, _ms_query = 3, 4.5, 6, 70
_lms_latent, _lms_reach, _lms_window, _lms_query = 5, 6, 6, 8


def load_stencil(graph_path):
    """Latent node lat/lon (deg, token order) and the sparse [receiver, sender] mask."""
    graph = torch.load(graph_path, weights_only=False, map_location="cpu")
    lat, lon = np.degrees(graph[HIDDEN_NAME].x.numpy()).T
    senders, receivers = graph[HIDDEN_NAME, "to", HIDDEN_NAME].edge_index.numpy()
    n = len(lat)
    mask = sp.csr_matrix((np.ones(len(senders), dtype=bool), (receivers, senders)), shape=(n, n))
    mask = (mask + sp.identity(n, dtype=bool, format="csr")).astype(bool)
    return lat, lon % 360.0, mask


def propagate(mask, qi, n_passes):
    reach = mask[qi].toarray().ravel()
    for _ in range(n_passes - 1):
        reach = (mask @ reach) > 0
    return reach


def nonzero_blocks(mask, order):
    """Number of BLOCK x BLOCK tiles with any nonzero when nodes are laid out in ``order``."""
    rows, cols = mask.nonzero()
    return len(set(zip(order[rows] // BLOCK, order[cols] // BLOCK)))


def cost_table(lat, lon, mask):
    n = len(lat)
    nside = hp.npix2nside(n)
    nest_index = hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)
    ring_index = hp.nest2ring(nside, nest_index)
    offsets = np.arange(n)
    band = (np.minimum(offsets + WINDOW_SIZE, n - 1) - np.maximum(offsets - WINDOW_SIZE, 0) + 1).sum()
    rows = {
        f"sliding window +/-{WINDOW_SIZE} (RING)": band,
        "exact disk (mask nonzeros)": mask.nnz,
        f"block-sparse {BLOCK}, NEST order": nonzero_blocks(mask, nest_index) * BLOCK**2,
        f"block-sparse {BLOCK}, RING order": nonzero_blocks(mask, ring_index) * BLOCK**2,
    }
    print(f"\n=== query-key pairs per head per layer, N={n} (dense = {n * n:,}) ===")
    for name, pairs in rows.items():
        print(f"{name:38s}: {pairs:>12,}  ({100 * pairs / n**2:5.1f}% of dense, {pairs / n:6.0f} keys/query)")


def report(lat, lon, qi, mask, reach):
    d_km = great_circle_km(lat, lon, lat[qi], lon[qi])
    disk = mask[qi].toarray().ravel()
    production = propagate(mask, qi, N_PASSES_PRODUCTION)
    print(f"\n--- query node {qi} (lat={lat[qi]:.2f}, lon={lon[qi]:.2f}) ---")
    print(f"single-layer stencil : {disk.sum()} nodes, max dist {d_km[disk].max():.0f} km")
    print(f"{N_PASSES}-layer reach        : {100 * reach.mean():.1f}% of mesh, max dist {d_km[reach].max():.0f} km")
    print(f"{N_PASSES_PRODUCTION}-layer reach       : {100 * production.mean():.1f}% of mesh, max dist {d_km[production].max():.0f} km")


def make_figure(lat, lon, qi, disk, reach, out_path):
    lon180 = ((lon + 180) % 360) - 180
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.Orthographic(central_longitude=lon180[qi], central_latitude=lat[qi])
    )
    ax.set_global()
    add_cartopy_features(ax, FEATURES)

    bg, reach_only = ~reach, reach & ~disk
    ax.scatter(lon180[bg], lat[bg], s=_ms_latent, c=C_BG, alpha=0.85, transform=pc, linewidths=0, zorder=2)
    ax.scatter(lon180[reach_only], lat[reach_only], s=_ms_reach, c=C_REACH, alpha=0.85, transform=pc, linewidths=0, zorder=2.5)
    ax.scatter(lon180[disk], lat[disk], s=_ms_window, c=C_WIN, alpha=0.9, transform=pc, linewidths=0, zorder=3)
    ax.scatter(lon180[qi], lat[qi], s=_ms_query, marker="D", c=C_QUERY, edgecolors="white", linewidths=0.8, transform=pc, zorder=4)

    handles = [
        Line2D([], [], marker="o", ls="", color=C_BG, ms=_lms_latent, label="Latent Node"),
        Line2D([], [], marker="o", ls="", color=C_REACH, ms=_lms_reach, label=f"Reach after {N_PASSES} passes"),
        Line2D([], [], marker="o", ls="", color=C_WIN, ms=_lms_window, label="Attention Stencil"),
        Line2D([], [], marker="D", ls="", color=C_QUERY, mec="white", ms=_lms_query, label="Query Node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, 0.075), handlelength=0.8, handletextpad=0.2, columnspacing=1.2, borderaxespad=0.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat, lon, mask = load_stencil(GRAPH)
    cost_table(lat, lon, mask)

    for qlat, qlon in QUERY_POINTS:
        dlon = np.minimum(np.abs(lon - qlon), 360.0 - np.abs(lon - qlon))
        qi = int(np.argmin((lat - qlat) ** 2 + dlon**2))
        disk = mask[qi].toarray().ravel()
        reach = propagate(mask, qi, N_PASSES)
        report(lat, lon, qi, mask, reach)
        make_figure(lat, lon, qi, disk, reach, out_dir / f"radius_q{qlat:g}_{qlon:g}.jpeg")


if __name__ == "__main__":
    main()
