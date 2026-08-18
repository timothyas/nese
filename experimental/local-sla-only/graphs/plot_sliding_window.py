"""Visualize the sliding-window transformer processor on the HEALPix latent mesh.

Analogous to Figure 2 of the AIFS preprint (arXiv:2406.01465), and a sibling of
nested-eagle-paper/notebooks/plot_sliding_window.py, but for the single-resolution
HEALPix hidden mesh used here. The processor is a transformer with sliding-window
attention over the latent tokens (see training.yaml: TransformerProcessor,
window_size=512, num_layers=4).

Key mechanics:
  - The hidden mesh comes straight out of the graph .pt file. Node order in
    ``graph["hidden"].x`` *is* the transformer token-sequence order, so a window in
    sequence index maps directly onto nodes on the sphere.
  - anemoi's window_size is symmetric: attention.py builds the mask as
    ``abs(q_idx - kv_idx) <= window_size``, so window_size=W means +/-W tokens, a
    2W+1 token window. Each processor layer widens the receptive field by another
    +/-W, clipped at the sequence ends (information cannot wrap).
  - Unlike the nested-eagle mesh, this sequence is NOT latitude-sorted. anemoi's
    HEALPixNodes emits ``hp.pix2ang(nside, ..., nest=True)``, i.e. HEALPix NESTED
    order: a hierarchical Z-order (Morton) curve within each of the 12 base pixels.
    A contiguous window is therefore a quadtree patch, not a thin zonal band -- it
    is spatially clustered, but with the characteristic Morton jumps, and it breaks
    across base-pixel boundaries. That structure is the point of the figure.

Set GRAPHS and QUERY_POINTS below; every (graph, query) pair gets its own figure.

Run in the `nese-anemoi` conda env (needs both torch and cartopy):
    conda run -n nese-anemoi python plot_sliding_window.py
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
SHOW_TITLE = False
SHOW_CAPTION = False

# One figure per (graph, query point) pair.
GRAPHS = [
    "graph.rmData.smt10.cutoff200km.pt",
    "graph.rmDataHidden.smt10.cutoff200km.pt",
]
QUERY_POINTS = [            # (lat, lon) in deg; lon in 0..360
    (40.0, 330.0),          # North Atlantic
    (0.0, 200.0),           # equatorial Pacific
    (70.0, 0.0),            # Nordic Seas
]

HIDDEN_NAME = "hidden"      # node set the processor runs on
WINDOW_SIZE = 512           # anemoi window_size (+/-W); full window = 2W+1 tokens
N_PASSES = 4                # processor layers whose reach to shade; each pass
                            # widens the receptive field by +/-WINDOW_SIZE tokens
OUT_DIR = "figures/sliding-window"

# Hemisphere shown by the main map, centered on the query node. 180 is the full
# visible disk; less crops in around the query.
MAIN_HALF_ANGLE_DEG = 180.0

# Colors: recessive grey background, one accent hue for the window, a distinct
# marker shape (diamond) for the query node.
C_BG = "darkgray"           # all latent nodes (neutral, recessive)
C_REACH = "dodgerblue"       # reach after N_PASSES passes
C_WIN = "navy"        # single-pass attention window (accent hue)
C_QUERY = "goldenrod"       # query node

# Marker sizes
_ms_latent, _ms_reach, _ms_window, _ms_query = 3, 4.5, 6, 70

# Legend marker sizes
_lms_latent, _lms_reach, _lms_window, _lms_query = 5, 6, 6, 8

FEATURES = {
    "coastlines": {"resolution": "110m", "color": "0.35", "linewidth": 0.5},
    "land": {"facecolor": "0.93", "edgecolor": "none"},
}

_CARTOPY_FEATURES = {
    "borders": cfeature.BORDERS,
    "states": cfeature.STATES,
    "land": cfeature.LAND,
    "ocean": cfeature.OCEAN,
    "lakes": cfeature.LAKES,
    "rivers": cfeature.RIVERS,
}


def add_cartopy_features(ax, features):
    """Add cartopy map features (coastlines, land, borders, ...) to an axis.

    Local copy of eagle.tools.visualize.add_cartopy_features, so this script does
    not drag in eagle-tools for one function.
    """
    for name, kwargs in features.items():
        kwargs = kwargs or {}
        key = name.lower()
        if key == "coastlines":
            ax.coastlines(**kwargs)
        elif key in _CARTOPY_FEATURES:
            ax.add_feature(_CARTOPY_FEATURES[key], **kwargs)
        else:
            raise ValueError(
                f"Unknown cartopy feature '{name}'. Valid options are: "
                f"'coastlines', {', '.join(repr(k) for k in _CARTOPY_FEATURES)}."
            )


def load_hidden_mesh(graph_path, nodes_name=HIDDEN_NAME):
    """Latitude/longitude (deg) of the latent nodes, in token-sequence order.

    anemoi stores node coordinates in radians as ``graph[name].x``, with lon on
    0..2pi. Returns lon on 0..360.
    """
    graph = torch.load(graph_path, weights_only=False, map_location="cpu")
    if nodes_name not in graph.node_types:
        raise KeyError(
            f"{graph_path}: no node set '{nodes_name}'. Available: {graph.node_types}"
        )
    nodes = graph[nodes_name]
    x = np.degrees(nodes.x.cpu().numpy())
    lat, lon = x[:, 0], x[:, 1] % 360.0

    # Present only on graphs whose unconnected hidden nodes were pruned by
    # RemoveUnconnectedNodes; the token sequence is then that pruned subset.
    pruned = "indices_connected_nodes" in nodes.node_attrs()
    return lat, lon, pruned


def great_circle_km(lat, lon, qlat, qlon):
    """Great-circle distance from every node to the query point, in km."""
    v = hp.ang2vec(lon, lat, lonlat=True)
    q = hp.ang2vec(qlon, qlat, lonlat=True)
    return 6371.0 * np.arccos(np.clip(v @ q, -1.0, 1.0))


def analyze(lat, lon, qlat, qlon):
    """Resolve the query node and the token ranges it attends to."""
    n = len(lat)

    dlon = np.minimum(np.abs(lon - qlon), 360.0 - np.abs(lon - qlon))
    qi = int(np.argmin((lat - qlat) ** 2 + dlon**2))

    # Sliding-window token range around the query (contiguous in sequence).
    a, b = max(0, qi - WINDOW_SIZE), min(n, qi + WINDOW_SIZE + 1)
    win = np.zeros(n, dtype=bool)
    win[a:b] = True

    # Receptive field after N_PASSES passes, clipped at the sequence ends.
    ra, rb = max(0, qi - N_PASSES * WINDOW_SIZE), min(n, qi + N_PASSES * WINDOW_SIZE + 1)
    reach = np.zeros(n, dtype=bool)
    reach[ra:rb] = True

    # HEALPix base pixel (nside=1) of every node, computed from coordinates rather
    # than from the index, so it stays correct when the mesh has been pruned.
    base = hp.ang2pix(1, lon, lat, nest=True, lonlat=True)

    return dict(n=n, qi=qi, a=a, b=b, ra=ra, rb=rb, win=win, reach=reach, base=base)


def report(lat, lon, qlat, qlon, info, graph_path, pruned):
    """Print the numbers the figure is asserting."""
    n, qi, a, b, ra, rb, base = (
        info["n"], info["qi"], info["a"], info["b"], info["ra"], info["rb"], info["base"]
    )
    d_km = great_circle_km(lat, lon, lat[qi], lon[qi])
    wl, wd, wb = lat[a:b], d_km[a:b], base[a:b]
    rl, rb_pix = lat[ra:rb], base[ra:rb]

    print(f"\n=== {graph_path}  query ({qlat:g}, {qlon:g}) ===")
    print(f"total latent nodes    : {n}" + ("  (pruned mesh)" if pruned else ""))
    print(f"query node index      : {qi}  (lat={lat[qi]:.2f}, lon={lon[qi]:.2f}, base pixel {base[qi]})")
    print(f"window index range    : [{a}, {b})  n={b - a}  ({100 * (b - a) / n:.1f}% of mesh)")
    print(f"window latitude span  : {wl.min():.2f} .. {wl.max():.2f}  ({wl.max() - wl.min():.1f} deg)")
    print(f"window base pixels    : {sorted(set(wb.tolist()))}")
    print(f"window dist to query  : median {np.median(wd):.0f} km, max {wd.max():.0f} km")
    print(f"{N_PASSES}-pass reach range   : [{ra}, {rb})  n={rb - ra}  ({100 * (rb - ra) / n:.1f}% of mesh)")
    print(f"{N_PASSES}-pass latitude span : {rl.min():.2f} .. {rl.max():.2f}  ({rl.max() - rl.min():.1f} deg)")
    print(f"{N_PASSES}-pass base pixels   : {sorted(set(rb_pix.tolist()))}")


def make_figure(lat, lon, info, out_path, graph_path):
    """Main hemisphere map centered on the query, with a global locator inset."""
    qi, n, a, b = info["qi"], info["n"], info["a"], info["b"]
    win, reach = info["win"], info["reach"]
    qlat, qlon = lat[qi], lon[qi]

    show_reach = N_PASSES > 1
    bg = ~reach
    reach_plot = reach & ~win

    lon180 = ((lon + 180) % 360) - 180
    qlon180 = ((qlon + 180) % 360) - 180
    pc = ccrs.PlateCarree()

    def scatter_layers(ax, s_bg, s_reach, s_win, s_query):
        ax.scatter(lon180[bg], lat[bg], s=s_bg, c=C_BG, alpha=0.85,
                   transform=pc, linewidths=0, zorder=2)
        if show_reach:
            ax.scatter(lon180[reach_plot], lat[reach_plot], s=s_reach, c=C_REACH,
                       alpha=0.85, transform=pc, linewidths=0, zorder=2.5)
        ax.scatter(lon180[win], lat[win], s=s_win, c=C_WIN, alpha=0.9,
                   transform=pc, linewidths=0, zorder=3)
        ax.scatter(qlon180, qlat, s=s_query, marker="D", c=C_QUERY,
                   edgecolors="white", linewidths=0.8, transform=pc, zorder=4)

    fig = plt.figure(figsize=(8, 7))

    # --- Hemisphere centered on the query node -------------------------------
    # Orthographic keeps the window's shape on the sphere honest, which matters
    # because the NESTED-order window is a 2D patch rather than a zonal band.
    axA = fig.add_subplot(
        1, 1, 1,
        projection=ccrs.Orthographic(central_longitude=qlon180, central_latitude=qlat),
    )
    axA.set_global()
    add_cartopy_features(axA, FEATURES)
    scatter_layers(axA, _ms_latent, _ms_reach, _ms_window, _ms_query)
    if MAIN_HALF_ANGLE_DEG < 180.0:
        r = 6378137.0 * np.sin(np.radians(MAIN_HALF_ANGLE_DEG))
        axA.set_xlim(-r, r)
        axA.set_ylim(-r, r)

    # --- Legend + annotation -------------------------------------------------
    handles = [Line2D([], [], marker="o", ls="", color=C_BG, ms=_lms_latent,
                      label="Latent Node")]
    if show_reach:
        handles.append(Line2D([], [], marker="o", ls="", color=C_REACH, ms=_lms_reach,
                              label=f"Reach after {N_PASSES} passes"))
    handles += [
        Line2D([], [], marker="o", ls="", color=C_WIN, ms=_lms_window,
               label="Attention Window"),
        Line2D([], [], marker="D", ls="", color=C_QUERY, mec="white", ms=_lms_query,
               label="Query Node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.075),
               handlelength=0.8, handletextpad=0.2, columnspacing=1.2,
               borderaxespad=0.0)

    if SHOW_TITLE:
        fig.suptitle(
            "Sliding-window attention on the HEALPix latent mesh "
            f"(window_size={WINDOW_SIZE}: ±{WINDOW_SIZE} tokens = "
            f"{b - a} nodes, {100 * (b - a) / n:.0f}% of mesh)",
            fontsize=12, y=0.97,
        )
    if SHOW_CAPTION:
        d_km = great_circle_km(lat, lon, qlat, qlon)
        caption = (
            f"HEALPix NESTED order makes one window a quadtree patch spanning "
            f"{len(set(info['base'][a:b].tolist()))} base pixel(s), with half its "
            f"nodes within {np.median(d_km[a:b]):.0f} km of the query."
        )
        if show_reach:
            ra, rb = info["ra"], info["rb"]
            caption += (
                f" After {N_PASSES} passes the receptive field covers "
                f"{100 * (rb - ra) / n:.0f}% of the mesh."
            )
        fig.text(0.5, 0.065, caption, ha="center", fontsize=9.5, color="0.3", wrap=True)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for graph_path in GRAPHS:
        lat, lon, pruned = load_hidden_mesh(graph_path)
        stem = Path(graph_path).stem.removeprefix("graph.")

        for qlat, qlon in QUERY_POINTS:
            info = analyze(lat, lon, qlat, qlon)
            report(lat, lon, qlat, qlon, info, graph_path, pruned)
            out_path = out_dir / f"{stem}_q{qlat:g}_{qlon:g}.jpeg"
            make_figure(lat, lon, info, out_path, graph_path)


if __name__ == "__main__":
    main()
