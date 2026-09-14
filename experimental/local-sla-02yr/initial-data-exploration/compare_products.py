"""Compare the raw (pre-HEALPix) Jason-1 altimetry point clouds to decide which
product + field to carry forward.

Four point sets for one day (2008-02-27), all mono-mission Jason-1:
  A  L2P sea_level_anomaly, finite            (no validation flag)
  B  L2P sea_level_anomaly, finite & valid    (validation_flag == 0 -> "kept")
  R  = A \\ B, i.e. finite but flagged rejected (validation_flag == 1 -> "removed")
  C  L3 DUACS along-track (sla_filtered / sla_unfiltered share point locations)

Primary question: do the flag-removed L2P points (R) fall where L3 has no data?
L2P and L3 are different processing chains with no shared indices, so coincidence
is assessed spatially, two ways:
  (a) cell-level coverage overlap on a common HEALPix mesh (L9 ~12 km, L5 ~200 km)
  (b) great-circle nearest-neighbour distance from each point to the L3 cloud
Plus supporting stats (counts, latitude distribution) and SLA value distributions
to inform the field choice. A diagnostic figure is written to figures/.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import healpy as hp
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

HERE = Path(__file__).parent

# Jason-1 (kept for reference):
#MISSION = "j1"
#L2P_NC = HERE / "data/l2p/dataset-l2p-uncross-calibrated-ntc-sla-j1_20080227_merged.nc"
#L3_NC = HERE / "data/l3/dt_global_j1_phy_l3_1hz_20080227_20240205.nc"
#DATE = "2008-02-27"
MISSION = "c2"
L2P_NC = HERE / "data/l2p/dataset-l2p-uncross-calibrated-ntc-sla-c2_20100716_merged.nc"
L3_NC = HERE / "data/l3/dt_global_c2_phy_l3_1hz_20100716_20240205.nc"
DATE = "2010-07-16"

LEVELS = (9, 5)            # HEALPix levels for coverage overlap (primary first)
EARTH_RADIUS_KM = 6371.0
FIG_PATH = HERE / "figures" / f"compare_products_{MISSION}_{DATE.replace('-', '')}.png"


def day_mask(time, date):
    start = np.datetime64(date)
    return (time >= start) & (time < start + np.timedelta64(1, "D"))


def load_points():
    """Return the four point sets as (lon, lat[, value]) arrays, day-windowed."""
    l2p = xr.open_dataset(L2P_NC)
    m = day_mask(l2p["time"].values, DATE)
    lon, lat = l2p["longitude"].values, l2p["latitude"].values
    sla, flag = l2p["sea_level_anomaly"].values, l2p["validation_flag"].values
    finite = np.isfinite(sla) & m
    kept = finite & (flag == 0)          # B
    removed = finite & (flag == 1)       # R
    l2p_all = finite                     # A

    l3 = xr.open_dataset(L3_NC)
    m3 = day_mask(l3["time"].values, DATE)
    c3 = np.isfinite(l3["sla_filtered"].values) & m3
    return {
        "l2p_all": (lon[l2p_all], lat[l2p_all], sla[l2p_all]),
        "kept": (lon[kept], lat[kept], sla[kept]),
        "removed": (lon[removed], lat[removed], sla[removed]),
        "l3": (l3["longitude"].values[c3], l3["latitude"].values[c3]),
        "l3_filtered": l3["sla_filtered"].values[c3],
        "l3_unfiltered": l3["sla_unfiltered"].values[c3],
    }


def to_xyz(lon, lat):
    """lon/lat degrees -> unit vectors on the sphere (for chord-distance KD-tree)."""
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def nn_gc_km(src, ref):
    """Great-circle distance (km) from each src point to the nearest ref point."""
    tree = cKDTree(to_xyz(*ref))
    chord, _ = tree.query(to_xyz(*src[:2]), k=1)
    # chord = 2 sin(theta/2)  ->  theta = 2 arcsin(chord/2)
    return 2 * np.arcsin(np.clip(chord, 0, 2) / 2) * EARTH_RADIUS_KM


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def coastline_xyz(resolution="50m", spacing_km=2.0):
    """Natural Earth coastline as dense unit vectors: each segment is subdivided to
    ~spacing_km so a nearest-vertex query approximates true distance-to-coastline."""
    lons, lats = [], []
    for geom in NaturalEarthFeature("physical", "coastline", resolution).geometries():
        lines = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in lines:
            x, y = (np.asarray(c) for c in line.coords.xy)
            for i in range(len(x) - 1):
                if abs(x[i + 1] - x[i]) > 180:      # dateline wrap: don't interpolate
                    lons.append([x[i]]); lats.append([y[i]])
                    continue
                n = max(1, int(np.ceil(haversine_km(x[i], y[i], x[i + 1], y[i + 1]) / spacing_km)))
                t = np.linspace(0, 1, n, endpoint=False)
                lons.append(x[i] + t * (x[i + 1] - x[i]))
                lats.append(y[i] + t * (y[i + 1] - y[i]))
            lons.append([x[-1]]); lats.append([y[-1]])
    return to_xyz(np.concatenate(lons), np.concatenate(lats))


def coast_dist_km(pts, coast_tree):
    """Great-circle distance (km) from each point to the nearest coastline."""
    chord, _ = coast_tree.query(to_xyz(pts[0], pts[1]), k=1)
    return 2 * np.arcsin(np.clip(chord, 0, 2) / 2) * EARTH_RADIUS_KM


def coverage_overlap(pts, level):
    """Cell-level coverage comparison at one HEALPix level.

    Returns a dict of the 2x2 kept-vs-L3 cell counts plus the headline fractions:
    what share of removed (and, as baseline, kept) points land in cells that hold
    no L3 data.
    """
    nside = 2**level

    def cells(key):
        lon, lat = pts[key][0], pts[key][1]
        return hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)

    kept_pix, removed_pix, l3_pix = cells("kept"), cells("removed"), cells("l3")
    kept_set, l3_set = set(kept_pix.tolist()), set(l3_pix.tolist())

    removed_no_l3 = np.mean([p not in l3_set for p in removed_pix])
    kept_no_l3 = np.mean([p not in l3_set for p in kept_pix])
    removed_isolated = np.mean(
        [(p not in l3_set) and (p not in kept_set) for p in removed_pix]
    )
    inter = len(kept_set & l3_set)
    union = len(kept_set | l3_set)
    return {
        "level": level, "nside": nside,
        "cells_kept": len(kept_set), "cells_l3": len(l3_set),
        "cells_both": inter, "cells_kept_only": len(kept_set) - inter,
        "cells_l3_only": len(l3_set) - inter, "jaccard": inter / union,
        "removed_frac_in_l3_empty": removed_no_l3,
        "kept_frac_in_l3_empty": kept_no_l3,
        "removed_frac_isolated": removed_isolated,
    }


def describe(name, v):
    p1, p50, p99 = np.percentile(v, [1, 50, 99])
    print(f"  {name:18s} n={v.size:6d}  mean={v.mean():+.4f}  std={v.std():.4f}  "
          f"min={v.min():+.4f}  p1={p1:+.4f}  p50={p50:+.4f}  p99={p99:+.4f}  "
          f"max={v.max():+.4f}")


def make_figure(pts, nn_removed, nn_kept, coast):
    wrap = lambda lon: ((np.asarray(lon) + 180) % 360) - 180  # 0-360 -> -180..180
    fig = plt.figure(figsize=(14, 14))

    ax = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.coastlines(linewidth=0.4, color="0.5")
    ax.set_global()
    ax.scatter(wrap(pts["l3"][0]), pts["l3"][1], s=1, c="#4C78A8", alpha=0.25,
               transform=ccrs.PlateCarree(), label="L3 coverage")
    ax.scatter(wrap(pts["removed"][0]), pts["removed"][1], s=10, c="#E45756",
               transform=ccrs.PlateCarree(), label="L2P flag-removed")
    ax.legend(loc="lower left", markerscale=2, framealpha=0.9)
    ax.set_title("Flag-removed L2P points over L3 coverage")

    ax = fig.add_subplot(3, 2, 2)
    for d, c, lab in [(nn_kept, "#54A24B", "kept -> L3"),
                      (nn_removed, "#E45756", "removed -> L3")]:
        xs = np.sort(d)
        ax.plot(xs, np.linspace(0, 1, xs.size), color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("great-circle distance to nearest L3 point (km)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Nearest-neighbour distance to L3")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(3, 2, 3)
    for key, c, lab in [("l3", "#4C78A8", "L3"), ("kept", "#54A24B", "L2P kept"),
                        ("removed", "#E45756", "L2P removed")]:
        xs = np.sort(coast[key])
        ax.plot(xs, np.linspace(0, 1, xs.size), color=c, label=lab)
    ax.set_xscale("log")
    for t in (5, 10, 20, 50):
        ax.axvline(t, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("distance to coast (km)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Distance to coast")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(3, 2, 4)
    bins = np.linspace(-90, 90, 61)
    for key, c, lab in [("l3", "#4C78A8", "L3"), ("kept", "#54A24B", "L2P kept"),
                        ("removed", "#E45756", "L2P removed")]:
        ax.hist(pts[key][1], bins=bins, density=True, histtype="step", color=c, label=lab)
    ax.set_xlabel("latitude (deg)")
    ax.set_ylabel("density")
    ax.set_title("Latitude distribution")
    ax.legend()

    ax = fig.add_subplot(3, 2, 5)
    bins = np.linspace(-1.5, 1.5, 121)
    for v, c, lab in [(pts["kept"][2], "#54A24B", "L2P sea_level_anomaly"),
                      (pts["l3_unfiltered"], "#F58518", "L3 sla_unfiltered"),
                      (pts["l3_filtered"], "#4C78A8", "L3 sla_filtered")]:
        ax.hist(v, bins=bins, density=True, histtype="step", color=c, label=lab)
    ax.set_xlabel("SLA (m)")
    ax.set_ylabel("density")
    ax.set_title("SLA value distributions")
    ax.legend()

    fig.suptitle(f"{MISSION.upper()} raw product comparison — {DATE}", fontsize=14)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    print(f"\nwrote {FIG_PATH.relative_to(HERE)}")


def main():
    pts = load_points()

    print(f"{'=' * 70}\n{MISSION.upper()} {DATE} — point counts (day window [00z, +1 day))\n{'=' * 70}")
    print(f"  L2P finite (no flag)        A = {pts['l2p_all'][0].size}")
    print(f"  L2P finite & valid (kept)   B = {pts['kept'][0].size}")
    print(f"  L2P finite & rejected (rm)  R = {pts['removed'][0].size}")
    print(f"  L3 finite (filtered=unfilt) C = {pts['l3'][0].size}")

    print(f"\n{'=' * 70}\n(a) HEALPix cell-level coverage overlap (kept vs L3)\n{'=' * 70}")
    for level in LEVELS:
        s = coverage_overlap(pts, level)
        print(f"L{s['level']} (nside={s['nside']}): cells kept={s['cells_kept']} "
              f"l3={s['cells_l3']} both={s['cells_both']} "
              f"kept-only={s['cells_kept_only']} l3-only={s['cells_l3_only']} "
              f"Jaccard={s['jaccard']:.3f}")
        print(f"    removed points in L3-empty cells : {s['removed_frac_in_l3_empty']:.1%}"
              f"   (baseline: kept points in L3-empty cells {s['kept_frac_in_l3_empty']:.1%})")
        print(f"    removed points isolated (no L3 & no kept-L2P nearby): "
              f"{s['removed_frac_isolated']:.1%}")

    print(f"\n{'=' * 70}\n(b) Great-circle nearest-neighbour distance to L3 cloud\n{'=' * 70}")
    nn_removed = nn_gc_km(pts["removed"], pts["l3"])
    nn_kept = nn_gc_km(pts["kept"], pts["l3"])
    for name, d in [("removed -> L3", nn_removed), ("kept -> L3", nn_kept)]:
        p50, p90, p99 = np.percentile(d, [50, 90, 99])
        print(f"  {name:14s} median={p50:8.2f} km  p90={p90:8.2f} km  "
              f"p99={p99:8.2f} km  max={d.max():8.2f} km")

    print(f"\n{'=' * 70}\nSLA value distributions (day window)\n{'=' * 70}")
    describe("L2P kept", pts["kept"][2])
    describe("L3 sla_filtered", pts["l3_filtered"])
    describe("L3 sla_unfiltered", pts["l3_unfiltered"])
    rms = np.sqrt(np.mean((pts["l3_filtered"] - pts["l3_unfiltered"]) ** 2))
    print(f"  L3 filtered vs unfiltered RMS difference (same points): {rms:.4f} m")

    print(f"\n{'=' * 70}\nDistance to coast (Natural Earth 50m, densified ~2 km)\n{'=' * 70}")
    coast_tree = cKDTree(coastline_xyz())
    coast = {k: coast_dist_km(pts[k], coast_tree) for k in ("kept", "removed", "l3")}
    thresholds = (5, 10, 20, 50)
    print("  " + " " * 12 + "".join(f"<{t:>3d} km   " for t in thresholds) + "  median")
    for key, lab in [("kept", "L2P kept"), ("removed", "L2P removed"), ("l3", "L3")]:
        d = coast[key]
        parts = "".join(f"{np.mean(d < t):5.1%}/{int((d < t).sum()):<5d} " for t in thresholds)
        print(f"  {lab:11s} {parts}  median={np.median(d):6.1f} km")

    make_figure(pts, nn_removed, nn_kept, coast)


if __name__ == "__main__":
    main()
