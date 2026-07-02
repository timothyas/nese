"""Bin along-track SLA observations onto a HEALPix mesh as a daily average.

Takes a NetCDF of along-track observations (default: the merged L2P file), keeps
one day's worth of points [00z, +1 day 00z), assigns each observation to the
HEALPix pixel it falls in, and computes a per-pixel weighted daily mean. Each
observation is weighted by a Gaussian of its great-circle distance to the pixel
center: w = exp(-(d / L)^2).

Conventions follow ecmwf/anemoi-graphs so the pixel indices are directly usable
by an anemoi-style model:
  - nside = 2**LEVEL   (anemoi calls LEVEL the "resolution")
  - NESTED ordering
  - pixel centers via healpy.pix2ang(..., nest=True, lonlat=True)

Everything below the config block is generic; change INPUT_NC / FIELD / LEVEL /
DATE to retarget. Straightforward first pass -- not yet optimized.
"""

from pathlib import Path

import healpy as hp
import numpy as np
import xarray as xr

HERE = Path(__file__).parent

# --- configuration (meant to be edited) --------------------------------------
# Options kept for reference (L2P field is sea_level_anomaly; L3 is sla_filtered):
#INPUT_NC = HERE / "data/l2p/dataset-l2p-uncross-calibrated-ntc-sla-c2_20100716_merged.nc"
#INPUT_NC = HERE / "data/l3/dt_global_c2_phy_l3_1hz_20100716_20240205.nc"   # CryoSat-2 L3
#INPUT_NC = HERE / "data/l2p/dataset-l2p-uncross-calibrated-ntc-sla-j1_20080227_merged.nc"  # Jason-1 L2P
INPUT_NC = HERE / "data/l3/dt_global_j1_phy_l3_1hz_20080227_20240205.nc"    # Jason-1 L3
FIELD = "sla_filtered"           # variable to bin (L3 SLA field name)
DATE = "2008-02-27"             # day to average, window is [DATE 00z, +1 day 00z)
LEVEL = 9                        # nside = 2**LEVEL; level 5 -> nside 32 (~200 km)
ORDERING_NESTED = True           # anemoi-graphs uses NESTED
KERNEL_LENGTH_KM = None          # Gaussian length scale L; None -> pixel resolution
SPARSE = True                    # True:  store only occupied cells (float32) keyed by
                                 #        HEALPix index, plus a shared mesh file (once).
                                 # False: store the full dense mesh every day.

_STEM = f"data/healpix/l3_j1_{FIELD}_L{LEVEL}_{DATE.replace('-', '')}"
OUTPUT_NC = HERE / (f"{_STEM}_sparse.nc" if SPARSE else f"{_STEM}.nc")
MESH_NC = HERE / f"data/healpix/mesh_L{LEVEL}_nested.nc"   # fixed grid, written once

EARTH_RADIUS_KM = 6371.0
# -----------------------------------------------------------------------------


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) between arrays of lon/lat points (degrees)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def select_day(ds, field, date):
    """Return (values, lon, lat) for one day, dropping missing observations."""
    start = np.datetime64(date)
    end = start + np.timedelta64(1, "D")
    day = ds.where((ds["time"] >= start) & (ds["time"] < end), drop=True)

    values = day[field]
    valid = np.isfinite(values)
    # NOTE: no quality-control filtering yet. To use AVISO's flag, AND in e.g.
    #   valid &= day["validation_flag"] == 1
    values = values.where(valid, drop=True)
    lon = day["longitude"].where(valid, drop=True)
    lat = day["latitude"].where(valid, drop=True)
    return values.values, lon.values, lat.values


def bin_to_healpix(values, lon, lat, nside, nested, length_km):
    """Weighted daily mean of `values` per HEALPix pixel (Gaussian by distance)."""
    npix = hp.nside2npix(nside)

    # Pixel each observation falls in, and that pixel's center.
    pix = hp.ang2pix(nside, lon, lat, nest=nested, lonlat=True)
    center_lon, center_lat = hp.pix2ang(nside, pix, nest=nested, lonlat=True)

    dist = haversine_km(lon, lat, center_lon, center_lat)
    weight = np.exp(-((dist / length_km) ** 2))

    # Scatter-add into pixel bins: weighted mean = Σ(w·x) / Σw.
    wsum = np.bincount(pix, weights=weight, minlength=npix)
    wxsum = np.bincount(pix, weights=weight * values, minlength=npix)
    count = np.bincount(pix, minlength=npix)

    binned = np.divide(wxsum, wsum, out=np.full(npix, np.nan), where=wsum > 0)
    return binned, count


def _attrs(nside, length_km):
    return {
        "healpix_nside": nside,
        "healpix_level": LEVEL,
        "healpix_ordering": "nested" if ORDERING_NESTED else "ring",
        "kernel": "gaussian exp(-(d/L)^2)",
        "kernel_length_km": length_km,
        "source_file": INPUT_NC.name,
        "source_field": FIELD,
    }


def build_dense(binned, count, nside, attrs):
    """Full-mesh dataset: every pixel, carrying its own lon/lat coordinates."""
    npix = hp.nside2npix(nside)
    hpx_lon, hpx_lat = hp.pix2ang(nside, np.arange(npix), nest=ORDERING_NESTED, lonlat=True)
    return xr.Dataset(
        data_vars={
            FIELD: ("cell", binned, {"long_name": f"daily weighted-mean {FIELD}"}),
            "obs_count": ("cell", count.astype("int32"), {"long_name": "observations in cell"}),
        },
        coords={
            "lon": ("cell", hpx_lon, {"units": "degrees_east"}),
            "lat": ("cell", hpx_lat, {"units": "degrees_north"}),
            "time": np.datetime64(DATE),
        },
        attrs=attrs,
    )


def build_sparse(binned, count, attrs):
    """Occupied cells only. The `cell` coordinate holds the HEALPix nested index;
    lon/lat live once in MESH_NC and are recovered by indexing it with `cell`."""
    occupied = np.nonzero(count > 0)[0].astype("int32")
    return xr.Dataset(
        data_vars={
            FIELD: ("cell", binned[occupied].astype("float32"),
                    {"long_name": f"daily weighted-mean {FIELD}"}),
            "obs_count": ("cell", count[occupied].astype("int32"),
                          {"long_name": "observations in cell"}),
        },
        coords={
            "cell": ("cell", occupied, {"long_name": "HEALPix nested pixel index"}),
            "time": np.datetime64(DATE),
        },
        attrs=attrs,
    )


def open_sparse_hpx(path, mesh_path=None):
    """Reopen a sparse HEALPix file as a dense full-mesh dataset for inspection.

    Inverse of build_sparse: scatters the occupied cells back onto all npix
    pixels (NaN where no observation fell) and attaches each pixel's lon/lat, so
    the result matches build_dense's layout and plots identically. nside, ordering
    and the field name are read from the file's own attrs, so this works on any
    sparse file. lon/lat come from mesh_path if given, else are recomputed.
    """
    sparse = xr.open_dataset(path)
    nside = int(sparse.attrs["healpix_nside"])
    nested = sparse.attrs["healpix_ordering"] == "nested"
    field = sparse.attrs["source_field"]
    npix = hp.nside2npix(nside)

    cells = sparse["cell"].values
    binned = np.full(npix, np.nan)
    binned[cells] = sparse[field].values
    count = np.zeros(npix, dtype="int32")
    count[cells] = sparse["obs_count"].values

    if mesh_path is not None:
        mesh = xr.open_dataset(mesh_path)
        hpx_lon, hpx_lat = mesh["lon"].values, mesh["lat"].values
    else:
        hpx_lon, hpx_lat = hp.pix2ang(nside, np.arange(npix), nest=nested, lonlat=True)

    return xr.Dataset(
        data_vars={
            field: ("cell", binned, {"long_name": f"daily weighted-mean {field}"}),
            "obs_count": ("cell", count, {"long_name": "observations in cell"}),
        },
        coords={
            "lon": ("cell", hpx_lon, {"units": "degrees_east"}),
            "lat": ("cell", hpx_lat, {"units": "degrees_north"}),
            "time": sparse["time"].values,
        },
        attrs=sparse.attrs,
    )


def write_mesh_once(nside, path):
    """Write the fixed HEALPix lon/lat grid once (skipped if it already exists)."""
    if path.exists():
        return
    npix = hp.nside2npix(nside)
    hpx_lon, hpx_lat = hp.pix2ang(nside, np.arange(npix), nest=ORDERING_NESTED, lonlat=True)
    mesh = xr.Dataset(
        coords={
            "cell": ("cell", np.arange(npix, dtype="int32")),
            "lon": ("cell", hpx_lon, {"units": "degrees_east"}),
            "lat": ("cell", hpx_lat, {"units": "degrees_north"}),
        },
        attrs={k: _attrs(nside, 0)[k] for k in
               ("healpix_nside", "healpix_level", "healpix_ordering")},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_netcdf(path)
    print(f"wrote mesh {path.name} ({path.stat().st_size / 1e6:.2f} MB, shared/once)")


def main():
    nside = 2**LEVEL
    npix = hp.nside2npix(nside)
    length_km = KERNEL_LENGTH_KM
    if length_km is None:
        length_km = hp.nside2resol(nside) * EARTH_RADIUS_KM  # ~pixel resolution

    ds = xr.open_dataset(INPUT_NC)
    values, lon, lat = select_day(ds, FIELD, DATE)
    print(f"{DATE}: {values.size} valid '{FIELD}' obs -> HEALPix L{LEVEL} "
          f"(nside={nside}, {npix} cells), Gaussian L={length_km:.1f} km")

    binned, count = bin_to_healpix(values, lon, lat, nside, ORDERING_NESTED, length_km)

    filled = int((count > 0).sum())
    print(f"filled {filled}/{npix} cells ({100 * filled / npix:.1f}%); "
          f"{FIELD} min/mean/max = "
          f"{np.nanmin(binned):.4f} / {np.nanmean(binned):.4f} / {np.nanmax(binned):.4f}")

    attrs = _attrs(nside, length_km)
    OUTPUT_NC.parent.mkdir(parents=True, exist_ok=True)
    if SPARSE:
        write_mesh_once(nside, MESH_NC)
        out = build_sparse(binned, count, attrs)
    else:
        out = build_dense(binned, count, nside, attrs)
    out.to_netcdf(OUTPUT_NC)
    print(f"wrote {OUTPUT_NC.name} ({OUTPUT_NC.stat().st_size / 1e6:.2f} MB, "
          f"{'sparse' if SPARSE else 'dense'})")


if __name__ == "__main__":
    main()
