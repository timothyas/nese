"""Estimate on-disk size of the HEALPix-binned DUACS L3 SLA store.

Before building the full pull+bin+write pipeline we need an accurate storage-size
estimate, because the storage format is not yet decided. This script:

  1. Discovers every reprocessed (MY) mission in SEALEVEL_GLO_PHY_L3_MY_008_062 and
     its temporal coverage / day count (copernicusmarine.describe()).
  2. Pulls ONE mid-mission day per mission as an NNZ / compression proxy.
  3. Bins each day onto a HEALPix mesh at level 6 (~100 km) and level 8 (~25 km),
     reusing bin_to_healpix.py. Two per-cell fields: sla_filtered (daily
     weighted-mean) and obs_count (points averaged per cell).
  4. Writes a representative multi-day block (all missions' mid-days stacked) in
     three candidate formats -- Zarr COO, Zarr dense, Parquet COO -- benchmarking
     zlib-4 vs Blosc-zstd, and measures real bytes on disk.
  5. Extrapolates each mission by its real day count and emits:
        data/healpix/size_estimate_v2.csv   -- per level/format/compressor totals
        ./mission_coverage.csv              -- appendix table (mission, range, days)
        data/healpix/mission_nnz.csv        -- per-mission NNZ at L6/L8

Precision note: binning and the weighted average run in float64; values are cast
to float32 only at the final write step (single precision on disk, full precision
through every intermediate op).

Run in the seso env:
    conda run -n seso python estimate_store_size.py
Needs COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD (or cached credentials).
"""

import csv
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from numcodecs import Blosc, Zlib

# Reuse the validated binning machinery (functions are fully parameterized).
from bin_to_healpix import bin_to_healpix

HERE = Path(__file__).parent
PRODUCT_ID = "SEALEVEL_GLO_PHY_L3_MY_008_062"
FIELD = "sla_filtered"
LEVELS = (6, 8)                       # ~100 km and ~25 km
EARTH_RADIUS_KM = 6371.0

DATA_DIR = HERE / "data"                        # mission coverage table lives here
DL_DIR = DATA_DIR / "l3_estimate"               # one mid-mission day per mission
OUT_DIR = DATA_DIR / "healpix"                  # size/NNZ outputs + existing meshes
BENCH_DIR = Path(
    os.environ.get("SCRATCH_BENCH")
    or "/private/tmp/claude-502/-Users-timith-work-nese-observations-sea-level-altimetry"
      "/6f0363af-e6f0-462c-9418-71dea3235a0b/scratchpad/bench"
)

# Token (between `_my_` and `-l3-duacs`) -> readable full name. Each orbit phase
# is its own dataset/variable (g = geodetic drifting phase, n = interleaved/new
# orbit). Names are lower_snake_case, ready to be zarr variable names.
MISSION_NAMES = {
    "tp": "topex_poseidon", "tpn": "topex_poseidon_interleaved",
    "e1": "ers_1", "e1g": "ers_1_geodetic", "e2": "ers_2",
    "en": "envisat", "enn": "envisat_new_orbit",
    "g2": "gfo",
    "j1": "jason_1", "j1g": "jason_1_geodetic", "j1n": "jason_1_interleaved",
    "j2": "jason_2", "j2g": "jason_2_geodetic", "j2n": "jason_2_interleaved",
    "j3": "jason_3", "j3g": "jason_3_geodetic", "j3n": "jason_3_interleaved",
    "al": "saral", "alg": "saral_geodetic",
    "c2": "cryosat_2", "c2n": "cryosat_2_interleaved",
    "h2a": "haiyang_2a", "h2ag": "haiyang_2a_geodetic", "h2b": "haiyang_2b",
    "s3a": "sentinel_3a", "s3b": "sentinel_3b",
    "s6a-lr": "sentinel_6a",
    "swon": "swot_nadir", "swonc": "swot_nadir_calval",
}

# Token -> pretty display name for the coverage appendix table.
DISPLAY_NAMES = {
    "tp": "TOPEX/Poseidon", "tpn": "TOPEX/Poseidon (interleaved)",
    "e1": "ERS-1", "e1g": "ERS-1 (geodetic)", "e2": "ERS-2",
    "en": "Envisat", "enn": "Envisat (new orbit)",
    "g2": "GFO",
    "j1": "Jason-1", "j1g": "Jason-1 (geodetic)", "j1n": "Jason-1 (interleaved)",
    "j2": "Jason-2", "j2g": "Jason-2 (geodetic)", "j2n": "Jason-2 (interleaved)",
    "j3": "Jason-3", "j3g": "Jason-3 (geodetic)", "j3n": "Jason-3 (interleaved)",
    "al": "SARAL/AltiKa", "alg": "SARAL/AltiKa (geodetic)",
    "c2": "CryoSat-2", "c2n": "CryoSat-2 (interleaved)",
    "h2a": "HY-2A", "h2ag": "HY-2A (geodetic)", "h2b": "HY-2B",
    "s3a": "Sentinel-3A", "s3b": "Sentinel-3B",
    "s6a-lr": "Sentinel-6A",
    "swon": "SWOT nadir", "swonc": "SWOT nadir (cal/val)",
}

TOKEN_RE = re.compile(r"_my_(?P<token>.+?)-l3-duacs")

COMPRESSORS = {
    "blosc_lz4": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),  # zarr v2 default
    "blosc_zstd": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
    "zlib4": Zlib(level=4),
}
PARQUET_COMPRESSORS = ("zstd", "snappy")


# --- discovery ---------------------------------------------------------------
def _time_extent(dataset):
    """(start, end) datetimes from the arco-geo-series time coordinate."""
    part = dataset.versions[0].parts[0]
    for service in part.services:
        for var in service.variables:
            for coord in var.coordinates:
                if str(getattr(coord, "coordinate_id", "")).lower() == "time":
                    lo = coord.minimum_value
                    hi = coord.maximum_value
                    if lo and hi:
                        fmt = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
                        return fmt(lo).replace(tzinfo=None), fmt(hi).replace(tzinfo=None)
    return None, None


def discover_missions():
    """Return a list of mission dicts sorted by start date."""
    import copernicusmarine as cm

    resp = cm.describe(product_id=PRODUCT_ID)
    product = resp.products[0]
    missions = []
    for ds in product.datasets:
        m = TOKEN_RE.search(ds.dataset_id)
        token = m.group("token") if m else ds.dataset_id
        start, end = _time_extent(ds)
        if start is None:
            print(f"  !! no time extent for {ds.dataset_id}, skipping")
            continue
        n_days = (end.date() - start.date()).days + 1
        mid = start + (end - start) / 2
        missions.append({
            "dataset_id": ds.dataset_id,
            "version": ds.versions[0].label,
            "token": token,
            "name": MISSION_NAMES.get(token, token.replace("-", "_")),
            "start": start,
            "end": end,
            "n_days": n_days,
            "mid_date": mid.date(),
        })
    missions.sort(key=lambda d: d["start"])
    return missions


# --- pull one mid-mission day ------------------------------------------------
def pull_mid_day(mission):
    """Download the mission's mid-mission day (try nearby days if absent).

    Returns a local .nc Path, or None if nothing could be fetched.
    """
    import copernicusmarine as cm

    DL_DIR.mkdir(parents=True, exist_ok=True)
    token, mid = mission["token"], mission["mid_date"]
    fname_token = token.replace("-", "_")   # filenames use s6a_lr, not s6a-lr

    # Reuse an already-downloaded day for this mission if present.
    existing = sorted(DL_DIR.glob(f"dt_global_{fname_token}_phy_l3_*.nc"))
    if existing:
        return existing[0]

    for delta in range(0, 21):
        for day in ({mid + timedelta(days=delta), mid - timedelta(days=delta)}
                    if delta else {mid}):
            stamp = day.strftime("%Y%m%d")
            try:
                result = cm.get(
                    dataset_id=mission["dataset_id"],
                    dataset_version=mission["version"],
                    filter=f"*{stamp}*",
                    output_directory=str(DL_DIR),
                    no_directories=True,
                    overwrite=False,
                    disable_progress_bar=True,
                )
            except Exception as e:                       # noqa: BLE001
                print(f"    get({token} {stamp}) failed: {e}")
                continue
            files = [Path(f.file_path) for f in (result.files if result else [])]
            ncs = [f for f in files if f.suffix == ".nc"]
            if ncs:
                return ncs[0]
    return None


# --- bin one day -------------------------------------------------------------
def bin_day(path, level):
    """Bin one L3 file's along-track sla_filtered onto HEALPix level `level`.

    DUACS L3 granules are already one measurement day, so we bin every valid
    (finite) observation in the file -- no time-windowing (a file's first sample
    can sit just before midnight, which would otherwise drop the whole day).
    Returns (occupied_cells int32, values float64, counts int32). Binning stays
    in float64; the float32 cast happens later, at write time only.
    """
    import healpy as hp

    nside = 2 ** level
    length_km = hp.nside2resol(nside) * EARTH_RADIUS_KM
    ds = xr.open_dataset(path)
    valid = np.isfinite(ds[FIELD].values)
    values = ds[FIELD].values[valid]
    lon = ds["longitude"].values[valid]
    lat = ds["latitude"].values[valid]
    binned, count = bin_to_healpix(values, lon, lat, nside, True, length_km)
    occupied = np.nonzero(count > 0)[0].astype("int32")
    return occupied, binned[occupied], count[occupied].astype("int32")


# --- disk measurement --------------------------------------------------------
def du_bytes(path):
    """Total bytes on disk under `path` (file or directory)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fresh(path):
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_zarr_coo(block, level, cname, codec):
    """COO store: shared cell/day index + float32 sla + int32 obs_count."""
    path = _fresh(BENCH_DIR / f"coo_L{level}_{cname}.zarr")
    ds = xr.Dataset(
        {
            "sla_filtered": ("obs", block["sla"].astype("float32")),
            "obs_count": ("obs", block["count"].astype("int32")),
            "cell": ("obs", block["cell"].astype("int32")),
            "day_offsets": ("day1", block["day_offsets"].astype("int32")),
        }
    )
    enc = {v: {"compressor": codec} for v in
           ("sla_filtered", "obs_count", "cell", "day_offsets")}
    ds.to_zarr(path, mode="w", encoding=enc, consolidated=False)
    return du_bytes(path)


def write_zarr_dense(block, level, cname, codec):
    """Dense (day, cell=npix) store: sla float32 NaN-fill, obs_count int32 0-fill."""
    npix = 12 * (4 ** level)
    nday = block["nday"]
    sla = np.full((nday, npix), np.nan, dtype="float32")
    cnt = np.zeros((nday, npix), dtype="int32")
    for d in range(nday):
        s, e = block["day_offsets"][d], block["day_offsets"][d + 1]
        cells = block["cell"][s:e]
        sla[d, cells] = block["sla"][s:e].astype("float32")
        cnt[d, cells] = block["count"][s:e]
    path = _fresh(BENCH_DIR / f"dense_L{level}_{cname}.zarr")
    ds = xr.Dataset({
        "sla_filtered": (("day", "cell"), sla),
        "obs_count": (("day", "cell"), cnt),
    })
    chunk = {"day": nday, "cell": min(npix, 1 << 20)}
    enc = {v: {"compressor": codec} for v in ("sla_filtered", "obs_count")}
    ds.chunk(chunk).to_zarr(path, mode="w", encoding=enc, consolidated=False)
    return du_bytes(path)


def write_parquet_coo(block, level, compression):
    """Single COO table [day, cell, sla_filtered(f32), obs_count(i32)]."""
    path = _fresh(BENCH_DIR / f"coo_L{level}_{compression}.parquet")
    day = np.empty(block["cell"].size, dtype="int32")
    for d in range(block["nday"]):
        day[block["day_offsets"][d]:block["day_offsets"][d + 1]] = d
    table = pa.table({
        "day": pa.array(day, pa.int32()),
        "cell": pa.array(block["cell"], pa.int32()),
        "sla_filtered": pa.array(block["sla"].astype("float32"), pa.float32()),
        "obs_count": pa.array(block["count"], pa.int32()),
    })
    kwargs = dict(compression=compression, use_dictionary=False,
                  data_page_version="2.0")
    try:
        # Delta-pack the integer index columns (cell/day are sorted-ish within a
        # day -> tiny deltas); byte-stream-split the float value column.
        pq.write_table(
            table, path, use_byte_stream_split=["sla_filtered"],
            column_encoding={"cell": "DELTA_BINARY_PACKED",
                             "day": "DELTA_BINARY_PACKED",
                             "obs_count": "DELTA_BINARY_PACKED"},
            **kwargs)
    except Exception as err:                     # noqa: BLE001
        print(f"    parquet encoded write failed ({err}); plain fallback")
        pq.write_table(table, path, **kwargs)   # fallback: plain + compression
    return du_bytes(path)


# --- coverage table (appendix) -----------------------------------------------
def write_coverage(missions):
    """Write the appendix coverage table as CSV and Markdown.

    Human-readable: pretty display names ("Jason-3") and months spelled out
    ("October 1992"). Columns: mission (display name), token (CMEMS id token),
    start, end, n_days. Written to data/ (not the healpix subdir).
    """
    header = ["Mission", "Token", "Start", "End", "Days"]
    rows = [[DISPLAY_NAMES.get(m["token"], m["name"]), m["token"],
             m["start"].strftime("%B %Y"), m["end"].strftime("%B %Y"),
             m["n_days"]] for m in missions]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HERE / "mission_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    align = ["---", "---", "---", "---", "---:"]   # right-align the Days column
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (HERE / "mission_coverage.md").write_text("\n".join(md) + "\n")


# --- main --------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering missions...")
    missions = discover_missions()
    print(f"  {len(missions)} mission datasets in {PRODUCT_ID}")

    # Appendix coverage table (CSV + Markdown), months spelled out.
    write_coverage(missions)
    total_days = sum(m["n_days"] for m in missions)
    print(f"  total mission-days across all missions: {total_days:,}")

    # Pull + bin one mid-mission day per mission, per level.
    # Per-level block = all missions' mid-days concatenated along an obs axis,
    # with day_offsets marking mission-day boundaries (a CSR indptr).
    nnz = {lvl: {} for lvl in LEVELS}
    blocks = {}
    for lvl in LEVELS:
        cells, slas, counts, offsets = [], [], [], [0]
        for m in missions:
            path = m.setdefault("_path", pull_mid_day(m))
            if path is None:
                print(f"  !! no file for {m['name']}; skipping")
                nnz[lvl][m["name"]] = 0
                continue
            occ, val, cnt = bin_day(path, lvl)
            nnz[lvl][m["name"]] = int(occ.size)
            cells.append(occ); slas.append(val); counts.append(cnt)
            offsets.append(offsets[-1] + occ.size)
        blocks[lvl] = {
            "cell": np.concatenate(cells),
            "sla": np.concatenate(slas),
            "count": np.concatenate(counts),
            "day_offsets": np.asarray(offsets, dtype="int64"),
            "nday": len(offsets) - 1,
        }
        b = blocks[lvl]
        print(f"  L{lvl}: {b['nday']} mission-days, "
              f"{b['cell'].size:,} total occupied cells "
              f"(mean {b['cell'].size / b['nday']:.0f}/day)")

    # Per-mission NNZ CSV.
    with open(OUT_DIR / "mission_nnz.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mission", "n_days"] + [f"nnz_L{lvl}" for lvl in LEVELS])
        for m in missions:
            w.writerow([m["name"], m["n_days"]]
                       + [nnz[lvl][m["name"]] for lvl in LEVELS])

    # Benchmark formats on each per-level block, then extrapolate.
    GiB = 1024 ** 3
    rows = []
    for lvl in LEVELS:
        b = blocks[lvl]
        npix = 12 * (4 ** lvl)
        total_obs_block = b["cell"].size
        # sum over missions of NNZ_m * ndays_m  (obs written in the full record)
        total_obs_all = sum(nnz[lvl][m["name"]] * m["n_days"] for m in missions)

        for cname, codec in COMPRESSORS.items():
            coo = write_zarr_coo(b, lvl, cname, codec)
            dense = write_zarr_dense(b, lvl, cname, codec)
            bpo = coo / total_obs_block                 # bytes / obs (both fields+idx)
            bpd = dense / b["nday"]                      # bytes / day (both fields)
            rows.append(("zarr_coo", lvl, cname, bpo * total_obs_all / GiB, bpo))
            rows.append(("zarr_dense", lvl, cname, bpd * total_days / GiB, bpd))

        for comp in PARQUET_COMPRESSORS:
            pqb = write_parquet_coo(b, lvl, comp)
            bpo = pqb / total_obs_block
            rows.append(("parquet_coo", lvl, comp, bpo * total_obs_all / GiB, bpo))

    # Mesh-once cost (existing files).
    mesh_gib = {}
    for lvl in LEVELS:
        mp = OUT_DIR / f"mesh_L{lvl}_nested.nc"
        mesh_gib[lvl] = (mp.stat().st_size / GiB) if mp.exists() else float("nan")

    with open(OUT_DIR / "size_estimate_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["format", "level", "compressor", "total_GiB",
                    "bytes_per_unit", "mesh_once_GiB"])
        for fmt, lvl, cname, gib, bpu in rows:
            w.writerow([fmt, lvl, cname, f"{gib:.3f}", f"{bpu:.2f}",
                        f"{mesh_gib[lvl]:.4f}"])

    # Console summary.
    print("\n" + "=" * 74)
    print(f"{'format':<13}{'level':<7}{'compressor':<13}{'total (GiB)':>14}")
    print("-" * 74)
    for fmt, lvl, cname, gib, _ in sorted(rows, key=lambda r: (r[1], r[3])):
        print(f"{fmt:<13}L{lvl:<6}{cname:<13}{gib:>14.2f}")
    print("=" * 74)
    for lvl in LEVELS:
        print(f"  mesh_L{lvl}_nested.nc one-time: {mesh_gib[lvl] * 1024:.1f} MiB")
    print(f"\nWrote {OUT_DIR / 'size_estimate_v2.csv'}, mission_coverage.csv, "
          "mission_nnz.csv")


if __name__ == "__main__":
    main()
