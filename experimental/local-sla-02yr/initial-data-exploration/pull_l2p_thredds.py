"""Pull all Jason-1 L2P along-track SLA passes for one day and merge them.

L2P = mono-mission, "uncross-calibrated" along-track sea level anomalies. This
is the rawer product (no inter-mission crossover calibration). It is served by
the ODATIS THREDDS server: the catalog is public and browsable, but downloading
the actual granules requires an AVISO+ account (HTTP Basic Auth).

Unlike the CMEMS L3 product (one file per mission per day), L2P is delivered as
one file per pass inside cycle folders. To get a day comparable to the L3
day-file, this script lists a cycle, keeps the passes whose measurements start
on the target day, downloads them, and concatenates them along the record
dimension into a single dataset.

Register + pick the "Sea Level Anomalies Along-track Level-2+ (L2P)" product:
    https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html

Then export credentials before running:
    export AVISO_USERNAME=...
    export AVISO_PASSWORD=...

TODO:
* convert validation_flag to integer
"""

import gzip
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import xarray as xr

TDS = "https://tds-odatis.aviso.altimetry.fr/thredds"

# One mission cycle + day to sample. To retarget a mission, swap DATASET and pick
# a CYCLE/DAY from its catalog: .../thredds/catalog/<DATASET>/<VERSION>/catalog.xml
# lists the cycles. Cycle numbers and dates are per-mission.
#
# CryoSat-2: cycle_0007 opens on 2010-07-16, the very start of the record, which
# also exists in the CMEMS L3 product for a clean comparison. (~369-day repeats.)
#   DATASET = "dataset-l2p-uncross-calibrated-ntc-sla-c2"
#   CYCLE   = "cycle_0007"
#   DAY     = "20100716"
#
# Jason-1: flew ~2002-2013 on a ~10-day repeat, so its 408 cycles are short;
# cycle_0226 spans 2008-02-24 -> 2008-03-05, and 2008-02-27 is a full day.
DATASET = "dataset-l2p-uncross-calibrated-ntc-sla-j1"
VERSION = "version_04_00"
CYCLE = "cycle_0226"
DAY = "20080227"  # YYYYMMDD; must fall within CYCLE

DATA_DIR = Path(__file__).parent / "data" / "l2p"

# THREDDS catalog namespace, needed to parse catalog.xml
NS = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}


def list_pass_files(dataset, version, cycle):
    """Return the .nc.gz pass filenames in a cycle (public, no auth needed)."""
    url = f"{TDS}/catalog/{dataset}/{version}/{cycle}/catalog.xml"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    names = [ds.get("name") for ds in root.findall(".//t:dataset", NS)]
    return sorted(n for n in names if n and n.endswith(".nc.gz"))


def passes_for_day(files, day):
    """Keep passes whose measurement start date matches day (YYYYMMDD).

    Filename pattern: ..._<startYYYYMMDD>T<hhmmss>_<endYYYYMMDD>T<hhmmss>_...
    Filtering on the start date gives each pass to exactly one day.
    """
    out = []
    for f in files:
        m = re.search(r"_(\d{8})T\d{6}_\d{8}T\d{6}_", f)
        if m and m.group(1) == day:
            out.append(f)
    return out


def download(dataset, version, cycle, filename, auth):
    """Download one granule via the THREDDS fileServer (needs AVISO+ auth).

    Idempotent: if the unzipped .nc is already present, reuse it.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / filename
    nc_path = gz_path.with_suffix("")  # strip .gz
    if nc_path.exists():
        return nc_path

    url = f"{TDS}/fileServer/{dataset}/{version}/{cycle}/{filename}"
    with requests.get(url, auth=auth, stream=True, timeout=120) as resp:
        if resp.status_code == 401:
            raise SystemExit(
                "401 Unauthorized. Set AVISO_USERNAME / AVISO_PASSWORD and make "
                "sure your AVISO+ account is approved for the L2P product."
            )
        resp.raise_for_status()
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)

    with gzip.open(gz_path, "rb") as fin, open(nc_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return nc_path


def merge_passes(nc_paths):
    """Concatenate per-pass along-track files into one dataset, sorted by time."""
    datasets = [xr.open_dataset(p) for p in nc_paths]
    # The along-track record dimension is whatever latitude is indexed by.
    record_dim = datasets[0]["latitude"].dims[0]

    # pass_number and cycle_number are per-file global attributes; broadcast them
    # to along-track variables so provenance survives the merge into one dataset.
    for ds in datasets:
        for attr, long_name in (
            ("pass_number", "along-track pass number"),
            ("cycle_number", "along-track cycle number"),
        ):
            var = xr.full_like(ds["latitude"], int(ds.attrs.get(attr, -1)))
            var = var.astype("int32")
            var.attrs = {"long_name": long_name}
            ds[attr] = var

    merged = xr.concat(
        datasets,
        dim=record_dim,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    if "time" in merged:
        merged = merged.sortby("time")
    return merged


def summarize(ds, source):
    """Print a compact summary so L2P and L3 can be compared side by side."""
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    print(ds)
    print("\n-- coordinate ranges --")
    for name in ("time", "latitude", "longitude"):
        if name in ds:
            v = ds[name]
            print(f"  {name:10s}: {v.min().values}  ->  {v.max().values}  (n={v.size})")
    for sla_name in ("sea_level_anomaly", "sla", "sla_unfiltered", "SLA"):
        if sla_name in ds:
            sla = ds[sla_name]
            print(f"\n-- {sla_name} (m) --")
            print(f"  min/mean/max: {float(sla.min()):.4f} / "
                  f"{float(sla.mean()):.4f} / {float(sla.max()):.4f}")
            break


def main():
    user = os.environ.get("AVISO_USERNAME")
    pw = os.environ.get("AVISO_PASSWORD")
    if not (user and pw):
        raise SystemExit("Set AVISO_USERNAME and AVISO_PASSWORD env vars first.")

    all_files = list_pass_files(DATASET, VERSION, CYCLE)
    day_files = passes_for_day(all_files, DAY)
    print(f"{CYCLE}: {len(all_files)} passes total, {len(day_files)} on {DAY}.")
    if not day_files:
        raise SystemExit(f"No passes start on {DAY} within {CYCLE}.")

    nc_paths = []
    for i, filename in enumerate(day_files, 1):
        print(f"  [{i}/{len(day_files)}] {filename}")
        nc_paths.append(download(DATASET, VERSION, CYCLE, filename, auth=(user, pw)))

    ds = merge_passes(nc_paths)

    merged_path = DATA_DIR / f"{DATASET}_{DAY}_merged.nc"
    ds.to_netcdf(merged_path)
    print(f"\nMerged {len(nc_paths)} passes -> {merged_path}")

    summarize(ds, source=f"AVISO L2P (mono-mission, uncross-calibrated) — {DAY}")


if __name__ == "__main__":
    main()
