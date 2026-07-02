"""Pull a single Jason-1 L3 along-track SLA file from Copernicus Marine.

L3 = the DUACS along-track product derived from L2P, but cross-calibrated /
homogenized across missions to a reference. Less "raw" than L2P, but consistent
across missions. Distributed by Copernicus Marine (CMEMS) via the official
`copernicusmarine` toolbox, which is the scalable, cloud-friendly access path.

Register (free) for a Copernicus Marine account:
    https://data.marine.copernicus.eu/register

Auth is non-interactive. The toolbox picks up these env vars automatically
(nothing is written to disk -- ideal for distributed workers with injected
secrets):
    export COPERNICUSMARINE_SERVICE_USERNAME=...
    export COPERNICUSMARINE_SERVICE_PASSWORD=...
There is no API token; auth is username/password (AWS Cognito). Alternatives:
run `copernicusmarine login` once to cache a credentials file, or pass
username=/password= straight to get().

Product: SEALEVEL_GLO_PHY_L3_MY_008_062 (reprocessed, "tailored for data
assimilation"). Each mission is a separate dataset id (..._<mission>-l3-duacs...);
Jason-1 alone has three phases: j1 (main), j1g (geodetic), j1n (interleaved/new
orbit). We use j1 on 2008-02-27, matching the L2P sample pulled by
pull_l2p_thredds.py. The CryoSat-2 counterpart (kept for reference below) is
cmems_obs-sl_glo_phy-ssh_my_c2-l3-duacs_PT1S, starting 2010-07-16.
"""

import os
from pathlib import Path

import copernicusmarine
import xarray as xr

# CryoSat-2 (kept for reference):
#DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_c2-l3-duacs_PT1S"
#DAY = "20100716"
DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_j1-l3-duacs_PT1S"
DATASET_VERSION = "202411"  # the "202411" is a version label, not part of the id
DAY = "20080227"  # same day as the L2P sample, for a like-for-like comparison

DATA_DIR = Path(__file__).parent / "data" / "l3"


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
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Credentials come from COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars
    # (or a cached credentials file). Passing None lets the toolbox resolve them.
    username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

    # `get` downloads the original NetCDF granule(s). DUACS L3 along-track files
    # are one-per-day per mission; filter to the single day of interest.
    result = copernicusmarine.get(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        filter=f"*{DAY}*",
        output_directory=str(DATA_DIR),
        no_directories=True,
        overwrite=True,
        username=username,
        password=password,
    )

    files = [Path(f.file_path) for f in result.files] if result else []
    files = [f for f in files if f.suffix == ".nc"]
    if not files:
        raise SystemExit(f"No files matched *{DAY}* in {DATASET_ID}.")
    print(f"Downloaded {len(files)} file(s); inspecting {files[0].name}")

    ds = xr.open_dataset(files[0])
    summarize(ds, source="CMEMS L3 (DUACS, cross-calibrated, DA-tailored)")


if __name__ == "__main__":
    main()
