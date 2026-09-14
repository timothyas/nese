"""Pull CMEMS in-situ tide-gauge sea level (product INSITU_GLO_PHY_SSH_DISCRETE_MY_013_053).

What we learned exploring this product (so future-me doesn't have to re-derive it):

  * Three datasets. We use the global HOURLY one, which is already resampled to a
    clean 60-min cadence (nice for a fixed time axis):
        cmems_obs-ins_glo_phy-ssh_my_na_PT1H   -> 934 stations, all "_60minute"
    The rawer alternative keeps each gauge's native, gappy sampling:
        cmems_obs-ins_glo_phy-ssh_my_na_irr    -> 1306 files (some locations appear
        at several rates, e.g. Ajaccio at 1/2/5/10/60 min; plain names carry their
        cadence in the `time_coverage_resolution` attr, not the filename).

  * Data is ONE NetCDF per tide-gauge station (fixed scalar LATITUDE/LONGITUDE).
    There is no single combined file. Discover them with a dry-run `get`, then
    bulk-download with a `filter`. (The `arco-platform-series` service would let you
    open all stations lazily as one store, but it errors in copernicusmarine 2.2.3.)

  * Variables: SLEV (water level, m), SLEV_QC (1 = good_data), TIME, LATITUDE,
    LONGITUDE. SLEV is referenced to a per-station NAMED datum in
    `SLEV.attrs["sea_level_datum"]` (e.g. "BSCD2000(DVR90)", "Admiralty Chart
    Datum") -- heterogeneous and NOT a subtractable number.

  * To get an SLA comparable to the DUACS altimetry `sla_filtered`: subtract each
    station's TEMPORAL MEAN (see `to_sla`). That removes the arbitrary datum and the
    local mean sea surface at once -- no MSS/geoid model needed. De-mean over the
    altimetry reference window [1993, 2012] so offsets match. The trend survives
    (as it does in altimetry SLA); the gauge trend is RELATIVE sea level, so it
    differs from altimetry by vertical land motion (GIA/subsidence) -- a trend-only
    correction to apply later if you ever need absolute agreement.

Credentials: COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars, or a cached
`copernicusmarine login`.

    conda run -n seso python pull_insitu_ssh.py
"""

from pathlib import Path

import numpy as np
import xarray as xr
import copernicusmarine

DATASET_ID = "cmems_obs-ins_glo_phy-ssh_my_na_PT1H"   # global hourly tide gauges
DATA_DIR = Path(__file__).parent / "data" / "hourly"

# Which stations to pull. None -> a small sample (first N). A glob (e.g.
# "*IR_TS_TG*" for one provider, or "*" for all 934) -> pass straight to `get`.
STATION_FILTER = None
SAMPLE_N = 3

SLA_REF_PERIOD = ("1993-01-01", "2013-01-01")   # DUACS SLA reference window


def list_stations():
    """All station filenames in the dataset (no download)."""
    res = copernicusmarine.get(dataset_id=DATASET_ID, dry_run=True,
                               disable_progress_bar=True)
    return sorted(Path(f.file_path).name for f in res.files)


def pull(pattern):
    """Download station file(s) matching `pattern` into DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    copernicusmarine.get(dataset_id=DATASET_ID, filter=pattern,
                         output_directory=str(DATA_DIR), no_directories=True,
                         disable_progress_bar=True)


def pull_station(name):
    """Download one station by exact filename, skipping if already present.

    (copernicusmarine's overwrite=False writes a numbered "_(1)" copy rather than
    skipping, so we guard on existence ourselves to keep re-runs idempotent.)
    """
    if not (DATA_DIR / name).exists():
        pull(f"*/{name}")


def to_sla(ds, ref_period=SLA_REF_PERIOD):
    """Good-QC SLEV as an anomaly: subtract the mean over `ref_period`.

    Falls back to the full-record mean if the station doesn't cover the window.
    Returns a 1-D DataArray on TIME (metres), datum/MSS removed.
    """
    slev = ds["SLEV"].where(ds["SLEV_QC"] == 1).squeeze(drop=True)
    ref = slev.sel(TIME=slice(*ref_period))
    baseline = ref.mean() if ref.count() > 0 else slev.mean()
    return (slev - baseline).rename("sla")


def summarize(ds):
    name = ds.attrs.get("platform_code", "?")
    lat, lon = float(ds["LATITUDE"]), float(ds["LONGITUDE"])
    datum = ds["SLEV"].attrs.get("sea_level_datum", "?")
    sla = to_sla(ds)
    print(f"  {name:16s} ({lat:6.2f}, {lon:7.2f})  datum={datum}")
    print(f"      {ds.sizes.get('TIME', 0):>8} hourly obs, "
          f"{ds.attrs.get('time_coverage_start', '?')[:10]} -> "
          f"{ds.attrs.get('time_coverage_end', '?')[:10]}")
    print(f"      SLA (de-meaned) std = {float(sla.std()):.3f} m")


def main():
    stations = list_stations()
    print(f"{len(stations)} stations in {DATASET_ID}")

    if STATION_FILTER is None:
        for name in stations[:SAMPLE_N]:
            pull_station(name)
    else:
        pull(STATION_FILTER)

    files = sorted(DATA_DIR.glob("*.nc"))
    print(f"\nInspecting {len(files)} downloaded station(s):")
    for f in files:
        summarize(xr.open_dataset(f))


if __name__ == "__main__":
    main()
