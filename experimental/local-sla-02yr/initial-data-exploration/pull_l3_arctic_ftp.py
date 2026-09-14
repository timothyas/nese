"""Pull one CryoSat-2 Arctic along-track (L3) "leads" SLA file from AVISO+ FTP.

This is the SSALTO/DUACS *experimental* polar product: sea level anomalies
retrieved inside sea-ice leads, 50N-88N, ~5 Hz (~1 km) along-track, delayed-time.
Unlike the global L2P/L3 products (ODATIS THREDDS / Copernicus Marine), it is
distributed only through the authenticated AVISO+ FTP, one plain NetCDF file per
mission per day -- so there is no cycle/pass listing and nothing to merge; this
mirrors pull_l3_cmems.py far more than pull_l2p_thredds.py.

Register + sign the license, selecting the product "Ssalto/Duacs Experimental
products: along-track and gridded Sea Level Heights and velocities":
    https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html
AVISO+ emails you a dedicated FTP login/password (may differ from the web login)
plus the exact directory tree for your account. Export them before running:
    export AVISO_USERNAME=...
    export AVISO_PASSWORD=...

Filename convention (handbook SALP-MU-P-EA-23555-CLS, along-track leads files):
    dt_arctic_<mission>_leads_phy_l3_<dataset_date>_<production_date>.nc
where <dataset_date> is the measurement day (YYYYMMDD). SLA lives in the
`sla_filtered` / `sla_unfiltered` variables (int16, scale_factor 0.001, metres).

NOTE: REMOTE_DIR below is a best guess. The authoritative path is the one AVISO+
sends you; the version label appears in the FTP folder name. If a listing comes
back empty, set REMOTE_DIR to what you see when browsing the FTP, and check
whether files are foldered by year.
"""

import ftplib
import os
import re
import ssl
from pathlib import Path

import xarray as xr

FTP_HOST = "ftp-access.aviso.altimetry.fr"

# Mission token as it appears in the filename. CryoSat-2 here; SARAL/AltiKa and
# Sentinel-3A are the other two missions (the latter appears as "s3a" in sample
# filenames, so confirm the token if you retarget).
MISSION = "cryosat2"
VERSION = "version_02_00"
DAY = "20100716"  # YYYYMMDD; CryoSat-2 along-track spans ~2012-12-21 -> 2021-12-31

# Best-guess remote directory -- correct this to your account's actual tree.
REMOTE_DIR = (
    f"/duacs-experimental/arctic-ocean/{VERSION}/along-track/leads/{MISSION}"
)

DATA_DIR = Path(__file__).parent / "data" / "l3_arctic"

# dt_arctic_<mission>_leads_phy_l3_<YYYYMMDD>_<prodYYYYMMDD>.nc
LEADS_RE = re.compile(
    r"dt_arctic_(?P<mission>[^_]+)_leads_phy_l3_(?P<date>\d{8})_\d{8}\.nc$"
)


def connect(user, pw):
    """Open an AVISO+ FTP session, preferring FTPS and falling back to plain FTP."""
    try:
        ftp = ftplib.FTP_TLS(FTP_HOST, timeout=60, context=ssl.create_default_context())
        ftp.login(user, pw)
        ftp.prot_p()
        return ftp
    except ftplib.all_errors:
        ftp = ftplib.FTP(FTP_HOST, timeout=60)
        ftp.login(user, pw)
        return ftp


def list_day_files(ftp, remote_dir, mission, day):
    """Return remote leads filenames for `mission` whose measurement day is `day`."""
    names = [name.rsplit("/", 1)[-1] for name in ftp.nlst(remote_dir)]
    out = []
    for name in names:
        m = LEADS_RE.match(name)
        if m and m.group("mission") == mission and m.group("date") == day:
            out.append(name)
    return sorted(out)


def download(ftp, remote_dir, filename):
    """Fetch one granule over FTP. Idempotent: reuse an already-downloaded file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = DATA_DIR / filename
    if local.exists():
        return local
    with open(local, "wb") as f:
        ftp.retrbinary(f"RETR {remote_dir}/{filename}", f.write)
    return local


def summarize(ds, source):
    """Print a compact summary so this can be compared against the other products."""
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    print(ds)
    print("\n-- coordinate ranges --")
    for name in ("time", "latitude", "longitude"):
        if name in ds:
            v = ds[name]
            print(f"  {name:10s}: {v.min().values}  ->  {v.max().values}  (n={v.size})")
    for sla_name in ("sla_filtered", "sla_unfiltered", "sla", "sea_level_anomaly"):
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

    ftp = connect(user, pw)
    try:
        try:
            day_files = list_day_files(ftp, REMOTE_DIR, MISSION, DAY)
        except ftplib.all_errors as e:
            raise SystemExit(
                f"Could not list {REMOTE_DIR!r} ({e}). Fix REMOTE_DIR to match the "
                "path AVISO+ gave your account (mind the version and any year folders)."
            )
        if not day_files:
            raise SystemExit(
                f"No {MISSION} leads file for {DAY} in {REMOTE_DIR}. Check the day is "
                "in range and that MISSION matches the filename token on the FTP."
            )

        local_paths = []
        for i, filename in enumerate(day_files, 1):
            print(f"  [{i}/{len(day_files)}] {filename}")
            local_paths.append(download(ftp, REMOTE_DIR, filename))
    finally:
        ftp.quit()

    # One file per day is the norm; concat only on the rare chance of several.
    if len(local_paths) == 1:
        ds = xr.open_dataset(local_paths[0])
    else:
        ds = xr.concat(
            [xr.open_dataset(p) for p in local_paths],
            dim="time", data_vars="minimal", coords="minimal", compat="override",
        ).sortby("time")

    summarize(ds, source=f"AVISO Arctic leads L3 (experimental, {MISSION}) — {DAY}")


if __name__ == "__main__":
    main()
