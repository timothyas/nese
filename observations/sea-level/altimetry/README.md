# Satellite Altimetry SLA

Will pull L3 Altimetry product from Copernicus, rather than L2P AVISO.
Data from,

the reprocessed (MY) DUACS L3 along-track product
`SEALEVEL_GLO_PHY_L3_MY_008_062`, one dataset/variable per mission and orbit
phase (geodetic / interleaved kept separate). `Days` is the calendar span of
each record's temporal coverage.

| Mission | Token | Start | End | Days |
| --- | --- | --- | --- | ---: |
| TOPEX/Poseidon | tp | October 1992 | April 2002 | 3481 |
| ERS-1 | e1 | October 1992 | May 1995 | 935 |
| ERS-1 (geodetic) | e1g | April 1994 | March 1995 | 346 |
| ERS-2 | e2 | May 1995 | May 2002 | 2557 |
| GFO | g2 | January 2000 | September 2008 | 3167 |
| Jason-1 | j1 | April 2002 | October 2008 | 2371 |
| Envisat | en | May 2002 | October 2010 | 3077 |
| TOPEX/Poseidon (interleaved) | tpn | September 2002 | October 2005 | 1120 |
| Jason-2 | j2 | October 2008 | May 2016 | 2777 |
| Jason-1 (interleaved) | j1n | February 2009 | March 2012 | 1118 |
| CryoSat-2 | c2 | July 2010 | July 2020 | 3669 |
| Envisat (new orbit) | enn | October 2010 | April 2012 | 531 |
| Jason-1 (geodetic) | j1g | May 2012 | June 2013 | 391 |
| SARAL/AltiKa | al | March 2013 | March 2015 | 748 |
| HY-2A | h2a | April 2014 | March 2016 | 704 |
| SARAL/AltiKa (geodetic) | alg | March 2015 | January 2026 | 3945 |
| HY-2A (geodetic) | h2ag | March 2016 | June 2020 | 1532 |
| Jason-3 | j3 | May 2016 | December 2021 | 2044 |
| Sentinel-3A | s3a | June 2016 | January 2026 | 3490 |
| Jason-2 (interleaved) | j2n | October 2016 | May 2017 | 213 |
| Jason-2 (geodetic) | j2g | July 2017 | September 2017 | 66 |
| Sentinel-3B | s3b | November 2018 | January 2026 | 2608 |
| HY-2B | h2b | December 2019 | January 2026 | 2220 |
| CryoSat-2 (interleaved) | c2n | August 2020 | January 2026 | 1995 |
| Sentinel-6A | s6a-lr | December 2021 | January 2026 | 1480 |
| Jason-3 (interleaved) | j3n | April 2022 | January 2025 | 989 |
| SWOT nadir (cal/val) | swonc | January 2023 | July 2023 | 175 |
| SWOT nadir | swon | July 2023 | January 2026 | 911 |
| Jason-3 (geodetic) | j3g | June 2025 | January 2026 | 213 |

## Which Dataset? Scripts for AVISO L2P vs L3 CMEMS Comparison

Two standalone scripts that pull the **same along-track SLA track**
from two different products, so
you can inspect both and decide which to build the pipeline on.
Two mission/dates exist in the scripts (maybe commented out) for CryoSat-2 and
Jason-1.

| | `pull_l2p_thredds.py` | `pull_l3_cmems.py` |
|---|---|---|
| Product | AVISO **L2P** | CMEMS **L3** (DUACS) |
| Access | ODATIS THREDDS over HTTPS | `copernicusmarine` toolbox |
| Account | AVISO+ | Copernicus Marine |

Both require a (free) account. The THREDDS *catalog* is public — file discovery
needs no login — but downloading granules needs AVISO+ credentials.

### Setup

L2P (AVISO+): register and select "Sea Level Anomalies Along-track Level-2+
(L2P) for other missions" at
<https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html>, then:

```bash
export AVISO_USERNAME=...
export AVISO_PASSWORD=...
conda run -n seso python pull_l2p_thredds.py
```

L3 (CMEMS): register at <https://data.marine.copernicus.eu/register>. Auth is
non-interactive via env vars (no API token exists; it's username/password).

```bash
export COPERNICUSMARINE_SERVICE_USERNAME=...
export COPERNICUSMARINE_SERVICE_PASSWORD=...
conda run -n seso python pull_l3_cmems.py
```

For a persistent local setup instead, run `conda run -n seso copernicusmarine
login` once to cache a credentials file; the script picks it up automatically.

Downloaded files land in `data/l2p/` and `data/l3/`. Each script prints a
matching summary (dims, variables, SLA min/mean/max, coordinate ranges).
