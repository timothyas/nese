# Satellite Altimetry SLA

Will pull L3 Altimetry product from Copernicus, rather than L2P AVISO, to have a
cross-calibrated SLA product to start with.
Will not use SSH data from tide gauges, since this involves a variable mapping
that I want to avoid for this early stage of development.
Will not use Arctic products
The L3 product is now hooked up to nese-dataprep, but this documents pulling the
L2P product from AVISO for future reference.

## Scripts

* `pull_l2p_thredds.py` pulls the L2P data over ODATIS THREDDS, instead of HTTPS
* `pull_l3_cmems.py` pulls the L3 data, legacy, now use nese-dataprep
* `compare_products.py` compares the L2P AVISO data with CMEMS L3 product
* `pull_insitu_ssh.py` pulls in the in situ ssh tide gauge data
* `pull_l3_arctic_ftp.py` pulls L3 Arctic product from AVISO

### Setup

The two services require a free account. The THREDDS *catalog* is public — file discovery
needs no login — but downloading granules needs AVISO+ credentials.

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
