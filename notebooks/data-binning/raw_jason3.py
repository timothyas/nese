import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nese.dataprep.sources import CMEMSL3SLAARCO

mission = "jason_3"
freq = "24h"

mission_start, mission_end = CMEMSL3SLAARCO.list_groups()[mission]
rng = np.random.default_rng()
date = pd.Timestamp(rng.choice(pd.date_range(mission_start.ceil("D"), mission_end.floor("D") - pd.Timedelta(freq), freq="D")))

source = CMEMSL3SLAARCO(
    time={"start": date, "end": date, "freq": freq},
    variables=["sla_filtered"],
    groups=[mission],
)
raw = source.open_record_dataset({"time": date})[mission]
print(date, raw.sizes)

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(8, 4))
ax.coastlines(linewidth=0.5)
ax.set_global()
mappable = ax.scatter(
    raw.longitude,
    raw.latitude,
    c=raw.sla_filtered,
    s=1,
    cmap="RdBu_r",
    vmin=-0.3,
    vmax=0.3,
    transform=ccrs.PlateCarree(),
)
fig.colorbar(mappable, ax=ax, label="sla_filtered (m)", shrink=0.7)
ax.set_title(f"{mission} raw along-track, [{date:%Y-%m-%d}, +{freq})")
