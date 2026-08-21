"""Map regions and cartopy axes for plotting the HEALPix fields.

Extents are in cartopy's ``set_extent`` order, ``[lon_min, lon_max, lat_min, lat_max]``,
which is *not* the ``[lat_min, lon_min, lat_max, lon_max]`` order anemoi uses for
``diagnostics.plot.focus_areas``.

Usage::

    import sys; sys.path.append("../..")
    from regions import REGIONS, subplots, wrap_longitudes

    ds = wrap_longitudes(ds)
    fig, ax = subplots("gulf_stream")
    ax.scatter(ds.longitude, ds.latitude, c=ds.sla, s=15, transform=ccrs.PlateCarree())
"""

import cartopy.crs as ccrs
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

REGIONS = {
    "global": [-180, 180, -90, 90],
    # North Atlantic
    "north_atlantic": [-80, 0, 0, 65],  # the basin, equator to the Greenland-Scotland ridge
    "subpolar_gyre": [-70, -10, 40, 65],  # highest mesoscale SLA variance up north
    "gulf_stream": [-80, -40, 30, 45],  # sharpest gradients in the basin
    "labrador_sea": [-65, -40, 50, 67],
    "nordic_seas": [-20, 20, 55, 80],
    # Runs past +180 rather than wrapping: the basin straddles the dateline, and a box
    # with lon_min > lon_max is not something set_extent accepts. Everything downstream
    # centres on the box, so the projection's seam lands at 25E, well outside it.
    "tropical_pacific": [120, 290, -30, 30],
    # Other eddy-rich regions, useful as contrast
    "kuroshio": [130, 180, 25, 45],
    "agulhas": [10, 60, -45, -25],
    "southern_ocean": [-180, 180, -70, -35],
    "mediterranean": [-6, 36, 30, 46],
    # Matches the focus_areas in training.yaml, for comparing against the training plots
    "europe": [-20, 40, 30, 60],
}


def wrap_longitudes(xds, name="longitude"):
    """Put longitudes on -180..180.

    The anemoi datasets store 0..360. ``scatter`` copes with either, but every extent
    here is negative west of Greenwich, so subsetting and ``tricontourf`` need this.
    """
    return xds.assign_coords({name: ((xds[name] + 180) % 360) - 180})


def projection(region):
    """A projection that suits the region's aspect and latitude band."""
    lon0, lon1, lat0, lat1 = REGIONS[region]
    lon_c, lat_c = (lon0 + lon1) / 2, (lat0 + lat1) / 2

    if region == "global":
        return ccrs.Robinson()
    if lat1 < -30 and lon1 - lon0 > 180:
        return ccrs.SouthPolarStereo(central_longitude=lon_c)
    # Conformal conics degenerate across the equator, so they are for one hemisphere only.
    if lat0 > 30:
        return ccrs.LambertConformal(central_longitude=lon_c, central_latitude=lat_c)
    return ccrs.Mercator(central_longitude=lon_c)


def subplots(region="north_atlantic", figsize=(7, 6), coastline_resolution="50m", **kwargs):
    """A figure and cartopy axis cropped to ``region``."""
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": projection(region)},
        **kwargs,
    )
    return fig, ax


def set_region(ax, region, coastline_resolution="50m"):
    """Crop an existing cartopy axis, e.g. one panel of a subplot grid."""
    if region != "global":
        ax.set_extent(REGIONS[region], crs=ccrs.PlateCarree())
    ax.coastlines(resolution=coastline_resolution, lw=0.5)
    return ax


def to_grid(values, region="global", resolution=0.25, nest=True, interp=False, pad=3.0):
    """Sample a HEALPix map onto a regular lat/lon grid, for ``pcolormesh``.

    The anemoi ``(cell,)`` arrays are healpy maps as they stand -- nside 64, nested -- so
    this is a lookup, not a regrid. Nearest-neighbour by default, which keeps the true
    cell boundaries visible; ``interp=True`` bilinearly smooths them instead.

    Returns ``(lon, lat, grid)`` for ``ax.pcolormesh(lon, lat, grid,
    transform=ccrs.PlateCarree())``. Preferable to ``scatter`` for a dense field: real
    cell shapes, and no marker size to tune. Sparse per-mission observations still want
    ``scatter`` with :func:`cell_sizes`, since nearest-neighbour would smear the gaps.

    ``pad`` widens the grid past the region so a conic projection's corners, which bow
    outside the lat/lon box, do not come up empty.
    """
    lon0, lon1, lat0, lat1 = REGIONS[region]
    # Longitude is deliberately not clamped to +/-180: a dateline-straddling region is
    # written as a continuous run past 180, and ang2pix wraps it for us.
    lon0, lon1 = lon0 - pad, lon1 + pad
    lat0, lat1 = max(lat0 - pad, -90), min(lat1 + pad, 90)

    lon = np.arange(lon0, lon1 + resolution, resolution)
    lat = np.arange(lat0, lat1 + resolution, resolution)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)

    nside = hp.npix2nside(np.size(values))
    if interp:
        grid = hp.get_interp_val(values, mesh_lon, mesh_lat, nest=nest, lonlat=True)
    else:
        grid = np.asarray(values)[hp.ang2pix(nside, mesh_lon, mesh_lat, nest=nest, lonlat=True)]
    return lon, lat, grid


def draw_cells(ax, values, nest=True, vmin=None, vmax=None, **kwargs):
    """Draw each HEALPix cell as its true quadrilateral. No resampling at all.

    Unlike :func:`to_grid`, which rasterises onto a lat/lon grid and so quantises the
    cell edges, this takes the cell corners from ``healpy.boundaries`` and hands cartopy
    the polygons. Exact geometry, and about as fast, so it is the better choice whenever
    the cells are large enough to see individually.

    Returns the ``PolyCollection``, e.g. for ``fig.colorbar(coll, ax=ax)``. Pass the
    usual ``cmap``/``vmin``/``vmax``/``edgecolors`` through ``kwargs``.

    Edges default to the face colour. The cells tile the sphere exactly and neighbours
    share vertices, but each polygon is antialiased against the background
    independently, which leaves a hairline of background showing between them. Drawing
    the edge in the polygon's own colour covers it. ``edgecolors="none"`` brings the
    hairlines back; a visible grid wants an explicit colour and a ``linewidths``.
    """
    kwargs.setdefault("edgecolors", "face")
    values = np.asarray(values)
    nside = hp.npix2nside(values.size)
    corners = hp.boundaries(nside, np.arange(values.size), step=1, nest=nest)
    lon, lat = hp.vec2ang(np.moveaxis(corners, 1, -1).reshape(-1, 3), lonlat=True)
    lon = (((lon + 180) % 360) - 180).reshape(-1, 4)
    lat = lat.reshape(-1, 4)

    # Project here rather than passing transform=PlateCarree(). Handing cartopy 49152
    # polygons makes it clip each one as a geometry, which takes two minutes; projecting
    # the vertices ourselves and drawing in axes coordinates takes well under a second.
    xy = ax.projection.transform_points(ccrs.PlateCarree(), lon.ravel(), lat.ravel())
    xy = xy[:, :2].reshape(-1, 4, 2)

    # A cell straddling the projection's seam has vertices at both edges of the map and
    # would draw as a band right across it. The seam sits at central_longitude +/- 180,
    # not at the dateline, so this has to be judged in projected space: a real cell is
    # ~1 degree wide and can never span an appreciable fraction of the whole domain.
    # Cells off the projection's domain entirely come back non-finite.
    span = xy[..., 0].max(axis=1) - xy[..., 0].min(axis=1)
    domain = ax.projection.x_limits
    drawable = np.isfinite(xy).all(axis=(1, 2)) & (span < 0.25 * (domain[1] - domain[0]))

    collection = PolyCollection(xy[drawable], array=values[drawable], **kwargs)
    # PolyCollection takes limits through set_clim, not the constructor.
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    return collection


def cell_sizes(ax, longitude, latitude, nside=64, fill=1.0):
    """Per-point ``scatter`` sizes that make the HEALPix cells tile the map.

    The cells are equal-area on the sphere, but ``s`` is an on-screen area, so a single
    value leaves gaps where the projection compresses and overlaps where it stretches.
    This measures the projection's local scale by walking one cell north and one cell
    east of each point, so it holds for any projection without a per-projection
    cos(latitude) exponent.

    Call it after the axis is cropped and the figure is sized -- it reads both. It does
    not follow interactive zoom.

    ``fill`` shrinks (<1) or pads (>1) the markers relative to the true cell.
    """
    diameter = np.degrees(np.sqrt(4 * np.pi / (12 * nside**2)))
    longitude, latitude = np.asarray(longitude), np.asarray(latitude)

    def display(lon, lat):
        xy = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)[:, :2]
        return ax.transData.transform(xy)

    here = display(longitude, latitude)
    # Step away from the pole so the offset point stays on the globe.
    north = display(longitude, latitude - np.sign(latitude) * diameter)
    east = display(longitude + diameter / np.cos(np.radians(latitude)), latitude)

    pixels = np.hypot(*(north - here).T) * np.hypot(*(east - here).T)
    points_per_pixel = 72 / ax.figure.dpi
    sizes = fill * pixels * points_per_pixel**2

    # Off-frame points would otherwise carry the projection's blow-up -- Mercator near
    # the poles reaches ~1e6 pt^2 -- and a marker that large, centred just outside, still
    # paints inside. Non-finite coordinates fail this comparison too, which is the point.
    box = ax.bbox.expanded(1.1, 1.1)
    on_screen = ((here > [box.x0, box.y0]) & (here < [box.x1, box.y1])).all(axis=1)
    return np.where(on_screen, sizes, 0.0)
