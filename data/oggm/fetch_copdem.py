#!/usr/bin/env python3
"""
Fetch Copernicus DEM GLO-30 (~2010-2019 in the European Alps) onto the same
OGGM grid as the L3 prepro gdir produced by fetch_oggm.py.

OGGM's COPDEM30 path goes through prism-dem-open.copernicus.eu, which has
been unreliable from our network. We bypass it and pull the same tiles from
the AWS Open Data mirror (s3://copernicus-dem-30m, no auth needed), then
reproject them onto the gdir grid using rasterio.

Output: <gdir>/copdem30.tif on the gdir's local TMerc grid, consumed by
build_input_nc.py as the end-of-period (≈2020) surface anchor — replacing the
previous `topo + dhdt × 20` extrapolation. The 2000 surface is then backed
out via `copdem30 - dhdt × 20`.

Usage:
    python fetch_copdem.py [RGI_ID]
"""

import os
import sys
import math
import shutil
import urllib.request
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RGI_ID = "RGI60-11.01450"
S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
CACHE_DIR = os.path.expanduser("~/.cache/copdem30_tiles")


def tile_name(lat_floor, lon_floor):
    """Tile naming: N46E007 covers 46..47°N, 7..8°E."""
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00"


def download_tile(lat_floor, lon_floor, out_dir):
    name = tile_name(lat_floor, lon_floor)
    fname = f"Copernicus_DSM_COG_10_{name}_DEM.tif"
    url = f"{S3_BASE}/Copernicus_DSM_COG_10_{name}_DEM/{fname}"
    dst = os.path.join(out_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"   cached: {fname}")
        return dst
    print(f"   downloading {url}")
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dst)
    return dst


def main(rgi_id):
    target_dir = os.path.join(HERE, rgi_id)
    gridded = os.path.join(target_dir, "gridded_data.nc")
    if not os.path.exists(gridded):
        sys.exit(f"missing gdir gridded data: {gridded} (run fetch_oggm.py first)")
    os.makedirs(CACHE_DIR, exist_ok=True)

    ds = xr.open_dataset(gridded)
    src_crs = ds.attrs.get("pyproj_srs")
    if not src_crs:
        sys.exit("gridded_data.nc has no pyproj_srs attribute — cannot reproject.")
    x = ds.x.values.astype(np.float64)
    y = ds.y.values.astype(np.float64)
    dx = float(abs(x[1] - x[0]))
    nx, ny = len(x), len(y)

    print(f"== Fetching COP-DEM 30 for {rgi_id}")
    print(f"   gdir grid: {nx} × {ny} @ dx={dx:.1f} m, CRS={src_crs!r}")

    # Determine needed COP-DEM tiles from the lat/lon bbox of the grid corners.
    to_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    xs = np.array([x.min(), x.max(), x.max(), x.min()])
    ys = np.array([y.min(), y.min(), y.max(), y.max()])
    lons, lats = to_ll.transform(xs, ys)
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    print(f"   lon/lat bbox: lon [{lon_min:.3f}, {lon_max:.3f}], "
          f"lat [{lat_min:.3f}, {lat_max:.3f}]")

    tiles = []
    for lat_floor in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lon_floor in range(math.floor(lon_min), math.floor(lon_max) + 1):
            tiles.append(download_tile(lat_floor, lon_floor, CACHE_DIR))
    print(f"   {len(tiles)} tile(s) ready")

    # Mosaic in source CRS (EPSG:4326), then reproject to gdir grid.
    open_tiles = [rasterio.open(t) for t in tiles]
    mosaic, mosaic_transform = merge(open_tiles)
    mosaic_crs = open_tiles[0].crs
    mosaic_dtype = open_tiles[0].dtypes[0]
    for s in open_tiles:
        s.close()
    print(f"   mosaic: shape={mosaic.shape}, crs={mosaic_crs}, dtype={mosaic_dtype}")

    # Destination raster: same grid as gridded_data (pixel-centered x, y).
    # Build the affine from pixel centers — rasterio transforms are
    # upper-left-origin, so y_top corresponds to max(y) + dx/2 if y is
    # ascending, max(y) - (-dx/2) = max(y) + dx/2 either way.
    # rasterio writes top-down (row 0 = max y). For our destination we mirror
    # that convention. The output will be y-descending in the GeoTIFF; that's
    # fine because build_input_nc.py flips on read.
    x_left = x.min() - dx / 2.0
    y_top = y.max() + dx / 2.0
    dst_transform = rasterio.transform.from_origin(x_left, y_top, dx, dx)

    dst = np.full((ny, nx), np.nan, dtype=np.float32)
    reproject(
        source=mosaic[0],  # 2D; reproject silently drops pixels if given (1,H,W)
        destination=dst,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    n_nan = int(np.isnan(dst).sum())
    print(f"   reprojected to gdir grid: dst NaN count = {n_nan} / {dst.size}")
    if n_nan > 0:
        print("   WARNING: NaN holes in COP-DEM mosaic; check tile coverage.")

    target_tif = os.path.join(target_dir, "copdem30.tif")
    with rasterio.open(
        target_tif, "w", driver="GTiff",
        height=ny, width=nx, count=1, dtype="float32",
        crs=src_crs, transform=dst_transform,
        nodata=np.nan, compress="deflate",
    ) as out:
        out.write(dst, 1)
    print(f"\nWrote {target_tif}")

    # Tiny provenance file alongside dem_source.txt.
    src_txt = os.path.join(target_dir, "copdem30_source.txt")
    with open(src_txt, "w") as f:
        f.write("Copernicus DEM GLO-30 (COP-DEM 30)\n")
        f.write(f"Tiles ({len(tiles)}) from {S3_BASE}/:\n")
        for t in tiles:
            f.write(f"  {os.path.basename(t)}\n")
        f.write("\nReprojected to gdir grid using rasterio bilinear.\n")
        f.write("Citation: ESA / Copernicus, 2022. CC BY 4.0.\n")
    print(f"      {src_txt}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RGI_ID)
