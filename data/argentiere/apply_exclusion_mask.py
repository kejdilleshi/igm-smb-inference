#!/usr/bin/env python3
"""
Rasterize out_mask.geojson (EPSG:32632) onto the NetCDF grids and zero out
`icemask` / `icemaskobs` pixels inside the polygon — i.e. exclude the small
tributary glacier from the DA domain.
"""

import json
import os
import shutil
import numpy as np
import netCDF4 as nc
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
GEOJSON = os.path.join(HERE, "out_mask.geojson")
TARGETS = [
    os.path.normpath(os.path.join(HERE, "..", "input_argentiere.nc")),
    os.path.join(HERE, "input_100m.nc"),
]


def load_polygons(path):
    with open(path) as f:
        gj = json.load(f)
    geoms = [shape(ft["geometry"]) for ft in gj["features"] if ft.get("geometry")]
    if not geoms:
        raise ValueError(f"No geometries in {path}")
    return geoms


def build_transform(x, y):
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    # Pixel corners: x/y are pixel centres.
    x_origin = float(x[0]) - dx / 2.0
    # If y is descending (north-up raster), top-left is y[0] + |dy|/2.
    # If y is ascending, top-left is y[-1] + |dy|/2 and we must flip.
    if dy < 0:
        y_top = float(y[0]) - dy / 2.0  # dy<0 so this adds |dy|/2
        flip = False
    else:
        y_top = float(y[-1]) + dy / 2.0
        flip = True
    return from_origin(x_origin, y_top, abs(dx), abs(dy)), flip


def patch(nc_path, geoms):
    if not os.path.exists(nc_path):
        print(f"skip (missing): {nc_path}")
        return

    backup = nc_path + ".bak_exclusion"
    if not os.path.exists(backup):
        shutil.copy2(nc_path, backup)
        print(f"backup: {backup}")

    with nc.Dataset(nc_path, "r+") as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        ny, nx = len(y), len(x)
        transform, flip = build_transform(x, y)

        excl = rasterize(
            [(g, 1) for g in geoms],
            out_shape=(ny, nx),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        if flip:
            excl = excl[::-1, :]

        n_excl = int(excl.sum())
        print(f"{nc_path}: polygon covers {n_excl} pixels on the grid")

        for name in ("icemask", "icemaskobs"):
            if name not in ds.variables:
                print(f"  {name}: not present — skipped")
                continue
            m = ds.variables[name][:].astype(np.float32)
            before = int((m > 0.5).sum())
            m[excl == 1] = 0.0
            ds.variables[name][:, :] = m
            after = int((m > 0.5).sum())
            print(f"  {name}: {before} -> {after} on-glacier pixels "
                  f"({before - after} dropped)")

    print(f"patched: {nc_path}")


def main():
    geoms = load_polygons(GEOJSON)
    print(f"loaded {len(geoms)} polygon(s) from {GEOJSON}")
    for t in TARGETS:
        patch(t, geoms)


if __name__ == "__main__":
    main()
