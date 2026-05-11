#!/usr/bin/env python3
"""
Add observed ice thickness (thkobs) to data/input_argentiere.nc from the
Rabatel-2018 GPR profile measurements stored in
data/argentiere/zbed_arg_measured_UTM32N.gpkg.

The geopackage stores measured *bed* elevations (column `Field_3`) at GPR
points along transects. Observed thickness at each point is computed as

    thkobs_pt = usurf(point) - zbed_measured

using the 2012 surface from the NetCDF (NaN-filled `usurf`, which matches
`thkinit`'s 2012 epoch). Points are then rasterized onto the model grid by
binning to the nearest pixel and averaging per pixel — same recipe as IGM's
`read_glathida_v6` (igm/inputs/oggm_shop/read_glathida.py). Pixels with no
measurement remain NaN, so the DA cost term `misfit_thk` (which gates on
`~tf.math.is_nan(state.thkobs)`) only sees observed pixels.
"""

import os
import shutil
import numpy as np
import netCDF4 as nc
import geopandas as gpd
from scipy.interpolate import RegularGridInterpolator

NC_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "input_argentiere.nc")
)
GPKG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "zbed_arg_measured_UTM32N.gpkg"
)


def main():
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(NC_PATH)
    if not os.path.exists(GPKG_PATH):
        raise FileNotFoundError(GPKG_PATH)

    backup = NC_PATH + ".bak_thkobs"
    if not os.path.exists(backup):
        shutil.copy2(NC_PATH, backup)
        print(f"Backup written: {backup}")

    gdf = gpd.read_file(GPKG_PATH)
    xx = gdf.geometry.x.values.astype(np.float64)
    yy = gdf.geometry.y.values.astype(np.float64)
    zbed = gdf["Field_3"].values.astype(np.float64)
    print(f"Loaded {len(gdf)} measurement points (CRS={gdf.crs}).")

    with nc.Dataset(NC_PATH, "r+") as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        usurf = np.asarray(ds.variables["usurf"][:], dtype=np.float64)  # NaN-filled

        # Sample the 2012 surface at every measurement point. usurf is laid out
        # (y, x); we need an interpolator over an ascending axis so flip y if
        # it's descending (the Argentière grid has y descending, dy < 0).
        if y[0] > y[-1]:
            y_asc = y[::-1]
            usurf_asc = usurf[::-1, :]
        else:
            y_asc = y
            usurf_asc = usurf
        interp = RegularGridInterpolator(
            (y_asc, x), usurf_asc, bounds_error=False, fill_value=np.nan
        )
        zsurf = interp(np.column_stack([yy, xx]))

        thk_pts = zsurf - zbed
        n_neg = int(np.sum(thk_pts < 0))
        n_nan = int(np.sum(np.isnan(thk_pts)))
        thk_pts = np.maximum(thk_pts, 0.0)
        print(
            f"Pointwise thickness: median={np.nanmedian(thk_pts):.1f} m, "
            f"min={np.nanmin(thk_pts):.1f}, max={np.nanmax(thk_pts):.1f}, "
            f"clipped<0={n_neg}, NaN(off-grid)={n_nan}"
        )

        # Bin to pixel indices. dx > 0 (x ascending), dy may be < 0 (y descending).
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        col = np.floor((xx - x[0]) / dx + 0.5).astype(int)
        row = np.floor((yy - y[0]) / dy + 0.5).astype(int)

        ny, nx = usurf.shape
        valid = (
            np.isfinite(thk_pts)
            & (col >= 0) & (col < nx)
            & (row >= 0) & (row < ny)
        )
        print(f"Points falling on grid: {int(valid.sum())}/{len(thk_pts)}")

        # Mean thickness per pixel (skip zero-thickness contributions, mirroring
        # read_glathida_v6 which sets gridded zeros to NaN).
        sums = np.zeros((ny, nx), dtype=np.float64)
        counts = np.zeros((ny, nx), dtype=np.int64)
        v = valid & (thk_pts > 0)
        np.add.at(sums, (row[v], col[v]), thk_pts[v])
        np.add.at(counts, (row[v], col[v]), 1)

        thkobs = np.full((ny, nx), np.nan, dtype=np.float32)
        observed = counts > 0
        thkobs[observed] = (sums[observed] / counts[observed]).astype(np.float32)
        print(
            f"Pixels with thkobs: {int(observed.sum())} "
            f"(mean={np.nanmean(thkobs):.1f} m, "
            f"max={np.nanmax(thkobs):.1f} m)"
        )

        # Cross-check vs icemask: points should fall on glacier.
        icemask = np.asarray(ds.variables["icemask"][:], dtype=np.float32)
        off_ice = int(((icemask < 0.5) & observed).sum())
        if off_ice:
            print(
                f"Note: {off_ice} observed pixels fall outside icemask "
                "(left as-is; misfit_thk gates on NaN, not icemask)."
            )

        if "thkobs" not in ds.variables:
            v = ds.createVariable("thkobs", "f4", ("y", "x"))
            v.long_name = "Ice Thickness Observations (Rabatel 2018 GPR)"
            v.units = "m"
            v.standard_name = "thkobs"
        else:
            ds.variables["thkobs"].long_name = (
                "Ice Thickness Observations (Rabatel 2018 GPR)"
            )
        ds.variables["thkobs"][:, :] = thkobs

    print(f"Patched: {NC_PATH}")


if __name__ == "__main__":
    main()
