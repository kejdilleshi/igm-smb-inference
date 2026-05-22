#!/usr/bin/env python3
"""
Build an IGM-ready input NetCDF from an OGGM gdir folder produced by
fetch_oggm.py + fetch_copdem.py. Output layout matches data/input_argentiere.nc
and is consumed unchanged by experiment/params_oggm_aletsch.yaml (and any
future per-glacier params YAML).

Period anchor (NEW pipeline — anchored on 2020):
  • End DEM   ← COP-DEM 30 (Copernicus, ≈2010-2019 in the European Alps),
                fetched onto the gdir grid by fetch_copdem.py. Treated as the
                ~2020 surface — this is the inversion's reference.
  • dhdt      ← Hugonnet 2021 mean dh/dt over 2000-01-01 → 2020-01-01 (m yr⁻¹)
  • Start DEM ← copdem30 − dhdt × N_YEARS (default N_YEARS=20 → ~Jan 2000),
                derived rather than loaded from OGGM. OGGM's NASADEM `topo`
                is still read for a sanity-check Δh print, then ignored.

Velocities use Millan 2022 (2017-18 composite). ITS_LIVE was unavailable for
Aletsch when this was written, so Millan is the fallback. Slidingco recovered
by DA will reflect that velocity epoch, not 2000.

Usage:
    python build_input_nc.py [RGI_ID]
"""

import os
import sys
import shutil
import json
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import rasterio
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RGI_ID = "RGI60-11.01450"
N_YEARS = 20.0   # Hugonnet window: 2000-01-01 → 2020-01-01

# Sanity-check thresholds (Aletsch-scale; loose enough for any Alpine glacier).
EXPECTED_DHDT_MEAN_RANGE = (-3.0, 1.0)    # m yr⁻¹, on-glacier; outside → suspect units


def to_nan(a):
    if hasattr(a, "filled"):
        a = a.filled(np.nan)
    return np.asarray(a, dtype=np.float32)


def fill_nan_nearest(arr):
    m = np.isnan(arr)
    if not m.any():
        return arr
    _, (iy, ix) = distance_transform_edt(m, return_indices=True)
    out = arr.copy()
    out[m] = arr[iy[m], ix[m]]
    return out


def rasterize_glathida(csv_path, x, y, icemask):
    """Bin GlaThiDa point thicknesses onto the (y, x) grid by mean per pixel,
    matching IGM's read_glathida_v6/v7 recipe. CSV columns x_proj, y_proj are
    already in the OGGM gdir's local TMerc, so no reprojection needed."""
    df = pd.read_csv(csv_path)
    df = df[df["thickness"] > 0]
    n_total = len(df)

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    # Pixel index relative to pixel-center grid.
    col = np.floor((df["x_proj"].values - x[0]) / dx + 0.5).astype(int)
    row = np.floor((df["y_proj"].values - y[0]) / dy + 0.5).astype(int)
    nx = len(x); ny = len(y)
    valid = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    print(f"  GlaThiDa: {valid.sum()}/{n_total} points fall on grid")

    sums = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.int64)
    thk = df["thickness"].values.astype(np.float64)
    np.add.at(sums, (row[valid], col[valid]), thk[valid])
    np.add.at(counts, (row[valid], col[valid]), 1)

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    obs = counts > 0
    out[obs] = (sums[obs] / counts[obs]).astype(np.float32)
    n_on_ice = int(((icemask > 0.5) & obs).sum())
    n_off_ice = int(((icemask < 0.5) & obs).sum())
    print(f"  GlaThiDa raster: {int(obs.sum())} pixels with thkobs "
          f"({n_on_ice} on-ice, {n_off_ice} off-ice; mean={np.nanmean(out):.1f} m, "
          f"max={np.nanmax(out):.1f} m)")
    return out


def main(rgi_id):
    gdir = os.path.join(HERE, rgi_id)
    if not os.path.isdir(gdir):
        sys.exit(f"missing OGGM gdir: {gdir}  (run fetch_oggm.py first)")
    gridded = os.path.join(gdir, "gridded_data.nc")
    glathida = os.path.join(gdir, "glathida_data.csv")
    copdem_tif = os.path.join(gdir, "copdem30.tif")
    out_nc = os.path.normpath(os.path.join(HERE, "..", f"input_{rgi_id.replace('-', '_')}.nc"))

    print(f"== Building IGM-ready NetCDF for {rgi_id}")
    print(f"   src: {gridded}")
    print(f"   out: {out_nc}")

    ds = xr.open_dataset(gridded)

    # OGGM stores y descending. Flip to ascending (IGM convention) and flip
    # every 2-D array along axis 0.
    x = ds.x.values.astype(np.float32)
    y = ds.y.values.astype(np.float32)
    if y[0] > y[-1]:
        y = y[::-1]
        flipy = True
    else:
        flipy = False

    def fetch(name):
        if name not in ds:
            return None
        arr = to_nan(ds[name].values)
        if flipy:
            arr = arr[::-1, :]
        return arr

    topo = fetch("topo")
    glacier_mask = fetch("glacier_mask")
    dhdt = fetch("hugonnet_dhdt")
    vx = fetch("millan_vx")
    vy = fetch("millan_vy")
    thk_prior = fetch("millan_ice_thickness")

    assert topo is not None and glacier_mask is not None, "topo / glacier_mask missing"
    assert dhdt is not None, "hugonnet_dhdt missing — re-run fetch_oggm.py"

    icemask = (glacier_mask > 0.5).astype(np.float32)
    on_ice = icemask > 0.5
    n_ice = int(on_ice.sum())
    dx = abs(float(x[1] - x[0]))
    area_km2 = n_ice * dx * dx * 1e-6
    print(f"   grid: {len(x)} × {len(y)} @ dx={dx:.1f} m, "
          f"icemask area = {area_km2:.2f} km² ({n_ice} pixels)")

    # Sanity-check dhdt units. Hugonnet 2021 publishes dh/dt in m/yr; OGGM's
    # units attribute incorrectly says 'm'. We expect a slightly-negative
    # on-glacier mean for any Alpine glacier in 2000-2020.
    on_ice_dhdt = dhdt[on_ice]
    dhdt_mean = float(np.nanmean(on_ice_dhdt))
    dhdt_min, dhdt_max = float(np.nanmin(on_ice_dhdt)), float(np.nanmax(on_ice_dhdt))
    print(f"   on-glacier dhdt: mean={dhdt_mean:.3f}, "
          f"range [{dhdt_min:.3f}, {dhdt_max:.3f}] (assumed m yr⁻¹)")
    lo, hi = EXPECTED_DHDT_MEAN_RANGE
    if not (lo <= dhdt_mean <= hi):
        sys.exit(f"on-glacier dhdt mean {dhdt_mean:.3f} outside expected range "
                 f"{EXPECTED_DHDT_MEAN_RANGE} — units may not be m/yr; bailing.")

    # End-of-period DEM ← COP-DEM 30 (≈2010-2019), treated as the ~2020
    # anchor. Start-of-period ← copdem30 − dhdt × N_YEARS (derived, not OGGM
    # NASADEM). This inverts the prior pipeline, which anchored on NASADEM
    # 2000 and extrapolated forward.
    if not os.path.exists(copdem_tif):
        sys.exit(f"missing COP-DEM file: {copdem_tif}  "
                 f"(run fetch_copdem.py first)")
    with rasterio.open(copdem_tif) as src:
        copdem = src.read(1).astype(np.float32)
        if (src.width, src.height) != (len(x), len(y)):
            sys.exit(f"COP-DEM grid {src.width}×{src.height} doesn't match "
                     f"gridded_data {len(x)}×{len(y)} — re-run fetch_copdem.py.")
    # rasterio reads y-descending; flip to match our y-ascending internal arrays
    # (gridded_data was flipped above iff flipy=True).
    if flipy:
        copdem = copdem[::-1, :]

    diff_oi = copdem[on_ice] - topo[on_ice]
    print(f"   COP-DEM 30 vs OGGM NASADEM on-ice: mean Δh = "
          f"{float(np.nanmean(diff_oi)):.1f} m, "
          f"std = {float(np.nanstd(diff_oi)):.1f} m, "
          f"range [{float(np.nanmin(diff_oi)):.1f}, "
          f"{float(np.nanmax(diff_oi)):.1f}] m")

    usurfinfer = copdem.astype(np.float32)              # ≈ Jan 2020
    usurfobs   = (copdem - dhdt * N_YEARS).astype(np.float32)  # ≈ Jan 2000 (derived)
    usurf      = fill_nan_nearest(usurfobs)             # NaN-filled (emulator input)

    print(f"   2000→2020 surface change: dh range "
          f"[{float(np.nanmin(usurfinfer - usurfobs)):.1f}, "
          f"{float(np.nanmax(usurfinfer - usurfobs)):.1f}] m total")

    # Thickness prior — Millan22; replace NaN off-glacier with 0.
    if thk_prior is None:
        sys.exit("No millan_ice_thickness in OGGM output; thickness prior missing.")
    thkinit = np.where(np.isnan(thk_prior) | ~on_ice, 0.0, thk_prior).astype(np.float32)
    print(f"   thkinit (Millan22): on-glacier mean={thkinit[on_ice].mean():.1f} m, "
          f"max={thkinit.max():.1f} m")

    # Velocity observations — keep NaN where Millan has no data.
    if vx is None or vy is None:
        sys.exit("Millan22 velocity missing; cannot proceed.")
    n_vx_valid = int(np.isfinite(vx[on_ice]).sum())
    print(f"   Millan22 vx/vy: {n_vx_valid}/{n_ice} on-glacier pixels valid")

    # GlaThiDa thkobs (rasterized point cloud).
    thkobs = rasterize_glathida(glathida, x, y, icemask)

    # icemaskobs = icemask ∧ obs are all finite (matches Argentière pipeline).
    invalid_obs = (np.isnan(usurfobs) | np.isnan(usurfinfer)
                   | np.isnan(vx) | np.isnan(vy))
    icemaskobs = icemask.copy()
    icemaskobs[invalid_obs] = 0.0
    n_dropped = int(((icemask > 0.5) & invalid_obs).sum())
    print(f"   icemaskobs: dropped {n_dropped} on-glacier pixels with NaN obs "
          f"({int((icemaskobs > 0.5).sum())} valid pixels remain)")

    # Write the NetCDF.
    if os.path.exists(out_nc):
        shutil.copy2(out_nc, out_nc + ".bak")
        print(f"   backup: {out_nc}.bak")

    ny, nx = topo.shape
    with nc.Dataset(out_nc, "w", format="NETCDF4") as ncf:
        ncf.createDimension("x", nx)
        ncf.createDimension("y", ny)

        xv = ncf.createVariable("x", "f4", ("x",))
        xv.units = "m"; xv.long_name = "x"; xv.standard_name = "x"; xv.axis = "X"
        xv[:] = x
        yv = ncf.createVariable("y", "f4", ("y",))
        yv.units = "m"; yv.long_name = "y"; yv.standard_name = "y"; yv.axis = "Y"
        yv[:] = y

        def add(name, data, units, long_name):
            v = ncf.createVariable(name, "f4", ("y", "x"))
            v.units = units
            v.long_name = long_name
            v.standard_name = name
            v[:, :] = data

        add("usurf",       usurf,      "m",       "Surface Topography (NaN-filled start surface ≈ 2000, emulator input)")
        add("usurfobs",    usurfobs,   "m",       f"Observed Surface Topography (COP-DEM 30 − Hugonnet dhdt × {N_YEARS:.0f}, ≈ Jan 2000)")
        add("usurfinfer",  usurfinfer, "m",       "End-of-period Surface Topography (COP-DEM 30, ≈ Jan 2020 anchor)")
        add("thkinit",     thkinit,    "m",       "Ice Thickness prior (Millan 2022)")
        add("thk",         thkinit,    "m",       "Ice Thickness (DA initial state, copy of thkinit)")
        add("thkobs",      thkobs,     "m",       "Ice Thickness Observations (GlaThiDa, rasterized)")
        add("uvelsurfobs", vx,         "m/y",     "x surface velocity of ice (Millan 2022, ~2017-18)")
        add("vvelsurfobs", vy,         "m/y",     "y surface velocity of ice (Millan 2022, ~2017-18)")
        add("dhdt",        dhdt,       "m/y",     "Hugonnet 2021 mean dh/dt 2000-2020")
        add("icemask",     icemask,    "no unit", "Ice mask (OGGM glacier_mask)")
        add("icemaskobs",  icemaskobs, "no unit", "Observation validity mask (icemask AND obs non-NaN)")

        ncf.title  = f"{rgi_id} input — OGGM-sourced, IGM-ready"
        ncf.source = f"Built from {gridded} + {glathida}"
        ncf.history = (f"Created by data/oggm/build_input_nc.py; "
                       f"usurfinfer = COP-DEM 30 (≈2020 anchor); "
                       f"usurfobs   = usurfinfer − hugonnet_dhdt × {N_YEARS:.0f} yr")
        ncf.pyproj_srs = ds.attrs.get("pyproj_srs", "")

    print(f"\nWrote {out_nc}")
    ds.close()


if __name__ == "__main__":
    rgi_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RGI_ID
    main(rgi_id)
