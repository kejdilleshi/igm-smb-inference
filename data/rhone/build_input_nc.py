#!/usr/bin/env python3
"""
Build the IN-HOUSE Rhone input NetCDF (swisstopo DEMs), analogous to
data/aletsch/build_input_nc.py.

Geometry/velocity/thkobs come from the OGGM-built Hugonnet input
(data/input_RGI60_11.01238.nc): Millan22 surface velocities, GlaThiDa thickness
prior, glacier mask — all on the OGGM local-TMerc grid (dx=66 m).

The surfaces come from the user-provided swisstopo DEMs in
data/rhone/surface_rhone_/ (UTM 32N GeoTIFFs), reprojected onto the OGGM grid:

  usurf       — DA-epoch DEM (NaN-filled), emulator input
  usurfobs    — DA-epoch DEM (raw, NaNs preserved), DA reference
  usurfinfer  — end-of-period DEM, SMB-inference target
  dhdt        — geodetic (surf_end − surf_start)/Δyears (the less-smooth,
                in-house alternative to Hugonnet dh/dt; used by diagnostics)
  thkinit/thk — DA-epoch thickness on the OGGM bed (surf_da − bed)
  thkobs      — GlaThiDa thickness obs (from the OGGM input)
  uvelsurfobs, vvelsurfobs — Millan22 surface velocities (from the OGGM input)
  icemask     — glacier outline (from the OGGM input)
  icemaskobs  — icemask ∧ obs non-NaN (recomputed)
  usurfstart  — start-of-period DEM (NaN-filled); written ONLY for
                --da-epoch end (see below)

Period: 2007 → 2016 (9 years), swisstopo surfaces — more reliable than the
Hugonnet-derived surface for this glacier.

``--da-epoch`` selects the surface the data assimilation is performed against:

  start (default) -> data/input_rhone_insitu.nc
      DA inverts the thickness on the 2007 surface; the SMB forward run starts
      from that same DA state. This is the original setup.

  end             -> data/input_rhone_2016.nc
      DA inverts the thickness on the 2016 surface — the epoch the Millan22
      velocities actually sample (~2015-2017) — so the bed it implies,
      topg = usurf(2016) − thk_DA, is constrained by contemporaneous data. The
      SMB forward run is then re-anchored to the 2007 surface on that fixed bed
      via ``processes.smb_inference.initial_state.usurf_variable=usurfstart``,
      and still scored against usurfinfer (the 2016 surface).

Examples:
    python data/rhone/build_input_nc.py                    # input_rhone_insitu.nc
    python data/rhone/build_input_nc.py --da-epoch end     # input_rhone_2016.nc
"""

import argparse
import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_OGGM = os.path.normpath(os.path.join(HERE, "..", "input_RGI60_11.01238.nc"))
DEM_DIR  = os.path.join(HERE, "surface_rhone_")

START_YEAR = 2007   # → usurfstart (and usurf/usurfobs when --da-epoch start)
END_YEAR   = 2016   # → usurfinfer (and usurf/usurfobs when --da-epoch end)


def to_nan(a):
    if hasattr(a, "filled"):
        a = a.filled(np.nan)
    a = np.asarray(a, dtype=np.float32)
    return np.where(a > 1e30, np.nan, a)


def fill_nan_nearest(arr):
    arr = arr.astype(np.float32)
    m = np.isnan(arr)
    if not m.any():
        return arr
    _, (iy, ix) = distance_transform_edt(m, return_indices=True)
    out = arr.copy()
    out[m] = arr[iy[m], ix[m]]
    return out


def reproject_dem(tif_path, x, y, dx, dst_crs):
    """Reproject a DEM GeoTIFF onto the OGGM grid (pixel-centred x, y ascending).

    rasterio writes top-down (row 0 = max y), so flip to y-ascending to match
    the OGGM input arrays. Areas outside the DEM footprint become NaN.
    """
    with rasterio.open(tif_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
    if src_nodata is not None:
        src_arr = np.where(src_arr == src_nodata, np.nan, src_arr)
    # defensive: drop implausible elevations (any fill values)
    src_arr = np.where((src_arr > 500.0) & (src_arr < 5000.0), src_arr, np.nan)

    x_left = float(x.min()) - dx / 2.0
    y_top = float(y.max()) + dx / 2.0
    dst_transform = rasterio.transform.from_origin(x_left, y_top, dx, dx)
    dst = np.full((len(y), len(x)), np.nan, dtype=np.float32)
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear, dst_nodata=np.nan,
    )
    return np.flipud(dst)  # top-down -> y-ascending


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=START_YEAR,
                    help=f"start of the SMB period (default {START_YEAR})")
    ap.add_argument("--end-year", type=int, default=END_YEAR,
                    help=f"end of the SMB period / inference target (default {END_YEAR})")
    ap.add_argument("--da-epoch", choices=["start", "end"], default="start",
                    help="surface the DA inverts against: 'start' (original "
                         "setup) or 'end' (velocity-contemporaneous surface)")
    ap.add_argument("--out", default=None,
                    help="output NetCDF (default: input_rhone_insitu.nc for "
                         "--da-epoch start, input_rhone_<end-year>.nc for end)")
    args = ap.parse_args()

    start_year, end_year, da_epoch = args.start_year, args.end_year, args.da_epoch
    da_year = start_year if da_epoch == "start" else end_year
    out_nc = args.out or os.path.normpath(os.path.join(
        HERE, "..",
        "input_rhone_insitu.nc" if da_epoch == "start" else f"input_rhone_{end_year}.nc"))

    if not os.path.exists(SRC_OGGM):
        raise FileNotFoundError(SRC_OGGM)
    dem_start = os.path.join(DEM_DIR, f"surface_rhone_{start_year}_UTM.tif")
    dem_end = os.path.join(DEM_DIR, f"surface_rhone_{end_year}_UTM.tif")
    for p in (dem_start, dem_end):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # OGGM grid + geometry/velocity source (force ascending x, y).
    ds = xr.open_dataset(SRC_OGGM).sortby("y").sortby("x")
    x = ds.x.values.astype(np.float64)
    y = ds.y.values.astype(np.float64)
    dx = float(abs(x[1] - x[0]))
    dst_crs = ds.attrs.get("pyproj_srs")
    if not dst_crs:
        raise ValueError(f"{SRC_OGGM} has no pyproj_srs attribute")
    print(f"OGGM grid: {len(y)} × {len(x)} @ dx={dx:.0f} m, CRS={dst_crs!r}")

    # Reproject the two swisstopo surfaces onto the OGGM grid.
    surf_start = reproject_dem(dem_start, x, y, dx, dst_crs)
    surf_end = reproject_dem(dem_end, x, y, dx, dst_crs)

    # Anchor the bed to usurfinfer = the OBSERVED COP-DEM (~2020) that the OGGM
    # thickness is consistent with. usurf/usurfobs are Hugonnet-extrapolated and
    # carry a spurious offset; the swisstopo surfaces agree with COP-DEM (swiss
    # 2016 ≈ usurfinfer), so this keeps surface and bed in one datum.
    usurf_modern = to_nan(ds["usurfinfer"].values)  # observed COP-DEM ~2020
    thk_oggm = to_nan(ds["thk"].values)             # OGGM thickness on COP-DEM
    thkobs = to_nan(ds["thkobs"].values)
    vx = to_nan(ds["uvelsurfobs"].values)
    vy = to_nan(ds["vvelsurfobs"].values)
    icemask = to_nan(ds["icemask"].values)

    # Bed (fixed) from the OGGM modern geometry, then thickness at the DA epoch.
    # The swisstopo DEM box is smaller than the OGGM grid, so the surfaces are
    # NaN outside it; zero those (off-DEM = off-glacier = no ice). The emulator
    # inputs (thk, usurf) MUST be NaN-free — any NaN poisons the whole velocity
    # field and the DA diverges to NaN at iter 0.
    bed = usurf_modern - thk_oggm
    surf_da = surf_start if da_epoch == "start" else surf_end
    thkinit = np.clip(surf_da - bed, 0.0, None)
    thkinit = np.where(np.isfinite(thkinit), thkinit, 0.0).astype(np.float32)

    period_years = float(end_year - start_year)
    dhdt = ((surf_end - surf_start) / period_years).astype(np.float32)

    usurfobs = surf_da.astype(np.float32)
    usurf = fill_nan_nearest(usurfobs)
    assert not np.isnan(usurf).any(), "usurf (emulator input) must not contain NaN"
    usurfinfer = surf_end.astype(np.float32)
    # Only needed when the DA epoch is the END of the period: the SMB forward
    # run is re-anchored to this surface on the DA bed (see module docstring).
    # NaN-filled because it feeds the emulator as thk = usurfstart − topg.
    usurfstart = fill_nan_nearest(surf_start.astype(np.float32))

    invalid_obs = (np.isnan(usurfobs) | np.isnan(vx) | np.isnan(vy)
                   | np.isnan(usurfinfer))
    icemaskobs = icemask.copy()
    icemaskobs[invalid_obs] = 0.0

    # ── sanity / coverage ────────────────────────────────────────────────────
    oi = icemask > 0.5
    n_ice = int(oi.sum())
    covered = oi & np.isfinite(surf_start) & np.isfinite(surf_end)
    print(f"DEM coverage of icemask: {int(covered.sum())}/{n_ice} pixels "
          f"({100*covered.sum()/max(n_ice,1):.0f}%)")
    print(f"thkinit({da_year}) on-ice mean {np.nanmean(thkinit[covered]):.1f} m "
          f"(clipped {int(((surf_da - bed) < 0)[covered].sum())} pixels < 0)")
    print(f"dhdt on-ice: mean {np.nanmean(dhdt[covered]):.3f} m/yr, "
          f"range [{np.nanmin(dhdt[covered]):.2f}, {np.nanmax(dhdt[covered]):.2f}]")
    print(f"surf_{da_year}−COP-DEM on-ice mean {np.nanmean((surf_da-usurf_modern)[covered]):.1f} m "
          f"(positive = thicker in {da_year} than modern)")
    print(f"icemaskobs: {int((icemaskobs>0.5).sum())} valid pixels "
          f"({n_ice-int((icemaskobs>0.5).sum())} dropped)")

    ny, nx = usurf.shape
    print(f"\nWriting {out_nc}  ({ny} × {nx})  [DA epoch = {da_year}]")
    with nc.Dataset(out_nc, "w", format="NETCDF4") as out:
        out.createDimension("x", nx)
        out.createDimension("y", ny)
        xv = out.createVariable("x", "f4", ("x",))
        xv.units = "m"; xv.long_name = "x"; xv.standard_name = "x"; xv.axis = "X"
        xv[:] = x.astype(np.float32)
        yv = out.createVariable("y", "f4", ("y",))
        yv.units = "m"; yv.long_name = "y"; yv.standard_name = "y"; yv.axis = "Y"
        yv[:] = y.astype(np.float32)

        def add(name, data, units, long_name):
            v = out.createVariable(name, "f4", ("y", "x"))
            v.units = units
            v.long_name = long_name
            v.standard_name = name
            v[:, :] = data

        add("usurf",       usurf,      "m",       f"Surface Topography (NaN-filled, swisstopo {da_year}, emulator input)")
        add("usurfobs",    usurfobs,   "m",       f"Observed Surface Topography (raw swisstopo {da_year}, DA reference)")
        add("usurfinfer",  usurfinfer, "m",       f"End-of-period Surface Topography (swisstopo {end_year}, SMB-inference target)")
        add("dhdt",        dhdt,       "m/y",     f"Geodetic dh/dt (swisstopo {end_year}−{start_year})/{int(period_years)} yr")
        add("thkinit",     thkinit,    "m",       f"Ice Thickness prior at {da_year} (OGGM bed + swisstopo surface)")
        add("thk",         thkinit,    "m",       "Ice Thickness (DA initial state, copy of thkinit)")
        add("thkobs",      thkobs,     "m",       "Ice Thickness Observations (GlaThiDa, from OGGM input)")
        add("uvelsurfobs", vx,         "m/y",     "x surface velocity of ice (Millan22, from OGGM input)")
        add("vvelsurfobs", vy,         "m/y",     "y surface velocity of ice (Millan22, from OGGM input)")
        add("icemask",     icemask,    "no unit", "Ice mask")
        add("icemaskobs",  icemaskobs, "no unit", "Observation validity mask (icemask AND obs non-NaN)")
        if da_epoch == "end":
            add("usurfstart", usurfstart, "m",
                f"Start-of-period Surface Topography (NaN-filled, swisstopo "
                f"{start_year}); SMB forward-run start geometry on the DA bed")

        out.pyproj_srs = dst_crs
        out.title = "Rhonegletscher Input Data — in-house swisstopo DEMs"
        out.source = (f"Surfaces: swisstopo {start_year}/{end_year} (UTM 32N, "
                      f"reprojected to OGGM grid). Velocity/thk/mask: {os.path.basename(SRC_OGGM)}")
        out.history = (f"Created with data/rhone/build_input_nc.py; "
                       f"start={start_year}, end={end_year}, da_epoch={da_year}")
        out.crs = "OGGM local TMerc (see pyproj_srs)"

    print("Done.")


if __name__ == "__main__":
    main()
