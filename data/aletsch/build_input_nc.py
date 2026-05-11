#!/usr/bin/env python3
"""
Build data/input_aletsch.nc from the pre-existing IGM aletsch tutorial inputs:

  - source geometry/velocity/thkobs : igm-examples/aletsch/data/input.nc
  - multi-epoch DEMs                : igm-examples/aletsch/data/past_surf.nc

The output NetCDF has the same variable layout as input_argentiere.nc so the
existing DA + SMB-inference pipeline (params_aletsch.yaml) can ingest it:

  usurf       — start-of-period DEM (NaN-filled), emulator input
  usurfobs    — start-of-period DEM (raw, NaNs preserved), DA reference
  usurfinfer  — end-of-period DEM, SMB-inference target
  thkinit     — start-of-period thickness prior, back-corrected by ΔDEM
  thk         — copy of thkinit (DA initial state)
  thkobs      — GPR thickness observations (from source input.nc)
  uvelsurfobs, vvelsurfobs — surface velocities (from source input.nc)
  icemask     — glacier outline (from source input.nc)
  icemaskobs  — icemask ∧ obs non-NaN (recomputed)

Period defaults to 2009 → 2017 (a stand-in for the requested 2009 → 2023; the
end DEM will be swapped to a real 2023 DEM later by re-running this script).
"""

import os
import shutil
import numpy as np
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_INPUT = "/home/klleshi/Documents/igm-examples/aletsch/data/input.nc"
SRC_PAST  = "/home/klleshi/Documents/igm-examples/aletsch/data/past_surf.nc"
OUT_NC    = os.path.normpath(os.path.join(HERE, "..", "input_aletsch.nc"))

START_SURF = "surf_2009"     # → usurf / usurfobs
END_SURF   = "surf_2017"     # → usurfinfer (stand-in for surf_2023)


def to_nan(a):
    if hasattr(a, "filled"):
        a = a.filled(np.nan)
    return np.asarray(a, dtype=np.float32)


def fill_nan_nearest(arr):
    arr = arr.astype(np.float32)
    m = np.isnan(arr)
    if not m.any():
        return arr
    _, (iy, ix) = distance_transform_edt(m, return_indices=True)
    out = arr.copy()
    out[m] = arr[iy[m], ix[m]]
    return out


def main():
    for p in (SRC_INPUT, SRC_PAST):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    if os.path.exists(OUT_NC):
        shutil.copy2(OUT_NC, OUT_NC + ".bak")
        print(f"Backup: {OUT_NC}.bak")

    with nc.Dataset(SRC_INPUT) as ds_in, nc.Dataset(SRC_PAST) as ds_past:
        x = np.asarray(ds_in.variables["x"][:], dtype=np.float32)
        y = np.asarray(ds_in.variables["y"][:], dtype=np.float32)

        usurf_modern = to_nan(ds_in.variables["usurf"][:])
        thk_modern   = to_nan(ds_in.variables["thk"][:])
        thkobs       = to_nan(ds_in.variables["thkobs"][:])
        vx           = to_nan(ds_in.variables["uvelsurfobs"][:])
        vy           = to_nan(ds_in.variables["vvelsurfobs"][:])
        icemask      = to_nan(ds_in.variables["icemask"][:])

        surf_start = to_nan(ds_past.variables[START_SURF][:])
        surf_end   = to_nan(ds_past.variables[END_SURF][:])

    # Back-correct thickness to the start epoch under fixed-bed assumption:
    # thk_start = thk_modern + (surf_start - usurf_modern), clipped >= 0.
    # surf_start/usurf_modern from past_surf agree off-glacier (same terrain),
    # so off-glacier thk stays 0.
    dh = surf_start - usurf_modern
    thkinit = np.clip(thk_modern + dh, 0.0, None).astype(np.float32)
    print(f"thkinit: dh range [{np.nanmin(dh):.1f}, {np.nanmax(dh):.1f}] m, "
          f"on-glacier mean thickness {thkinit[icemask > 0.5].mean():.1f} m")

    # usurfobs = raw start DEM; usurf = NaN-filled clean copy for the emulator.
    usurfobs = surf_start.astype(np.float32)
    usurf    = fill_nan_nearest(usurfobs)
    assert not np.isnan(usurf).any(), "usurf (emulator input) must not contain NaN"

    # usurfinfer = end-of-period DEM (NaNs preserved; the SMB-inference cost
    # gates on ~is_nan(usurfinfer) just like Argentière).
    usurfinfer = surf_end.astype(np.float32)

    # icemaskobs: gate DA costs to pixels where every observation is finite.
    invalid_obs = (np.isnan(usurfobs) | np.isnan(vx) | np.isnan(vy)
                   | np.isnan(usurfinfer))
    icemaskobs = icemask.copy()
    icemaskobs[invalid_obs] = 0.0
    n_dropped = int(((icemask > 0.5) & invalid_obs).sum())
    print(f"icemaskobs: dropped {n_dropped} on-glacier pixels with NaN obs "
          f"({int((icemaskobs > 0.5).sum())} valid pixels remain)")

    ny, nx = usurf.shape
    print(f"\nWriting {OUT_NC}  ({ny} × {nx}, dx={abs(x[1]-x[0]):.0f} m)")
    with nc.Dataset(OUT_NC, "w", format="NETCDF4") as ds:
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)

        xv = ds.createVariable("x", "f4", ("x",))
        xv.units = "m"; xv.long_name = "x"; xv.standard_name = "x"; xv.axis = "X"
        xv[:] = x
        yv = ds.createVariable("y", "f4", ("y",))
        yv.units = "m"; yv.long_name = "y"; yv.standard_name = "y"; yv.axis = "Y"
        yv[:] = y

        def add(name, data, units, long_name):
            v = ds.createVariable(name, "f4", ("y", "x"))
            v.units = units
            v.long_name = long_name
            v.standard_name = name
            v[:, :] = data

        add("usurf",       usurf,      "m",        f"Surface Topography (NaN-filled, {START_SURF}, emulator input)")
        add("usurfobs",    usurfobs,   "m",        f"Observed Surface Topography (raw {START_SURF}, NaNs preserved)")
        add("usurfinfer",  usurfinfer, "m",        f"End-of-period Surface Topography ({END_SURF}, SMB-inference target)")
        add("thkinit",     thkinit,    "m",        f"Ice Thickness prior at {START_SURF} epoch (modern thk + ΔDEM)")
        add("thk",         thkinit,    "m",        f"Ice Thickness (DA initial state, copy of thkinit)")
        add("thkobs",      thkobs,     "m",        "Ice Thickness Observations (GPR, from igm-examples/aletsch)")
        add("uvelsurfobs", vx,         "m/y",      "x surface velocity of ice")
        add("vvelsurfobs", vy,         "m/y",      "y surface velocity of ice")
        add("icemask",     icemask,    "no unit",  "Ice mask")
        add("icemaskobs",  icemaskobs, "no unit",  "Observation validity mask (icemask AND obs non-NaN)")

        ds.title  = "Grosser Aletschgletscher Input Data — 100 m resolution"
        ds.source = f"Built from {SRC_INPUT} and {SRC_PAST}"
        ds.history = (f"Created with build_input_nc.py; "
                      f"start={START_SURF}, end={END_SURF}")
        ds.crs = "EPSG:32632 (UTM zone 32N / WGS84)"

    print("Done.")
    print(f"\nNext: build_stake_csv.py to produce the SMB stake CSV, then run\n"
          f"  igm_run +experiment=params_aletsch")


if __name__ == "__main__":
    main()
