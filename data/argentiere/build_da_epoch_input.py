#!/usr/bin/env python3
"""
Rebuild data/input_argentiere.nc so the data assimilation runs on the END of the
SMB period (2021) instead of the start (2012), matching aletsch_insitu and
rhone_insitu.

Source is the existing input_argentiere.nc (IGE 2012/2021 DEMs, measured bed,
surface velocities). Only the surface-related fields are re-derived:

  usurf       — 2021 DEM, NaN-filled (emulator input; MUST be NaN-free)
  usurfobs    — 2021 DEM raw, NaNs preserved (DA reference)
  usurfinfer  — 2021 DEM raw (SMB-inference target; unchanged)
  usurfstart  — 2012 DEM, NaN-filled (SMB forward-run start geometry on the
                DA-fixed bed, via smb_inference.initial_state.usurf_variable)
  thkinit     — thickness prior at 2021 = thkinit(2012) + (usurf2021 - usurf2012)
  dhdt        — (2021 - 2012)/9 yr, NEW: enables the continuity diagnostics
                (plot_smb_results / smb_vs_continuity), which previously had no
                dhdt for this glacier
  icemaskobs  — icemask ∧ (all obs finite), recomputed for the new usurfobs

thkobs is carried through untouched and is NOT used by the pipeline (the DA is
velocity-only); it exists for validation only.

The previous file is backed up to input_argentiere.nc.bak_da2012.

Usage:
    python data/argentiere/build_da_epoch_input.py [--da-epoch end|start]
"""

import argparse
import os
import shutil

import numpy as np
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.normpath(os.path.join(HERE, "..", "input_argentiere.nc"))

START_YEAR = 2012
END_YEAR = 2021


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--da-epoch", choices=["start", "end"], default="end")
    args = ap.parse_args()
    da_epoch = args.da_epoch
    da_year = START_YEAR if da_epoch == "start" else END_YEAR

    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(NC_PATH)

    with nc.Dataset(NC_PATH) as ds:
        data = {v: to_nan(ds.variables[v][:]) for v in ds.variables
                if v not in ("x", "y")}
        x = np.asarray(ds.variables["x"][:], dtype=np.float32)
        y = np.asarray(ds.variables["y"][:], dtype=np.float32)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}

    # The file on disk is the DA@2012 layout: usurf/usurfobs = 2012 surface.
    # If it has already been converted (usurfstart present) refuse to re-derive
    # from the wrong baseline.
    if "usurfstart" in data:
        raise SystemExit(
            f"{NC_PATH} already has usurfstart — it was already converted. "
            f"Restore {NC_PATH}.bak_da2012 first if you need to rebuild.")

    surf_start = data["usurf"]          # 2012, already NaN-filled upstream
    surf_end = data["usurfinfer"]       # 2021, raw (has NaNs)
    thk_start = data["thkinit"]         # measured 2012 thickness prior

    surf_da = surf_start if da_epoch == "start" else surf_end

    # Thickness prior at the DA epoch, fixed bed: thk(t) = thk(2012) + Δsurface.
    # Only a starting point for the inversion — it is NOT a cost term.
    thkinit = np.clip(thk_start + (surf_da - surf_start), 0.0, None)
    thkinit = np.where(np.isfinite(thkinit), thkinit, 0.0).astype(np.float32)

    usurfobs = surf_da.astype(np.float32)
    usurf = fill_nan_nearest(usurfobs)
    assert not np.isnan(usurf).any(), "usurf (emulator input) must not contain NaN"
    usurfstart = fill_nan_nearest(surf_start.astype(np.float32))
    usurfinfer = surf_end.astype(np.float32)

    period_years = float(END_YEAR - START_YEAR)
    dhdt = ((surf_end - surf_start) / period_years).astype(np.float32)

    icemask = data["icemask"]
    vx, vy = data["uvelsurfobs"], data["vvelsurfobs"]
    invalid = (np.isnan(usurfobs) | np.isnan(vx) | np.isnan(vy) | np.isnan(usurfinfer))
    icemaskobs = icemask.copy()
    icemaskobs[invalid] = 0.0

    oi = icemask > 0.5
    n_ice = int(oi.sum())
    print(f"DA epoch = {da_year}")
    print(f"  thkinit({da_year}) on-ice mean {np.nanmean(thkinit[oi]):.1f} m "
          f"(was {np.nanmean(thk_start[oi]):.1f} m at {START_YEAR})")
    print(f"  dhdt on-ice mean {np.nanmean(dhdt[oi]):.3f} m/yr")
    print(f"  usurfstart-usurf on-ice mean {np.nanmean((usurfstart-usurf)[oi]):+.2f} m "
          f"(expect {-period_years*np.nanmean(dhdt[oi]):+.2f})")
    print(f"  icemaskobs {int((icemaskobs>0.5).sum())}/{n_ice} valid "
          f"({n_ice-int((icemaskobs>0.5).sum())} dropped)")

    shutil.copy2(NC_PATH, NC_PATH + ".bak_da2012")
    print(f"  backup -> {NC_PATH}.bak_da2012")

    out_fields = dict(data)
    out_fields.update(usurf=usurf, usurfobs=usurfobs, usurfinfer=usurfinfer,
                      thkinit=thkinit, icemaskobs=icemaskobs,
                      dhdt=dhdt)
    if da_epoch == "end":
        out_fields["usurfstart"] = usurfstart

    ny, nx = usurf.shape
    with nc.Dataset(NC_PATH, "w", format="NETCDF4") as ds:
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        xv = ds.createVariable("x", "f4", ("x",))
        xv.units = "m"; xv.long_name = "x"; xv.standard_name = "x"; xv.axis = "X"
        xv[:] = x
        yv = ds.createVariable("y", "f4", ("y",))
        yv.units = "m"; yv.long_name = "y"; yv.standard_name = "y"; yv.axis = "Y"
        yv[:] = y
        for name, arr in out_fields.items():
            v = ds.createVariable(name, "f4", ("y", "x"))
            v.standard_name = name
            v[:, :] = arr
        for k, val in attrs.items():
            ds.setncattr(k, val)
        ds.history = (attrs.get("history", "")
                      + f" | DA epoch retargeted to {da_year}; added usurfstart"
                        f"({START_YEAR}) + dhdt by build_da_epoch_input.py")

    print(f"Wrote {NC_PATH}  ({ny} × {nx})  vars={sorted(out_fields)}")


if __name__ == "__main__":
    main()
