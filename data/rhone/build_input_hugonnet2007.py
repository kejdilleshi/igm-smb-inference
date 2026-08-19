#!/usr/bin/env python3
"""
Build the CORRECTED Hugonnet-anchored Rhone input NetCDF for Appendix A2.

Fixes a methodological mismatch in the first attempt (build_input_hugonnet_da2020.py,
now deleted): that version anchored DA on the ~2020 COP-DEM and derived a ~2006
start surface, a DIFFERENT 9/14-year window than the main text's dedicated-DEM
experiment (2007-2016), so the two were not a clean like-for-like comparison.

This version keeps EVERYTHING identical to the main-text experiment
(data/input_rhone_insitu.nc: DA anchored on the real swisstopo 2016 DEM, same
bed, same thickness/velocity/icemask fields, same 2007-2016 window) and
changes ONLY the source of the 2007 start-of-period surface:

  main text (input_rhone_insitu.nc):
      usurfstart = real swisstopo 2007 DEM (surveyed)

  this experiment (input_rhone_hugonnet2007.nc):
      usurfstart = usurf(2016) - hugonnet_dhdt * 9   (Hugonnet-implied 2007)

hugonnet_dhdt is the Hugonnet et al. (2021) mean dh/dt (2000-01-01 -> 2020-01-01,
m/yr) already present as "dhdt" in data/input_RGI60_11.01238.nc (fetched via
OGGM's hugonnet_maps shop task; same source used by data/oggm/build_input_nc.py
for the OGGM-pipeline glaciers). Both files share the exact same OGGM grid
(verified: identical x/y arrays), so no reprojection is needed -- just a
direct elementwise combination.

Everything else (usurf, usurfobs, usurfinfer, thk, thkinit, thkobs,
uvelsurfobs, vvelsurfobs, icemask, icemaskobs) is copied byte-for-byte from
input_rhone_insitu.nc, so the DA state is numerically IDENTICAL between the
two experiments for a given tau_ref -- only the smb_inference forward-run
start geometry differs.
"""
import os
import shutil

import netCDF4 as nc
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC_INSITU = os.path.join(PROJECT, "data", "input_rhone_insitu.nc")
SRC_OGGM = os.path.join(PROJECT, "data", "input_RGI60_11.01238.nc")
OUT_NC = os.path.join(PROJECT, "data", "input_rhone_hugonnet2007.nc")

START_YEAR = 2007
END_YEAR = 2016
N_YEARS = float(END_YEAR - START_YEAR)


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
    if not os.path.exists(SRC_INSITU):
        raise FileNotFoundError(SRC_INSITU)
    if not os.path.exists(SRC_OGGM):
        raise FileNotFoundError(SRC_OGGM)

    insitu = xr.open_dataset(SRC_INSITU)
    oggm = xr.open_dataset(SRC_OGGM).sortby("y").sortby("x")

    assert insitu["usurf"].shape == oggm["dhdt"].shape, "grid mismatch"
    assert np.array_equal(insitu.x.values, oggm.x.values.astype(insitu.x.dtype)), "x mismatch"
    assert np.array_equal(insitu.y.values, oggm.y.values.astype(insitu.y.dtype)), "y mismatch"

    usurf_2016 = insitu["usurf"].values.astype(np.float32)          # NaN-filled DA-epoch DEM
    usurfstart_real2007 = insitu["usurfstart"].values.astype(np.float32)  # surveyed, for comparison only
    icemask = insitu["icemask"].values

    dhdt_hugonnet = fill_nan_nearest(oggm["dhdt"].values.astype(np.float32))  # m/yr, 2000-2020 mean

    usurfstart_hugonnet = (usurf_2016 - dhdt_hugonnet * N_YEARS).astype(np.float32)
    assert not np.isnan(usurfstart_hugonnet).any(), "usurfstart (emulator input) must not contain NaN"

    oi = icemask > 0.5
    diff = (usurfstart_hugonnet - usurfstart_real2007)[oi]
    print(f"On-ice ({int(oi.sum())} px) 2007 surface, Hugonnet-implied vs surveyed swisstopo:")
    print(f"  mean diff = {np.nanmean(diff):+.2f} m,  RMS diff = {np.sqrt(np.nanmean(diff**2)):.2f} m")
    print(f"  Hugonnet on-ice mean dh/dt = {np.nanmean(dhdt_hugonnet[oi]):.3f} m/yr "
          f"(x {N_YEARS:.0f} yr = {np.nanmean(dhdt_hugonnet[oi]) * N_YEARS:+.2f} m)")

    shutil.copyfile(SRC_INSITU, OUT_NC)
    with nc.Dataset(OUT_NC, "a") as out:
        out.variables["usurfstart"][:, :] = usurfstart_hugonnet
        # Also replace "dhdt" (inherited byte-for-byte from input_rhone_insitu.nc,
        # i.e. real swisstopo (2016-2007)/9) with the Hugonnet rate actually used
        # to derive usurfstart above. "dhdt" doesn't feed the DA/SMB optimisation
        # (diagnostic-only, saved as dhdt_obs.npy for the stationary-flux
        # continuity comparison in smb_vs_continuity.py) but must reflect the
        # Hugonnet-only scenario this experiment represents, not leak the real
        # surveyed rate into that comparison.
        out.variables["dhdt"][:, :] = dhdt_hugonnet
        out.title = "Rhonegletscher Input Data — Hugonnet-anchored 2007 start (Appendix A2, corrected)"
        out.source = (
            "DA-epoch (2016) geometry/velocity/thk/mask: identical to input_rhone_insitu.nc "
            "(swisstopo 2016 DEM). Start-of-period (2007) surface: usurf(2016) - "
            f"hugonnet_dhdt * {N_YEARS:.0f}yr, hugonnet_dhdt from {os.path.basename(SRC_OGGM)} "
            "(Hugonnet et al. 2021, mean 2000-01-01 to 2020-01-01 rate). 'dhdt' field is "
            "this same hugonnet_dhdt (replaces the inherited real swisstopo rate; "
            "diagnostic-only, does not feed the DA/SMB optimisation)."
        )
        out.history = (
            f"Created with data/rhone/build_input_hugonnet2007.py from {os.path.basename(SRC_INSITU)} "
            f"+ {os.path.basename(SRC_OGGM)}; start={START_YEAR}, end={END_YEAR}"
        )

    print(f"\nWrote {OUT_NC}")


if __name__ == "__main__":
    main()
