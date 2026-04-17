#!/usr/bin/env python3
"""
Patch data/input_argentiere.nc in place:

  - usurf       : overwritten with a NaN-filled (nearest-neighbour) copy, used
                  as the iceflow emulator input (no NaN allowed).
  - usurfobs    : added — raw DEM with NaNs preserved, DA observation reference.
  - icemaskobs  : added — icemask ∧ (obs are non-NaN). Gates DA cost terms so
                  NaN obs pixels never contribute to cost or regularization.

Leaves every other variable (usurfinfer, thkobs, thkinit, icemask, velsurfobs,
attrs, coords, ...) untouched.
"""

import os
import shutil
import numpy as np
import netCDF4 as nc
from scipy.ndimage import distance_transform_edt

NC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "input_argentiere.nc")
NC_PATH = os.path.normpath(NC_PATH)


def fill_nan_nearest(arr):
    arr = arr.astype(np.float32)
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    _, (iy, ix) = distance_transform_edt(nan_mask, return_indices=True)
    filled = arr.copy()
    filled[nan_mask] = arr[iy[nan_mask], ix[nan_mask]]
    return filled


def main():
    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(NC_PATH)

    backup = NC_PATH + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(NC_PATH, backup)
        print(f"Backup written: {backup}")

    with nc.Dataset(NC_PATH, "r+") as ds:
        vars_in = set(ds.variables.keys())
        print(f"Existing variables: {sorted(vars_in)}")

        usurf_raw = ds.variables["usurf"][:].astype(np.float32)
        icemask = ds.variables["icemask"][:].astype(np.float32)
        vx = ds.variables["uvelsurfobs"][:].astype(np.float32)
        vy = ds.variables["vvelsurfobs"][:].astype(np.float32)

        # Masked arrays -> ndarray with NaN where masked
        def to_nan(a):
            if hasattr(a, "filled"):
                a = a.filled(np.nan)
            return np.asarray(a, dtype=np.float32)
        usurf_raw = to_nan(usurf_raw)
        icemask = to_nan(icemask)
        vx = to_nan(vx)
        vy = to_nan(vy)

        n_nan_usurf = int(np.isnan(usurf_raw).sum())
        print(f"usurf NaN count: {n_nan_usurf}")

        # 1) usurfobs — raw DEM with NaN preserved (add if missing, else overwrite).
        if "usurfobs" not in vars_in:
            v = ds.createVariable("usurfobs", "f4", ("y", "x"))
            v.long_name = "Observed Surface Topography (raw DEM, NaNs preserved)"
            v.units = "m"
            v.standard_name = "usurfobs"
        ds.variables["usurfobs"][:, :] = usurf_raw

        # 2) usurf — NaN-filled clean copy (emulator input).
        usurf_clean = fill_nan_nearest(usurf_raw)
        assert not np.isnan(usurf_clean).any(), "usurf must not contain NaN"
        ds.variables["usurf"][:, :] = usurf_clean
        # Reflect the new role in the attribute.
        ds.variables["usurf"].long_name = \
            "Surface Topography (NaN-filled, emulator input)"

        # 3) icemaskobs — icemask ∧ non-NaN obs.
        invalid_obs = np.isnan(usurf_raw) | np.isnan(vx) | np.isnan(vy)
        icemaskobs = icemask.copy()
        icemaskobs[invalid_obs] = 0.0
        n_dropped = int(((icemask > 0.5) & invalid_obs).sum())
        print(f"icemaskobs: dropped {n_dropped} on-glacier pixels due to NaN obs "
              f"({int(np.sum(icemaskobs))} valid pixels remain)")

        if "icemaskobs" not in vars_in:
            v = ds.createVariable("icemaskobs", "f4", ("y", "x"))
            v.long_name = "Observation validity mask (icemask AND obs non-NaN)"
            v.units = "no unit"
            v.standard_name = "icemaskobs"
        ds.variables["icemaskobs"][:, :] = icemaskobs

        print(f"Final variables: {sorted(ds.variables.keys())}")

    print(f"Patched: {NC_PATH}")


if __name__ == "__main__":
    main()
