#!/usr/bin/env python3
"""
Bake a DA-optimized geometry into an IGM input NetCDF (pipeline step 3 prep).

Produces `input_<RGI>_daopt.nc`: the input grid cropped to the DA grid extent,
with `thk` and `usurf` overwritten by the chosen DA run's geology-optimized.nc.
All observation fields (usurfinfer, usurfobs, thkobs, vel obs, icemask) are kept
from the input. IGM's complete_data derives `topg = usurf - thk`, so baking
thk+usurf reproduces the exact post-DA bedrock (DA conserves topg).

This lets the SMB regularization L-curve (step 3.1) run SMB inference on a fixed
geometry without re-running expensive DA per regularization value.

`slidingco` is NOT baked: DA here keeps it uniform (control_list=[thk]), so the
smbonly config just sets iceflow.physics.init_slidingco to the chosen value.

Usage:
    python bake_da_geometry.py --input data/input_RGI60_11.02773.nc \
        --da-run results/findelen/da_lcurve/<best>/geology-optimized.nc \
        --out data/input_RGI60_11.02773_daopt.nc
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Source input grid NetCDF")
    ap.add_argument("--da-run", required=True,
                    help="geology-optimized.nc (or its dir) from the chosen DA run")
    ap.add_argument("--out", required=True, help="Output baked NetCDF")
    args = ap.parse_args()

    geo = args.da_run
    if os.path.isdir(geo):
        geo = os.path.join(geo, "geology-optimized.nc")
    if not os.path.exists(geo):
        sys.exit(f"missing DA output: {geo}")
    if not os.path.exists(args.input):
        sys.exit(f"missing input grid: {args.input}")

    inp = xr.open_dataset(args.input)
    opt = xr.open_dataset(geo)

    # Crop the input to the DA grid extent so the grids align exactly.
    x0, x1 = float(opt.x.min()), float(opt.x.max())
    y0, y1 = float(opt.y.min()), float(opt.y.max())
    sub = inp.sel(x=slice(x0, x1), y=slice(y0, y1))

    if sub.x.size != opt.x.size or sub.y.size != opt.y.size:
        sys.exit(
            f"grid mismatch after crop: input {sub.y.size}x{sub.x.size} vs "
            f"DA {opt.y.size}x{opt.x.size}. The input and DA gdir grids differ — "
            f"check that --input matches the experiment used for DA.")

    out = sub.copy(deep=True)
    out["thk"] = (("y", "x"), opt["thk"].values.astype(np.float32))
    out["usurf"] = (("y", "x"), opt["usurf"].values.astype(np.float32))
    out.attrs["history"] = (out.attrs.get("history", "")
                            + f" | DA geometry baked from {os.path.relpath(geo)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_netcdf(args.out)
    inp.close(); opt.close()

    on_ice = out["icemask"].values > 0.5 if "icemask" in out else np.ones_like(out["thk"].values, bool)
    print(f"Baked DA geometry -> {args.out}")
    print(f"  grid {out.y.size}x{out.x.size}, "
          f"on-ice thk mean={float(np.nanmean(out['thk'].values[on_ice])):.1f} m, "
          f"max={float(np.nanmax(out['thk'].values)):.1f} m")


if __name__ == "__main__":
    main()
