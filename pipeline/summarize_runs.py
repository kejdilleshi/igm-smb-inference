#!/usr/bin/env python3
"""
One-line-per-glacier summary of finished pipeline runs — the paper table.

For each glacier it reports what the run ACTUALLY used and what it produced:

  params      init_slidingco / init_arrhenius / reg.thk, read from
              results/<g>/final/.hydra/overrides.yaml
  velRMSE     surface-velocity misfit of the DA (the DA's own data term)
  meanH/vol   inverted geometry
  smb         inferred profile: value in the lowest bin, and the ELA
  stakeRMSE   inferred profile vs the stake observations, in m w.e./yr —
              vs repeat-cell anchors when the CSV has a Year column
              (aletsch/rhone), else vs the raw stakes (argentiere)
  contRMSE    same, for the dh/dt + div(flux) continuity estimate
  thkobs      DIAGNOSTIC ONLY — bias vs measured radar thickness. Never used
              by the pipeline; the DA is velocity-only for every glacier.

`.hydra/overrides.yaml` is the source of truth for parameters, NOT
pipeline_summary.json — the latter could be rewritten by a --dry-run that
executed nothing (fixed in run_pipeline.py, but old files may still be stale).

Usage:
    python pipeline/summarize_runs.py                       # all known runs
    python pipeline/summarize_runs.py --glaciers rhone_insitu argentiere
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "pipeline"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "user", "code", "processes", "smb_inference"))

from data.homogenize import prepare_stake_overlay  # noqa: E402
from smb_vs_continuity import continuity_profile, prepare_cells, _interp_at, _rmse  # noqa: E402

DEFAULT_GLACIERS = ["aletsch_insitu", "rhone_insitu", "argentiere"]
# glacier -> (input nc, stakes csv) ; stakes may be None
DATA = {
    "aletsch_insitu": ("data/input_aletsch.nc",
                       "data/aletsch/SMB_aletsch_2009-2017_rawstakes_utm32N.csv"),
    "rhone_insitu":   ("data/input_rhone_insitu.nc",
                       "data/rhone/SMB_rhone_2007-2016_rawstakes_utm32N.csv"),
    "argentiere":     ("data/input_argentiere.nc",
                       "data/argentiere/SMB_argentiere_2012-2021_utm32N.csv"),
}


def read_overrides(final_dir):
    """Actual run parameters from hydra's override record."""
    p = os.path.join(final_dir, ".hydra", "overrides.yaml")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for item in yaml.safe_load(f) or []:
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            out[k.split(".")[-1]] = v
    return out


def profile(run_dir, rho_ice=917.0):
    v = np.load(os.path.join(run_dir, "smb_vec.npy")) * (rho_ice / 1000.0)
    z0 = float(np.load(os.path.join(run_dir, "z_min.npy")))
    dz = float(np.load(os.path.join(run_dir, "dz.npy")))
    return z0 + dz * np.arange(v.size), v


def ela_of(z, we):
    s = np.where(np.diff(np.sign(we)))[0]
    if not s.size:
        return float("nan")
    i = s[0]
    return z[i] + (z[i + 1] - z[i]) * (0 - we[i]) / (we[i + 1] - we[i])


def stake_rmse(run_dir, stakes, z, we, rho_ice=917.0):
    """(inferred RMSE, continuity RMSE, n, mode) against the stake data."""
    if not stakes or not os.path.exists(stakes):
        return None, None, 0, "none"
    prep, _ = prepare_cells(stakes, 100.0, 10, 4)
    if prep is None:
        return None, None, 0, "none"
    cont, _ = continuity_profile(run_dir, z[0], z[1] - z[0], z.size, rho_ice)
    if prep["clu_alt"].size >= 2:                       # repeat-cell anchors
        alt, obs, mode = prep["clu_alt"], prep["clu_mean"], "cells"
    else:                                               # single-snapshot stakes
        alt, obs, mode = prep["raw_z"], prep["raw_b"], "raw"
    inf_at = _interp_at(z, we, alt)
    cont_at = _interp_at(z, cont, alt) if cont is not None else np.full(alt.shape, np.nan)
    common = np.isfinite(inf_at)
    if cont is not None:
        common &= np.isfinite(cont_at)
    return (_rmse(inf_at, obs, common),
            _rmse(cont_at, obs, common) if cont is not None else None,
            int(common.sum()), mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glaciers", nargs="*", default=DEFAULT_GLACIERS)
    ap.add_argument("--results", default=os.path.join(PROJECT_DIR, "results"))
    args = ap.parse_args()

    hdr = (f"{'glacier':16s} {'slid':>6s} {'A':>6s} {'reg':>6s} {'velRMSE':>7s} "
           f"{'meanH':>6s} {'vol':>6s} {'smb_low':>7s} {'ELA':>5s} "
           f"{'stakeR':>6s} {'contR':>6s} {'n':>3s} {'thkobs bias':>11s}")
    print(hdr); print("-" * len(hdr))

    for g in args.glaciers:
        final = os.path.join(args.results, g, "final")
        run = os.path.join(final, "smb_inference")
        if not os.path.exists(os.path.join(run, "smb_vec.npy")):
            print(f"{g:16s} (no finished run)"); continue
        ov = read_overrides(final)
        inp_rel, stakes_rel = DATA.get(g, (None, None))
        inp = os.path.join(PROJECT_DIR, inp_rel) if inp_rel else None
        stakes = os.path.join(PROJECT_DIR, stakes_rel) if stakes_rel else None

        o = xr.open_dataset(os.path.join(final, "geology-optimized.nc"))
        m = o["icemask"].values > 0.5
        thk = o["thk"].values
        vm, vo = o["velsurf_mag"].values, o["velsurfobs_mag"].values
        vv = m & np.isfinite(vo)
        dx = float(o.x.values[1] - o.x.values[0])
        velr = float(np.sqrt(np.nanmean((vm[vv] - vo[vv]) ** 2)))

        bias = np.nan
        if inp and os.path.exists(inp):
            di = xr.open_dataset(inp).sortby("y").sortby("x").sel(
                x=slice(float(o.x.min()), float(o.x.max())),
                y=slice(float(o.y.min()), float(o.y.max())))
            if "thkobs" in di and di["thkobs"].shape == thk.shape:
                t = di["thkobs"].values
                sel = np.isfinite(t) & (t > 0) & m
                if sel.any():
                    bias = float((thk[sel] - t[sel]).mean())

        z, we = profile(run)
        sr, cr, n, mode = stake_rmse(run, stakes, z, we)
        f = lambda v, s="{:6.3f}": "   n/a" if v is None or not np.isfinite(v) else s.format(v)
        print(f"{g:16s} {float(ov.get('init_slidingco', 'nan')):6.3f} "
              f"{float(ov.get('init_arrhenius', 'nan')):6.1f} "
              f"{float(ov.get('thk', 'nan')):6.0f} {velr:7.2f} "
              f"{thk[m].mean():6.1f} {thk[m].sum()*dx*dx/1e9:6.3f} "
              f"{we[0]:7.2f} {ela_of(z, we):5.0f} "
              f"{f(sr)} {f(cr)} {n:3d}{'*' if mode=='raw' else ' '} "
              f"{bias:+10.1f}")
    print("\n* RMSE vs raw stakes (no Year column); others vs repeat-cell anchors."
          "\n  thkobs bias is DIAGNOSTIC ONLY — never used by the velocity-only DA.")


if __name__ == "__main__":
    main()
