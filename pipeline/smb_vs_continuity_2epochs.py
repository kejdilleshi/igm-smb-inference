#!/usr/bin/env python3
"""smb_vs_continuity with BOTH epochs' flux divergence.

Same figure as pipeline/smb_vs_continuity.py but overlays a second continuity
curve built from the 2021-surface DA:

  * black ●    inferred SMB profile (unchanged — the SMB we already inferred)
  * grey/green raw stakes + repeat-cell anchors (unchanged)
  * blue  --   dh/dt + div(flux)_2012   (the continuity we currently have)
  * red   --   dh/dt + div(flux)_2021   (new: div(flux) from the 2021-DEM DA)

dh/dt (observed 2012->2021 thinning) is the same in both; only div(flux) and the
surface used to bin it differ between epochs.
"""
import argparse, os, sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "pipeline"))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "user", "code", "processes", "smb_inference"))

from smb_vs_continuity import continuity_profile, prepare_cells, _interp_at, _rmse  # noqa
from compare_smb import load_profile  # noqa
from plot_smb_results import _bin_by_elevation  # noqa
from data.homogenize import draw_stake_overlay  # noqa


def continuity_2021(run2012_dir, da2021_nc, z_min, dz, n, rho):
    """dh/dt(2012->2021) + div(flux) from the 2021-surface DA, binned by the
    2021 surface, in m w.e./yr."""
    dhdt = np.load(os.path.join(run2012_dir, "dhdt_obs.npy"))
    ds = xr.open_dataset(da2021_nc)
    div = ds["divflux"].values
    usurf = ds["usurf"].values
    icemask = ds["icemask"].values if "icemask" in ds else None
    mask = (icemask > 0.5) if icemask is not None else np.isfinite(usurf)
    mask = mask & np.isfinite(div) & np.isfinite(dhdt)
    dhdt_p = _bin_by_elevation(dhdt, usurf, mask, float(z_min), float(dz), n)
    div_p = _bin_by_elevation(div, usurf, mask, float(z_min), float(dz), n)
    return (dhdt_p + div_p) * (rho / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/argentiere/sens_slidingco/slid_0.2689/smb_inference",
                    help="2012 smb_inference dir (inferred SMB, dhdt, 2012 div via ../geology-optimized.nc)")
    ap.add_argument("--da2021-nc", default="results/argentiere/epoch2021/da_2021_slid0.2689/geology-optimized.nc")
    ap.add_argument("--stakes", default="data/argentiere/SMB_argentiere_2012-2021_utm32N.csv")
    ap.add_argument("--out", default="results/argentiere/epoch2021/smb_vs_continuity_2epochs.png")
    ap.add_argument("--rho-ice", type=float, default=917.0)
    ap.add_argument("--title", default="Argentiere — continuity at 2012 vs 2021 surface")
    args = ap.parse_args()

    A = lambda p: p if os.path.isabs(p) else os.path.join(PROJECT_DIR, p)
    run, da2021, stakes, out = A(args.run), A(args.da2021_nc), A(args.stakes), A(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    z, smb_we = load_profile(run, args.rho_ice)
    n = z.shape[0]
    z_min, dz = float(z[0]), float(z[1] - z[0]) if n > 1 else 200.0

    cont12, _ = continuity_profile(run, z_min, dz, n, args.rho_ice)   # 2012 div(flux)
    cont21 = continuity_2021(run, da2021, z_min, dz, n, args.rho_ice)  # 2021 div(flux)

    # optional stake overlay (argentiere CSV: X,Y,Alt,Annual_SMB)
    prep = None
    try:
        prep, _ = prepare_cells(stakes, 100.0, 10, 4)
    except Exception as e:
        print(f"[2epochs] stake overlay unavailable ({e}); scattering raw stakes")

    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.plot(smb_we, z, "-o", color="k", markersize=3, label="inferred SMB (our method)")
    if prep is not None:
        draw_stake_overlay(ax, prep)
    else:
        import pandas as pd
        df = pd.read_csv(stakes)
        ax.scatter(df.iloc[:, 3], df["Alt"], s=10, color="grey", alpha=0.5, label="stakes")
    ax.plot(cont12, z, "--", color="royalblue", lw=2.0, zorder=4,
            label=r"dh/dt + $\nabla\!\cdot$flux (2012 surface)")
    ax.plot(cont21, z, "--", color="firebrick", lw=2.0, zorder=4,
            label=r"dh/dt + $\nabla\!\cdot$flux (2021 surface)")

    # RMSE of each curve vs repeat-cell means (same cells for all)
    if prep is not None and prep["clu_alt"].size:
        clu_alt, clu_mean = prep["clu_alt"], prep["clu_mean"]
        inf_at = _interp_at(z, smb_we, clu_alt)
        c12_at = _interp_at(z, cont12, clu_alt)
        c21_at = _interp_at(z, cont21, clu_alt)
        common = np.isfinite(inf_at) & np.isfinite(c12_at) & np.isfinite(c21_at)
        nc = int(common.sum())
        for nm, arr in (("inferred", inf_at), ("2012 cont.", c12_at), ("2021 cont.", c21_at)):
            r = _rmse(arr, clu_mean, common)
            if r is not None:
                ax.plot([], [], " ", label=f"RMSE {nm} = {r:.3f} ({nc} cells)")

    ax.axvline(0.0, color="k", lw=0.5)
    ax.set_xlabel("SMB (m w.e./yr)"); ax.set_ylabel("Elevation (m a.s.l.)")
    ax.set_title(args.title, fontsize=11)
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[2epochs] wrote {out}")

    # headline: how much the continuity estimate shifts between epochs
    d = cont21 - cont12
    fin = np.isfinite(d)
    print(f"[2epochs] continuity(2021)-continuity(2012): mean {np.nanmean(d[fin]):+.3f}  "
          f"RMS {np.sqrt(np.nanmean(d[fin]**2)):.3f}  max|Δ| {np.nanmax(np.abs(d[fin])):.3f} m w.e./yr")


if __name__ == "__main__":
    main()
