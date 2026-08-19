#!/usr/bin/env python3
"""velsurf-RMSE vs stake-RMSE across the full sliding sweep (all 3 glaciers)."""
import glob, os, sys
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
                            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                            "legend.fontsize": 7.5, "savefig.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "user", "code", "processes", "smb_inference"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_smb import load_profile                                   # noqa: E402
from smb_vs_continuity import prepare_cells, _interp_at, _rmse          # noqa: E402
from _sens import da_velsurf                                           # noqa: E402

RHO = 917.0
GL = {
    "argentiere":     dict(name="Argentière", color="#d1495b",
                           stakes=f"{ROOT}/data/argentiere/SMB_argentiere_2012-2021_utm32N.csv", year=False),
    "aletsch_insitu": dict(name="Great Aletsch", color="#2c6fa8",
                           stakes=f"{ROOT}/data/aletsch_insitu/SMB_aletsch_insitu_2009-2017_rawstakes_utm32N.csv", year=True),
    "rhone_insitu":   dict(name="Rhône", color="#3a923a",
                           stakes=f"{ROOT}/data/rhone_insitu/SMB_rhone_insitu_2007-2016_rawstakes_utm32N.csv", year=True),
}


# driver-reported da_velsurf (Table 2 metric), read from each glacier's
# sens_slidingco/summary_<g>.csv so it always matches the runs on disk.
DA_VELSURF = {g: da_velsurf(g) for g in GL}


def velsurf_rms(slid_dir):
    g = slid_dir.split("/results/")[1].split("/")[0]
    tau = round(float(os.path.basename(slid_dir)[5:]), 4)
    return DA_VELSURF.get(g, {}).get(tau, np.nan)


def stake_rmse(run, cfg):
    z, smb = load_profile(run, RHO)
    if cfg["year"]:
        prep, _ = prepare_cells(cfg["stakes"], 100.0, 10, 4)
        if prep is None or prep["clu_alt"].size == 0:
            return np.nan
        at = _interp_at(z, smb, prep["clu_alt"])
        return _rmse(at, prep["clu_mean"], np.isfinite(at))
    df = pd.read_csv(cfg["stakes"])
    alt = df["Alt"].to_numpy(float); obs = df["Annual_SMB (m w.e.)"].to_numpy(float)
    p = np.interp(alt, z, smb, left=np.nan, right=np.nan); k = np.isfinite(p)
    return float(np.sqrt(np.mean((p[k] - obs[k]) ** 2)))


rec = []
for g, cfg in GL.items():
    for sd in sorted(glob.glob(f"{ROOT}/results/{g}/sens_slidingco/slid_*")):
        run = os.path.join(sd, "smb_inference")
        if not os.path.exists(os.path.join(run, "smb_vec.npy")):
            continue
        v = velsurf_rms(sd); s = stake_rmse(run, cfg)
        tau = float(os.path.basename(sd)[5:])
        if np.isfinite(v) and s is not None and np.isfinite(s):
            rec.append(dict(g=g, tau=tau, vel=v, stk=s))
            print(f"{cfg['name']:14s} tau={tau:.3f}  velsurf={v:7.2f}  stakeRMSE={s:.3f}")

df = pd.DataFrame(rec)
r_all, p_all = pearsonr(df["vel"], df["stk"])
rho_all, _ = spearmanr(df["vel"], df["stk"])
print(f"\nPooled: Pearson r={r_all:.2f} (p={p_all:.1e})  Spearman rho={rho_all:.2f}  n={len(df)}")
 
fig, ax = plt.subplots(figsize=(3.6, 3.2))
handles = []
for g, cfg in GL.items():
    d = df[df.g == g]
    ax.scatter(d.vel, d.stk, s=34, color=cfg["color"], edgecolor="k", lw=0.4,
               zorder=3)
    # Per-glacier least-squares trend, in that glacier's colour.
    r_g, p_g = pearsonr(d["vel"], d["stk"])
    rho_g, _ = spearmanr(d["vel"], d["stk"])
    b, a = np.polyfit(d["vel"], d["stk"], 1)
    xs = np.linspace(d["vel"].min(), d["vel"].max(), 50)
    ax.plot(xs, a + b * xs, ls="--", color=cfg["color"], lw=1.3, alpha=0.8, zorder=2)
    print(f"{cfg['name']:14s} n={len(d)}  Pearson r={r_g:+.2f} (p={p_g:.2f})  "
          f"Spearman rho={rho_g:+.2f}")
    # One legend entry per glacier showing both the points and their fit.
    handles.append(Line2D([0], [0], color=cfg["color"], ls="--", lw=1.3, alpha=0.8,
                          marker="o", ms=5, mec="k", mew=0.4,
                          label=f"{cfg['name']}: $r$={r_g:.2f}, $\\rho$={rho_g:.2f}"))
ax.set_xlabel("velsurf RMSE (m yr$^{-1}$)")
ax.set_ylabel("Stake RMSE (m w.e. yr$^{-1}$)")
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(handles=handles, loc="lower right", handletextpad=0.4, borderpad=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{ROOT}/results/velsurf_vs_stake_rmse.{ext}", bbox_inches="tight")
print("\nwrote results/velsurf_vs_stake_rmse.{png,pdf}")
