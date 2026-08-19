#!/usr/bin/env python3
"""Rhone: SMB inferred from Hugonnet (2021) dh/dt vs from the dedicated two-epoch
DEMs, both validated against the same stake averages. Appendix figure."""
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "savefig.dpi": 300})
import matplotlib.pyplot as plt
ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "user/code/processes/smb_inference"))
from compare_smb import load_profile
from smb_vs_continuity import prepare_cells, _interp_at, _rmse
from data.homogenize import draw_stake_overlay
RHO = 917.0

STK = f"{ROOT}/data/rhone_insitu/SMB_rhone_insitu_2007-2016_rawstakes_utm32N.csv"
prep, _ = prepare_cells(STK, 100.0, 10, 4)
runs = [
    dict(lab="dedicated DEMs", run=f"{ROOT}/results/rhone_insitu/sens_slidingco/slid_0.1991/smb_inference",
         c="k", ls="-"),
    dict(lab="Hugonnet dh/dt", run=f"{ROOT}/results/rhone/sens_slidingco/slid_0.1205/smb_inference",
         c="tab:blue", ls="-"),
]
fig, ax = plt.subplots(figsize=(3.7, 3.9))
draw_stake_overlay(ax, prep)
for r in runs:
    z, smb = load_profile(r["run"], RHO)
    at = _interp_at(z, smb, prep["clu_alt"])
    rmse = _rmse(at, prep["clu_mean"], np.isfinite(at))
    ax.plot(smb, z, r["ls"], color=r["c"], lw=1.9, marker="o", ms=2.5, zorder=5,
            label=f"{r['lab']} (RMSE {rmse:.2f})")
ax.axvline(0, color="0.5", lw=0.8, ls=":")
ax.set_xlabel("SMB (m w.e. yr$^{-1}$)"); ax.set_ylabel("Elevation (m a.s.l.)")
ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper left")
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"{ROOT}/results/rhone/fig_hugonnet_rhone.{e}", bbox_inches="tight")
print("wrote fig_hugonnet_rhone")
