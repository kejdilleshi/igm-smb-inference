#!/usr/bin/env python3
"""SMB-profile smoothness weight: L-curve + stake check (Rhone). 2-column.

Only four sweep points carry each panel, so the previous 7.4 x 3.0 in canvas
left most of the figure empty and shrank the type to ~8 pt once LaTeX scaled it
down to \linewidth (it also clipped the right-most point label). Drawn narrower
than the text block, LaTeX scales it up instead and the type grows with it;
styling follows fig_continuity_3panel, the paper's reference figure.

Reads results/rhone_insitu/smb_reg_sweep/reg_<lambda>/, the sweep run at the
adopted tau_ref / A / reg.thk of the Rhone run the paper reports, so the only
thing varying between points is lambda. Scored with the same stake reference as
paper_figures/_sens.stake_rmse, not the older score_smb_reg_sweep convention.
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.titlepad": 3.5,
                     "axes.labelpad": 2.0, "savefig.dpi": 300})
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.join(ROOT, "paper_figures"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "user", "code", "processes", "smb_inference"))

from _sens import STAKES, RHO                                    # noqa: E402
from smb_vs_continuity import prepare_cells, _interp_at, _rmse   # noqa: E402

SWEEP = os.path.join(ROOT, "results", "rhone_insitu", "smb_reg_sweep")
ADOPTED = 0.05

csv_path, _ = STAKES["rhone_insitu"]
prep, _ = prepare_cells(csv_path, 100.0, 10, 4)
alt, obs = prep["clu_alt"], prep["clu_mean"]

lam, misfit, rough, rmse = [], [], [], []
dirs = [d for d in glob.glob(os.path.join(SWEEP, "reg_*")) if os.path.isdir(d)]
for d in sorted(dirs, key=lambda p: float(re.sub(r"^reg_", "", os.path.basename(p)))):
    run = os.path.join(d, "smb_inference")
    if not os.path.exists(os.path.join(run, "smb_vec.npy")):
        continue
    lam.append(float(re.sub(r"^reg_", "", os.path.basename(d))))

    smb_ice = np.load(os.path.join(run, "smb_vec.npy")).ravel()
    curv = smb_ice[:-2] - 2 * smb_ice[1:-1] + smb_ice[2:]
    rough.append(float(np.sum(curv ** 2)))
    misfit.append(float(np.load(os.path.join(run, "data_history.npy")).ravel()[-1]))

    smb = smb_ice * (RHO / 1000.0)
    z0 = float(np.load(os.path.join(run, "z_min.npy")))
    dz = float(np.load(os.path.join(run, "dz.npy")))
    z = z0 + dz * np.arange(smb.size)
    at = _interp_at(z, smb, alt)
    rmse.append(float(_rmse(at, obs, np.isfinite(at))))

lam = np.array(lam); misfit = np.array(misfit)
rough = np.array(rough); rmse = np.array(rmse)
k = int(np.argmin(np.abs(lam - ADOPTED)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.35))

ax1.plot(rough, misfit, "-o", color="tab:blue", ms=4.5, lw=1.4, zorder=3)
ax1.plot(rough[k], misfit[k], "*", color="crimson", ms=13, zorder=4,
         label=f"adopted $\\lambda={ADOPTED:g}$")
# Label every point with its lambda. The right-most goes above rather than to
# the right, where "0.01" used to be clipped by the axes edge.
i_right = int(np.argmax(rough))
for i, (x, y, l) in enumerate(zip(rough, misfit, lam)):
    off, ha = ((0, 7), "center") if i == i_right else ((6, 2), "left")
    ax1.annotate(f"{l:g}", (x, y), textcoords="offset points", xytext=off,
                 ha=ha, fontsize=7.5, color="crimson" if i == k else "0.15")
# The full stencil is too wide for the narrower panel at label size, so this one
# label is set a point smaller rather than truncated to a bare "roughness".
ax1.set_xlabel("Profile roughness $\\sum_i (m_{i-1}-2m_i+m_{i+1})^2$", fontsize=8)
ax1.set_ylabel("Geometry misfit (m ice)")
ax1.set_title("(a) Misfit vs smoothness")
ax1.grid(True, ls=":", lw=0.5, alpha=0.6)
ax1.margins(x=0.12, y=0.16)
ax1.legend(loc="upper right", handlelength=1.2, handletextpad=0.4,
           borderpad=0.3, borderaxespad=0.3, framealpha=0.9)

ax2.axvspan(0.05, 0.1, color="0.88", zorder=0)
ax2.plot(lam, rmse, "-s", color="tab:orange", ms=4.5, lw=1.4, zorder=3)
ax2.plot(lam[k], rmse[k], "*", color="crimson", ms=13, zorder=4)
ax2.set_xscale("log")
ax2.set_xlabel("$\\lambda$")
ax2.set_ylabel("Stake RMSE (m w.e. yr$^{-1}$)")
ax2.set_title("(b) Stake RMSE vs $\\lambda$")
ax2.grid(True, ls=":", lw=0.5, alpha=0.6)
ax2.margins(x=0.12, y=0.16)

fig.tight_layout(pad=0.3, w_pad=1.0)
for e in ("pdf", "png"):
    out = os.path.join(ROOT, "SMB_inference_Abstract", "figs", f"fig_lambda_lcurve.{e}")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
