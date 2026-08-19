#!/usr/bin/env python3
"""DA smoothness weight: the Tikhonov L-curve, shown for Great Aletsch. 1-column.

Replaces figs/l-curve_aletsch.png, which was a raster crop of the left half of
pipeline/analyze_da_lcurve.py's diagnostic `lcurve_step1.png` -- default
matplotlib styling at ~6 pt effective type once scaled to 0.5\\linewidth.

Data-driven: reads the `da_lcurve_result.json` that analyze_da_lcurve.py writes
alongside that diagnostic, so the figure cannot drift from the sweep that
actually selected the weight. Styling follows fig_continuity_3panel (the paper's
reference figure); the canvas is drawn narrower than the 0.5\\linewidth slot it
occupies so LaTeX scales it up and enlarges the type with it.
"""
import json
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.titlepad": 3.5,
                     "axes.labelpad": 2.0, "savefig.dpi": 300})
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/klleshi/Documents/igm-smb-inference"
SRC = os.path.join(ROOT, "results", "aletsch_insitu", "da_lcurve",
                   "da_lcurve_result.json")

with open(SRC) as f:
    d = json.load(f)
reg = np.array(d["regs"], float)
vel = np.array(d["vel_rmse"], float)
rough = np.array(d["grad_norm"], float)
best = float(d["best_reg"])
k = int(np.argmin(np.abs(reg - best)))
print(f"{len(reg)} assimilations, corner at reg.thk={best:g} "
      f"(vel_rmse={vel[k]:.2f} m/yr, roughness={rough[k]:.1f})")

fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.plot(rough, vel, "-o", color="tab:blue", ms=4.5, lw=1.4, zorder=3)
ax.plot(rough[k], vel[k], "*", color="crimson", ms=14, zorder=4,
        label=f"corner: {best:g}")

# Label every point with its Tikhonov weight. The right-most point is labelled
# above instead of to the right, where it used to run off the axes.
i_right = int(np.argmax(rough))
for i, r in enumerate(reg):
    off, ha = ((0, 7), "center") if i == i_right else ((6, 2), "left")
    ax.annotate(f"{r:g}", (rough[i], vel[i]), textcoords="offset points",
                xytext=off, ha=ha, fontsize=7.5,
                color="crimson" if i == k else "0.15")

ax.set_xlabel("Thickness roughness (m per grid cell)")
ax.set_ylabel("Velocity RMSE (m yr$^{-1}$)")
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
# Room for the annotations, which tight axes limits would otherwise clip.
ax.margins(x=0.11, y=0.13)
ax.legend(loc="upper right", handlelength=1.2, handletextpad=0.4,
          borderpad=0.3, borderaxespad=0.3, framealpha=0.9)
fig.tight_layout(pad=0.3)
for e in ("pdf", "png"):
    out = os.path.join(ROOT, "SMB_inference_Abstract", "figs", f"fig_da_lcurve.{e}")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
