#!/usr/bin/env python3
"""Argentiere DA state (adopted scalar run): calibrated thickness +
surface-velocity fit. 2-column (179 mm), publication fonts."""
import os, sys
import numpy as np, netCDF4 as nc
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "savefig.dpi": 300})
import matplotlib.pyplot as plt

ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sens import adopted_tau, run_dir                                # noqa: E402

TAU = adopted_tau("argentiere")
print(f"argentiere adopted tau_ref={TAU:.4f}")
G = os.path.join(run_dir("argentiere", TAU), "geology-optimized.nc")
d = nc.Dataset(G)
x = d["x"][:] / 1000.0; y = d["y"][:] / 1000.0
mask = np.asarray(d["icemask"][:]) > 0.5
thk = np.where(mask, np.asarray(d["thk"][:]), np.nan)
vobs = np.where(mask, np.asarray(d["velsurfobs_mag"][:]), np.nan)
vmod = np.where(mask, np.asarray(d["velsurf_mag"][:]), np.nan)
d.close()
ext = [x.min(), x.max(), y.min(), y.max()]
vmax = np.nanpercentile(np.concatenate([vobs[np.isfinite(vobs)], vmod[np.isfinite(vmod)]]), 99)

# Zoom to glacier: trim empty white margin around the icemask (pad in cells)
def crop_limits(x, y, mask, pad=3):
    cx = np.where(np.any(mask, axis=0))[0]; ry = np.where(np.any(mask, axis=1))[0]
    xi = slice(max(cx[0] - pad, 0), min(cx[-1] + pad, len(x) - 1))
    yi = slice(max(ry[0] - pad, 0), min(ry[-1] + pad, len(y) - 1))
    return (float(x[xi.start]), float(x[xi.stop])), (float(y[yi.start]), float(y[yi.stop]))
xlim, ylim = crop_limits(x, y, mask)

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
kw = dict(extent=ext, origin="lower", aspect="equal")
im0 = ax[0].imshow(thk, cmap="viridis", **kw)
ax[0].set_title(r"(a) Calibrated thickness $H_{t_2}$")
fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03, label="m")
im1 = ax[1].imshow(vobs, cmap="magma", vmin=0, vmax=vmax, **kw)
ax[1].set_title(r"(b) Observed speed $u_s^{\mathrm{obs}}$")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.03, label=r"m yr$^{-1}$")
im2 = ax[2].imshow(vmod, cmap="magma", vmin=0, vmax=vmax, **kw)
ax[2].set_title(r"(c) Modelled speed $u_s^{\mathrm{mod}}$")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03, label=r"m yr$^{-1}$")
for a in ax:
    a.set_xlabel("x (km)")
    a.set_xlim(*xlim); a.set_ylim(*ylim)
ax[0].set_ylabel("y (km)")

# Hide y-axis on panels (b) and (c)
for a in ax[1:]:
    a.tick_params(axis="y", left=False, labelleft=False)

fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"{ROOT}/results/argentiere/fig_da_state.{e}", bbox_inches="tight")
print("wrote fig_da_state")
