#!/usr/bin/env python3
"""Illustrate b(x)=beta(z_s(x)): SMB profile -> surface elevation -> 2D SMB field.
Argentiere adopted run. 2-column (179 mm)."""
import os, sys
import numpy as np, netCDF4 as nc
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "savefig.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sens import adopted_tau, run_dir                                # noqa: E402

RUN = run_dir("argentiere", adopted_tau("argentiere"))
SI = f"{RUN}/smb_inference"
RHO = 0.917
prof = np.load(f"{SI}/smb_profile.npy") * RHO
z_min = float(np.load(f"{SI}/z_min.npy")); dz = float(np.load(f"{SI}/dz.npy"))
z = z_min + dz * np.arange(prof.size)
field = np.load(f"{SI}/smb_field.npy") * RHO

d = nc.Dataset(f"{RUN}/geology-optimized.nc")
x = d["x"][:] / 1000.0; y = d["y"][:] / 1000.0
mask = np.asarray(d["icemask"][:]) > 0.5
usurf = np.where(mask, np.asarray(d["usurf"][:]), np.nan)
d.close()
field = np.where(mask, field, np.nan)
ext = [x.min(), x.max(), y.min(), y.max()]
lim = np.nanmax(np.abs(field))

# Zoom to glacier: trim empty white margin around the icemask (matches fig_da_state)
def crop_limits(x, y, mask, pad=3):
    cx = np.where(np.any(mask, axis=0))[0]; ry = np.where(np.any(mask, axis=1))[0]
    xi = slice(max(cx[0] - pad, 0), min(cx[-1] + pad, len(x) - 1))
    yi = slice(max(ry[0] - pad, 0), min(ry[-1] + pad, len(y) - 1))
    return (float(x[xi.start]), float(x[xi.stop])), (float(y[yi.start]), float(y[yi.stop]))
xlim, ylim = crop_limits(x, y, mask)

fig, ax = plt.subplots(1, 3, figsize=(7.8, 3.0), gridspec_kw={"width_ratios":  [0.6, 1.4, 1.4]})
# (a) profile beta(z)
ax[0].plot(prof, z, "-o", color="k", ms=3, lw=1.6)
ax[0].axvline(0, color="0.5", lw=0.8, ls=":")
ax[0].set_xlabel(r"$smb(z)$ (m w.e. yr$^{-1}$)"); ax[0].set_ylabel("Elevation (m a.s.l.)")
ax[0].set_title(r"(a) 1-D profile $smb(z)$"); ax[0].grid(True, ls=":", lw=0.5, alpha=0.6)
# (b) surface elevation map
im1 = ax[1].imshow(usurf, cmap="terrain", extent=ext, origin="lower")
ax[1].set_title(r"(b) Surface elevation $z_s(\mathbf{x})$")
ax[1].set_xlabel("x (km)"); ax[1].set_ylabel("y (km)")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.03, label="m a.s.l.")
# (c) SMB field = beta(z_s). Keep red/white (ablation) at full range, cap blue
# (accumulation) at +2 so the narrow positive range [0, ~1.3] uses the full blue.
BLUE_CAP = 2.0
im2 = ax[2].imshow(field, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, BLUE_CAP),
                   extent=ext, origin="lower")
ax[2].set_title(r"(c) $smb$ field $smb(\mathbf{x})=smb(z_s(\mathbf{x}))$")
ax[2].set_xlabel("x (km)");ax[2].tick_params(axis='y', left=False, labelleft=False)
ax[2].set_ylabel("")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03, label=r"m w.e. yr$^{-1}$")
for a in (ax[1], ax[2]):
    a.xaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    a.set_xlim(*xlim); a.set_ylim(*ylim)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"{ROOT}/results/argentiere/fig_profile_to_field.{e}", bbox_inches="tight")
print("wrote fig_profile_to_field")
