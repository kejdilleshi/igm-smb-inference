#!/usr/bin/env python3
"""SMB-profile fans vs basal sliding. Adopted = best sRMSE among non-excluded.
Emits argentiere & aletsch single panels + 3-panel.

The tau values and their da_velsurf come from each glacier's
results/<g>/sens_slidingco/summary_<g>.csv, so this tracks whatever sweep is
currently on disk. Arrhenius is per-glacier (the adopted final run's value),
not the unified A=78 the earlier DA-at-t1 sweeps used."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
                            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                            "legend.fontsize": 7, "savefig.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "user", "code", "processes", "smb_inference"))
from smb_vs_continuity import prepare_cells, _interp_at, _rmse   # noqa: E402
from data.homogenize import draw_stake_overlay                   # noqa: E402
from _sens import da_velsurf, taus, excluded, vel_cap             # noqa: E402
RHO = 917.0

GLACIERS = ["argentiere", "aletsch_insitu", "rhone_insitu"]
VELSURF = {g: da_velsurf(g) for g in GLACIERS}
# Plot only the N best-fitting runs per glacier. The sweeps now extend into a
# high-velsurf arm (up to 140 on Aletsch) that exists for the correlation
# figure; those runs would swamp the fan without informing it. The adopted run
# is inside the top 4 on all three glaciers.
NTOP = 4
PLOT = {g: sorted(sorted(taus(g), key=lambda t: VELSURF[g][t])[:NTOP])
        for g in GLACIERS}
STK = {
    "argentiere":     dict(csv=f"{ROOT}/data/argentiere/SMB_argentiere_2012-2021_utm32N.csv", year=False),
    "aletsch_insitu": dict(csv=f"{ROOT}/data/aletsch_insitu/SMB_aletsch_insitu_2009-2017_rawstakes_utm32N.csv", year=True),
    "rhone_insitu":   dict(csv=f"{ROOT}/data/rhone_insitu/SMB_rhone_insitu_2007-2016_rawstakes_utm32N.csv", year=True),
}
TITLE = {"argentiere": "Argentière", "aletsch_insitu": "Great Aletsch", "rhone_insitu": "Rhône"}
_CELLS = {}


def profile(g, tau):
    run = f"{ROOT}/results/{g}/sens_slidingco/slid_{tau:.4f}/smb_inference"
    smb = np.load(f"{run}/smb_vec.npy").ravel() * (RHO / 1000.0)
    zmin = float(np.load(f"{run}/z_min.npy")); dz = float(np.load(f"{run}/dz.npy"))
    return zmin + dz * np.arange(smb.size), smb


def ela(z, s):
    k = np.where(np.diff(np.sign(s)) != 0)[0]
    if not len(k):
        return float(z[np.argmin(np.abs(s))])
    i = k[0]
    return float(z[i] - s[i] * (z[i + 1] - z[i]) / (s[i + 1] - s[i]))


def srmse(g, z, s):
    c = STK[g]
    if c["year"]:
        if g not in _CELLS:
            _CELLS[g] = prepare_cells(c["csv"], 100.0, 10, 4)[0]
        p = _CELLS[g]
        at = _interp_at(z, s, p["clu_alt"])
        return _rmse(at, p["clu_mean"], np.isfinite(at))
    df = pd.read_csv(c["csv"]); alt = df["Alt"].to_numpy(float); obs = df["Annual_SMB (m w.e.)"].to_numpy(float)
    q = np.interp(alt, z, s, left=np.nan, right=np.nan); m = np.isfinite(q)
    return float(np.sqrt(np.mean((q[m] - obs[m]) ** 2)))


def rows(g):
    out = []
    for tau in PLOT[g]:
        z, s = profile(g, tau)
        v = VELSURF[g][tau]
        out.append(dict(c=tau, z=z, s=s, vs=v, ela=ela(z, s), rmse=srmse(g, z, s),
                        excl=excluded(g, tau)))
    return out


def label_for(r):
    return f"$\\tau_{{\\rm ref}}$={r['c']:.3f}"


def stakes(ax, g):
    """Same stake rendering as fig_continuity_3panel: repeat-cell anchors with
    their spread where the record allows it, raw stakes otherwise."""
    c = STK[g]
    if c["year"]:
        if g not in _CELLS:
            _CELLS[g] = prepare_cells(c["csv"], 100.0, 10, 4)[0]
        draw_stake_overlay(ax, _CELLS[g])
    else:
        df = pd.read_csv(c["csv"])
        ax.scatter(df["Annual_SMB (m w.e.)"].to_numpy(float),
                   df["Alt"].to_numpy(float), s=22, marker="D",
                   color="tab:green", zorder=6, label="stakes")


def panel(ax, g):
    rr = rows(g)
    stakes(ax, g)
    inc = [r for r in rr if not r["excl"]]
    adopted = min(inc, key=lambda r: r["rmse"])
    band = [r["ela"] for r in inc]
    ax.axhspan(min(band), max(band), color="orange", alpha=0.12, zorder=0)
    ax.axvline(0, color="0.4", lw=0.8, ls=":", zorder=1)
    cs = PLOT[g]; norm = Normalize(min(cs), max(cs)); cmap = cm.viridis
    for r in rr:
        lab = label_for(r)
        if r["excl"]:
            ax.plot(r["s"], r["z"], "--", color="0.6", lw=1.2, label=lab, zorder=2)
        else:
            ax.plot(r["s"], r["z"], "-", color=cmap(norm(r["c"])), lw=1.5,
                    marker="o", ms=2.5, label=lab, zorder=4)
            ax.plot(0, r["ela"], "o", ms=5, mfc=cmap(norm(r["c"])), mec="k", mew=0.6, zorder=5)
    ax.set_xlabel("SMB (m w.e. yr$^{-1}$)"); ax.set_title(TITLE[g])
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    leg = ax.legend(loc="upper left", handletextpad=0.4)
    # De-emphasise excluded (high-vRMSE) runs in grey; mark the adopted run with
    # bold legend text (lines are left at uniform weight). Matched by label, not
    # by position — the stake overlay also contributes legend entries, so a
    # positional zip against rr would style the wrong rows.
    by_label = {label_for(r): r for r in rr}
    for txt in leg.get_texts():
        r = by_label.get(txt.get_text())
        if r is None:
            continue
        if r["excl"]:
            txt.set_color("0.6")
        elif r["c"] == adopted["c"]:
            txt.set_fontweight("bold")


def save_single(g, out):
    f, ax = plt.subplots(figsize=(3.4, 4.4)); panel(ax, g)
    ax.set_ylabel("Elevation (m a.s.l.)"); f.tight_layout()
    for ext in ("png", "pdf"):
        f.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(f)


save_single("argentiere", f"{ROOT}/results/argentiere/sens_slidingco/sens_slidingco_argentiere")
save_single("aletsch_insitu", f"{ROOT}/results/aletsch_insitu/sens_slidingco/sens_slidingco_aletsch")
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9))
for ax, g in zip(axes, ["argentiere", "aletsch_insitu", "rhone_insitu"]):
    panel(ax, g)
axes[0].set_ylabel("Elevation (m a.s.l.)")
fig.tight_layout()
fig.savefig(f"{ROOT}/results/sens_slidingco_3panel.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{ROOT}/results/sens_slidingco_3panel.pdf", bbox_inches="tight")
print("wrote sens_slidingco_aletsch, sens_slidingco_argentiere, sens_slidingco_3panel")
for g in PLOT:
    a = min((r for r in rows(g) if not r["excl"]), key=lambda r: r["rmse"])
    print(f"  {g:16s} adopted tau_ref={a['c']:.4f}  sRMSE={a['rmse']:.3f}  ELA={a['ela']:.0f}")
