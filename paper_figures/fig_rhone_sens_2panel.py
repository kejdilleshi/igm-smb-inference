#!/usr/bin/env python3
"""Rhone sliding-coefficient sensitivity fan for the TWO 2007-2016 experiments
side by side: (a) dedicated two-epoch swisstopo DEMs (results/rhone_insitu),
(b) Hugonnet-implied 2007 start surface on the SAME DA state/bed/2016 anchor
(results/rhone_hugonnet2007). The two experiments are identical except for the
source of the 2007 start-of-period surface (surveyed vs. Hugonnet dh/dt-
derived); see data/rhone/build_input_hugonnet2007.py. Data-driven (reads
costs.dat / smb_vec.npy per run + each experiment's own stake CSV, both n=5,
2007-2016) rather than hardcoded -- rerun after any sweep change. Appendix A2
figure.

"Adopted" is fixed to the SAME tau_ref as the main text (ADOPTED_TAU below,
0.1317 -- the ensemble-search value used throughout the paper for Rhone),
not a per-panel argmin of anything: the appendix's point is that the
velocity DA / ensemble search is left unchanged and only the DEM/dh-dt
source differs. A run is drawn dashed/excluded if its DA velsurf misfit
exceeds 1.5x the panel's own best (same cutoff convention as
pipeline/sensitivity_slidingco.py's auto-selection).

Both panels are restricted to the same five tau_ref (KEEP_TAUS): the
rhone_insitu sweep also has 0.2689 and 0.4592 on disk, but the comparison is
only legible if the two panels fan over an identical set of sliding
coefficients, and seven lines per panel crowd the legend without adding
information.
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 10,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "savefig.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from smb_vs_continuity import prepare_cells, _interp_at, _rmse  # noqa: E402

RHO = 917.0
ADOPTED_TAU = 0.1317
VEL_CUTOFF_RATIO = 1.5  # exclude runs with velsurf > best*ratio (matches
                         # pipeline/sensitivity_slidingco.select_slidingco)
# The tau_ref both panels fan over. Runs on disk outside this set are skipped.
KEEP_TAUS = (0.0978, 0.1317, 0.1991, 0.7442, 1.0779)

CFG = [
    dict(title="(a) Dedicated DEMs, 2007 surveyed",
         d=f"{ROOT}/results/rhone_insitu/sens_slidingco",
         stakes=f"{ROOT}/data/rhone_insitu/SMB_rhone_insitu_2007-2016_rawstakes_utm32N.csv"),
    dict(title="(b) Hugonnet dh/dt, 2007 implied",
         d=f"{ROOT}/results/rhone_hugonnet2007/sens_slidingco",
         stakes=f"{ROOT}/data/rhone_hugonnet2007/SMB_rhone_hugonnet2007_2007-2016_rawstakes_utm32N.csv"),
]


def discover_runs(sens_dir):
    """KEEP_TAUS runs with a completed smb_vec.npy -> sorted list of tau_ref.

    Missing ones are reported rather than dropped silently, so a panel that
    quietly loses a line to a half-finished sweep is visible on stderr.
    """
    taus, missing = [], []
    for tau in KEEP_TAUS:
        d = os.path.join(sens_dir, f"slid_{tau:.4f}")
        if os.path.exists(os.path.join(d, "smb_inference", "smb_vec.npy")):
            taus.append(tau)
        else:
            missing.append(tau)
    if missing:
        print(f"WARNING [{os.path.basename(os.path.dirname(sens_dir))}]: no finished "
              f"run for tau_ref {['%.4f' % t for t in missing]} -- not plotted",
              file=sys.stderr)
    return sorted(taus)


def profile(d, tau):
    r = f"{d}/slid_{tau:.4f}/smb_inference"
    smb = np.load(f"{r}/smb_vec.npy").ravel() * (RHO / 1000.0)
    zmin = float(np.load(f"{r}/z_min.npy")); dz = float(np.load(f"{r}/dz.npy"))
    return zmin + dz * np.arange(smb.size), smb


def da_velsurf(d, tau):
    p = f"{d}/slid_{tau:.4f}/costs.dat"
    with open(p) as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]
    hdr = lines[0].split()
    starts = [i for i, ln in enumerate(lines) if ln.split()[0] == hdr[0]]
    body = lines[starts[-1] + 1:]
    return float(body[-1].split()[hdr.index("velsurf")])


def ela(z, s):
    k = np.where(np.diff(np.sign(s)) != 0)[0]
    if not len(k):
        return float(z[np.argmin(np.abs(s))])
    i = k[0]
    return float(z[i] - s[i] * (z[i + 1] - z[i]) / (s[i + 1] - s[i]))


def panel(ax, c):
    taus = discover_runs(c["d"])
    prep, _ = prepare_cells(c["stakes"], 100.0, 10, 4)
    rows = []
    for tau in taus:
        z, s = profile(c["d"], tau)
        vs = da_velsurf(c["d"], tau)
        at = _interp_at(z, s, prep["clu_alt"])
        rows.append(dict(c=tau, z=z, s=s, vs=vs, ela=ela(z, s),
                         rmse=_rmse(at, prep["clu_mean"], np.isfinite(at))))
    best_vel = min(r["vs"] for r in rows)
    for r in rows:
        r["excl"] = r["vs"] > best_vel * VEL_CUTOFF_RATIO
    inc = [r for r in rows if not r["excl"]]

    ax.axhspan(min(r["ela"] for r in inc), max(r["ela"] for r in inc),
               color="orange", alpha=0.12, zorder=0)
    ax.axvline(0, color="0.5", lw=0.8, ls=":", zorder=1)
    norm = Normalize(min(taus), max(taus)); cmap = cm.viridis
    for r in rows:
        lab = f"$\\tau_{{\\rm ref}}$={r['c']:.3f}"
        is_adopted = abs(r["c"] - ADOPTED_TAU) < 1e-3
        if r["excl"]:
            ax.plot(r["s"], r["z"], "--", color="0.6", lw=1.2,
                    label=lab + f" (excl., vRMSE={r['vs']:.0f})", zorder=2)
        else:
            ax.plot(r["s"], r["z"], "-", color=cmap(norm(r["c"])),
                    lw=2.6 if is_adopted else 1.5, marker="o", ms=2.5, zorder=4,
                    label=lab + (f" (adopted, sRMSE {r['rmse']:.2f})" if is_adopted
                                  else f" (sRMSE {r['rmse']:.2f})"))
            ax.plot(0, r["ela"], "o", ms=5, mfc=cmap(norm(r["c"])), mec="k", mew=0.6, zorder=5)
    ax.errorbar(prep["clu_mean"], prep["clu_alt"], xerr=prep["clu_std"], fmt="D",
                color="tab:green", ecolor="tab:green", elinewidth=1.1, capsize=2.5,
                markersize=5, zorder=6, label=f"stake averages (n={prep['clu_alt'].size})")
    ax.set_xlabel("SMB (m w.e. yr$^{-1}$)"); ax.set_title(c["title"])
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper left", handletextpad=0.4)
    return rows


fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.0), sharex=True, sharey=True)
all_rows = {}
for ax, c in zip(axes, CFG):
    all_rows[c["title"]] = panel(ax, c)
axes[0].set_ylabel("Elevation (m a.s.l.)")
fig.tight_layout()
out_dir = f"{ROOT}/results/rhone_hugonnet2007"
for e in ("pdf", "png"):
    fig.savefig(f"{out_dir}/fig_rhone_sens_2panel.{e}", bbox_inches="tight")
print(f"wrote {out_dir}/fig_rhone_sens_2panel.{{pdf,png}}")

# Echo a plain-text table (source for Table A1 in the appendix).
for title, rows in all_rows.items():
    print(f"\n{title}")
    print(f"{'tau_ref':>8} {'velsurf':>8} {'stakeRMSE':>10} {'ELA':>6} {'excl':>5}")
    for r in sorted(rows, key=lambda r: r["c"]):
        print(f"{r['c']:>8.4f} {r['vs']:>8.2f} {r['rmse']:>10.3f} {r['ela']:>6.0f} {str(r['excl']):>5}")
