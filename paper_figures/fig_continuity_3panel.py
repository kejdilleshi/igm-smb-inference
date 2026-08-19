#!/usr/bin/env python3
"""Merged 3-panel: inferred SMB vs stakes vs stationary-flux continuity, for the
three adopted runs. Replaces the two separate continuity figures. 2-column."""
import os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "savefig.dpi": 300})
import matplotlib.pyplot as plt
ROOT = "/home/klleshi/Documents/igm-smb-inference"
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "user/code/processes/smb_inference"))
from compare_smb import load_profile
from smb_vs_continuity import prepare_cells, continuity_profile, _interp_at, _rmse
from data.homogenize import draw_stake_overlay
RHO = 917.0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sens import STAKES, adopted_tau, run_dir                        # noqa: E402

# The adopted run per glacier is derived, not hardcoded, so this panel always
# shows the same runs as the sensitivity fans.
G = []
for label, g in [("(a) Argentière", "argentiere"),
                 ("(b) Great Aletsch", "aletsch_insitu"),
                 ("(c) Rhône", "rhone_insitu")]:
    tau = adopted_tau(g)
    csv_path, has_year = STAKES[g]
    print(f"{label}: adopted tau_ref={tau:.4f}")
    G.append(dict(name=label, run=os.path.join(run_dir(g, tau), "smb_inference"),
                  stakes=csv_path, year=has_year))

fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=False)
for ax, g in zip(axes, G):
    z, smb = load_profile(g["run"], RHO)
    n = z.size; zmin = float(z[0]); dz = float(z[1] - z[0])
    cont, _ = continuity_profile(g["run"], zmin, dz, n, RHO)
    if g["year"]:
        prep, _ = prepare_cells(g["stakes"], 100.0, 10, 4)
        draw_stake_overlay(ax, prep)
        alt, obs = prep["clu_alt"], prep["clu_mean"]
    else:
        df = pd.read_csv(g["stakes"]); alt = df["Alt"].to_numpy(float); obs = df["Annual_SMB (m w.e.)"].to_numpy(float)
        ax.scatter(obs, alt, s=26, marker="D", color="tab:green", zorder=6, label="stake averages")
    ri = _rmse(_interp_at(z, smb, alt), obs, np.isfinite(_interp_at(z, smb, alt)))
    rc = _rmse(_interp_at(z, cont, alt), obs, np.isfinite(_interp_at(z, cont, alt))) if cont is not None else None
    ax.plot(smb, z, "-", color="k", lw=1.8, marker="o", ms=2.5, zorder=5,
            label=f"inferred (RMSE {ri:.2f})")
    if cont is not None:
        ax.plot(cont, z, "--", color="firebrick", lw=1.6, zorder=4,
                label=f"$\\mathrm{{d}}h/\\mathrm{{d}}t+\\nabla\\!\\cdot\\mathbf{{q}}$ (RMSE {rc:.2f})")
    ax.axvline(0, color="0.5", lw=0.8, ls=":")
    ax.set_title(g["name"]); ax.set_xlabel("SMB (m w.e. yr$^{-1}$)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6); ax.legend(loc="upper left", fontsize=6.5)
axes[0].set_ylabel("Elevation (m a.s.l.)")
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"{ROOT}/results/fig_continuity_3panel.{e}", bbox_inches="tight")
print("wrote fig_continuity_3panel")
