#!/usr/bin/env python3
"""
SMB-inference regularization L-curve (step 3.1 of the pipeline).

Reads the per-run smb_inference outputs from a Hydra-multirun sweep over
`processes.smb_inference.optimization.regularisation` and, for each run:
  * data misfit     = final surface-fit RMSE  (data_history.npy[-1], or
                      reconstructed from loss_history − reg·smoothness)
  * profile roughness = Σ (2nd difference of smb_vec)²   — the penalty term
plots `lcurve_smb.png`, and auto-detects the corner (roughness vs misfit) →
the chosen optimization.regularisation.

The two axes mirror smb_inference's loss: total = data_term + reg·smoothness
(see smb_inference.py::_compute_loss).

Usage:
    python analyze_smb_lcurve.py <results_dir>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcurve import parse_param, lcurve_corner  # noqa: E402

REG_PARAM = "processes.smb_inference.optimization.regularisation"


def _roughness(smb_vec):
    if smb_vec.shape[0] < 3:
        return 0.0
    d2 = smb_vec[:-2] - 2.0 * smb_vec[1:-1] + smb_vec[2:]
    return float(np.sum(d2 ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="Hydra sweep dir (contains <reg>/smb_inference/*.npy)")
    ap.add_argument("--out", default=None, help="Output PNG (default <results_dir>/lcurve_smb.png)")
    args = ap.parse_args()

    vec_files = sorted(glob.glob(os.path.join(args.results_dir, "**/smb_inference/smb_vec.npy"),
                                 recursive=True))
    if not vec_files:
        sys.exit(f"No smb_inference/smb_vec.npy under {args.results_dir}")

    regs, misfits, roughs = [], [], []
    for vec_path in vec_files:
        smb_dir = os.path.dirname(vec_path)
        run_dir = os.path.dirname(smb_dir)
        reg = parse_param(run_dir, REG_PARAM)
        if reg is None:
            print(f"  [skip] cannot parse reg from {os.path.relpath(run_dir, args.results_dir)}")
            continue

        smb_vec = np.load(vec_path)
        rough = _roughness(smb_vec)

        data_path = os.path.join(smb_dir, "data_history.npy")
        if os.path.exists(data_path):
            misfit = float(np.load(data_path)[-1])
        else:
            # Reconstruct: total = data + reg·roughness  →  data = total − reg·roughness.
            loss = np.load(os.path.join(smb_dir, "loss_history.npy"))
            misfit = float(loss[-1] - reg * rough)

        regs.append(reg); misfits.append(misfit); roughs.append(rough)
        print(f"  reg={reg:>10g}: data_misfit={misfit:.4f}, roughness={rough:.4g}")

    if len(regs) < 2:
        sys.exit("Not enough runs to build an L-curve.")

    idx = np.argsort(regs)
    regs = np.array(regs)[idx]
    misfits = np.array(misfits)[idx]
    roughs = np.array(roughs)[idx]

    corner = lcurve_corner(roughs, misfits)
    best_reg = float(regs[corner])
    print(f"\nAuto-detected corner: optimization.regularisation = {best_reg:g} "
          f"(data_misfit={misfits[corner]:.4f}, roughness={roughs[corner]:.4g})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(roughs, misfits, "o-", color="seagreen", markersize=8, linewidth=2)
    for i, r in enumerate(regs):
        ax.annotate(f"  {r:g}", (roughs[i], misfits[i]), fontsize=9)
    ax.plot(roughs[corner], misfits[corner], "*", color="red", markersize=20,
            label=f"corner reg={best_reg:g}")
    ax.set_xlabel("SMB profile roughness  Σ(2nd diff)²")
    ax.set_ylabel("Data misfit (surface RMSE)")
    ax.set_title("SMB L-curve: misfit vs roughness")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(regs, misfits, "o-", color="seagreen", label="Data misfit", linewidth=2)
    ax2 = ax.twinx()
    ax2.loglog(regs, roughs, "s--", color="purple", label="Roughness", linewidth=2)
    ax.axvline(best_reg, color="red", linestyle=":", linewidth=1.5)
    ax.set_xlabel("optimization.regularisation")
    ax.set_ylabel("Data misfit (surface RMSE)", color="seagreen")
    ax2.set_ylabel("Roughness", color="purple")
    ax.set_title("Misfit & roughness vs regularization")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Step 3.1: SMB regularization L-curve", fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = args.out or os.path.join(args.results_dir, "lcurve_smb.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    result = {
        "param": REG_PARAM,
        "best_reg": best_reg,
        "regs": regs.tolist(),
        "data_misfit": misfits.tolist(),
        "roughness": roughs.tolist(),
        "plot": out_png,
    }
    out_json = os.path.join(args.results_dir, "smb_lcurve_result.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
