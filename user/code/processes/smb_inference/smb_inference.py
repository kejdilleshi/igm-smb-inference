#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
SMB Inference process for IGM.

Performs gradient-based inversion to recover surface mass balance (SMB)
parameters from glacier thickness observations. Uses a differentiable
glacier dynamics simulation with a CNN or PINN ice flow emulator.

This process is analogous to data_assimilation but targets climate/SMB
parameters rather than ice dynamics parameters (thickness, sliding, etc.).

Like data_assimilation, the full optimization loop runs within initialize().
"""

import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IGM loads this file directly via SourceFileLoader (not as a package), so
# relative imports don't work. Add the smb_inference/ directory to sys.path
# so sub-modules can be imported by their package-relative names.
_smb_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _smb_pkg_dir not in sys.path:
    sys.path.insert(0, _smb_pkg_dir)

from core.glacier import GlacierDynamicsCheckpointed
from core.inversion import _eval_pair
from core.smb import update_smb_profile
from data.loader import load_observations_from_nc, load_smb_point_obs
from visualization.plots import plot_loss_components
from config.read_config import Config


# ─── Helper functions ────────────────────────────────────────────────────────


def _get_dx_dy(cfg, state):
    """Compute grid spacing from state coordinates, with config fallback."""
    iceflow_physics = cfg.processes.iceflow.physics

    if hasattr(state, "x") and state.x is not None and len(state.x) > 1:
        dx = float(state.x[1] - state.x[0])
    else:
        dx = float(iceflow_physics.get("dx", 100.0))

    if hasattr(state, "y") and state.y is not None and len(state.y) > 1:
        dy = float(state.y[1] - state.y[0])
    else:
        dy = float(iceflow_physics.get("dy", 100.0))

    return dx, dy


def _load_observation(smb_cfg, state, topg):
    """Load the observation target for inversion."""
    obs_cfg = smb_cfg.observations

    if getattr(obs_cfg, "geology_file", ""):
        obs_years = list(obs_cfg.observation_years)
        obs = load_observations_from_nc(obs_cfg.geology_file, topg, obs_years)
        target_key = obs_years[-1]
        observation = obs[target_key]
        print(
            f"[smb_inference] Loaded {len(obs)} observations from "
            f"{obs_cfg.geology_file}, target: {target_key}"
        )
        return observation, obs

    # Use a state variable as observation target
    source = getattr(obs_cfg, "state_variable", "thk")
    if hasattr(state, source) and getattr(state, source) is not None:
        raw = tf.cast(getattr(state, source), tf.float32)

        # If the observation is a surface elevation, convert to thickness
        if source in ("usurfinfer", "usurfobs", "usurf"):
            observation = raw - topg
            print(
                f"[smb_inference] Using state.{source} as observation target "
                f"(converted to thickness via {source} - topg)"
            )
        else:
            observation = raw
            print(f"[smb_inference] Using state.{source} as observation target")

        return observation, None
    else:
        raise ValueError(
            f"[smb_inference] No observation data found. "
            f"state.{source} is not available. "
            f"Set observations.geology_file or ensure state.{source} exists."
        )


def _run_profile_inversion(smb_cfg, glacier_model, observation, topg, H_init, ice_mask, outdir=None):
    """
    Optimize a 1D elevation-dependent SMB profile to match observed thickness.

    Returns
    -------
    H_sim : tf.Tensor
        Simulated ice thickness at end of optimization.
    smb_vec : tf.Variable
        Optimized 1D SMB elevation profile.
    z_min : tf.Tensor
        Minimum elevation of the profile.
    dz : tf.Tensor
        Elevation bin spacing.
    loss_history : list
        Total loss per iteration.
    data_history : list
        Data-fidelity term per iteration.
    """
    inv_cfg = smb_cfg.inversion
    opt_cfg = smb_cfg.optimization

    # Elevation range for profile discretization. Restrict to on-glacier
    # pixels — off-glacier terrain (outside ice_mask) can extend well above
    # or below the glacier and would otherwise inflate N_bins with bins
    # that never see any ice.
    Z_surf = topg + H_init
    on_ice = ice_mask > 0.5
    Z_on = tf.boolean_mask(Z_surf, on_ice)
    z_min = tf.reduce_min(Z_on)
    z_max = tf.reduce_max(Z_on)
    dz = tf.constant(float(inv_cfg.dz), dtype=tf.float32)
    N_bins = int(tf.math.ceil((z_max - z_min) / dz).numpy()) + 1

    # Trainable SMB profile
    smb_vec = tf.Variable(
        tf.linspace(float(inv_cfg.smb_init_low), float(inv_cfg.smb_init_high), N_bins),
        trainable=True,
        dtype=tf.float32,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=opt_cfg.learning_rate)
    reg_lambda = float(opt_cfg.regularisation)

    loss_history = []
    data_history = []
    best_loss = float("inf")
    no_improve_count = 0
    H_sim = None

    print(
        f"[smb_inference] Profile inversion: {N_bins} elevation bins, "
        f"z_min={float(z_min):.0f}, z_max={float(z_max):.0f}, dz={float(dz):.0f}"
        f", smb min={float(tf.reduce_min(smb_vec).numpy()):.2f}, "
        f"smb max={float(tf.reduce_max(smb_vec).numpy()):.2f}"
    )

    log_freq = max(1, opt_cfg.nbitmax // 50) # Log ~10 times during optimization

    smb_vec_dir = None
    if outdir is not None:
        smb_vec_dir = os.path.join(outdir, "smb_vec_iters")
        os.makedirs(smb_vec_dir, exist_ok=True)

    retrain_freq = getattr(opt_cfg, 'retrain_emulator_freq', 20)
    early_retrain_iters = getattr(opt_cfg, 'early_retrain_iters', 5)

    # Optional: overlay stake SMB observations on per-iter profile plots.
    # CSV has columns X, Y, Alt, Annual_SMB (m w.e.). We ignore X/Y and
    # convert to m ice eq./yr (divide by rho_ice=0.917) so it matches the
    # smb_vec axis.
    obs_alts, obs_smb = None, None
    smb_points_file = getattr(smb_cfg.observations, "smb_points_file", "") or ""
    if smb_points_file and os.path.exists(smb_points_file):
        try:
            obs_alts, obs_smb = load_smb_point_obs(smb_points_file)
            print(f"[smb_inference] Loaded {len(obs_alts)} SMB stake points "
                  f"from {smb_points_file}")
        except Exception as e:
            print(f"[smb_inference] Could not load {smb_points_file}: {e}")

    for i in range(opt_cfg.nbitmax):
        # # Retrain every iteration for the first early_retrain_iters, then every retrain_freq
        # if i < early_retrain_iters or (retrain_freq > 0 and i % retrain_freq == 0):
        #     glacier_model(
        #         precip_tensor=None,
        #         T_m_lowest=None,
        #         T_s=None,
        #         melt_factor=None,
        #         smb_method="profile",
        #         smb_vec=smb_vec,
        #         z_min=z_min,
        #         dz=dz,
        #         retrain_mode=True,
        #     )
        #     print(f"[smb_inference] Emulator retrained at iteration {i + 1}")

        


        with tf.GradientTape() as tape:
            # Forward glacier simulation (differentiable, tf.while_loop)
            H_sim = glacier_model(
                precip_tensor=None,
                T_m_lowest=None,
                T_s=None,
                melt_factor=None,
                smb_method="profile",
                smb_vec=smb_vec,
                z_min=z_min,
                dz=dz,
                retrain_mode=False,
            )

            # Data fidelity
            metrics = _eval_pair(H_sim, observation)
            data_term = metrics["rmse"]

            # Regularization: penalize curvature of SMB profile
            if reg_lambda > 0 and smb_vec.shape[0] > 2:
                second_deriv = smb_vec[:-2] - 2.0 * smb_vec[1:-1] + smb_vec[2:]
                smoothness = tf.reduce_sum(second_deriv ** 2)
                loss = data_term + reg_lambda * smoothness
            else:
                loss = data_term

        grads = tape.gradient(loss, [smb_vec])
        optimizer.apply_gradients(zip(grads, [smb_vec]))

        loss_val = float(loss.numpy())
        data_val = float(data_term.numpy())
        loss_history.append(loss_val)
        data_history.append(data_val)

        if i % log_freq == 0 or i == opt_cfg.nbitmax - 1:
            if smb_vec_dir is not None:
                z_axis = float(z_min) + float(dz) * np.arange(smb_vec.shape[0])
                fig, ax = plt.subplots(figsize=(5, 8))
                ax.plot(smb_vec.numpy(), z_axis, "-o", markersize=3, label="profile")
                if obs_alts is not None:
                    ax.scatter(obs_smb, obs_alts, s=18, c="tab:red", alpha=0.7,
                               edgecolors="k", linewidths=0.4, zorder=5,
                               label="stakes (m ice eq./yr)")
                    ax.legend(loc="best", fontsize=8)
                ax.axvline(0.0, color="k", linewidth=0.5)
                ax.set_xlabel("SMB (m ice eq./yr)")
                ax.set_ylabel("Elevation (m)")
                ax.set_title(f"SMB profile — iter {i + 1}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(smb_vec_dir, f"smb_vec_iter_{i + 1:04d}.png"), dpi=100)
                plt.close(fig)
            print(
                f"[smb_inference] Iter {i + 1}/{opt_cfg.nbitmax}: "
                f"loss={loss_val:.5f}, data={data_val:.5f}, "
                f"rmse={float(metrics['rmse']):.3f}"
            )

        # Early stopping
        if i > 10 and opt_cfg.early_stop_patience > 0:
            if best_loss - loss_val > opt_cfg.early_stop_threshold:
                best_loss = loss_val
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= opt_cfg.early_stop_patience:
                print(
                    f"[smb_inference] Early stopping at iteration {i + 1} "
                    f"(no improvement for {opt_cfg.early_stop_patience} iters)"
                )
                break

            if loss_val <= getattr(opt_cfg, "loss_threshold", 1.0e-4):
                print(f"[smb_inference] Loss threshold reached at iteration {i + 1}")
                break

    return H_sim, smb_vec, z_min, dz, loss_history, data_history


# ─── IGM process interface ───────────────────────────────────────────────────

from igm.processes.iceflow import initialize as iceflow_initialize

def initialize(cfg, state):
    """
    Initialize and run SMB inference (inversion).

    Like data_assimilation, the full optimization loop runs within
    initialize(). The optimized SMB parameters and resulting glacier
    state are stored back into the IGM state object.
    """
    iceflow_initialize(cfg, state) # initialize the iceflow model

    smb_cfg = cfg.processes.smb_inference

    # ── 1. Read data from IGM state ──────────────────────────────────────
    topg = tf.cast(state.topg, tf.float32)

    if hasattr(state, "icemask") and state.icemask is not None:
        ice_mask = tf.cast(state.icemask, tf.float32)
    else:
        ice_mask = tf.ones_like(topg)

    if hasattr(state, "thk") and state.thk is not None:
        H_init = tf.cast(state.thk, tf.float32)
    else:
        H_init = tf.zeros_like(topg)

    dx, dy = _get_dx_dy(cfg, state)
    print(f"[smb_inference] Grid: {topg.shape}, dx={dx}, dy={dy}")

    # ── 2. Load ice flow emulator (fallback if not on state) ────────────
    has_state_model = hasattr(state, 'iceflow_model') and state.iceflow_model is not None
    if has_state_model:
        print("[smb_inference] Emulator inherited from state.iceflow_model")
        # Detect input_fields from the iceflow config (pretrained CNN manifest)
        iceflow_cfg = cfg.processes.iceflow
        input_fields = list(getattr(iceflow_cfg.unified, 'inputs', []))
        if input_fields:
            print(f"[smb_inference] Using pretrained CNN with inputs: {input_fields}")
        else:
            input_fields = None
        model, V_bar, Nz = None, None, None
    else:
        print("[smb_inference] Emulator not found on state")
        model, V_bar, Nz, input_fields = None, None, None, None

    # ── 3. Build glacier dynamics model ──────────────────────────────────
    args = Config(
        ttot=float(smb_cfg.ttot),
        t_start=float(smb_cfg.t_start),
        dtmax=float(smb_cfg.dtmax),
        cfl=float(smb_cfg.cfl),
        rho=float(cfg.processes.iceflow.physics.ice_density),
        g=float(cfg.processes.iceflow.physics.gravity_cst),
        fd=0.25e-16,
        dx=dx,
        dy=dy,
        vis_freq=float(smb_cfg.output.vis_freq),
        outdir=smb_cfg.output.outdir,
    )

    glacier_model = GlacierDynamicsCheckpointed(
        Z_topo=topg,
        H_init=H_init,
        ice_mask=ice_mask,
        args=args,
        model=model,
        V_bar=V_bar,
        Nz=Nz,
        input_fields=input_fields,
        state=state,
        cfg=cfg,
    )

    # ── 4. Load observations ─────────────────────────────────────────────
    observation, all_obs = _load_observation(smb_cfg, state, topg)

    # ── 5. Run inversion ─────────────────────────────────────────────────
    method = smb_cfg.inversion.method

    if method == "profile":
        outdir = smb_cfg.output.outdir
        os.makedirs(outdir, exist_ok=True)
        H_sim, smb_vec, z_min, dz, loss_history, data_history = (
            _run_profile_inversion(
                smb_cfg, glacier_model, observation, topg, H_init, ice_mask,
                outdir=outdir,
            )
        )

        # Store profile-specific results
        state.smb_vec = smb_vec
        state.smb_z_min = z_min
        state.smb_dz = dz

        # Compute 2D SMB field from optimized profile
        Z_surf_final = topg + H_sim
        state.smb = update_smb_profile(Z_surf_final, smb_vec, z_min, dz) * ice_mask

    else:
        raise ValueError(
            f"[smb_inference] Unknown inversion method: '{method}'. "
            "Supported: 'profile'"
        )

    # ── 6. Update state ──────────────────────────────────────────────────
    state.thk = H_sim
    state.usurf = topg + H_sim
    state.smb_inference_loss_history = loss_history
    state.smb_inference_data_history = data_history

    # ── 7. Output ────────────────────────────────────────────────────────
    outdir = smb_cfg.output.outdir
    os.makedirs(outdir, exist_ok=True)

    if smb_cfg.output.plot_loss:
        try:
            plot_loss_components(loss_history, data_history, args)
        except Exception as e:
            print(f"[smb_inference] plot_loss_components failed: {e}")

    if smb_cfg.output.save_results:

        np.save(os.path.join(outdir, "thk_optimized.npy"), H_sim.numpy())
        np.save(os.path.join(outdir, "thk_observed.npy"), observation.numpy())
        np.save(os.path.join(outdir, "loss_history.npy"), np.array(loss_history))
        if hasattr(state, "smb_vec"):
            np.save(os.path.join(outdir, "smb_vec.npy"), state.smb_vec.numpy())
        if hasattr(state, "smb_z_min"):
            np.save(os.path.join(outdir, "z_min.npy"), np.array(float(state.smb_z_min)))
        if hasattr(state, "smb_dz"):
            np.save(os.path.join(outdir, "dz.npy"), np.array(float(state.smb_dz)))
        if hasattr(state, "smb") and state.smb is not None:
            np.save(os.path.join(outdir, "smb_field.npy"), state.smb.numpy())
            np.save(os.path.join(outdir, "smb_profile.npy"), smb_vec.numpy())

        print(f"[smb_inference] Results saved to {outdir}")

    print(f"[smb_inference] Optimization complete. Final loss: {loss_history[-1]:.5f}")


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass
