import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter
from igm.processes.smb_inference.utils.emulator_tools import apply_boundary_condition_tf


def make_pinn_emulator_step(model, V_bar, Nz, dx_value):
    """
    Factory function that creates a gradient-checkpointed PINN emulator step.

    The PINN model takes 5 inputs: [thk, usurf, arrhenius, slidingco, dX]
    and outputs 2*Nz channels: U at Nz levels followed by V at Nz levels.
    Depth-averaged ubar/vbar are computed using V_bar weights.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded FNO2 PINN emulator (with input_normalizer attached).
    V_bar : tf.Tensor
        Vertical averaging weights, shape (Nz,) or (Nz, 1, 1).
    Nz : int
        Number of vertical levels.
    dx_value : float
        Grid spacing value (used as the dX input channel).

    Returns
    -------
    pinn_emulator_step_differentiable : function
        A gradient-checkpointed function for use in tf.while_loop.
    """
    # Ensure V_bar is 1D (Nz,) for einsum
    V_bar_1d = tf.squeeze(V_bar)

    @tf.recompute_grad
    def pinn_emulator_step_differentiable(H_ice, Z_surf, smb, time, Z_topo, dx, dy,
                                          dtmax, cfl, arrhenius_field, slidingco_field):
        dx = tf.cast(dx, tf.float32)
        dy = tf.cast(dy, tf.float32)
        dtmax = tf.cast(dtmax, tf.float32)
        cfl = tf.cast(cfl, tf.float32)

        # Build the 5-channel input: [thk, usurf, arrhenius, slidingco, dX]
        dX_field = tf.ones_like(H_ice) * dx_value

        input_data = tf.stack([
            H_ice,
            Z_surf,
            arrhenius_field,
            slidingco_field,
            dX_field,
        ], axis=-1)
        input_data = tf.expand_dims(input_data, 0)  # (1, ny, nx, 5)

        # Model forward pass (normalization is handled internally by FNO2)
        velocity_pred = model(input_data, training=False)  # (1, ny, nx, 2*Nz)

        # Extract U and V at each vertical level
        U_all = velocity_pred[0, :, :, :Nz]   # (ny, nx, Nz)
        V_all = velocity_pred[0, :, :, Nz:]   # (ny, nx, Nz)

        # Depth-average using V_bar weights
        ubar = tf.einsum("ijn,n->ij", U_all, V_bar_1d)
        vbar = tf.einsum("ijn,n->ij", V_all, V_bar_1d)

        # CFL time step
        vel_max = tf.maximum(tf.reduce_max(tf.abs(ubar)), tf.reduce_max(tf.abs(vbar)))
        vel_max = tf.maximum(vel_max, 1e-4)
        dt = tf.minimum(cfl * dx / vel_max, dtmax)

        # Flux divergence and thickness update
        H_ice_new = H_ice + dt * (smb-compute_divflux_slope_limiter(ubar, vbar, H_ice, dx, dy, dt, "superbee")
)
        H_ice_new = tf.maximum(H_ice_new, 0.0)

        # Boundary condition
        H_ice_new = apply_boundary_condition_tf(H_ice_new)

        # Update surface
        Z_surf_new = Z_topo + H_ice_new

        return H_ice_new, Z_surf_new, time + dt, ubar, vbar

    return pinn_emulator_step_differentiable
