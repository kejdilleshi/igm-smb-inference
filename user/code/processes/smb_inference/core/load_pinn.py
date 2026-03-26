"""
Loading utilities for PINN emulators (e.g. FNO2) using IGM's components.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from omegaconf import OmegaConf

from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate.utils.normalizations import FixedChannelStandardization
from igm.processes.iceflow.emulate.utils.artifacts_schema_v3 import parse_manifest_v3
from igm.processes.iceflow.vertical.vertical_lagrange import LagrangeDiscr
from igm.utils.math.precision import normalize_precision


def load_pinn_emulator(artifact_dir, Nz=4, vert_spacing=1.0, precision='single',
                       basis_vertical='Lagrange', basis_horizontal='central',
                       inputs=None, architecture='FNO2', width=32,
                       modes1=8, modes2=8, padding=9, use_grid=True, output_scale=1.0):
    """
    Load a PINN emulator from an IGM artifact directory.

    Uses IGM's architecture classes, normalization layers, and manifest parsing,
    but handles the loading flow directly.

    Parameters
    ----------
    artifact_dir : str or Path
        Path to the emulator artifact directory (contains manifest.yaml).
    Nz : int
        Number of vertical levels.
    vert_spacing : float
        Vertical spacing parameter (1.0 = linear).
    precision : str
        Precision ('single' or 'double').
    basis_vertical : str
        Vertical basis type (e.g. 'Lagrange').
    basis_horizontal : str
        Horizontal basis type (e.g. 'central').
    inputs : list of str or None
        Input field names. Defaults to ['thk', 'usurf', 'arrhenius', 'slidingco', 'dX'].
    architecture : str
        Network architecture name (e.g. 'FNO2').
    width : int
        Network width (nb_out_filter).
    modes1, modes2 : int
        Fourier modes in each spatial dimension.
    padding : int
        Padding size.
    use_grid : bool
        Whether to use grid coordinates as extra input channels.
    output_scale : float
        Output scaling factor.

    Returns
    -------
    model : tf.keras.Model
        Loaded emulator model with input_normalizer attached.
    manifest : EmulatorManifestV3
        Parsed manifest.
    V_bar : tf.Tensor
        Vertical averaging weights, shape (Nz, 1, 1).
    """
    if inputs is None:
        inputs = ['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']

    artifact_dir = Path(artifact_dir)
    desired_dtype = normalize_precision(precision)

    # Parse manifest
    raw = yaml.safe_load((artifact_dir / "manifest.yaml").read_text())
    manifest = parse_manifest_v3(raw)

    # Build minimal OmegaConf config for the architecture constructor
    cfg = OmegaConf.create({
        'processes': {
            'iceflow': {
                'numerics': {
                    'Nz': Nz,
                    'precision': precision,
                    'vert_spacing': vert_spacing,
                    'basis_vertical': basis_vertical,
                    'basis_horizontal': basis_horizontal,
                },
                'unified': {
                    'inputs': inputs,
                    'network': {
                        'architecture': architecture,
                        'width': width,
                        'modes1': modes1,
                        'modes2': modes2,
                        'padding': padding,
                        'use_grid': use_grid,
                        'output_scale': output_scale,
                    },
                },
            }
        }
    })

    # Build model
    arch_name = str(manifest.architecture.name)
    model = Architectures[arch_name](cfg, manifest.nb_inputs, manifest.nb_outputs)
    model.input_normalizer = None

    # Use a dummy size large enough for spectral conv modes
    dummy_size = max(2 * max(modes1, modes2), 16)
    dummy = tf.zeros((1, dummy_size, dummy_size, manifest.nb_inputs), dtype=desired_dtype)
    _ = model(dummy, training=False)

    # Load weights
    weights_path = artifact_dir / "export" / "weights.weights.h5"
    model.load_weights(str(weights_path))

    # Attach normalizer from manifest stats
    p = manifest.normalization.params
    eps = float(p.get("epsilon", p.get("variance_epsilon", 1e-7)))
    mean_1d = np.asarray(p["mean_1d"], dtype=np.float64).reshape(-1)
    var_1d = np.asarray(p["var_1d"], dtype=np.float64).reshape(-1)

    model.input_normalizer = FixedChannelStandardization(
        mean_1d=mean_1d,
        var_1d=var_1d,
        epsilon=eps,
        dtype=desired_dtype,
        name="input_norm",
    )
    _ = model.input_normalizer(tf.zeros((1, 2, 2, manifest.nb_inputs), dtype=desired_dtype))

    # Compute V_bar for depth averaging using Lagrange vertical discretization
    vert_discr = LagrangeDiscr(cfg)
    V_bar = vert_discr.V_bar

    print(f"Loaded PINN emulator: {manifest.nb_inputs} inputs -> {manifest.nb_outputs} outputs (Nz={manifest.Nz})")

    return model, manifest, V_bar
