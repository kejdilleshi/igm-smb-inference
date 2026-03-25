# igm-smb-inference

A user module for [IGM (Instructed Glacier Model)](https://igm-model.org/) that performs **Surface Mass Balance (SMB) inversion** from glacier thickness observations using gradient-based optimization through a differentiable ice flow emulator.

## How it works

1. **Data assimilation** runs first (built-in IGM process) — optimizes ice thickness from observed surface velocities and trains an ice flow emulator.
2. **`smb_inference`** then takes the DA-optimized state (thickness + trained emulator) and runs a second inversion:
   - Parameterizes SMB as a 1D elevation-dependent profile (hat basis functions).
   - Runs a differentiable forward glacier simulation using `tf.GradientTape` + `tf.while_loop` with gradient checkpointing.
   - Minimizes MAE between simulated and observed thickness plus a curvature regularization term.
   - Uses Adam optimizer; supports early stopping and periodic emulator retraining.

## Installation

### 1. Clone into your experiment's user module directory

IGM discovers user modules from the `user/code/processes/` directory relative to your experiment working directory.

```bash
cd /path/to/your/experiment
mkdir -p user/code/processes
cd user/code/processes
git clone https://github.com/<your-username>/igm-smb-inference smb_inference
```

> **Important**: The clone target folder **must** be named `smb_inference` (matching the module name).

### 2. Copy the Hydra config

IGM needs the process config YAML in the Hydra search path:

```bash
cd /path/to/your/experiment
mkdir -p user/conf/processes
cp user/code/processes/smb_inference/smb_inference.yaml user/conf/processes/
```

### 3. Add to your experiment config

In your experiment's `params.yaml`:

```yaml
defaults:
  - override /processes:
    - time
    - iceflow
    - data_assimilation
    - smb_inference          # <-- add this
```

## Requirements

- IGM (installed and importable)
- TensorFlow >= 2.x
- NumPy, SciPy, matplotlib, netCDF4

## Configuration

All parameters are in `smb_inference.yaml` and accessed via `cfg.processes.smb_inference`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `ttot` | 2017.0 | End time of forward simulation (yr) |
| `t_start` | 1700.0 | Start time (yr) |
| `emulator.from_state` | false | If true, inherit emulator from `state.iceflow_model` |
| `inversion.method` | profile | Inversion method (1D elevation-dependent profile) |
| `inversion.dz` | 100.0 | Elevation bin spacing (m) |
| `optimization.nbitmax` | 300 | Maximum optimization iterations |
| `optimization.learning_rate` | 0.05 | Adam learning rate |
| `optimization.regularisation` | 0.05 | Curvature regularization weight |

See `smb_inference.yaml` for the full parameter list.

## Directory structure

```
igm-smb-inference/
├── smb_inference/              # Python package (must match module name)
│   ├── __init__.py             # Exports initialize/update/finalize
│   ├── smb_inference.py        # Main entry point
│   ├── core/
│   │   ├── glacier.py          # GlacierDynamicsCheckpointed model
│   │   ├── smb.py              # SMB computation (profile, PDD, ELA)
│   │   ├── inversion.py        # Loss functions and metrics
│   │   ├── climate.py          # Lapse rates, PDD sums
│   │   ├── load_pinn.py        # PINN/FNO2 emulator loading
│   │   └── forward_schemes/    # Differentiable emulator steps
│   ├── data/                   # Data loading (NetCDF)
│   ├── utils/                  # Numerical tools (flux divergence, etc.)
│   ├── visualization/          # Plotting
│   └── config/                 # Config parsing
├── smb_inference.yaml          # Hydra config (copy to user/conf/processes/)
└── README.md
```

## How IGM discovers this module

IGM's module loader ([`loader.py`](https://github.com/instructed-glacier-model/igm)) searches for user process modules in:

1. `.smb_inference.py` (local directory)
2. `user/code/processes/smb_inference.py` (flat file)
3. `user/code/processes/smb_inference/smb_inference.py` (package) ← **this is how it works**

The `user/code/processes/` directory is added to `sys.path`, so internal imports like `from smb_inference.core.glacier import ...` resolve correctly.

## IGM dependencies

This module depends on the following IGM internals (which are available when IGM is installed):

- `igm.processes.iceflow` — emulator infrastructure, vertical discretization, solver
- `igm.utils.grad` — slope-limited flux divergence
- `igm.utils.math` — precision utilities
- `igm.inputs` — IGM input handling

## License

GNU GPL v3 (same as IGM)
