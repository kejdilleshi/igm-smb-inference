# igm-smb-inference

A user module for [IGM (Instructed Glacier Model)](https://igm-model.org/) that performs **Surface Mass Balance (SMB) inversion** from glacier thickness observations using gradient-based optimization through a differentiable ice flow emulator.

## How it works

1. **Data assimilation** runs first (built-in IGM process) — optimizes ice thickness from observed surface velocities and trains an ice flow emulator.
2. **`smb_inference`** then takes the DA-optimized state (thickness + trained emulator) and runs a second inversion:
   - Parameterizes SMB as a 1D elevation-dependent profile (hat basis functions).
   - Runs a differentiable forward glacier simulation using `tf.GradientTape` + `tf.while_loop` with gradient checkpointing.
   - Minimizes MAE between simulated and observed thickness plus a curvature regularization term.
   - Uses Adam optimizer; supports early stopping and periodic emulator retraining.

## Usage

```bash
igm_run +experiment=params
```

See `experiment/params.yaml` for the full configuration. An alternative experiment using a pretrained CNN emulator is available:

```bash
igm_run +experiment=params_pretrained
```

## Configuration

All parameters are in `user/conf/processes/smb_inference.yaml` and accessed via `cfg.processes.smb_inference`. Key settings:

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

## Directory structure

```
igm-smb-inference/
├── experiment/
│   ├── params.yaml              # DA + SMB inference experiment
│   └── params_pretrained.yaml   # Pretrained CNN experiment
├── user/
│   ├── code/processes/smb_inference/   # Python package
│   │   ├── __init__.py                 # Exports initialize/update/finalize
│   │   ├── smb_inference.py            # Main entry point
│   │   ├── core/                       # Glacier model, SMB, inversion, climate
│   │   ├── data/                       # Data loading (NetCDF)
│   │   ├── utils/                      # Numerical tools (flux divergence, etc.)
│   │   ├── visualization/              # Plotting
│   │   └── config/                     # Config parsing
│   └── conf/processes/
│       └── smb_inference.yaml          # Hydra config (default parameters)
├── data/
│   └── input.nc
└── README.md
```

## License

GNU GPL v3 (same as IGM)
