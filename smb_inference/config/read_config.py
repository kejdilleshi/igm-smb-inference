import argparse
import tensorflow as tf
import sys


def parse_arguments(arg_list=None):
    """
    Parse command-line arguments for glacier evolution model parameters.

    Parameters:
    -----------
    arg_list : list, optional
        List of arguments to parse (default: sys.argv)

    Returns:
    --------
    args : argparse.Namespace
        Parsed arguments with tensor values for time parameters
    """
    parser = argparse.ArgumentParser(description="Parse glacier evolution model parameters.")

    parser.add_argument('--ttot', type=float, default=2017.0, help='Time limit (yr)')
    parser.add_argument('--t_start', type=float, default=1700.0, help='Start time (yr)')
    parser.add_argument('--dtmax', type=float, default=1.0, help='Maximum timestep (yr)')
    parser.add_argument('--cfl', type=float, default=0.2, help='CFL condition for numerical stability')
    parser.add_argument('--initial_mean_temp', type=float, default=7, help='Initial mean temperature (C)')
    parser.add_argument('--initial_precip', type=float, default=0.2, help='Initial precipitation (m/yr)')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--regularisation', type=float, default=0.05, help='Smoothness regularization')
    parser.add_argument('--rho', type=float, default=910.0, help='Density of ice (kg/m3)')
    parser.add_argument('--g', type=float, default=9.81, help='Acceleration due to gravity (m/s2)')
    parser.add_argument('--fd', type=float, default=0.25e-16, help='Flow rate factor (Pa-3 s-1)')
    parser.add_argument('--dx', type=float, default=100.0, help='Grid resolution in x direction (m)')
    parser.add_argument('--dy', type=float, default=100.0, help='Grid resolution in y direction (m)')
    parser.add_argument('--vis_freq', type=float, default=10.0, help='Visualization frequency (yr)')
    parser.add_argument('--outdir', type=str, default='./results/run1', help='Output directory')
    parser.add_argument('--forward_scheme', type=str, default='emulator',
                        choices=['SIA', 'emulator'], help='Forward scheme for ice dynamics')

    if arg_list is None:
        # Clear Jupyter's own arguments if running inside a notebook
        argv = sys.argv[:1]
    else:
        argv = arg_list

    args = parser.parse_args(argv[1:] if len(argv) > 1 else [])

    # Convert specific arguments to tensors if needed
    args.ttot = tf.constant(args.ttot, dtype=tf.float32)
    args.t_start = tf.constant(args.t_start, dtype=tf.float32)

    return args


class Config:
    """
    Configuration class for glacier model parameters.

    This provides an alternative to argparse for programmatic configuration.
    """

    def __init__(
        self,
        ttot=2017.0,
        t_start=1700.0,
        dtmax=1.0,
        cfl=0.2,
        initial_mean_temp=7.0,
        initial_precip=0.2,
        learning_rate=0.05,
        regularisation=0.05,
        rho=910.0,
        g=9.81,
        fd=0.25e-16,
        dx=100,
        dy=100,
        vis_freq=10.0,
        outdir='./results/run1',
        forward_scheme='emulator'
    ):
        self.ttot = tf.constant(ttot, dtype=tf.float32)
        self.t_start = tf.constant(t_start, dtype=tf.float32)
        self.dtmax = dtmax
        self.cfl = cfl
        self.initial_mean_temp = initial_mean_temp
        self.initial_precip = initial_precip
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.rho = rho
        self.g = g
        self.fd = fd
        self.dx = dx
        self.dy = dy
        self.vis_freq = vis_freq
        self.outdir = outdir
        self.forward_scheme = forward_scheme
