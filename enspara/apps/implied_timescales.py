"""Given assignments and a list of lagtimes, plot implied timescales.

Options are provided for using various forms of MSM and parallelization.
"""

import sys
import argparse

import numpy as np
import mdtraj as md

from tables.exceptions import NoSuchNodeError

from enspara import exception
from enspara.msm import implied_timescales, builders
from enspara.util import array as ra


def process_command_line(argv):

    parser = argparse.ArgumentParser(
        prog='implied',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--assignments", required=True,
        help="File containing assignments to states.")
    parser.add_argument(
        "--n-eigenvalues", default=5, type=int,
        help="Number of eigenvalues to compute for each lag time.")
    parser.add_argument(
        "--lag-times", default="5:100:2",
        help="List of lagtimes (in frames) to compute eigenspectra for. "
             "Format is min:max:step.")
    parser.add_argument(
        "--symmetrization", default="transpose",
        choices=['transpose', 'row_normalize', 'prior_counts'],
        help="The method to use to fit transition probabilities from "
             "the transition counts matrix.")
    parser.add_argument(
        "--trj-ids", default=None,
        help="Computed the implied timescales for only the given "
             "trajectory ids. This is useful for handling assignments "
             "for shared state space clusterings.")
    parser.add_argument(
        "--trim", default=False, action="store_true",
        help="Turn ergodic trimming on.")

    parser.add_argument(
        "--timestep", default=None, type=float,
        help='A conversion between frames and nanoseconds (i.e. frames '
             'per nanosecond) to scale the axes to physical units '
             '(rather than frames).')
    parser.add_argument(
        "--infer-timestep", default=None,
        help="An example trajectory from which to infer the conversion "
             "from frame to nanoseconds.")

    parser.add_argument(
        "--plot", default=None,
        help="Path for the implied timescales plot.")
    parser.add_argument(
        "--logscale", action='store_true',
        help="Flag to output y-axis log scale plot.")

    args = parser.parse_args(argv[1:])

    args.lag_times = range(*map(int, args.lag_times.split(':')))

    if args.trj_ids is not None:
        args.trj_ids = slice(*map(int, args.trj_ids.split(':')))

    if args.symmetrization == 'prior_counts':
        args.symmetrization = prior_counts
    else:
        args.symmetrization = getattr(builders, args.symmetrization)

    return args


def prior_counts(C):
    return builders.normalize(C, prior_counts=1/C.shape[0])


def process_units(timestep=None, infer_timestep=None):
    """Take the timestep parameter and infer_timestep parameters from
    the command line arguments and convert it to the string indicating
    units (ns) and the factor converting ns to frames.

    Parameters
    ----------
    timestep : float
        Ratio of ns to frames. This is typically 10 (for 100 ps
        timesteps) or 100 (for 10 ps timesteps).
    infer_timestep : str, path
        Path to a trajectory containing timestep information to infer
        the correct timestep from when plotting implied timescales.
    """

    if timestep and infer_timestep:
        raise exception.ImproperlyConfigured(
            'Only one of --timestep and --infer-timestep can be '
            'supplied, you supplied both --timestep=%s and '
            '--infer-timestep=%s' % (timestep, infer_timestep))

    if timestep:
        unit_factor = timestep
        unit_str = 'ns'
    elif infer_timestep:
        try:
            timestep = md.load(infer_timestep).timestep
        except ValueError:
            if infer_timestep[-4:] != '.xtc':
                raise exception.ImproperlyConfigured(
                    "Topologyless formats other than XTC are not supported.")
            with md.formats.xtc.XTCTrajectoryFile(infer_timestep) as f:
                xyz, time, step, box = f.read(n_frames=10)
                timesteps = time[1:] - time[0:-1]
                assert np.all(timesteps[0] == timesteps)
                timestep = timesteps[0]
        unit_factor = 1000 / timestep  # units are ps
        unit_str = 'ns'
    else:
        unit_factor = 1
        unit_str = 'frames'

    return unit_factor, unit_str


def main(argv=None):

    args = process_command_line(argv)

    try:
        assignments = ra.load(args.assignments, keys=None)
    except NoSuchNodeError:
        assignments = ra.load(args.assignments, keys=...)
    if args.trj_ids is not None:
        assignments = assignments[args.trj_ids]

    tscales = implied_timescales(
        assignments, args.lag_times, n_times=args.n_eigenvalues,
        sliding_window=True, trim=args.trim,
        method=args.symmetrization)

    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt

    unit_factor, unit_str = process_units(args.timestep, args.infer_timestep)

    # scale x and y axes to nanoseconds
    lag_times = np.array(args.lag_times) / unit_factor
    tscales /= unit_factor

    for i in range(args.n_eigenvalues):
        plt.plot(lag_times, tscales[:, i] / unit_factor,
                 label=r'$\lambda_{i}$'.format(i=i+1))

    if args.logscale:
        plt.yscale('log')

    plt.ylabel('Eigenmotion Speed [{u}]'.format(u=unit_str))
    plt.xlabel('Lag Time [{u}]'.format(u=unit_str))
    plt.legend(frameon=False)

    plt.savefig(args.plot, dpi=300)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
