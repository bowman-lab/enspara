import sys
import argparse

from multiprocessing import cpu_count

import numpy as np

from enspara.msm import implied_timescales, builders
from enspara.util import array as ra

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt


def process_command_line(argv):
    '''Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.'''

    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--assignments", required=True,
        help="File containing assignments to states.")
    parser.add_argument(
        "--n-eigenvalues", default=5, type=int,
        help="Number of eigenvalues to compute for each lag time.")
    parser.add_argument(
        "--lag-times",  default="5:100:2",
        help="List of lagtimes (in frames) to compute eigenspectra for. "
             "Format is min:max:step.")
    parser.add_argument(
        "--symmetrization", default="transpose",
        help="The method to use to enforce detailed balance in the"
             "counts matrix.")
    parser.add_argument(
        "--trj-ids", default=None,
        help="Computed the implied timescales for only the given "
             "trajectory ids. This is useful for handling assignments "
             "for shared state space clusterings.")
    parser.add_argument(
        "--processes", default=max(1, cpu_count()/4), type=int,
        help="Number of processes to use. Because eigenvector "
             "decompositions are thread-parallelized, this should "
             "usually be several times smaller than the number of "
             "cores availiable on your machine.")
    parser.add_argument(
        "--trim", default=False, action="store_true",
        help="Turn ergodic trimming on.")

    parser.add_argument(
        "--timestep", default=None, type=float,
        help='A conversion between frames and nanoseconds (i.e. frames '
             'per nanosecond) to scale the axes to physical units '
             '(rather than frames).')
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

    args.symmetrization = getattr(builders, args.symmetrization)

    return args


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    try:
        assignments = ra.load(args.assignments)
    except KeyError:
        assignments = ra.load(args.assignments, keys=...)
    if args.trj_ids is not None:
        assignments = assignments[args.trj_ids]

    tscales = implied_timescales(
        assignments, args.lag_times, n_times=args.n_eigenvalues,
        sliding_window=True, trim=args.trim,
        method=args.symmetrization, n_procs=args.processes)

    if args.timestep:
        unit_factor = args.timestep
        unit_str = 'ns'
    else:
        unit_factor = 1
        unit_str = 'frames'

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
