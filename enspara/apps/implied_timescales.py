import sys
import argparse

from multiprocessing import cpu_count

from mdtraj import io

from enspara.msm import implied_timescales, builders

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
        "--n-eigenvalues", default=5,
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
        "--processes", default=cpu_count(), help="Number of cores to use.")
    parser.add_argument(
        "--trim", default=False, action="store_true")
    parser.add_argument(
        "--plot", default=None,
        help="Path for the implied timescales plot.")

    args = parser.parse_args(argv[1:])

    args.lag_times = range(*map(int, args.lag_times.split(':')))
    args.symmetrization = getattr(builders, args.symmetrization)

    return args


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    assignments = io.loadh(args.assignments)['arr_0']

    tscales = implied_timescales(
        assignments, args.lag_times, n_times=args.n_eigenvalues,
        sliding_window=True, trim=args.trim,
        method=args.symmetrization, n_procs=args.processes)

    for i in range(args.n_eigenvalues):
        plt.plot(args.lag_times, tscales[:, i],
                 label=r'$\lambda_{i}$'.format(i=i+1))

    plt.ylabel('Eigenmotion Speed (frames)')
    plt.xlabel('Lag Time (frames)')
    plt.legend(frameon=False)

    plt.savefig(args.plot, dpi=300)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))