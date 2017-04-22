import sys
import argparse
import logging

import numpy as np
import mdtraj as md

from enspara.cluster.util import assign_to_nearest_center
from enspara.util import array as ra


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):
    '''Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.'''

    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--centers", required=True,
        help="The centers to use for reassignment.")
    parser.add_argument(
        '--trajectories', required=True, nargs="+", action='append',
        help="The aligned xtc files to cluster.")
    parser.add_argument(
        '--topology', required=True, action='append',
        help="The topology file for the trajectories.")
    parser.add_argument(
        '--atoms', default='name C or name O or name CA or name N or name CB',
        help="The atoms from the trajectories (using MDTraj atom-selection"
             "syntax) to cluster based upon.")

    args = parser.parse_args(argv[1:])

    return args


def reassign(topologies, trajectories, atoms, centers, processes):

    logger.info("Reassigning dataset.")
    assignments = []
    distances = []

    try:
        for topfile, trjfiles in zip(topologies, trajectories):
            top = md.load(topfile).top

            for trjfile in trjfiles:
                trj = md.load(trjfile, top=top, atom_indices=top.select(atoms))

                _, single_assignments, single_distances = \
                    assign_to_nearest_center(trj, centers, md.rmsd)

                assignments.append(single_assignments)
                distances.append(single_distances)
    except:
        print("topfile was", topfile)
        print("trjfile was", trjfile)
        raise

    if all([len(assignments[0]) == len(a) for a in assignments]):
        logger.info("Trajectory lengths are homogenous. Output will "
                    "be np.ndarrays.")
        assert all([len(distances[0]) == len(d) for d in distances])
        return np.array(assignments), np.array(distances)
    else:
        logger.info("Trajectory lengths are heterogenous. Output will "
                    "be ra.RaggedArrays.")
        return ra.RaggedArray(assignments), ra.RaggedArray(distances)


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
