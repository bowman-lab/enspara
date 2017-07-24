import os
import sys
import argparse
import logging
import pickle
import time

from functools import partial
from multiprocessing import cpu_count

import numpy as np
import mdtraj as md
from sklearn.externals.joblib import Parallel, delayed

from enspara import exception

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
        '--centers', required=True,
        help="The centers to use for reassignment.")
    parser.add_argument(
        '--trajectories', required=True, nargs="+", action='append',
        help="The aligned xtc files to cluster.")
    parser.add_argument(
        '--topology', required=True, action='append', dest='topologies',
        help="The topology file for the trajectories.")
    parser.add_argument(
        '--atoms', default="(name CA or name C or name N or name CB)",
        help="The atoms from the trajectories (using MDTraj atom-selection"
             "syntax) to cluster based upon.")
    parser.add_argument(
        '--output-path', default=None,
        help="Output path for results (distances, assignments). "
             "Default is in the same directory as the input centers.")
    parser.add_argument(
        '-j', '--n_procs', default=cpu_count(), type=int,
        help="The number of cores to use while reassigning.")
    parser.add_argument(
        '--output-tag', default='',
        help="An optional extra string prepended to output filenames (useful"
             "for giving this choice of parameters a name to separate it from"
             "other clusterings or proteins.")

    args = parser.parse_args(argv[1:])

    if len(args.topologies) != len(args.trajectories):
        raise exception.ImproperlyConfigured(
            "The number of --topology and --trajectory flags must agree.")

    if args.output_path is None:
        args.output_path = os.path.dirname(args.centers)

    return args


def reassign_single(trjfile, topfile, centers, atoms):
    """Reassign a single trjfile using centers and the selection atoms.
    """

    top = md.load(topfile).top
    trj = md.load(trjfile, top=top, atom_indices=top.select(atoms))

    try:
        _, single_assignments, single_distances = \
            assign_to_nearest_center(
                trj, centers, partial(md.rmsd, parallel=False))

    except ValueError:
        raise DataInvalid(
            "Failed to assign trajectory %s with topology %s" %
            (trjfile, topfile))

    return single_assignments, single_distances


def reassign(topologies, trajectories, atoms, centers, n_procs=None):

    if len(topologies) != len(trajectories):
        raise exception.ImproperlyConfigured(
            "Number of topologies (%s) didn't match number of sets of "
            "trajectories (%s)." % (len(topologies), len(trajectories)))
    if len(topologies) != len(atoms):
        raise exception.ImproperlyConfigured(
            "Number of topologies (%s) didn't match number of atom selection "
            "strings (%s)." % (len(topologies), len(atoms)))

    logger.info("Reassigning dataset of %s trajectories and %s topologies.",
                sum(len(t) for t in trajectories), len(topologies))

    tick = time.clock()

    targets = []
    for topfile, trjfiles, atoms in zip(topologies, trajectories, atoms):
        targets.extend([(trjfile, topfile, atoms) for trjfile in trjfiles])

    reassignments = Parallel(n_jobs=n_procs)(
        delayed(reassign_single)(trj, top, centers, atoms)
        for trj, top, atoms in targets)

    assignments = [r[0] for r in reassignments]
    distances = [r[1] for r in reassignments]

    tock = time.clock()
    logger.info("Reassignment took %s seconds.", tock - tick)

    if all([len(assignments[0]) == len(a) for a in assignments]):
        logger.info("Trajectory lengths are homogenous. Output will "
                    "be np.ndarrays.")
        assert all([len(distances[0]) == len(d) for d in distances])
        return np.array(assignments), np.array(distances)
    else:
        logger.info("Trajectory lengths are heterogenous. Output will "
                    "be ra.RaggedArrays.")
        return ra.RaggedArray(assignments), ra.RaggedArray(distances)


def extract_center_atoms(c, atoms):
    """Slice down a single center (number i) from a centers list using
    atoms.
    """
    return c.atom_slice(c.top.select(atoms))


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    with open(args.centers, 'rb') as f:
        centers = pickle.load(f)

    assert(all(centers[0].n_atoms == c.n_atoms for c in centers))
    logger.info("Loaded %s centers with %s atoms each.",
                len(centers), centers[0].n_atoms)

    centers = Parallel(n_jobs=args.n_procs)(
        delayed(extract_center_atoms)(ctr, args.atoms) for ctr in centers)
    logger.info("Sliced %s centers down to %s atoms each.",
                len(centers), centers[0].n_atoms)

    assig, dist = reassign(
        args.topologies, args.trajectories, [args.atoms]*len(args.topologies),
        centers=centers, n_procs=args.n_procs)

    fstem = os.path.join(args.output_path, args.output_tag)
    ra.save(fstem+'-distances.h5', dist)
    ra.save(fstem+'-assignments.h5', assig)

    logger.info("Finished reassignments. Data deposited in " +
                "%s-{distances,assignments}.h5", fstem)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
