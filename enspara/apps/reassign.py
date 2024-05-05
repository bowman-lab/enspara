"""Given cluster centers, reassign trajectories in batches.

Options are provided for modifying what fraction of memory will be used
and which atoms to use for reassignment.
"""

import os
import sys
import argparse
import logging
import pickle
import time
import resource

from functools import partial

import psutil

import numpy as np
import mdtraj as md

from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s %(name)-8s %(levelname)-7s %(message)s'),
    datefmt='%m-%d-%Y %H:%M:%S')

import enspara

from enspara.cluster.util import assign_to_nearest_center, partition_list
from enspara.util.load import (concatenate_trjs, sound_trajectory,
                               load_as_concatenated)

from enspara.cluster.util import *
from enspara import ra
from enspara.util.log import timed


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):

    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--centers', required=True,
        help="Center structures (as a pickle) to use for reassignment.")
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
        '-m', '--mem-fraction', default=0.5, type=float,
        help="The fraction of total RAM to use in deciding the batch size. "
             "Genrally, this number shouldn't be much higher than 0.5.")

    # OUTPUT ARGS
    parser.add_argument(
        '--distances', required=True,
        help="Path to h5 file where distance to nearest cluster center "
             "will be output.")
    parser.add_argument(
        '--assignments', required=True,
        help="Path to h5 file where assignments to nearest center will "
             "be ouput")

    args = parser.parse_args(argv[1:])

    if args.mem_fraction >= 1 or args.mem_fraction <= 0:
        raise enspara.exception.ImproperlyConfigured(
            "Flag --mem-fraction must be in range (0, 1). Got %s"
            % args.mem_fraction)

    if len(args.topologies) != len(args.trajectories):
        raise enspara.exception.ImproperlyConfigured(
            "The number of --topology and --trajectory flags must agree.")

    if args.output_path is None:
        args.output_path = os.path.dirname(args.centers)

    for trjset in args.trajectories:
        for trj in trjset:
            f = open(trj, 'r')
            f.close()

    return args


def main(argv=None):

    args = process_command_line(argv)

    tick = time.perf_counter()

    with open(args.centers, 'rb') as f:
        centers = concatenate_trjs(
            pickle.load(f), args.atoms,
            enspara.util.parallel.auto_nprocs())
    logger.info('Loaded %s centers with %s atoms using selection "%s" '
                'in %.1f seconds.',
                len(centers), centers.n_atoms, args.atoms,
                time.perf_counter() - tick)

    assig, dist = reassign(
        args.topologies, args.trajectories, [args.atoms]*len(args.topologies),
        centers=centers, frac_mem=args.mem_fraction)

    mem_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(
        "Finished reassignments in %.1f seconds. Process memory high-water "
        "mark was %.2f GB (VRAM size is %.2f GB).",
        time.perf_counter() - tick,
        (mem_highwater / 1024**2),
        psutil.virtual_memory().total / 1024**3)

    ra.save(args.distances, dist)
    ra.save(args.assignments, assig)

    logger.info("Wrote distances at %s.", args.distances)
    logger.info("Wrote assignments at %s.", args.assignments)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
