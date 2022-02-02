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
from enspara.util import array as ra
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


def compute_batches(lengths, batch_size):
    """Compute batches (slices into lengths) of combined length at most
    size batch_size.
    """

    batch_sizes = [[]]
    batch_indices = [[]]
    for i, l in enumerate(lengths):
        if sum(batch_sizes[-1]) + l < batch_size:
            batch_sizes[-1].append(l)
            batch_indices[-1].append(i)
        else:
            batch_sizes.append([l])
            batch_indices.append([i])

    return batch_indices


def determine_batch_size(n_atoms, dtype_bytes, frac_mem):
    floats_per_frame = n_atoms * 3
    bytes_per_frame = floats_per_frame * dtype_bytes

    mem = psutil.virtual_memory()
    bytes_total = mem.total

    batch_size = int(bytes_total * frac_mem / bytes_per_frame)
    batch_gb = batch_size * bytes_per_frame / (1024**3)

    return batch_size, batch_gb


def batch_reassign(targets, centers, lengths, frac_mem, n_procs=None):

    example_center = centers[0]

    DTYPE_BYTES = 4
    batch_size, batch_gb = determine_batch_size(
        example_center.n_atoms, DTYPE_BYTES, frac_mem)

    logger.info(
        'Batch max size set to %s frames (~%.2f GB, %.1f%% of total RAM).' %
        (batch_size, batch_gb, frac_mem*100))

    if batch_size < max(lengths):
        raise enspara.exception.ImproperlyConfigured(
            'Batch size of %s was smaller than largest file (size %s).' %
            (batch_size, max(lengths)))

    batches = compute_batches(lengths, batch_size)

    assignments = []
    distances = []

    for i, batch_indices in enumerate(batches):
        tick = time.perf_counter()
        logger.info("Starting batch %s of %s", i+1, len(batches))
        batch_targets = [targets[i] for i in batch_indices]

        with timed("Loaded frames for batch in %.1f seconds", logger.info):
            batch_lengths, xyz = load_as_concatenated(
                [tfile for tfile, top, aids in batch_targets],
                lengths=[lengths[i] for i in batch_indices],
                args=[{'top': top, 'atom_indices': aids}
                      for t, top, aids in batch_targets],
                processes=n_procs)

        # mdtraj loads as float32, and load_as_concatenated should thus
        # also load as float32. This should _never_ be hit, but there might be
        # some platform-specific situation where double != float64?
        assert xyz.dtype.itemsize == DTYPE_BYTES

        trj = md.Trajectory(xyz, topology=example_center.top)

        with timed("Precentered trajectories in %.1f seconds", logger.debug):
            trj.center_coordinates()

        with timed("Assigned trajectories in %.1f seconds", logger.debug):
            batch_assignments, batch_distances = assign_to_nearest_center(
                    trj, centers, partial(md.rmsd, precentered=True))

        # clear memory of xyz and trj to allow cleanup to deallocate
        # these large arrays; may help with memory high-water mark
        with timed("Cleared array from memory in %.1f seconds", logger.debug):
            xyz_size = xyz.size
            del trj, xyz

        assignments.extend(partition_list(batch_assignments, batch_lengths))
        distances.extend(partition_list(batch_distances, batch_lengths))

        logger.info(
            "Finished batch %s of %s in %.1f seconds. Coordinates array had "
            "memory footprint of %.2f GB (of memory high-water mark %.2f/%.2f "
            "GB).",
            i, len(batches), time.perf_counter() - tick,
            xyz_size * DTYPE_BYTES / 1024**3,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            psutil.virtual_memory().total / 1024**3)

    return assignments, distances


def reassign(topologies, trajectories, atoms, centers, frac_mem=0.5):
    """Reassign a set of trajectories based on a subset of atoms and centers.

    Parameters
    ----------
    topologies : list
        List of topologies corresponding to the trajectories to be
        reassigned.
    trajectories : list of lists
        List of lists of tajectories to be loaded in batches and
        reassigned.
    atoms : list
        List of MDTraj atom query strings. Each string is applied to the
        corresponding topology to choose which atoms will be used for
        the reassignment.
    centers : md.Trajectory or list of trajectories
        The atoms representing the centers to reassign to.
    frac_mem : float, default=0.5
        The fraction of main RAM to use for trajectories. A lower number
        will mean more batches.
    """

    n_procs = enspara.util.parallel.auto_nprocs()

    # check input validity
    if len(topologies) != len(trajectories):
        raise enspara.exception.ImproperlyConfigured(
            "Number of topologies (%s) didn't match number of sets of "
            "trajectories (%s)." % (len(topologies), len(trajectories)))
    if len(topologies) != len(atoms):
        raise enspara.exception.ImproperlyConfigured(
            "Number of topologies (%s) didn't match number of atom selection "
            "strings (%s)." % (len(topologies), len(atoms)))

    # iteration across md.Trajectory is insanely slow. Do it only once here.
    if isinstance(centers, md.Trajectory):
        tick = time.perf_counter()
        logger.info('Centers are an md.Trajectory. Creating trj-list to '
                    'avoid repeated iteration.')
        # using in-place copies to reduce memory usage (and for speed)
        centers = [centers.slice(i, copy=False) for i in range(len(centers))]
        logger.info('Built trj list in %.1f seconds.',
                    time.perf_counter() - tick)

    # precenter centers (there will be many RMSD calcs here)
    for c in centers:
        c.center_coordinates()

    with timed("Reassignment took %.1f seconds.", logger.info):
        # build flat list of targets
        targets = []
        for topfile, trjfiles, atoms in zip(topologies, trajectories, atoms):
            t = md.load(topfile).top
            atom_ids = t.select(atoms)
            for trjfile in trjfiles:
                assert os.path.exists(trjfile)
                targets.append((trjfile, t, atom_ids))

        # determine trajectory length
        tick_sounding = time.perf_counter()
        logger.info("Sounding dataset of %s trajectories and %s topologies.",
                    sum(len(t) for t in trajectories), len(topologies))

        lengths = Parallel(n_jobs=n_procs)(
            delayed(sound_trajectory)(f) for f, _, _ in targets)

        logger.info("Sounded %s trajectories with %s frames (median length "
                    "%i frames) in %.1f seconds.",
                    len(lengths), sum(lengths), np.median(lengths),
                    time.perf_counter() - tick_sounding)

        assignments, distances = batch_reassign(
            targets, centers, lengths, frac_mem=frac_mem, n_procs=n_procs)

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
