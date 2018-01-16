import sys
import argparse
import logging
import pickle
import time
import resource
import psutil
import warnings

import multiprocessing as mp

from glob import glob

import numpy as np
import mdtraj as md

from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

RANKSTR = "[Rank %s]" % RANK

logging.basicConfig(
    level=logging.DEBUG if RANK == 0 else logging.INFO,
    format=('%(asctime)s ' + RANKSTR +
            ' %(name)-8s %(levelname)-7s %(message)s'),
    datefmt='%m-%d-%Y %H:%M:%S')

from enspara.util import array as ra
from enspara.util.log import timed

from enspara.cluster.util import load_frames, partition_indices
from enspara.cluster.kcenters import kcenters_mpi
from enspara.cluster.kmedoids import _kmedoids_update_mpi

from enspara.apps.util import readable_dir
from enspara.util.load import load_as_concatenated


def process_command_line(argv):
    '''Parse the command line and do a first-pass on processing them into a
    format appropriate for the rest of the script.'''

    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--trajectories", required=True,
        help="Trajectories to cluster.")
    parser.add_argument(
        "--topology", required=True,
        help="Topology of files.")
    parser.add_argument(
        "--selection", required=True,
        help="MDTraj DSL for atoms to be clustered.")
    parser.add_argument(
        "--subsample", default=1, metavar='N', type=int,
        help="Load only every Nth frame.")

    parser.add_argument(
        "--cluster-radii", required=True, type=float, nargs="+",
        help="Minimum maximum distance between any point and a center. "
             "More than one value can be supplied, and a kcenters set"
             "will be saved at each value")
    parser.add_argument(
        "--kmedoids-iters", default=5, metavar='I', type=int,
        help="Number of iterations of kmedoids to run.")

    parser.add_argument(
        '--processes', default=mp.cpu_count(), type=int,
        help="Number processes to use (on each node) for loading and "
             "clustering.")

    parser.add_argument(
        "--distances", action=readable_dir,
        help="Path to output distances h5.")
    parser.add_argument(
        "--assignments", action=readable_dir,
        help="Path to output assignments h5.")
    parser.add_argument(
        "--center-indices", required=True, action=readable_dir,
        help="Path to output center-indices pickle.")
    parser.add_argument(
        "--center-structures", required=True, action=readable_dir,
        help="Path to output center-structures h5.")

    args = parser.parse_args(argv[1:])

    args.trajectories = glob(args.trajectories)

    if args.subsample > 1 and args.distances:
        warnings.warn('When subsampling > 1, distances are also subsampled.')
    if args.subsample > 1 and args.assignments:
        warnings.warn('When subsampling > 1, assignments are also subsampled.')

    return args


def staged_kcenters(trjs, dist_func, cluster_radii, n_clusters,
                    stage_callback=None):

    local_dists = np.empty(shape=(len(trjs)), dtype=np.float32)
    local_dists.fill(np.inf)
    local_assigs = np.empty(shape=(len(trjs)), dtype=np.int)
    local_assigs.fill(0)
    local_ctr_inds = []
    cur_radius = np.inf

    for radius in sorted(cluster_radii):
        while len(local_ctr_inds) < n_clusters and cur_radius > radius:
            cur_radius, local_dists, local_assigs, local_ctr_inds = \
                _kcenters_iteration_mpi(
                    trjs, md.rmsd,
                    distances=local_dists,
                    assignments=local_assigs,
                    center_inds=local_ctr_inds)

        stage_callback(cur_radius, local_dists, local_assigs,
                       local_ctr_inds)

    return local_dists, local_assigs, local_ctr_inds


def distribute_lengths(n_files, local_lengths):

    global_lengths = np.zeros((n_files,), dtype=local_lengths.dtype) - 1

    with timed("Distributed length information in %s", logging.info):
        logging.info("%s", local_lengths)
        for i in range(SIZE):
            global_lengths[i::SIZE] = COMM.bcast(local_lengths, root=i)

    assert np.all(global_lengths > 0)

    return global_lengths


def assemble_per_frame_array(global_lengths, local_array):

    global_array = np.zeros((np.sum(global_lengths),)) - 1
    global_ra = ra.RaggedArray(global_array, lengths=global_lengths)

    for rank in range(SIZE):
        rank_array = COMM.bcast(local_array, root=rank)
        rank_ra = ra.RaggedArray(
            rank_array, lengths=global_lengths[rank::SIZE])

        global_ra[rank::SIZE] = rank_ra

    assert np.all(global_ra._data) >= 0

    return global_ra._data


def convert_center_indices(local_ctr_inds, global_lengths):

    global_indexing = np.arange(np.sum(global_lengths))
    file_origin_ra = ra.RaggedArray(global_indexing, lengths=global_lengths)

    ctr_inds = []
    for rank, local_frameid in local_ctr_inds:
        global_frameid = file_origin_ra[rank::SIZE].flatten()[local_frameid]
        ctr_inds.append(global_frameid)

    return ctr_inds


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    top = md.load(args.topology).top
    atom_ids = top.select(args.selection)

    logging.info("Running with %s total workers.", SIZE)

    logging.info(
        "Loading trajectories [%s::%s]; selection == '%s' w/ "
        "subsampling %s", RANK, SIZE, args.selection, args.subsample)

    with timed("load_as_concatenated took %.2f sec", logging.info):
        local_lengths, my_xyz = load_as_concatenated(
            filenames=args.trajectories[RANK::SIZE],
            top=top,
            atom_indices=atom_ids,
            stride=args.subsample,
            processes=args.processes)
        local_lengths = np.array(local_lengths, dtype=int)

    with timed("Turned over array in %.2f min", logging.info):
        xyz = my_xyz.copy()
        del my_xyz
        my_xyz = xyz

    logging.info(
        "Loaded %s frames in %s trjs (%.2fG).",
        len(my_xyz), len(local_lengths),
        my_xyz.data.nbytes / 1024**3)

    trjs = md.Trajectory(my_xyz, topology=top.subset(atom_ids))

    global_lengths = distribute_lengths(len(args.trajectories), local_lengths)

    logging.info(
        "Beginning kcenters clustering with memory footprint of %.2fG "
        "RAM (coords are %.2fG; total VRAM is %.2fG)",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        trjs.xyz.nbytes / 1024**3,
        psutil.virtual_memory().total / 1024**3)

    if len(args.cluster_radii) > 1:
        raise NotImplementedError(
            "Multiple cluster radii are not yet supported")

    tick = time.perf_counter()
    local_dists, local_assigs, local_ctr_inds = kcenters_mpi(
        trjs, md.rmsd, dist_cutoff=args.cluster_radii[0])
    tock = time.perf_counter()

    logging.info(
        "Finished kcenters clustering using %.2fG RAM (coords are "
        "%.2fG) in %.2f min.",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        trjs.xyz.nbytes / 1024**3,
        (tock - tick)/60)

    for i in range(args.kmedoids_iters):
        with timed("KMedoids iteration {i} took %.2f sec".format(i=i),
                   logging.info):
            local_ctr_inds, local_assigs, local_dists = _kmedoids_update_mpi(
                trjs, md.rmsd, local_ctr_inds, local_assigs, local_dists)

    with timed("Reassembled dist and assign arrays in %.2f sec", logging.info):
        all_dists = assemble_per_frame_array(global_lengths, local_dists)
        all_assigs = assemble_per_frame_array(global_lengths, local_assigs)
        ctr_inds = convert_center_indices(local_ctr_inds, global_lengths)
        ctr_inds = partition_indices(ctr_inds, global_lengths)

    if RANK == 0:
        logging.info("Dumping center indices to %s", args.center_indices)
        with open(args.center_indices, 'wb') as f:
            pickle.dump(ctr_inds, f)

        if args.distances:
            ra.save(args.distances,
                    ra.RaggedArray(all_dists, lengths=global_lengths))
        if args.assignments:
            ra.save(args.assignments,
                    ra.RaggedArray(all_assigs, lengths=global_lengths))

        centers = load_frames(
            args.trajectories,
            ctr_inds,
            stride=args.subsample,
            top=md.load(args.topology).top)

        with open(args.center_structures, 'wb') as f:
            pickle.dump(centers, f)
        logging.info("Wrote %s centers to %s", len(centers),
                     args.center_structures)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
