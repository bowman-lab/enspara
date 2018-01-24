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

import mdtraj as md

from sklearn.utils import check_random_state

from mpi4py.MPI import COMM_WORLD as COMM
RANKSTR = "[Rank %s]" % COMM.Get_rank()

logging.basicConfig(
    level=logging.DEBUG if COMM.Get_rank() == 0 else logging.INFO,
    format=('%(asctime)s ' + RANKSTR +
            ' %(name)-26s %(levelname)-7s %(message)s'),
    datefmt='%m-%d-%Y %H:%M:%S')

from enspara.mpi import MPI_RANK, MPI_SIZE
from enspara import mpi

from enspara.cluster.util import load_frames, partition_indices
from enspara.cluster.kcenters import kcenters_mpi
from enspara.cluster.kmedoids import _kmedoids_pam_update

from enspara.apps.util import readable_dir

from enspara.util import array as ra
from enspara.util.log import timed


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
        "--kmedoids-iters", default=5, type=int,
        help="Number of iterations of kmedoids to run.")

    parser.add_argument(
        '--processes', default=mp.cpu_count(), type=int,
        help="Number processes to use (on each node) for loading and "
             "clustering.")
    parser.add_argument(
        '--random-state', default=None, type=int,
        help="Give a fixed random seed to ensure reproducible results")

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
    args.radom_state = check_random_state(args.random_state)

    if args.subsample > 1 and args.distances:
        warnings.warn('When subsampling > 1, distances are also subsampled.')
    if args.subsample > 1 and args.assignments:
        warnings.warn('When subsampling > 1, assignments are also subsampled.')

    return args


def main(argv=None):
    '''Run the driver script for this module. This code only runs if we're
    being run as a script. Otherwise, it's silent and just exposes methods.'''
    args = process_command_line(argv)

    top = md.load(args.topology).top
    atom_ids = top.select(args.selection)

    logging.info("Running with %s total workers.", MPI_SIZE)

    logging.info(
        "Loading trajectories [%s::%s]; selection == '%s' w/ "
        "subsampling %s", MPI_RANK, MPI_SIZE, args.selection, args.subsample)

    with timed("load_as_concatenated took %.2f sec", logging.info):
        global_lengths, my_xyz = mpi.io.load_as_striped(
            filenames=args.trajectories,
            top=top,
            atom_indices=atom_ids,
            stride=args.subsample,
            processes=args.processes)

    with timed("Turned over array in %.2f min", logging.info):
        xyz = my_xyz.copy()
        del my_xyz
        my_xyz = xyz

    logging.info(
        "Loaded %s frames in %s trjs (%.2fG).",
        len(my_xyz), len(args.trajectories) // MPI_SIZE,
        my_xyz.data.nbytes / 1024**3)

    trjs = md.Trajectory(my_xyz, topology=top.subset(atom_ids))

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
            local_ctr_inds, local_dists, local_assigs = _kmedoids_pam_update(
                X=trjs, metric=md.rmsd,
                medoid_inds=local_ctr_inds,
                assignments=local_assigs,
                distances=local_dists,
                random_state=args.random_state)

    with timed("Reassembled dist and assign arrays in %.2f sec", logging.info):
        all_dists = mpi.ops.assemble_striped_ragged_array(
            local_dists, global_lengths)
        all_assigs = mpi.ops.assemble_striped_ragged_array(
            local_assigs, global_lengths)

        ctr_inds = mpi.ops.convert_local_indices(
            local_ctr_inds, global_lengths)
        ctr_inds = partition_indices(ctr_inds, global_lengths)

    if MPI_RANK == 0:
        logging.info("Dumping center indices to %s", args.center_indices)

        with open(args.center_indices, 'wb') as f:
            pickle.dump(
                [(trj, frame*args.subsample) for trj, frame in ctr_inds], f)

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
