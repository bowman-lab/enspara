"""The cluster app allows you to cluster your trajectories based on
distances from one another in a feature space. The entire protein or
specific residue locations can be used for the clustering. Parameters
such as the clustering algorithm and cluster radius can be specified.
The app will return information about cluster centers and frame
assignments. See the apps tab for more information.
"""

import sys
import argparse
import os
import logging
import itertools
import pickle
import json
from glob import glob

import numpy as np
import mdtraj as md

try:
    # this mpi will get overriden by the enspara mpi module in a few lines.
    from mpi4py.MPI import COMM_WORLD as mpi

    # this happens now and here becuase otherwise these changes to logging
    # don't propagate to enspara submodules.
    if mpi.Get_size() > 1:
        RANKSTR = "[Rank %s]" % mpi.Get_rank()
        logging.basicConfig(
            level=logging.DEBUG if mpi.Get_rank() == 0 else logging.INFO,
            format=('%(asctime)s ' + RANKSTR +
                    ' %(name)-26s %(levelname)-7s %(message)s'),
            datefmt='%m-%d-%Y %H:%M:%S')
        mpi_mode = True
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=('%(asctime)s %(name)-8s %(levelname)-7s %(message)s'),
            datefmt='%m-%d-%Y %H:%M:%S')
        mpi_mode = False

except ModuleNotFoundError:
    logging.basicConfig(
        level=logging.INFO,
        format=('%(asctime)s %(name)-8s %(levelname)-7s %(message)s'),
        datefmt='%m-%d-%Y %H:%M:%S')
    mpi_mode = False

from enspara.apps.util import readable_dir

from enspara import mpi
from enspara.cluster import KHybrid, KCenters, KMedoids
from enspara import ra
from enspara.util import load_as_concatenated
from enspara.util.log import timed
from enspara.util.parallel import auto_nprocs
from enspara.cluster import util 

from enspara.geometry import libdist

from enspara import exception
from enspara import mpi


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):

    FEATURE_DISTANCES = ['euclidean', 'manhattan']
    TRAJECTORY_DISTANCES = ['rmsd']
    ALGORITHMS = {
                  'kcenters': KCenters,
                  'khybrid': KHybrid,
                  'kmedoids': KMedoids
}

    parser = argparse.ArgumentParser(
        prog='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cluster a set (or several sets) of trajectories "
                    "into a single state space based upon RMSD.")

    # INPUTS
    input_args = parser.add_argument_group("Input Settings")
    input_data_group = parser.add_mutually_exclusive_group(required=True)
    input_data_group.add_argument(
        "--features", nargs='+',
        help="The h5 file containin observations and features.")
    input_data_group.add_argument(
        '--trajectories', nargs="+", action='append',
        help="List of paths to aligned trajectory files to cluster. "
             "All file types that MDTraj supports are supported here.")
    input_args.add_argument(
        '--topology', action='append', dest='topologies',
        help="The topology file for the trajectories. This flag must be"
             " specified once for each instance of the --trajectories "
             "flag. The first --topology flag is taken to be the "
             "topology to use for the first instance of the "
             "--trajectories flag, and so forth.")

    # PARAMETERS
    cluster_args = parser.add_argument_group("Clustering Settings")
    cluster_args.add_argument(
        '--algorithm', required=True,
        choices=["khybrid", "kcenters", "kmedoids"],
        help="The clustering algorithm to use.")
    cluster_args.add_argument(
        '--atoms', action="append",
        help="When clustering trajectories, specifies which atoms from the "
             "trajectories (using MDTraj atom-selection syntax) to cluster "
             "based upon. Specify once to apply this selection to every set "
             "of trajectories specified by the --trajectories flag, or "
             "once for each different topology (i.e. the number of "
             "times --trajectories and --topology was specified.)")
    cluster_args.add_argument(
        '--cluster-radius', default=None, type=float,
        help="Produce clusters with a maximum distance to cluster "
             "center of this value.")
    cluster_args.add_argument(
        '--cluster-number', default=None, type=int,
        help="Produce at least this number of clusters.")
    cluster_args.add_argument(
        "--cluster-distance", default=None,
        choices=FEATURE_DISTANCES + TRAJECTORY_DISTANCES,
        help="The metric for measuring distances. Some metrics (e.g. rmsd) "
             "only apply to trajectories, and others only to features.")
    cluster_args.add_argument(
        "--cluster-iterations", default=None, type=int,
        help="The number of refinement iterations to perform. This is only "
             "relevant to khybrid clustering.")
    cluster_args.add_argument(
        "--save_intermediates", default=False, type=bool,
        help="Save intermediate clustering results when doing khybrid? ")
    cluster_args.add_argument(
        "--init-center-inds", default=None, type=str,
        help="Path to a .npy file that is a list giving the position of "
             "each cluster center in traj. Useful for restarting clustering.")
    cluster_args.add_argument(
        "--init-assignments", default=None, type=str,
        help="Path to an .h5 file that indicates which cluster center each "
             "data point is closest to. Useful for restarting clustering")
    cluster_args.add_argument(
        "--init-distances", default=None, type=str,
        help="Path to an .h5 file that indicates how far each data point is"
             "to its cluster center. Useful for restarting clustering")
    cluster_args.add_argument(
        '--subsample', default=1, type=int,
        help="Take only every nth frame when loading trajectories. "
             "1 implies no subsampling.")

    # OUTPUT
    output_args = parser.add_argument_group("Output Settings")
    output_args.add_argument(
        '--no-reassign', default=False, action='store_true',
        help="Do not do a reassigment step. Ignored if --subsample is "
             "not supplied or 1.")

    output_args.add_argument(
        '--distances', required=True, action=readable_dir,
        help="The location to write the distances file.")
    output_args.add_argument(
        '--center-features', required=True, action=readable_dir,
        help="The location to write the cluster center structures.")
    output_args.add_argument(
        '--assignments', required=True, action=readable_dir,
        help="The location to write assignments of frames to clusters.")
    output_args.add_argument(
        "--center-indices", required=False, action=readable_dir,
        help="Location for cluster center indices output (pickle).")

    args = parser.parse_args(argv[1:])

    if args.features:
        args.features = util.expand_files([args.features])[0]

        if args.cluster_distance in FEATURE_DISTANCES:
            args.cluster_distance = getattr(libdist, args.cluster_distance)
        else:
            raise exception.ImproperlyConfigured(
                "The given distance (%s) is not compatible with features." %
                args.cluster_distance)

        if args.subsample != 1 and len(args.features) == 1:
                raise exception.ImproperlyConfigured(
                    "Subsampling is not supported for h5 inputs.")

        # TODO: not necessary if mutually exclusvie above works
        if args.trajectories:
            raise exception.ImproperlyConfigured(
                "--features and --trajectories are mutually exclusive. "
                "Either trajectories or features, not both, are clustered.")
        if args.topologies:
            raise exception.ImproperlyConfigured(
                "When --features is specified, --topology is unneccessary.")
        if args.atoms:
            raise exception.ImproperlyConfigured(
                "Option --atoms is only meaningful when clustering "
                "trajectories.")
        if not args.cluster_distance:
            raise exception.ImproperlyConfigured(
                "Option --cluster-distance is required when clustering "
                "features.")

    elif args.trajectories and args.topologies:
        args.trajectories = util.expand_files(args.trajectories)

        if not args.cluster_distance or args.cluster_distance == 'rmsd':
            args.cluster_distance = md.rmsd
        else:
            raise exception.ImproperlyConfigured(
                "Option --cluster-distance must be rmsd when clustering "
                "trajectories.")

        if not args.atoms:
            raise exception.ImproperlyConfigured(
                "Option --atoms is required when clustering trajectories.")
        elif len(args.atoms) == 1:
            args.atoms = args.atoms * len(args.trajectories)
        elif len(args.atoms) != len(args.trajectories):
            raise exception.ImproperlyConfigured(
                "Flag --atoms must be provided either once (selection is "
                "applied to all trajectories) or the same number of times "
                "--trajectories is supplied.")

        if len(args.topologies) != len(args.trajectories):
            raise exception.ImproperlyConfigured(
                "The number of --topology and --trajectory flags must agree.")

    else:
        # CANNOT CLUSTER
        raise exception.ImproperlyConfigured(
            "Either --features or both of --trajectories and --topologies "
            "are required.")

    if args.cluster_radius is None and args.cluster_number is None:
        raise exception.ImproperlyConfigured(
            "At least one of --cluster-radius and --cluster-number is "
            "required to cluster.")

    args.Clusterer = ALGORITHMS[args.algorithm]
    if args.Clusterer is KCenters:
        if args.cluster_iterations is not None:
            raise exception.ImproperlyConfigured(
                "--cluster-iterations only has an effect when using an "
                "interative clustering scheme (e.g. khybrid).")
    if args.Clusterer is KMedoids:
        if args.cluster_radius is not None:
            raise exception.ImproperlyConfigured(
                "--cluster-radius only has an effect when using kcenters"
                " or khybrid.")
    else:
        restart_arg_names = [args.init_center_inds, args.init_distances,
            args.init_assignments]
        for name in restart_arg_names:
            if name:
                raise exception.ImproperlyConfigured(
                    "--init-center-inds, --init-distances, and"
                    "--init-assignments are only implemented for kmedoids")


    if args.no_reassign and args.subsample == 1:
        logger.warn("When subsampling is 1 (or unspecified), "
                    "--no-reassign has no effect.")
    if not args.no_reassign and mpi_mode and args.subsample > 1:
        logger.warn("Reassignment is suppressed in MPI mode.")
        args.no_reassign = True

    if args.trajectories:
        if os.path.splitext(args.center_features)[1] == '.h5':
            logger.warn(
                "You provided a centers file (%s) that looks like it's "
                "an h5... centers are saved as pickle. Are you sure this "
                "is what you want?")
    else:
        if os.path.splitext(args.center_features)[1] != '.npy':
            logger.warn(
                "You provided a centers file (%s) that looks like it's not "
                "an npy, but this is how they are saved. Are you sure "
                "this is what you want?" %
                os.path.basename(args.center_features))

    return args


def main(argv=None):

    args = process_command_line(argv)

    # note that in MPI mode, lengths will be global, whereas data will
    # be local (i.e. only this node's data).
    lengths, data = util.load_trjs_or_features(args)

    kwargs = {}
    if args.cluster_iterations is not None:
        if args.Clusterer is KHybrid:
            kwargs['kmedoids_updates'] = int(args.cluster_iterations)
        elif args.Clusterer is KMedoids:
            kwargs['n_iters'] = int(args.cluster_iterations)
        if args.Clusterer is not KCenters:
            kwargs['args']=args
            kwargs['lengths']=lengths

    #kmedoids doesn't need a cluster radius, but kcenters does
    if args.cluster_radius is not None:
        kwargs['cluster_radius']=args.cluster_radius
        kwargs['mpi_mode']=mpi_mode

    clustering = args.Clusterer(
        metric=args.cluster_distance,
        n_clusters=args.cluster_number,
        **kwargs)
    
    # Note to self: Need to implement restarts for KCenters as well
    kwargs_restart = {}
    if args.Clusterer is KMedoids:
        if args.init_distances:
            _, kwargs_restart['distances'] = \
                 mpi.io.load_h5_as_striped(args.init_distances)
        if args.init_assignments:
            kwargs_restart['X_lengths'], kwargs_restart['assignments'] = \
                mpi.io.load_h5_as_striped(args.init_assignments)
        if args.save_intermediates:
            kwargs_restart['args']=args
        if args.init_center_inds:
            kwargs_restart['cluster_center_inds'] = \
                np.load(args.init_center_inds) 
        clustering.fit(data,**kwargs_restart)
    else:
        clustering.fit(data)
    # release the RAM held by the trajectories (we don't need it anymore)
    del data

    logger.info(
        "Clustered %s frames into %s clusters in %s seconds.",
        sum(lengths), len(clustering.centers_), clustering.runtime_)

    result = clustering.result_
    if mpi_mode:
        local_ctr_inds, local_dists, local_assigs = \
            result.center_indices, result.distances, result.assignments

        with timed("Reassembled dist and assign arrays in %.2f sec",
                   logging.info):
            all_dists = mpi.ops.assemble_striped_ragged_array(
                local_dists, lengths)
            all_assigs = mpi.ops.assemble_striped_ragged_array(
                local_assigs, lengths)
            ctr_inds = mpi.ops.convert_local_indices(local_ctr_inds, lengths)

        result = util.ClusterResult(
            center_indices=ctr_inds,
            distances=all_dists,
            assignments=all_assigs,
            centers=result.centers)
    result = result.partition(lengths)

    if mpi.rank() == 0:
        with timed("Wrote center indices in %.2f sec.", logger.info):
            util.write_centers_indices(
                args.center_indices,
                [(t, f * args.subsample) for t, f in result.center_indices])
        with timed("Wrote center structures in %.2f sec.", logger.info):
            util.write_centers(result, args)
        util.write_assignments_and_distances_with_reassign(result, args)

    mpi.comm.barrier()

    logger.info("Success! Data can be found in %s.",
                os.path.dirname(args.distances))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
