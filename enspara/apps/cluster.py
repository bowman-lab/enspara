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

from enspara.apps.reassign import reassign
from enspara.apps.util import readable_dir

from enspara import mpi
from enspara.cluster import KHybrid, KCenters
from enspara.util import array as ra
from enspara.util import load_as_concatenated
from enspara.util.log import timed
from enspara.util.parallel import auto_nprocs
from enspara.cluster.util import load_frames, partition_indices, ClusterResult

from enspara.geometry import libdist

from enspara import exception
from enspara import mpi


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_command_line(argv):

    FEATURE_DISTANCES = ['euclidean', 'manhattan']
    TRAJECTORY_DISTANCES = ['rmsd']

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
        '--algorithm', required=True, choices=["khybrid", "kcenters"],
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
        args.features = expand_files([args.features])[0]

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
        args.trajectories = expand_files(args.trajectories)

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

    if args.algorithm == 'kcenters':
        args.Clusterer = KCenters
        if args.cluster_iterations is not None:
            raise exception.ImproperlyConfigured(
                "--cluster-iterations only has an effect when using an "
                "interative clustering scheme (e.g. khybrid).")
    elif args.algorithm == 'khybrid':
        args.Clusterer = KHybrid

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


def expand_files(pgroups):
    expanded_pgroups = []
    for pgroup in pgroups:
        expanded_pgroups.append([])
        for p in pgroup:
            expanded_pgroups[-1].extend(sorted(glob(p)))
    return expanded_pgroups


def load_features(features, stride):
    try:
        if len(features) == 1:
            with timed("Loading features took %.1f s.", logger.info):
                lengths, data = mpi.io.load_h5_as_striped(features[0], stride)

        else:  # and len(features) > 1
            with timed("Loading features took %.1f s.", logger.info):
                lengths, data = mpi.io.load_npy_as_striped(features, stride)

        with timed("Turned over array in %.2f min", logger.info):
            tmp_data = data.copy()
            del data
            data = tmp_data
    except MemoryError:
        logger.error(
            "Ran out of memory trying to allocate features array"
            " from file %s", features[0])
        raise

    logger.info("Loaded %s trajectories with %s frames with stride %s.",
                len(lengths), len(data), stride)

    return lengths, data


def load_trajectories(topologies, trajectories, selections, stride, processes):

    for top, selection in zip(topologies, selections):
        sentinel_trj = md.load(top)
        try:
            # noop, but causes fast-fail w/bad args.atoms
            sentinel_trj.top.select(selection)
        except:
            raise exception.ImproperlyConfigured((
                "The provided selection '{s}' didn't match the topology "
                "file, {t}").format(s=selection, t=top))

    flat_trjs = []
    configs = []
    n_inds = None

    for topfile, trjset, selection in zip(topologies, trajectories,
                                          selections):
        top = md.load(topfile).top
        indices = top.select(selection)

        if n_inds is not None:
            if n_inds != len(indices):
                raise exception.ImproperlyConfigured(
                    ("Selection on topology %s selected %s atoms, but "
                     "other selections selected %s atoms.") %
                    (topfile, len(indices), n_inds))
        n_inds = len(indices)

        for trj in trjset:
            flat_trjs.append(trj)
            configs.append({
                'top': top,
                'stride': stride,
                'atom_indices': indices,
            })

    logger.info(
        "Loading %s trajectories with %s atoms using %s processes "
        "(subsampling %s)",
        len(flat_trjs), len(top.select(selection)), processes, stride)
    assert len(top.select(selection)) > 0, "No atoms selected for clustering"

    with timed("Loading took %.1f sec", logger.info):
        lengths, xyz = mpi.io.load_trajectory_as_striped(
            flat_trjs, args=configs, processes=auto_nprocs())

    with timed("Turned over array in %.2f min", logger.info):
        tmp_xyz = xyz.copy()
        del xyz
        xyz = tmp_xyz

    logger.info("Loaded %s frames.", len(xyz))

    return lengths, xyz, top.subset(top.select(selection))


def load_asymm_frames(center_indices, trajectories, topology, subsample):

    frames = []
    begin_index = 0
    for topfile, trjset in zip(topology, trajectories):
        end_index = begin_index + len(trjset)
        target_centers = [c for c in center_indices
                          if begin_index <= c[0] < end_index]

        try:
            subframes = load_frames(
                list(itertools.chain(*trajectories)),
                target_centers,
                top=md.load(topfile).top,
                stride=subsample)
        except exception.ImproperlyConfigured:
            logger.error('Failure to load cluster centers %s for topology %s',
                         topfile, target_centers)
            raise

        frames.extend(subframes)
        begin_index += len(trjset)

    return frames


def load_trjs_or_features(args):

    if args.features:
        with timed("Loading features took %.1f s.", logger.info):
            lengths, data = load_features(args.features, stride=args.subsample)
    else:
        assert args.trajectories
        assert len(args.trajectories) == len(args.topologies)

        targets = {os.path.basename(topf): "%s files" % len(trjfs)
                   for topf, trjfs
                   in zip(args.topologies, args.trajectories)
                   }
        logger.info("Beginning clustering; targets:\n%s",
                    json.dumps(targets, indent=4))

        with timed("Loading trajectories took %.1f s.", logger.info):
            lengths, xyz, select_top = load_trajectories(
                args.topologies, args.trajectories, selections=args.atoms,
                stride=args.subsample, processes=auto_nprocs())

        logger.info("Clustering using %s atoms matching '%s'.", xyz.shape[1],
                    args.atoms)

        # md.rmsd requires an md.Trajectory object, so wrap `xyz` in
        # the topology.
        data = md.Trajectory(xyz=xyz, topology=select_top)

    return lengths, data


def write_centers_indices(path, indices):
    if path:
        with open(path, 'wb') as f:
            np.save(f, indices)
    else:
        logger.info("--center-indices not provided, not writing center "
                    "indices to file.")


def write_centers(result, args):
    if args.features:
        np.save(args.center_features, result.centers)
    else:
        outdir = os.path.dirname(args.center_features)
        logger.info("Saving cluster centers at %s", outdir)

        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass

        centers = load_asymm_frames(result.center_indices, args.trajectories,
                                    args.topologies, args.subsample)
        with open(args.center_features, 'wb') as f:
            pickle.dump(centers, f)


def write_assignments_and_distances_with_reassign(result, args):
    if args.subsample == 1:
        logger.debug("Subsampling was 1, not reassigning.")
        ra.save(args.distances, result.distances)
        ra.save(args.assignments, result.assignments)
    elif not args.no_reassign:
        logger.debug("Reassigning data from subsampling of %s", args.subsample)
        assig, dist = reassign(
            args.topologies, args.trajectories, args.atoms,
            centers=result.centers)

        ra.save(args.distances, dist)
        ra.save(args.assignments, assig)
    else:
        logger.debug("Got --no-reassign, not doing reassigment")


def main(argv=None):

    args = process_command_line(argv)

    # note that in MPI mode, lengths will be global, whereas data will
    # be local (i.e. only this node's data).
    lengths, data = load_trjs_or_features(args)

    kwargs = {}
    if args.cluster_iterations is not None:
        kwargs['kmedoids_updates'] = int(args.cluster_iterations)

    clustering = args.Clusterer(
        metric=args.cluster_distance,
        n_clusters=args.cluster_number,
        cluster_radius=args.cluster_radius,
        mpi_mode=mpi_mode,
        **kwargs)

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

        result = ClusterResult(
            center_indices=ctr_inds,
            distances=all_dists,
            assignments=all_assigs,
            centers=result.centers)
    result = result.partition(lengths)

    if mpi.rank() == 0:
        with timed("Wrote center indices in %.2f sec.", logger.info):
            write_centers_indices(
                args.center_indices,
                [(t, f * args.subsample) for t, f in result.center_indices])
        with timed("Wrote center structures in %.2f sec.", logger.info):
            write_centers(result, args)
        write_assignments_and_distances_with_reassign(result, args)

    mpi.comm.barrier()

    logger.info("Success! Data can be found in %s.",
                os.path.dirname(args.distances))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
