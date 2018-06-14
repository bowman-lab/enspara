import sys
import argparse
import pickle
import os
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s %(name)-8s %(levelname)-7s %(message)s'),
    datefmt='%m-%d-%Y %H:%M:%S')

from enspara import exception
from enspara.apps.util import readable_dir
from enspara.cluster import KHybrid
from enspara.util import array as ra
from enspara.geometry.libdist import euclidean


def process_command_line(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--features", required=True,
        help="The h5 file containin observations and features.")

    parser.add_argument(
        "--cluster-algorithm", required=True, choices=['khybrid'],
        help="The algorithm to use for clustering.")
    parser.add_argument(
        "--cluster-radius", required=True, type=float,
        help="The radius to cluster to.")
    parser.add_argument(
        "--cluster-distance", default='euclidean',
        choices=['euclidean', 'manhattan'],
        help="The metric for measuring distances")
    parser.add_argument(
        "--kmedoids-updates", default=5, type=int,
        help="Number of iterations of kemedoids to perform when "
             "refining the kcenters cluster assignments. Valid only "
             "when --cluster-algorithm is 'khybrid'.")

    parser.add_argument(
        "--assignments", action=readable_dir, required=True,
        help="Location for assignments output (h5 file).")
    parser.add_argument(
        "--distances", action=readable_dir, required=True,
        help="Location for distances output (h5 file).")
    parser.add_argument(
        "--center-indices", action=readable_dir, required=True,
        help="Location for indices output (pickle).")
    parser.add_argument(
        "--cluster-centers", action=readable_dir, required=True,
        help="Location for cluster centers output (h5 file). These are "
             "the feature vectors (from --features) that are found to "
             "be cluster centers.")

    parser.add_argument(
        "--overwrite", action='store_true',
        help="Flag to overwrite when output file exists.")

    args = parser.parse_args(argv[1:])

    if args.cluster_distance.lower() == 'euclidean':
        args.cluster_distance = euclidean
    elif args.cluster_distance.lower() == 'manhattan':
        args.cluster_distance = diff_manhattan

    a_file_exists = any(
        os.path.isfile(getattr(args, o)) for o in
        ['assignments', 'distances', 'center_indices', 'cluster_centers'])
    if a_file_exists and not args.overwrite:
        raise FileExistsError

    assert args.cluster_algorithm.lower() == 'khybrid'

    return args


def diff_manhattan(trj, ref):
    return np.abs(trj - ref)


def main(argv=None):
    args = process_command_line(argv)

    try:
        features = ra.load(
            args.features, keys=...)
    except exception.DataInvalid:
        features = ra.load(args.features)

    clustering = KHybrid(
        metric=args.cluster_distance,
        cluster_radius=args.cluster_radius,
        kmedoids_updates=args.kmedoids_updates)

    clustering.fit(features._data)

    result = clustering.result_.partition(features.lengths)
    del features

    ra.save(args.distances, result.distances)
    ra.save(args.assignments, result.assignments)
    ra.save(args.cluster_centers, result.centers)
    pickle.dump(result.center_indices, open(args.center_indices, 'wb'))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
