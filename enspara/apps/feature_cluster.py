import sys
import argparse
import pickle

from mdtraj import io
from msmbuilder.libdistance import cdist

from enspara import exception
from enspara.cluster import KHybrid
from enspara.util import array as ra


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
        "--assignments", required=True,
        help="Location for assignments output (h5 file).")
    parser.add_argument(
        "--distances", required=True,
        help="Location for distances output (h5 file).")
    parser.add_argument(
        "--center-indices", required=True,
        help="Location for indices output (pickle).")
    parser.add_argument(
        "--cluster-centers", required=True,
        help="Location for cluster centers output (h5 file). These are "
             "the feature vectors (from --features) that are found to "
             "be cluster centers.")

    args = parser.parse_args(argv[1:])

    if args.cluster_distance.lower() == 'euclidean':
        args.cluster_distance = diff_euclidean
    elif args.cluster_distance.lower() == 'manhattan':
        args.cluster_distance = diff_manhattan

    assert args.cluster_algorithm.lower() == 'khybrid'

    return args


def diff_euclidean(trj, ref):
    return cdist(ref.reshape(1, -1), trj, 'euclidean')[0]


def diff_manhattan(trj, ref):
    return trj - ref


def main(argv=None):
    args = process_command_line(argv)

    keys = io.loadh(args.features).keys()

    try:
        features = ra.load(
            args.features, keys=sorted(keys, key=lambda x: x.split('_')[-1]))
    except exception.DataInvalid:
        features = ra.load(args.features)

    clustering = KHybrid(
        metric=args.cluster_distance,
        cluster_radius=args.cluster_radius,
        kmedoids_updates=args.kmedoids_updates)

    clustering.fit(features._data)

    result = clustering.result_.partition(features.lengths)

    ra.save(args.distances, result.distances)
    ra.save(args.assignments, result.assignments)
    ra.save(args.cluster_centers, result.centers)
    pickle.dump(result.center_indices, open(args.center_indices, 'wb'))

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
