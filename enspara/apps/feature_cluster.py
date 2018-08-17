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
from enspara.cluster import KHybrid, KCenters
from enspara.util import array as ra
from enspara.geometry.libdist import euclidean

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_command_line(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--features", required=True,
        help="The h5 file containin observations and features.")

    parser.add_argument(
        "--cluster-algorithm", required=True, choices=['khybrid', 'kcenters'],
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

    if not args.overwrite:
        for outf in ['assignments', 'distances', 'center_indices',
                     'cluster_centers']:
            fname = getattr(args, outf)
            if os.path.isfile(fname):
                raise FileExistsError(
                    ("The file '%s' already exists. To overwrite, pass "
                     "--overwrite.") % fname)

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

    logger.info(
        "Loaded data from %s with shape %s",
        args.features, features.shape)

    if args.cluster_algorithm == 'khybrid':
        clustering = KHybrid(
            metric=args.cluster_distance,
            cluster_radius=args.cluster_radius,
            kmedoids_updates=args.kmedoids_updates)
    elif args.cluster_algorithm == 'kcenters':
        clustering = KCenters(
            cluster_radius=args.cluster_radius,
            metric=args.cluster_distance)

    logger.info("Clustering with %s", clustering)

    clustering.fit(features._data)

    result = clustering.result_.partition(features.lengths)
    del features

    ra.save(args.distances, result.distances)
    logger.info("Wrote distances with shape %s to %s",
                result.distances.shape, args.distances)

    ra.save(args.assignments, result.assignments)
    logger.info("Wrote assignments with shape %s to %s",
                result.assignments.shape, args.cluster_centers)

    ra.save(args.cluster_centers, result.centers)
    logger.info("Wrote cluster_centers with shape %s to %s",
                result.centers.shape, args.cluster_centers)

    pickle.dump(result.center_indices, open(args.center_indices, 'wb'))
    logger.info("Wrote %s center_indices with shape to %s",
                len(result.center_indices), args.center_indices)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
