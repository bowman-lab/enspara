# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import sys

from .kcenters import _kcenters_helper
from .kmedoids import _kmedoids_update
from .utils import requires_concatenated_trajectories


def _hybrid_medoids_update(
        traj, distance_method, cluster_center_inds, assignments, distances,
        output):

    proposed_center_inds, proposed_assignments, proposed_distances =\
        _kmedoids_update(traj, distance_method, cluster_center_inds,
                         assignments, distances, output)

    max_orig_dist_to_center = distances.max()
    max_proposed_dist_to_center = proposed_distances.max()
    if max_proposed_dist_to_center <= max_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances


@requires_concatenated_trajectories
def hybrid(
        traj, distance_method, n_iters=5, n_clusters=None, dist_cutoff=None,
        random_first_center=False, cluster_centers=None, output=sys.stdout):

    cluster_center_inds, assignments, distances = _kcenters_helper(
        traj,
        distance_method,
        n_clusters=n_clusters,
        dist_cutoff=dist_cutoff,
        cluster_centers=cluster_centers,
        random_first_center=random_first_center,
        output=output)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _hybrid_medoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances, output=output)
        output.write("KMedoids update %s of %s\n" % (i, n_iters))

    return cluster_centers, assignments, distances
