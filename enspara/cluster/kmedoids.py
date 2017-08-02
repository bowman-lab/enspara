
# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging

from .util import (assign_to_nearest_center, _get_distance_method,
                   ClusterResult, find_cluster_centers)

import numpy as np

logger = logging.getLogger(__name__)


def kmedoids(traj, distance_method, n_clusters, n_iters=5):

    distance_method = _get_distance_method(distance_method)

    n_frames = len(traj)

    # for short lists, np.random.random_integers sometimes forgets to assign
    # something to each cluster. This will simply repeat the assignments if
    # that is the case.
    cluster_center_inds = np.array([])
    while len(np.unique(cluster_center_inds)) < n_clusters:
        cluster_center_inds = np.random.randint(0, n_frames, n_clusters)

    assignments, distances = assign_to_nearest_center(
        traj, traj[cluster_center_inds], distance_method)
    cluster_center_inds = find_cluster_centers(assignments, distances)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _kmedoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances)
        logger.info("KMedoids update %s", i)

    return ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=cluster_center_inds)


def _kmedoids_update(
        traj, distance_method, cluster_center_inds, assignments,
        distances):

    assert assignments.dtype == np.int

    proposed_center_inds = np.zeros(len(cluster_center_inds), dtype=int)
    for i in range(len(cluster_center_inds)):
        state_inds = np.where(assignments == i)[0]
        proposed_center_inds[i] = np.random.choice(state_inds)
    proposed_cluster_centers = traj[proposed_center_inds]

    proposed_assignments, proposed_distances = assign_to_nearest_center(
        traj, proposed_cluster_centers, distance_method)
    proposed_center_inds = find_cluster_centers(
        proposed_assignments, proposed_distances)

    mean_orig_dist_to_center = distances.dot(distances)/len(traj)
    mean_proposed_dist_to_center = proposed_distances.dot(
        proposed_distances)/len(traj)
    if mean_proposed_dist_to_center <= mean_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances
