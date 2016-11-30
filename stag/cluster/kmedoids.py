
# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import sys
import os

from .utils import assign_to_nearest_center

import numpy as np


def _kmedoids_update(
        traj, distance_method, cluster_center_inds, assignments,
        distances, output=os.devnull):

    n_clusters = len(cluster_center_inds)
    n_frames = len(traj)

    proposed_center_inds = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        state_inds = np.where(assignments == i)[0]
        proposed_center_inds[i] = np.random.choice(state_inds)
    proposed_cluster_centers = traj[proposed_center_inds]
    proposed_center_inds, proposed_assignments, proposed_distances =\
        assign_to_nearest_center(traj, proposed_cluster_centers,
                                 distance_method)

    mean_orig_dist_to_center = distances.dot(distances)/n_frames
    mean_proposed_dist_to_center = proposed_distances.dot(
        proposed_distances)/n_frames
    if mean_proposed_dist_to_center <= mean_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances


def kmedoids(traj, distance_method, n_clusters, n_iters=5, output=sys.stdout):

    n_frames = len(traj)

    # for short lists, np.random.random_integers sometimes forgets to assign
    # something to each cluster. This will simply repeat the assignments if
    # that is the case.
    cluster_center_inds = np.array([])
    while len(np.unique(cluster_center_inds)) < n_clusters:
        cluster_center_inds = np.random.random_integers(0, n_frames-1,
                                                        n_clusters)

    cluster_center_inds, assignments, distances = assign_to_nearest_center(
        traj, traj[cluster_center_inds], distance_method)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _kmedoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances, output=output)
        output.write("KMedoids update %s\n" % i)

    return cluster_center_inds, assignments, distances
