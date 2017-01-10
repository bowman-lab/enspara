# Author: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
#         Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import time
import logging

import numpy as np

from .util import assign_to_nearest_center, _get_distance_method, \
    ClusterResult, Clusterer

from ..exception import ImproperlyConfigured


class KCenters(Clusterer):

    def __init__(self, metric, n_clusters, cluster_radius):

        super(KCenters, self).__init__(metric)

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius

    def fit(self, X):

        t0 = time.clock()

        self.result_ = kcenters(
            X,
            distance_method=self.metric,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            random_first_center=False)

        self.runtime_ = time.clock() - t0


def kcenters(
        traj, distance_method, n_clusters=np.inf, dist_cutoff=0,
        init_cluster_centers=None, random_first_center=False):

    if (n_clusters is np.inf) and (dist_cutoff is 0):
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

    distance_method = _get_distance_method(distance_method)

    if n_clusters is None and dist_cutoff is None:
        raise ImproperlyConfigured(
            "KCenters must specify 'n_clusters' or 'distance_cutoff'")
    elif n_clusters is None and dist_cutoff is not None:
        n_clusters = np.inf
    elif n_clusters is not None and dist_cutoff is None:
        dist_cutoff = 0

    cluster_center_inds, assignments, distances = _kcenters_helper(
        traj, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        cluster_centers=init_cluster_centers,
        random_first_center=random_first_center)

    return ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=traj[cluster_center_inds])


def _kcenters_helper(
        traj, distance_method, n_clusters, dist_cutoff,
        cluster_centers, random_first_center):

    new_center_index = 0
    n_frames = len(traj)
    assignments = np.zeros(n_frames, dtype=int)
    distances = np.empty(n_frames, dtype=float)
    distances.fill(np.inf)
    cluster_center_inds = []
    max_distance = np.inf
    cluster_num = 0

    if cluster_centers is not None:
        logging.info("Updating assignments to previous cluster centers")
        cluster_center_inds, assignments, distances = assign_to_nearest_center(
            traj, cluster_centers, distance_method)
        cluster_num = len(cluster_center_inds) + 1
        new_center_index = np.argmax(distances)
        max_distance = np.max(distances)

    while (cluster_num < n_clusters) and (max_distance > dist_cutoff):
        dist = distance_method(traj, traj[new_center_index])

        # scipy distance metrics return shape (n, 1) instead of (n), which
        # causes breakage here.
        assert len(dist.shape) == len(distances.shape)

        inds = (dist < distances)
        distances[inds] = dist[inds]
        assignments[inds] = cluster_num
        cluster_center_inds.append(new_center_index)
        new_center_index = np.argmax(distances)
        max_distance = np.max(distances)
        logging.info(
            "kCenters cluster "+str(cluster_num) +
            " will continue until max-distance, " +
            '{0:0.6f}'.format(max_distance) + ", falls below " +
            '{0:0.6f}'.format(dist_cutoff) +
            " or num-clusters reaches "+str(n_clusters))
        cluster_num += 1
    cluster_centers = traj[cluster_center_inds]

    return cluster_center_inds, assignments, distances
