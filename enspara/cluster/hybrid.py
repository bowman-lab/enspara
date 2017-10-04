# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import time
import logging

import numpy as np

from .kcenters import kcenters
from .kmedoids import _kmedoids_update
from .util import _get_distance_method, ClusterResult, Clusterer

from ..exception import ImproperlyConfigured

logger = logging.getLogger(__name__)


class KHybrid(Clusterer):

    def __init__(self, metric, n_clusters=None, cluster_radius=None,
                 kmedoids_updates=5, random_first_center=False):

        super(KHybrid, self).__init__(metric)

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.kmedoids_updates = kmedoids_updates
        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

    def fit(self, X, init_cluster_centers=None):
        """Takes trajectories, X, and performs KHybrid clustering.
        Optionally continues clustering from an initial set of cluster
        centers.
        """

        starttime = time.clock()

        self.result_ = hybrid(
            X, self.metric,
            n_iters=self.kmedoids_updates,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            random_first_center=self.random_first_center,
            init_cluster_centers=init_cluster_centers)

        self.runtime_ = time.clock() - starttime


def hybrid(
        traj, distance_method, n_iters=5, n_clusters=np.inf,
        dist_cutoff=0, random_first_center=False,
        init_cluster_centers=None):

    distance_method = _get_distance_method(distance_method)

    result = kcenters(
        traj, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        init_cluster_centers=init_cluster_centers,
        random_first_center=random_first_center)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _hybrid_medoids_update(
            traj, distance_method,
            result.center_indices, result.assignments, result.distances)
        logger.info("KMedoids update %s of %s", i, n_iters)

    return ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=traj[cluster_center_inds])


def _hybrid_medoids_update(
        traj, distance_method, cluster_center_inds, assignments, distances):

    proposed_center_inds, proposed_assignments, proposed_distances =\
        _kmedoids_update(traj, distance_method, cluster_center_inds,
                         assignments, distances)

    max_orig_dist_to_center = distances.max()
    max_proposed_dist_to_center = proposed_distances.max()
    if max_proposed_dist_to_center <= max_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances
