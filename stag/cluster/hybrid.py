# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import sys
import time
import os

import numpy as np

from .kcenters import kcenters, KCenters
from .kmedoids import _kmedoids_update
from .util import _get_distance_method, ClusterResult

from ..util import partition_list, partition_indices
from ..exception import ImproperlyConfigured


class KHybrid(object):

    def __init__(self, metric, n_clusters=None, cluster_radius=None,
                 kmedoids_updates=5, verbose=False):

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.metric = metric
        self.kmedoids_updates = kmedoids_updates
        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius

        self.kcenters = KCenters(
            metric=self.metric, n_clusters=self.n_clusters,
            cluster_radius=self.cluster_radius)

        self.output = sys.stdout if verbose else open(os.devnull, 'w')
        self.kcenters.output = self.output

    def fit(self, X):

        starttime = time.clock()

        self.kcenters.fit(X)

        center_inds = self.kcenters.center_indices_
        assignments = self.kcenters.labels_
        distances = self.kcenters.distances_

        for i in range(self.kmedoids_updates):
            self.output.write("KMedoids update %s of %s\n" %
                              (i, self.kmedoids_updates))

            center_inds, assignments, distances = \
                _hybrid_medoids_update(
                    X,
                    distance_method=self.metric,
                    cluster_center_inds=center_inds,
                    assignments=assignments,
                    distances=distances)

        self.labels_ = assignments
        self.distances_ = distances
        self.center_indices_ = center_inds
        self.result_ = ClusterResult(
            assignments=assignments,
            distances=distances,
            center_indices=center_inds)

        self.runtime_ = time.clock() - starttime


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


def hybrid(
        traj, distance_method, n_iters=5, n_clusters=np.inf,
        dist_cutoff=0, random_first_center=False,
        init_cluster_centers=None, output=sys.stdout):

    if output is None:
        output = os.devnull

    distance_method = _get_distance_method(distance_method)

    result = kcenters(
        traj, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        init_cluster_centers=init_cluster_centers,
        random_first_center=random_first_center,
        output=output)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _hybrid_medoids_update(
            traj, distance_method,
            result.center_indices, result.assignments, result.distances)
        if output:
            output.write("KMedoids update %s of %s\n" % (i, n_iters))

    return ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances)
