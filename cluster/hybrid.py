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

from .kcenters import _kcenters_helper
from .kmedoids import _kmedoids_update
from .utils import requires_concatenated_trajectories, _partition_list, \
    _partition_indices

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
        self.verbose = verbose

    def fit(self, X):

        starttime = time.clock()

        if self.verbose:
            output = sys.stdout
        else:
            output = open(os.devnull, 'w')

        cluster_center_inds, assignments, distances = _kcenters_helper(
            X,
            distance_method=self.metric,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            random_first_center=False,
            cluster_centers=None,
            output=output)

        for i in range(self.kmedoids_updates):
            output.write("KMedoids update %s of %s\n" %
                         (i, self.kmedoids_updates))

            cluster_center_inds, assignments, distances = \
                _hybrid_medoids_update(
                    X,
                    distance_method=self.metric,
                    cluster_center_inds=cluster_center_inds,
                    assignments=assignments,
                    distances=distances)

        self.runtime_ = time.clock() - starttime
        self.labels_ = assignments
        self.distances_ = distances
        self.cluster_center_indices_ = cluster_center_inds

    def partitioned_labels(self, lengths):
        return _partition_list(self.labels_, lengths)

    def partitioned_distances(self, lengths):
        return _partition_list(self.distances_, lengths)

    def partitioned_center_indices(self, lengths):
        return _partition_indices(self.cluster_center_indices_, lengths)


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


@requires_concatenated_trajectories
def hybrid(
        traj, distance_method, n_iters=5, n_clusters=None, dist_cutoff=None,
        random_first_center=False, init_cluster_centers=None,
        output=sys.stdout):

    cluster_center_inds, assignments, distances = _kcenters_helper(
        traj,
        distance_method,
        n_clusters=n_clusters,
        dist_cutoff=dist_cutoff,
        cluster_centers=init_cluster_centers,
        random_first_center=random_first_center,
        output=output)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _hybrid_medoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances)
        if output:
            output.write("KMedoids update %s of %s\n" % (i, n_iters))

    return cluster_center_inds, assignments, distances
