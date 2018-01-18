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

from sklearn.utils import check_random_state

from . import kcenters
from . import kmedoids
from .util import _get_distance_method, ClusterResult, Clusterer

from ..exception import ImproperlyConfigured

logger = logging.getLogger(__name__)


class KHybrid(Clusterer):
    """Sklearn-style object for khybrid clustering.

    KHybrid clustering uses the k-centers protocol to define cluster
    centers and the kmedoids protocol to refine the clustering.

    Parameters
    ----------
    metric : required
        Distance metric used while comparing data points.
    n_clusters : int, default=None
        The number of clusters to build using kcenters. When none,
        only `cluster_radius` is used.
    cluster_radius : float, default=None
        The minimum maximum cluster-datum distance to use in when
        adding cluster centers in the kcenters step. When `None`,
        only `n_clusters` is used.
    kmedoids_updates : it, default=None
        Number of rounds of kmedoids to run.
    random_first_center : bool, default=False
        Choose a random center as the first center, rather than
        choosing the zeroth element (default)
    random_state : int or np.RandomState
        Random state to use to seed the random number generator.

    References
    ----------
    .. [1] Beauchamp, K. A. et al. MSMBuilder2: Modeling Conformational Dynamics at the Picosecond to Millisecond Scale. J. Chem. Theory Comput. 7, 3412–3419 (2011).
    """

    def __init__(self, n_clusters=None, cluster_radius=None,
                 kmedoids_updates=5, random_first_center=False, *args,
                 **kwargs):

        super(KHybrid, self).__init__(self, *args, **kwargs)

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.kmedoids_updates = kmedoids_updates
        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

    def fit(self, X, init_centers=None):
        """Takes trajectories, X, and performs KHybrid clustering.
        Optionally continues clustering from an initial set of cluster
        centers.

        Parameters
        ----------
        X : array-like, shape=(n_observations, n_features(, n_atoms))
            Data to cluster.
        """

        t0 = time.perf_counter()

        self.result_ = hybrid(
            X, self.metric,
            n_iters=self.kmedoids_updates,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            random_first_center=self.random_first_center,
            init_centers=init_centers)

        self.runtime_ = time.perf_counter() - t0


class KHybridMPI(Clusterer):

    def __init__(self, n_clusters=None, cluster_radius=None,
                 kmedoids_updates=5, random_first_center=False,
                 random_state=None, *args, **kwargs):

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.kmedoids_updates = kmedoids_updates
        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

        super(KHybrid, self).__init__(*args, **kwargs)


    def fit(self, X, init_centers=None):
        """Takes trajectories, X, and performs KHybrid clustering.
        Optionally continues clustering from an initial set of cluster
        centers.

        Parameters
        ----------
        X : array-like, shape=(n_observations, n_features(, n_atoms))
            Data to cluster.
        """

        t0 = time.perf_counter()

        dists, assigs, ctr_inds = kcenters.kcenters_mpi(
            X, self.metric, dist_cutoff=self.cluster_radius)

        for i in range(self.kmedoids_updates):
            ctr_inds, assigs, dists = kmedoids._kmedoids_update_mpi(
                X, self.metric,
                ctr_inds, assigs, dists,
                random_state=self.random_state)

        self.labels_ = assigs
        self.distances_ = dists
        self.center_indices_ = ctr_inds

        self.runtime_ = time.perf_counter() - t0


def hybrid(
        traj, distance_method, n_iters=5, n_clusters=np.inf,
        dist_cutoff=0, random_first_center=False,
        init_centers=None, random_state=None):

    distance_method = _get_distance_method(distance_method)

    result = kcenters.kcenters(
        traj, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        init_centers=init_centers, random_first_center=random_first_center)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _hybrid_medoids_update(
            traj, distance_method,
            result.center_indices, result.assignments, result.distances,
            random_state)
        logger.info("KMedoids update %s of %s", i, n_iters)

    return ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=traj[cluster_center_inds])


def _hybrid_medoids_update(
        traj, distance_method, cluster_center_inds, assignments, distances,
        random_state=None):

    proposed_center_inds, proposed_assignments, proposed_distances =\
        kmedoids._kmedoids_update(traj, distance_method, cluster_center_inds,
                                  assignments, distances, random_state)

    max_orig_dist_to_center = distances.max()
    max_proposed_dist_to_center = proposed_distances.max()
    if max_proposed_dist_to_center <= max_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances
