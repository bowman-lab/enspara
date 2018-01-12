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

from ..util import log
from ..exception import ImproperlyConfigured
from . import util

logger = logging.getLogger(__name__)


class KCenters(util.Clusterer):

    def __init__(
            self, metric, n_clusters=None, cluster_radius=None,
            random_first_center=False):

        super(KCenters, self).__init__(metric)

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

    def fit(self, X, init_centers=None):
        """Takes trajectories, X, and performs KCenters clustering.
        Optionally continues clustering from an initial set of cluster
        centers.

        Parameters
        ----------
        X : array-like, shape=(n_observations, n_features(, n_atoms))
            Data to cluster.
        init_centers : array-like, shape=(n_centers, n_features(, n_atoms))
            Begin clustring with these centers as cluster centers.
        """

        t0 = time.clock()

        self.result_ = kcenters(
            X,
            distance_method=self.metric,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            init_centers=init_centers,
            random_first_center=self.random_first_center)

        self.runtime_ = time.clock() - t0


def kcenters(
        traj, distance_method, n_clusters=np.inf, dist_cutoff=0,
        init_centers=None, random_first_center=False):
    """The functional (rather than object-oriented) implementation of
    the k-centers clustering algorithm.

    K-centers is essentially an outlier detection algorithm. It
    iteratively searches out the point that is most distant from all
    existing cluster centers, and adds it as a new cluster centers.
    Its worst-case runtime is O(kn), where k is the number of cluster
    centers and n is the number of observations.

    Parameters
    ----------
        traj : array-like
            The data to cluster with kcenters.
        distance_method : callable
            A callable that takes two arguments: an array of shape `traj.shape` and and array of shape `traj.shape[1:]`, and
            returns an array of shape `traj.shape[0]`, representing the
            'distance' between each element of the `traj` and a proposed
            cluster center.
        n_clusters : int (default=np.inf)
            Stop finding new cluster centers when the number of clsuters
            reaches this value.
        dist_cutoff : float (default=0)
            Stop finding new cluster centers when the maximum minimum
            distance between any point and a cluster center reaches this
            value.
        init_centers : array-like, shape=(n_centers, n_featers)
            A list of observations to use as the first `n_centers`
            centers before discovering new centers with the kcenters
            algorithm.
        random_first_center : bool, default=False
            When false, center 0 is always frame 0. If True, this value
            is chosen randomly.
    Returns
    -------
        result : ClusterResult
            Subclass of NamedTuple containing assignments, distances,
            and center indices for this function.

    References
    ----------
    .. [1] Gonzalez, T. F. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science 38, 293–306 (1985).
    """

    if (n_clusters is np.inf) and (dist_cutoff is 0):
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

    distance_method = util._get_distance_method(distance_method)

    if n_clusters is None and dist_cutoff is None:
        raise ImproperlyConfigured(
            "KCenters must specify 'n_clusters' or 'distance_cutoff'")
    elif n_clusters is None and dist_cutoff is not None:
        n_clusters = np.inf
    elif n_clusters is not None and dist_cutoff is None:
        dist_cutoff = 0

    cluster_center_inds, assignments, distances = _kcenters_helper(
        traj, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        cluster_centers=init_centers, random_first_center=random_first_center)

    return util.ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=traj[cluster_center_inds])


def _kcenters_helper(
        traj, distance_method, n_clusters, dist_cutoff,
        cluster_centers, random_first_center):

    if random_first_center:
        raise NotImplementedError(
            "We haven't implemented kcenters 'random_first_center' yet.")

    new_center_index = 0
    n_frames = len(traj)
    assignments = np.zeros(n_frames, dtype=int)
    distances = np.empty(n_frames, dtype=float)
    distances.fill(np.inf)
    cluster_center_inds = []
    max_distance = np.inf
    cluster_num = 0

    if cluster_centers is not None:
        logger.info("Updating assignments to previous cluster centers")
        assignments, distances = util.assign_to_nearest_center(
            traj, cluster_centers, distance_method)
        cluster_center_inds = list(
            util.find_cluster_centers(assignments, distances))

        cluster_num = len(cluster_center_inds)
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
        logger.info(
            "kCenters cluster "+str(cluster_num) +
            " will continue until max-distance, " +
            '{0:0.6f}'.format(max_distance) + ", falls below " +
            '{0:0.6f}'.format(dist_cutoff) +
            " or num-clusters reaches "+str(n_clusters))
        cluster_num += 1
    cluster_centers = traj[cluster_center_inds]

    return cluster_center_inds, assignments, distances


def kcenters_mpi(traj, distance_method, n_clusters=np.inf, dist_cutoff=0):
    """KCenters implementation for MPI.

    In this function, `traj` is assumed to be only a subset of the data
    in a SIMD execution environment. As a consequence, some
    inter-process communication is required. The user is responsible for
    partitioning the data in `traj` appropriately across the workers and
    for assembling the results correctly.

    Parameters
    ----------
        traj : array-like
            The data to cluster with kcenters.
        distance_method : callable
            A callable that takes two arguments: an array of shape `traj.shape` and and array of shape `traj.shape[1:]`, and
            returns an array of shape `traj.shape[0]`, representing the
            'distance' between each element of the `traj` and a proposed
            cluster center.
        n_clusters : int (default=np.inf)
            Stop finding new cluster centers when the number of clsuters
            reaches this value.
        dist_cutoff : float (default=0)
            Stop finding new cluster centers when the maximum minimum
            distance between any point and a cluster center reaches this
            value.

    Returns
    -------
        world_distances : np.ndarray, shape=(n_observations,)
            For each observation in this MPI worker's world, the
            distance between that observation and the nearest cluster
            center
        world_assignments : np.ndarray, shape=(n_observations,)
            For each observation in this MPI worker's world, the
            assignment of that observation to the nearest cluster center
        world_center_indices : list of tuples
            For each cluster center, a list of the pairs
            (owner_rank, world_index).
    """

    from mpi4py import MPI
    COMM = MPI.COMM_WORLD

    if (n_clusters is np.inf) and (dist_cutoff is 0):
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

    distance_method = util._get_distance_method(distance_method)

    if n_clusters is None and dist_cutoff is None:
        raise ImproperlyConfigured(
            "KCenters must specify 'n_clusters' or 'distance_cutoff'")
    elif n_clusters is None and dist_cutoff is not None:
        n_clusters = np.inf
    elif n_clusters is not None and dist_cutoff is None:
        dist_cutoff = 0

    min_max_dist = np.inf

    distances = np.full(shape=(len(traj),), fill_value=np.inf)
    assignments = np.zeros(shape=(len(traj),), dtype=np.int32) - 1
    ctr_inds = []

    while (len(ctr_inds) < n_clusters) and (min_max_dist > dist_cutoff):

        min_max_dist, distances, assignments, center_inds = \
            _kcenters_iteration_mpi(traj, distance_method, distances,
                                    assignments, ctr_inds)

        if COMM.Get_rank() == 0:
            logger.info(
                "Center %s gives max dist of %.6f (stopping @ %.6f).",
                len(center_inds), min_max_dist, dist_cutoff)

    if COMM.Get_rank() == 0:
        logger.info(
            "Found %s clusters @ %s",
            len(ctr_inds), ctr_inds)

    return distances, assignments, ctr_inds


def _kcenters_iteration_mpi(traj, distance_method, distances, assignments,
                            center_inds=None):
    """The core inner loop of the kcenters iteration protocol. This can
    be used to start and stop doing kcenters (for example to save
    frequently or do checkpointing).
    """

    from mpi4py import MPI
    COMM = MPI.COMM_WORLD

    assert len(traj) == len(distances)
    assert len(traj) == len(assignments)
    assert np.issubdtype(type(assignments[0]), np.integer)

    if center_inds is None:
        center_inds = []

    if len(center_inds) == 0:
        new_cluster_center_index = 0
        new_cluster_center_owner = 0

        min_max_dist = np.inf
    else:

        dist_locs = np.zeros((COMM.Get_size(),), dtype=int) - 1
        dist_vals = np.zeros((COMM.Get_size(),), dtype=float) - 1

        dist_locs[COMM.Get_rank()] = np.argmax(distances)
        dist_vals[COMM.Get_rank()] = np.max(distances)

        with log.timed("Gathered distances in %.2f sec", logger.debug):
            COMM.Allgather(
                [dist_locs[COMM.Get_rank()], MPI.DOUBLE],
                [dist_locs, MPI.DOUBLE])
            COMM.Allgather(
                [dist_vals[COMM.Get_rank()], MPI.FLOAT],
                [dist_vals, MPI.FLOAT])

        assert np.all(dist_locs >= 0)
        assert np.all(dist_vals >= 0)

        new_cluster_center_owner = np.argmax(dist_vals)
        new_cluster_center_index = dist_locs[new_cluster_center_owner]

        min_max_dist = np.max(dist_vals)

    with log.timed("Distributed cluster ctr in %.2f sec",
                   log_func=logger.info):
        new_center = util.mpi_distribute_frame(
            data=traj,
            world_index=new_cluster_center_index,
            owner_rank=new_cluster_center_owner)

    with log.timed("Computed distance in %.2f sec", log_func=logger.info):
        new_dists = distance_method(traj, new_center)

    inds = (new_dists < distances)

    distances[inds] = new_dists[inds]
    assignments[inds] = len(center_inds)

    center_inds.append(
        (new_cluster_center_owner, new_cluster_center_index))

    return min_max_dist, distances, assignments, center_inds
