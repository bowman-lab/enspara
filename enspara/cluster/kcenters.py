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

from .util import (assign_to_nearest_center, _get_distance_method,
                   ClusterResult, Clusterer, find_cluster_centers,
                   mpi_distribute_frame)

from ..exception import ImproperlyConfigured

logger = logging.getLogger(__name__)


class KCenters(Clusterer):

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
    """

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
        cluster_centers=init_centers, random_first_center=random_first_center)

    return ClusterResult(
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
        assignments, distances = assign_to_nearest_center(
            traj, cluster_centers, distance_method)
        cluster_center_inds = list(
            find_cluster_centers(assignments, distances))

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


def kcenters_mpi(
        traj, distance_method, n_clusters=np.inf, dist_cutoff=0):
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

    distance_method = _get_distance_method(distance_method)

    if n_clusters is None and dist_cutoff is None:
        raise ImproperlyConfigured(
            "KCenters must specify 'n_clusters' or 'distance_cutoff'")
    elif n_clusters is None and dist_cutoff is not None:
        n_clusters = np.inf
    elif n_clusters is not None and dist_cutoff is None:
        dist_cutoff = 0

    cluster_num = 0
    min_max_dist = np.inf

    distances = np.full(shape=(len(traj),), fill_value=np.inf)
    assignments = np.zeros(shape=(len(traj),)) - 1
    world_ctr_inds = []

    new_cluster_center_owner = 0
    new_cluster_center_index = 0

    while (cluster_num < n_clusters) and (min_max_dist > dist_cutoff):

        tick = time.perf_counter()
        new_center = mpi_distribute_frame(
            data=traj,
            world_index=new_cluster_center_index,
            owner_rank=new_cluster_center_owner)
        tock = time.perf_counter()
        logger.debug("Distributed cluster ctr in %.2f sec", tock-tick)

        new_dists = distance_method(traj, new_center)

        inds = (new_dists < distances)

        distances[inds] = new_dists[inds]
        assignments[inds] = cluster_num

        world_ctr_inds.append(
            (new_cluster_center_owner, new_cluster_center_index))

        dist_locs = np.zeros((COMM.Get_size(),), dtype=int) - 1
        dist_vals = np.zeros((COMM.Get_size(),), dtype=float) - 1

        dist_locs[COMM.Get_rank()] = np.argmax(distances)
        dist_vals[COMM.Get_rank()] = np.max(distances)

        tick = time.perf_counter()
        COMM.Allgather(
            [dist_locs[COMM.Get_rank()], MPI.DOUBLE],
            [dist_locs, MPI.DOUBLE])
        COMM.Allgather(
            [dist_vals[COMM.Get_rank()], MPI.FLOAT],
            [dist_vals, MPI.FLOAT])
        tock = time.perf_counter()
        logger.debug("Gathered distances in %.2f sec", tock-tick)

        assert np.all(dist_locs >= 0)
        assert np.all(dist_vals >= 0)

        min_max_dist = np.min(dist_vals)
        new_cluster_center_owner = np.argmax(dist_vals)
        new_cluster_center_index = dist_locs[new_cluster_center_owner]

        if COMM.Get_rank() == 0:
            logger.info(
                "Center %s from rank %s, frame %s, gives max dist of %.5f "
                "(stopping @ %s).",
                cluster_num, new_cluster_center_owner,
                new_cluster_center_index, dist_vals[new_cluster_center_owner],
                dist_cutoff)

        cluster_num += 1

    if COMM.Get_rank() == 0:
        logger.info(
            "Found %s clusters @ %s",
            len(world_ctr_inds), world_ctr_inds)

    return distances, assignments, world_ctr_inds
