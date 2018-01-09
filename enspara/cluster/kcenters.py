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
                   ClusterResult, Clusterer, find_cluster_centers)

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


def mpi_distribute_frame(data, world_index, owner_rank):
    """Distribute an element of an array to every node in an MPI swarm.

    Parameters
    ----------
    data : array-like or md.Trajectory
        Data array with frames to distribute. The frame will be taken
        from axis 0 of the input.
    world_index : int
        Position of the target frame in `data` on the node that owns it
    owner_rank : int
        Rank of the node that owns the datum that we'll broadcast.

    Returns
    -------
    frame : array-like or md.Trajectory
        A single slice of `data`, of shape `data.shape[1:]`.
    """

    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    if hasattr(data, 'xyz'):
        if rank == owner_rank:
            frame = data[world_index].xyz
        else:
            frame = np.empty_like(data[0].xyz)
    else:
        if rank == owner_rank:
            frame = data[world_index]
        else:
            frame = np.empty_like(data[0])

    MPI.COMM_WORLD.Bcast(frame, root=owner_rank)

    if hasattr(data, 'top'):
        return type(data)(frame, topology=data.top)
    else:
        return type(data)(frame)


def kcenters_mpi(
        traj, distance_method, n_clusters=np.inf, dist_cutoff=0,
        init_centers=None):

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
            index=new_cluster_center_index,
            owner=new_cluster_center_owner,
            trjs=traj)
        tock = time.perf_counter()
        logging.debug("Distributed cluster ctr in %.2f sec", tock-tick)

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
        logging.debug("Gathered distances in %.2f sec", tock-tick)
        # logging.debug("World max-dists are %s", dist_vals)

        assert np.all(dist_locs >= 0)
        assert np.all(dist_vals >= 0)

        # logging.debug("Gathered distance locations %s", dist_locs)
        # logging.debug("Gathered distance values %s", dist_vals)

        min_max_dist = np.min(dist_vals)
        new_cluster_center_owner = np.argmax(dist_vals)
        new_cluster_center_index = dist_locs[new_cluster_center_owner]

        if COMM.Get_rank() == 0:
            logging.info(
                "Center %s from rank %s, frame %s, gives max dist of %.5f "
                "(stopping @ %s).",
                cluster_num, new_cluster_center_owner,
                new_cluster_center_index, dist_vals[new_cluster_center_owner],
                dist_cutoff)

        cluster_num += 1

    if COMM.Get_rank() == 0:
        logging.info(
            "Found %s clusters @ %s",
            len(world_ctr_inds), world_ctr_inds)

    return distances, assignments, world_ctr_inds
