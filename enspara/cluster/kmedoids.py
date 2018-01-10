
# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging

import numpy as np

from ..util import log
from . import util

logger = logging.getLogger(__name__)


def kmedoids(traj, distance_method, n_clusters, n_iters=5):

    distance_method = util._get_distance_method(distance_method)

    n_frames = len(traj)

    # for short lists, np.random.random_integers sometimes forgets to assign
    # something to each cluster. This will simply repeat the assignments if
    # that is the case.
    cluster_center_inds = np.array([])
    while len(np.unique(cluster_center_inds)) < n_clusters:
        cluster_center_inds = np.random.randint(0, n_frames, n_clusters)

    assignments, distances = util.assign_to_nearest_center(
        traj, traj[cluster_center_inds], distance_method)
    cluster_center_inds = util.find_cluster_centers(assignments, distances)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _kmedoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances)
        logger.info("KMedoids update %s", i)

    return util.ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=traj[cluster_center_inds])


def mpi_mean(local_array):
    """Compute the mean of an array across all MPI nodes.
    """

    from mpi4py import MPI
    COMM = MPI.COMM_WORLD

    local_sum = np.sum(local_array)
    local_len = len(local_array)

    global_sum = np.zeros(1) - 1
    global_len = np.zeros(1) - 1

    global_sum = COMM.allreduce(local_sum, op=MPI.SUM)
    global_len = COMM.allreduce(local_len, op=MPI.SUM)

    assert global_len >= 0
    assert global_sum >= local_sum

    return global_sum / local_len


def _kmedoids_update_mpi(traj, distance_method, cluster_center_inds,
                         assignments, distances):
    """K-Medoids clustering using MPI to parallelze the computation
    across multiple computers over a network in a SIMD fashion.

    Parameters
    ----------
    traj : array-like, shape=(n_observations, n_features, *)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    distance_method : callable
        Function that takes a parameter like `traj` and a single frame
        of `traj` (_i.e._ traj.shape[1:]).
    cluster_center_inds : list, [(owner_rank, world_index), ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    assignments : ndarray, shape=(traj.shape[0],)
        Array indicating the assignment of each frame in `traj` to a
        cluster center.
    distances : ndarray, shape=(traj.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.

    Returns
    -------
    updated_cluster_center_inds : list, [(owner_rank, world_index), ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    updated_assignments : ndarray, shape=(traj.shape[0],)
        Array indicating the assignment of each frame in `traj` to a
        cluster center.
    updated_distances : ndarray, shape=(traj.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    """

    assert np.issubdtype(type(assignments[0]), np.integer)
    assert len(assignments) == len(traj)
    assert len(distances) == len(traj)

    proposed_center_inds = []

    with log.timed("Proposing new centers took %.2f sec",
                    log_func=logger.debug):
        for i in range(len(cluster_center_inds)):
            world_state_inds = np.where(assignments == i)[0]
            r, i = util.mpi_np_choice(world_state_inds)
            proposed_center_inds.append((r, i))

    assert len(proposed_center_inds) == len(cluster_center_inds)

    proposed_cluster_centers = [None] * len(proposed_center_inds)

    with log.timed("Distributing proposed cluster centers took %.2f sec",
                    log_func=logger.debug):
        for center_idx, (rank, frame_idx) in enumerate(proposed_center_inds):
            new_center = util.mpi_distribute_frame(
                data=traj, owner_rank=rank, world_index=frame_idx)
            proposed_cluster_centers[center_idx] = new_center

    with log.timed("Computing distances to new cluster centers took %.2f sec",
                    log_func=logger.debug):
        tu = util.assign_to_nearest_center(
            traj, proposed_cluster_centers, distance_method)
        proposed_assignments, proposed_distances = tu

    with log.timed("Computed quality of new clustering in %.3f.",
                    log_func=logger.debug):
        mean_proposed_dist_to_center = mpi_mean(np.square(proposed_distances))
        mean_orig_dist_to_center = mpi_mean(np.square(distances))

    if mean_proposed_dist_to_center <= mean_orig_dist_to_center:
        logger.info(
            "Accepted centers proposal with avg. dist %.4f (was %.4f).",
            mean_proposed_dist_to_center, mean_orig_dist_to_center)
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        logger.info(
            "Rejected centers proposal with avg. dist %.4f (orig. %.4f).",
            mean_proposed_dist_to_center, mean_orig_dist_to_center)
        return cluster_center_inds, assignments, distances


def _kmedoids_update(
        traj, distance_method, cluster_center_inds, assignments,
        distances):

    assert np.issubdtype(type(assignments[0]), np.integer)
    assert len(assignments) == len(traj)
    assert len(distances) == len(traj)

    proposed_center_inds = np.zeros(len(cluster_center_inds), dtype=int)
    for i in range(len(cluster_center_inds)):
        state_inds = np.where(assignments == i)[0]
        proposed_center_inds[i] = np.random.choice(state_inds)
    proposed_cluster_centers = traj[proposed_center_inds]

    proposed_assignments, proposed_distances = util.assign_to_nearest_center(
        traj, proposed_cluster_centers, distance_method)
    proposed_center_inds = util.find_cluster_centers(
        proposed_assignments, proposed_distances)

    mean_orig_dist_to_center = distances.dot(distances)/len(traj)
    mean_proposed_dist_to_center = proposed_distances.dot(
        proposed_distances)/len(traj)
    if mean_proposed_dist_to_center <= mean_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances
