
# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging

import numpy as np

from sklearn.utils import check_random_state

from ..util import log
from .. import mpi

from . import util

logger = logging.getLogger(__name__)


def kmedoids(X, distance_method, n_clusters, n_iters=5):
    """K-Medoids clustering.

    K-Medoids is a clustering algorithm similar to the k-means algorithm
    but the center of each cluster is required to actually be an
    observation in the input data.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, *)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    distance_method : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    cluster_center_inds : list, [(owner_rank, world_index), ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    n_iters : int, default=5
        Number of rounds of new proposed centers to run.

    Returns
    -------
    result : ClusterResult
        Subclass of NamedTuple containing assignments, distances,
        and center indices for this function.

    References
    ----------
    .. [1]  Chodera, J. D., Singhal, N., Pande, V. S., Dill, K. A. & Swope, W. C. Automatic discovery of metastable states for the construction of Markov models of macromolecular conformational dynamics. J. Chem. Phys. 126, 155101 (2007).
    """

    distance_method = util._get_distance_method(distance_method)

    n_frames = len(X)

    # for short lists, np.random.random_integers sometimes forgets to assign
    # something to each cluster. This will simply repeat the assignments if
    # that is the case.
    cluster_center_inds = np.array([])
    while len(np.unique(cluster_center_inds)) < n_clusters:
        cluster_center_inds = np.random.randint(0, n_frames, n_clusters)

    assignments, distances = util.assign_to_nearest_center(
        X, X[cluster_center_inds], distance_method)
    cluster_center_inds = util.find_cluster_centers(assignments, distances)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _kmedoids_update(
            X, distance_method, cluster_center_inds, assignments,
            distances)
        logger.info("KMedoids update %s", i)

    return util.ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=X[cluster_center_inds])


def _msq(x):
    return np.dot(x, x) / len(x)


def _kmedoids_update_mpi(
        X, distance_method, cluster_center_inds, assignments,
        distances, acceptance_criterion=_msq, random_state=None):
    """K-Medoids clustering using MPI to parallelze the computation
    across multiple computers over a network in a SIMD fashion.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, *)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    distance_method : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    cluster_center_inds : list, [(owner_rank, world_index), ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    acceptance_criterion : callable
        Function returning a number that should be lower to accept a set
        of proposed centers (i.e. the criterion minimized by the
        algorithm).
    random_state : numpy.RandomState
        RandomState object used to indentify new centers.

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
    assert len(assignments) == len(X)
    assert len(distances) == len(X)
    random_state = check_random_state(random_state)

    proposed_center_inds = []

    with log.timed("Proposing new centers took %.2f sec",
                    log_func=logger.debug):
        for i in range(len(cluster_center_inds)):
            world_state_inds = np.where(assignments == i)[0]
            r, i = mpi.ops.np_choice(world_state_inds, random_state)
            proposed_center_inds.append((r, i))

    assert len(proposed_center_inds) == len(cluster_center_inds)

    proposed_cluster_centers = [None] * len(proposed_center_inds)

    with log.timed("Distributing proposed cluster centers took %.2f sec",
                    log_func=logger.debug):
        for center_idx, (rank, frame_idx) in enumerate(proposed_center_inds):
            new_center = mpi.ops.distribute_frame(
                data=X, owner_rank=rank, world_index=frame_idx)
            proposed_cluster_centers[center_idx] = new_center

    with log.timed("Computing distances to new cluster centers took %.2f sec",
                    log_func=logger.debug):
        tu = util.assign_to_nearest_center(
            X, proposed_cluster_centers, distance_method)
        proposed_assignments, proposed_distances = tu

    with log.timed("Computed quality of new clustering in %.3f.",
                    log_func=logger.debug):
        mean_proposed_dist_to_center = mpi.ops.mean(np.square(
            proposed_distances))
        mean_orig_dist_to_center = mpi.ops.mean(np.square(distances))

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


def _kmedoids_pam_update(
        X, metric, cluster_center_inds, assignments,
        distances, cost=_msq, random_state=None):
    """Compute a kmedoids update using Partitioning Around Medoids (PAM)

    PAM iteratively proposes a new cluster center from among the points
    assigned to a cluster center, recomputes the cost function, and
    accepts the proposal iff cost goes down. Its time complexity is
    O(k*n*i), where k is the number of centers, n is the size of the data
    and i is the number of iterations (i.e. each invocation of this
    function costs O(kn).)

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, *)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    metric : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    cluster_center_inds : list, [(rank, index), ...] if MPI or [index, ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    cost : callable, default='meansquare'
        Function computing the cost of a particular clustering. Should
        take a vector of distances and returning a number. This value is
        minimzed.
    random_state : numpy.RandomState
        RandomState object used to indentify new centers.

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
    assert len(assignments) == len(X)
    assert len(distances) == len(X)
    random_state = check_random_state(random_state)

    # first we build a list of the actual coordinates of the cluster centers
    # this list will be updated as we go; this is primarily because we want
    # to limit the amount of communication that happens when we're running
    # MPI mode.
    medoid_coords = []
    if hasattr(cluster_center_inds[0], '__len__'):
        assert len(cluster_center_inds[0]) == 2
        for center_idx, (rank, frame_idx) in enumerate(cluster_center_inds):
            assert rank < mpi.MPI_SIZE
            new_center = mpi.ops.distribute_frame(
                data=X, owner_rank=rank, world_index=frame_idx)
            medoid_coords.append(new_center)
    else:
        medoid_coords = [X[i] for i in cluster_center_inds]

    assert len(cluster_center_inds) == len(np.unique(assignments))
    for cid in np.unique(assignments):
        state_inds = np.where(assignments == cid)[0]

        # first, we propose a new center. This works a bit differently
        # if we're running with MPI, because we want to make a choice
        # that uniformly distributed across any node.

        # TODO: make it impossible to choose the current center
        if hasattr(cluster_center_inds[0], '__len__'):
            assert len(cluster_center_inds[0]) == 2
            r, i = mpi.ops.np_choice(state_inds, random_state)
            proposed_center = mpi.ops.distribute_frame(
                data=X, owner_rank=r, world_index=i)
            proposed_center_ind = (r, i)
        else:
            proposed_center_ind = random_state.choice(state_inds)
            proposed_center = X[proposed_center_ind]

        # In the PAM method, once we have a new center, we recompute
        # the distance from this center to every point. Depending on if
        # the distance goes up or down, and which old center it was
        # assigned to (this or other), we update distances and assignents.
        new_ctr_dist = metric(X, proposed_center)

        new_dist = np.zeros_like(distances) - 1
        new_assig = np.zeros_like(assignments) - 1

        # if the new center decreases the distance below whatever it is
        # to its current medoid (cid or not cid), assign it to cid
        dst_dn = (distances > new_ctr_dist)
        new_assig[dst_dn] = cid
        new_dist[dst_dn] = new_ctr_dist[dst_dn]

        # if the new center increases the distance, we have to think
        # harder. If it's assigned to some other medoid, we just copy
        # the old assignments into the new assignments array.
        dst_up_assig_other = (distances <= new_ctr_dist) & (assignments != cid)
        new_assig[dst_up_assig_other] = assignments[dst_up_assig_other]
        new_dist[dst_up_assig_other] = distances[dst_up_assig_other]

        # if the new center increases the distance to cid and it was
        # previously assigned to cid, then we have to compute the
        # distance to _all_ other medoids :(
        dst_up_assig_this = (distances <= new_ctr_dist) & (assignments == cid)

        new_medoids = medoid_coords.copy()
        new_medoids[cid] = proposed_center
        logger.debug("Reassigning %s distances...",
                     np.count_nonzero(dst_up_assig_this))
        ambig_assigs, ambig_dists = util.assign_to_nearest_center(
            X[dst_up_assig_this], new_medoids, metric)

        new_assig[dst_up_assig_this] = ambig_assigs
        new_dist[dst_up_assig_this] = ambig_dists

        # every element of new_dist and new_assig should have been touched
        assert np.all(new_assig >= 0)
        assert np.all(new_dist >= 0)

        old_cost = cost(distances)
        new_cost = cost(new_dist)

        if new_cost < old_cost:
            logger.debug(
                "Accepted proposed center for k=%s, %s -> %s (cost %s -> %s).",
                cid, cluster_center_inds[cid], proposed_center_ind, old_cost,
                new_cost)
            distances, assignments = new_dist, new_assig
            medoid_coords = new_medoids
            cluster_center_inds[cid] = proposed_center_ind
        else:
            logger.debug(
                "Rejected proposed center for k=%s, %s -> %s (cost %s -> %s).",
                cid, cluster_center_inds[cid], proposed_center_ind, old_cost,
                new_cost)

    logger.info("Kmedoid sweep reduced cost to %.4f", min(old_cost, new_cost))

    return cluster_center_inds, distances, assignments


def _kmedoids_update(
        X, distance_method, cluster_center_inds, assignments,
        distances, acceptance_criterion=_msq, random_state=None):

    assert np.issubdtype(type(assignments[0]), np.integer)
    assert len(assignments) == len(X)
    assert len(distances) == len(X)
    random_state = check_random_state(random_state)

    proposed_center_inds = np.zeros(len(cluster_center_inds), dtype=int)
    for i in range(len(cluster_center_inds)):
        state_inds = np.where(assignments == i)[0]
        proposed_center_inds[i] = random_state.choice(state_inds)
    proposed_cluster_centers = X[proposed_center_inds]

    proposed_assignments, proposed_distances = util.assign_to_nearest_center(
        X, proposed_cluster_centers, distance_method)
    proposed_center_inds = util.find_cluster_centers(
        proposed_assignments, proposed_distances)

    mean_orig_dist_to_center = acceptance_criterion(distances)
    mean_proposed_dist_to_center = acceptance_criterion(proposed_distances)

    if mean_proposed_dist_to_center <= mean_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances
