# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

from ..traj_manipulation import sloopy_concatenate_trjs
from .utils import assign_to_nearest_center, _get_distance_method,\
    _partition_list

import mdtraj as md
import numpy as np


def _kmedoids_update(
        traj, distance_method, cluster_center_inds, assignments,
        distances, verbose=True):

    n_clusers = len(cluster_center_inds)
    n_frames = len(traj)

    proposed_center_inds = np.zeros(n_clusers, dtype=int)
    for i in range(n_clusers):
        state_inds = np.where(assignments == i)[0]
        proposed_center_inds[i] = np.random.choice(state_inds)
    proposed_cluster_centers = traj[proposed_center_inds]
    proposed_center_inds, proposed_assignments, proposed_distances =\
        assign_to_nearest_center(traj, proposed_cluster_centers,
                                 distance_method)

    mean_orig_dist_to_center = distances.dot(distances)/n_frames
    mean_proposed_dist_to_center = proposed_distances.dot(
        proposed_distances)/n_frames
    if mean_proposed_dist_to_center <= mean_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances


def kmedoids(
        traj_lst, n_clusters, metric='rmsd', n_iters=5,
        delete_trjs=True, verbose=True):

    distance_method = _get_distance_method(metric)

    traj_lengths = [len(t) for t in traj_lst]
    if isinstance(traj_lst[0], md.Trajectory):
        traj = sloopy_concatenate_trjs(traj_lst, delete_trjs=delete_trjs)
    else:
        traj = np.concatenate(traj_lst)
    n_frames = len(traj)

    cluster_center_inds = np.random.random_integers(0, n_frames-1, n_clusters)
    cluster_centers = traj[cluster_center_inds]
    cluster_center_inds, assignments, distances = assign_to_nearest_center(
        traj, cluster_centers, distance_method)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _kmedoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances, verbose=verbose)

    cluster_centers = traj[cluster_center_inds]
    assignments = _partition_list(assignments, traj_lengths)
    distances = _partition_list(distances, traj_lengths)

    return cluster_centers, assignments, distances
