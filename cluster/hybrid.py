# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

from .kcenters import _kcenters_helper
from .kmedoids import _kmedoids_update
from ..traj_manipulation import sloopy_concatenate_trjs
from .utils import _get_distance_method, _partition_list

import mdtraj as md
import numpy as np


def _hybrid_medoids_update(
        traj, distance_method, cluster_center_inds, assignments, distances,
        verbose=True):

    proposed_center_inds, proposed_assignments, proposed_distances =\
        _kmedoids_update(traj, distance_method, cluster_center_inds,
                         assignments, distances, verbose=verbose)

    max_orig_dist_to_center = distances.max()
    max_proposed_dist_to_center = proposed_distances.max()
    if max_proposed_dist_to_center <= max_orig_dist_to_center:
        return proposed_center_inds, proposed_assignments, proposed_distances
    else:
        return cluster_center_inds, assignments, distances


def hybrid(
        traj_lst, n_iters, n_clusters=None, dist_cutoff=None, metric='rmsd',
        random_first_center=False, delete_trjs=True, verbose=True):

    # TODO: this block of code is repeated between all three basic clustering
    # schemes

    distance_method = _get_distance_method(metric)

    traj_lengths = [len(t) for t in traj_lst]
    if isinstance(traj_lst[0], md.Trajectory):
        traj = sloopy_concatenate_trjs(traj_lst, delete_trjs=delete_trjs)
    else:
        traj = np.concatenate(traj_lst)
    # /ENDBLOCK

    cluster_center_inds, assignments, distances = _kcenters_helper(
        traj, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        random_first_center=random_first_center, verbose=verbose)

    for i in range(n_iters):
        cluster_center_inds, assignments, distances = _hybrid_medoids_update(
            traj, distance_method, cluster_center_inds, assignments,
            distances, verbose=verbose)

    # TODO: this block of code is repeated between all three basic clustering
    # schemes
    cluster_centers = traj[cluster_center_inds]
    assignments = _partition_list(assignments, traj_lengths)
    distances = _partition_list(distances, traj_lengths)

    return cluster_centers, assignments, distances
    # /ENDBLOCK
