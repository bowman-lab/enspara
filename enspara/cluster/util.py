# Authors: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
#          Gregory R. Bowman <gregoryrbowman@gmail.com>,
#          Justin R. Porter <justinrporter@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

from collections import namedtuple

import mdtraj as md
import numpy as np

from ..exception import ImproperlyConfigured, DataInvalid
from ..util import partition_list, partition_indices


class Clusterer(object):
    '''
    Clusterer class defines the base API for a clustering object in the
    sklearn style.
    '''

    def __init__(self, metric):
        self.metric = metric

    def fit(self, X):
        raise NotImplementedError("All Clusterers should implement fit().")

    @property
    def labels_(self):
        return self.result_.assignments

    @property
    def distances_(self):
        return self.result_.distances

    @property
    def center_indices_(self):
        return self.result_.cluster_center_inds


class ClusterResult(namedtuple('ClusterResult',
                               ['center_indices',
                                'distances',
                                'assignments',
                                'centers'])):
    __slots__ = ()

    def partition(self, lengths):
        return ClusterResult(
            assignments=partition_list(self.assignments, lengths),
            distances=partition_list(self.distances, lengths),
            center_indices=partition_indices(self.center_indices, lengths),
            centers=self.centers)


def assign_to_nearest_center(traj, cluster_centers, distance_method):
    n_frames = len(traj)
    assignments = np.zeros(n_frames, dtype=int)
    distances = np.empty(n_frames, dtype=float)
    distances.fill(np.inf)
    cluster_center_inds = []

    cluster_num = 0
    for center in cluster_centers:
        dist = distance_method(traj, center)
        inds = (dist < distances)
        distances[inds] = dist[inds]
        assignments[inds] = cluster_num
        new_center_index = np.argmin(dist)
        cluster_center_inds.append(new_center_index)
        cluster_num += 1

    return cluster_center_inds, assignments, distances


def load_frames(indices, filenames, **kwargs):
    '''
    Given a list of trajectory file names (`filenames`) and tuples
    indicating trajectory number and frame number (`center indices`),
    load the given frames into a list of md.Trajectory objects. All
    additional kwargs are passed on to md.load_frame.
    '''

    stride = kwargs.pop('stride', 1)

    centers = [md.load_frame(filenames[i], index=j*stride, **kwargs)
               for i, j in indices]
    return centers


def find_cluster_centers(traj_lst, distances):
    '''
    Use the distances matrix to calculate which of the trajectories in
    traj_lst are the center of a cluster. Return a list of indices.
    '''

    if len(traj_lst) != len(distances):
        raise DataInvalid(
            "Expected len(traj_lst) ({}) to match len(distances) ({})".
            format(len(traj_lst), distances.shape))

    center_indices = np.argwhere(distances == 0)

    try:
        # 3D case (trj index, frame)
        centers = [traj_lst[trj_indx][frame] for (trj_indx, frame)
                   in center_indices]
    except ValueError:
        # 2D case (just frame)
        centers = [traj_lst[trj_indx] for trj_indx
                   in center_indices]

    assert len(centers) == center_indices.shape[0]

    return centers


def _get_distance_method(metric):
    if metric == 'rmsd':
        return md.rmsd
    elif isinstance(metric, str):
        try:
            import msmbuilder.libdistance as libdistance
        except ImportError:
            raise ImproperlyConfigured(
                "To use '{}' as a clustering metric, STAG ".format(metric) +
                "uses MSMBuilder3's libdistance, but we weren't able to " +
                "import msmbuilder.libdistance.")

        def f(X, Y):
            return libdistance.dist(X, Y, metric)
        return f
    elif callable(metric):
        return metric
    else:
        raise ImproperlyConfigured(
            "'{}' is not a recognized metric".format(metric))
