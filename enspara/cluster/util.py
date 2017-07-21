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
from ..util import array as ra


class Clusterer(object):
    """Clusterer class defines the base API for a clustering object in
    the sklearn style.
    """

    def __init__(self, metric):
        self.metric = _get_distance_method(metric)

    def fit(self, X):
        raise NotImplementedError("All Clusterers should implement fit().")

    def predict(self, X):
        """Use an existing clustring fit to predict the assignments,
        distances, and center indices of on new data.new

        See also: assign_to_nearest_center()

        Parameters
        ----------
        X : array-like, shape=(n_states, n_features)
            New data to predict.

        Returns
        -------
        result : ClusterResult
            The result of assigning the given data to the pretrained
            centers.
        """

        if not hasattr(self, 'result_'):
            raise ImproperlyConfigured(
                "To predict the clustering result for new data, the "
                "clusterer first must have fit some data.")

        pred_centers, pred_assigs, pred_dists = assign_to_nearest_center(
            traj=X,
            cluster_centers=self.centers_,
            distance_method=self.metric)

        result = ClusterResult(
            assignments=pred_assigs,
            distances=pred_dists,
            center_indices=pred_centers,
            centers=self.centers_)

        return result

    @property
    def labels_(self):
        return self.result_.assignments

    @property
    def distances_(self):
        return self.result_.distances

    @property
    def center_indices_(self):
        return self.result_.center_indices

    @property
    def centers_(self):
        return self.result_.centers


class ClusterResult(namedtuple('ClusterResult',
                               ['center_indices',
                                'distances',
                                'assignments',
                                'centers'])):
    __slots__ = ()

    def partition(self, lengths):
        """Split each array in this ClusterResult into multiple
        subarrays of variable length.

        Parameters
        ----------
        lengths : array, shape=(n_subarrays)
            Length of each individual subarray.

        Returns
        -------
        result : ClusterResult
            ClusterResult object containing partitioned arrays.
            Assignments and distances are np.ndarrays if each row is the
            same length, and ra.RaggedArrays if trajectories differ.

        See Also
        --------
        partition_indices : for converting lists of concatenated-array
            indices into lists of partitioned-array indices.
        partition_list : for converting concatenated arrays into
            partitioned arrays
        """

        square = all(lengths[0] == l for l in lengths)

        if square:
            return ClusterResult(
                assignments=np.array(partition_list(self.assignments, lengths)),
                distances=np.array(partition_list(self.distances, lengths)),
                center_indices=partition_indices(self.center_indices, lengths),
                centers=self.centers)
        else:
            return ClusterResult(
                assignments=ra.RaggedArray(self.assignments, lengths=lengths),
                distances=ra.RaggedArray(self.distances, lengths=lengths),
                center_indices=partition_indices(self.center_indices, lengths),
                centers=self.centers)


def assign_to_nearest_center(traj, cluster_centers, distance_method):
    """Assign each frame from traj to one of the given cluster centers
    using the given distance metric.

    Parameters
    ----------
    traj: {array-like, trajectory}, shape=(n_frames, n_features, ...)
        The frames to assign to a cluster_center. This parameter need
        only implement `__len__` and  be accepted by `distance_method`.
    cluster_centers : iterable
        Iterable containing some number of exemplar data that each datum
        in `traj` can be compared to using distance_method.
    distance_method: function, params=(traj, cluster_centers[i])
        The distance method to use for assigning each observation in
        trajs to one of the cluster_centers. Must take the entire traj
        and one item from cluster_centers as parameters.

    Returns
    ----------
    tuple : (cluster_center_indices, assignments, distances)
        A tuple containing the assignment of each observation to a
        center (assignments), the distance to that center (distances),
        and a list of observations that are closest to a given center
        (cluster_center_indices.)
    """

    assignments = np.zeros(len(traj), dtype=int)
    distances = np.empty(len(traj), dtype=float)
    distances.fill(np.inf)
    cluster_center_inds = []

    for i, center in enumerate(cluster_centers):
        dist = distance_method(traj, center)
        inds = (dist < distances)
        distances[inds] = dist[inds]
        assignments[inds] = i
        cluster_center_inds.append(np.argmin(dist))

    return cluster_center_inds, assignments, distances


def load_frames(filenames, indices, **kwargs):
    """Load specific frame indices from a list of trajectory files.

    Given a list of trajectory file names (`filenames`) and tuples
    indicating trajectory number and frame number (`indices`), load the
    given frames into a list of md.Trajectory objects. All additional
    kwargs are passed on to md.load_frame.

    Parameters
    ----------
    indices: list, shape=(n_frames, 2)
        List of 2d coordinates, indicating filename to load from and
        which frame to load.
    filenames: list, shape=(n_files)
        List of files to load frames from. The first position in indices
        is taken to refer to a position in this list.
    stride: int
        Treat the indices as having been computed using a stride, so
        mulitply the second index (frame number) by this number (e.g.
        for stride 10, [2, 3] becomes [2, 30]).

    Returns
    ----------
    centers: list
        List of loaded trajectories.
    """

    stride = kwargs.pop('stride', 1)
    if stride is None:
        stride = 1

    centers = []
    for i, j in indices:
        try:
            c = md.load_frame(filenames[i], index=j*stride, **kwargs)
        except ValueError:
            raise ImproperlyConfigured(
                'Failed to load frame {fr} of {fn} using args {kw}.'.format(
                    fn=filenames[i], fr=j*stride, kw=kwargs))
        centers.append(c)

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
