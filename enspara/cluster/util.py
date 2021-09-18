# Authors: Maxwell I. Zimmerman <mizimmer@wustl.edu>,
#          Gregory R. Bowman <gregoryrbowman@gmail.com>,
#          Justin R. Porter <justinrporter@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import logging
from collections import namedtuple

import mdtraj as md
import numpy as np

from ..geometry.libdist import euclidean, manhattan

from ..exception import ImproperlyConfigured, DataInvalid
from ..util import partition_list, partition_indices
from ..util import array as ra

logger = logging.getLogger(__name__)

msmbuilder_libdistance_metrics = ["euclidean", "sqeuclidean", "cityblock",
                                  "chebyshev", "canberra", "braycurtis",
                                  "hamming", "jaccard"]

class MolecularClusterMixin:
    """Additional logic for clusterers in enspara that cluster molecular
    trajectories.
    """

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

        pred_assigs, pred_dists = assign_to_nearest_center(
            trajectory=X,
            cluster_centers=self.centers_,
            distance_method=self.metric)
        pred_centers = find_cluster_centers(pred_assigs, pred_dists)

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
            logger.debug(
                'Lengths are homogenous (%s); using numpy arrays '
                'as output to partitioning.', lengths[0])
            return ClusterResult(
                assignments=np.array(partition_list(self.assignments,
                                                    lengths)),
                distances=np.array(partition_list(self.distances, lengths)),
                center_indices=partition_indices(self.center_indices, lengths),
                centers=self.centers)
        else:
            logger.debug(
                'Lengths are nonhomogenous (median=%d, min=%d, max=%d); '
                'using RaggedArray as output to partitioning.',
                np.median(lengths), np.min(lengths), np.max(lengths))
            return ClusterResult(
                assignments=ra.RaggedArray(self.assignments, lengths=lengths),
                distances=ra.RaggedArray(self.distances, lengths=lengths),
                center_indices=partition_indices(self.center_indices, lengths),
                centers=self.centers)


def assign_to_nearest_center(trajectory, cluster_centers, distance_method):
    """Assign each frame from trajectory to one of the given cluster centers
    using the given distance metric.

    Parameters
    ----------
    trajectory: md.Trajectory or ndarray, shape=(n_frames, n_features, ...)
        The frames to assign to a cluster_center. This parameter need
        only implement `__len__` and  be accepted by `distance_method`.
    cluster_centers : iterable
        Iterable containing some number of exemplar data that each datum
        in `trajectory` can be compared to using distance_method.
    distance_method: function, params=(trajectory, cluster_centers[i])
        The distance method to use for assigning each observation in
        trajectorys to one of the cluster_centers. Must take the entire
        trajectory and one item from cluster_centers as parameters.

    Returns
    ----------
    assignments : ndarray, shape=(n_frames,)
        The assignment of each frame in `trajectory` to a frame in
        cluster_centers.
    distances : ndarray, shape=(n_frames,)
        The distance between each frame in `trajectory` and its assigned
        frame in cluster_centers.
    """

    assignments = np.zeros(len(trajectory), dtype=int)
    distances = np.empty(len(trajectory), dtype=float)
    distances.fill(np.inf)

    # if there are more cluster_centers than trajectory, significant
    # performance benefit can be realized by computing each frame's
    # distance to ALL cluster centers, rather than the reverse.
    if len(cluster_centers) > len(trajectory) and hasattr(cluster_centers, 'xyz'):
        for i, frame in enumerate(trajectory):
            dist = distance_method(cluster_centers, frame)
            assignments[i] = np.argmin(dist)
            distances[i] = np.min(dist)
    else:
        for i, center in enumerate(cluster_centers):
            dist = distance_method(trajectory, center)
            inds = (dist < distances)
            distances[inds] = dist[inds]
            assignments[inds] = i

    return assignments, distances


def find_cluster_centers(assignments, distances):
    """Given a list of distances and assignments, find the
    lowest-distance frame to each label in assignments.

    Parameters
    ----------
    distances: array-like, shape=(n_frames,)
        The distance of each observation to the cluster center.
    assignments : array-like, shape=(n_frames,)
        The assignment of each observation to a cluster.

    Returns
    ----------
    cluster_center_indices : array, shape=(n_labels,)
        A tuple containing the assignment of each observation to a
        center (assignments), the distance to that center (distances),
        and a list of observations that are closest to a given center
        (cluster_center_indices.)
    """

    if len(distances) != len(assignments):
        raise DataInvalid(
            "Length of distances (%s) must match length of assignments "
            "(%s)." % (len(distances), len(assignments)))

    unique_centers = np.unique(assignments)
    center_inds = np.zeros_like(unique_centers)

    for i, c in enumerate(unique_centers):
        assigned_frames = np.where(assignments == c)[0]
        ind = assigned_frames[np.argmin(distances[assigned_frames])]

        center_inds[i] = ind

    return center_inds


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


def _get_distance_method(metric):
    if metric == 'rmsd':
        return md.rmsd
    if metric == 'euclidean':
        return euclidean
    elif metric in ['cityblock', 'manhattan']:
        return manhattan
    elif metric in msmbuilder_libdistance_metrics:
        try:
            import msmbuilder.libdistance as libdistance
        except ImportError:
            raise ImproperlyConfigured(
                "Enspara needs the optional MSMBuilder dependency installed " +
                "to use '{}' as a clustering metric.".format(metric) +
                "It uses MSMBuilder3's libdistance, but we weren't able to " +
                "import msmbuilder.libdistance.")

        def f(X, Y):
            return libdistance.dist(X, Y, metric)
        return f
    elif callable(metric):
        return metric
    else:
        raise ImproperlyConfigured(
            "'{}' is not a recognized metric".format(metric))
