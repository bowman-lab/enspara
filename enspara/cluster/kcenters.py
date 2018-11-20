from __future__ import print_function, division, absolute_import

import time
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from ..util import log
from ..exception import ImproperlyConfigured
from .. import mpi

from . import util

logger = logging.getLogger(__name__)


class KCenters(BaseEstimator, ClusterMixin, util.MolecularClusterMixin):
    """Sklearn-style object for kcenters clustering.

    K-centers is essentially an outlier detection algorithm. It
    iteratively searches out the point that is most distant from all
    existing cluster centers, and adds it as a new cluster centers.
    Its worst-case runtime is O(kn), where k is the number of cluster
    centers and n is the number of observations.

    Parameters
    ----------
    metric : required
        Distance metric used while comparing data points.
    n_clusters : int, default=None
        The number of clusters to build using kcenters. When none,
        only `cluster_radius` is used.
    cluster_radius : float, default=None
        The minimum maximum cluster-datum distance to use in when
        adding cluster centers in the kcenters step. When `None`,
        only `n_clusters` is used.
    random_first_center : bool, default=False
        Choose a random center as the first center, rather than
        choosing the zeroth element (default)
    random_state : int or np.RandomState
        Random state to use to seed the random number generator.

    References
    ----------
    .. [1] Gonzalez, T. F. Clustering to minimize the maximum intercluster
    distance. Theoretical Computer Science 38, 293–306 (1985).
    """

    def __init__(
            self, n_clusters=None, cluster_radius=None,
            random_first_center=False, *args, **kwargs):

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

        self.metric = util._get_distance_method(kwargs.pop('metric'))
        self.random_state = check_random_state(
            kwargs.pop('random_state', None))

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
        return self


class KCentersMPI(KCenters):

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

        if self.random_first_center:
            raise NotImplementedError(
                "KCentersMPI doesn't implement random_first_center yet.")

        if self.random_first_center:
            raise NotImplementedError(
                "KCentersMPI doesn't implement init_centers yet.")

        t0 = time.clock()

        distances, assignments, ctr_inds = kcenters_mpi(
            X,
            distance_method=self.metric,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius)

        self.center

        self.runtime_ = time.clock() - t0
        return self


def _kcenters_iteration(
        traj, distance_method, distances, assignments, center_inds):

    # scipy distance metrics return shape (n, 1) instead of (n), which
    # causes breakage here.

    assert len(traj) == len(distances)
    assert len(traj) == len(assignments)
    assert np.issubdtype(type(assignments[0]), np.integer)

    new_center_index = np.argmax(distances)
    new_center = traj[new_center_index]

    dist = distance_method(traj, new_center)
    assert len(dist.shape) == len(distances.shape)

    inds = (dist < distances)
    distances[inds] = dist[inds]
    assignments[inds] = len(center_inds)

    center_inds.append(new_center_index)
    new_center_index = np.argmax(distances)
    max_distance = np.max(distances)

    return new_center, max_distance, distances, assignments, center_inds


def kcenters_mpi(*args, **kwargs):
    kwargs.pop('mpi_mode', None)
    return kcenters(*args, mpi_mode=True, **kwargs)


def kcenters(traj, distance_method, n_clusters=np.inf, dist_cutoff=0,
             init_centers=None, random_first_center=False, mpi_mode=False):
    """The functional (rather than object-oriented) implementation of
    the k-centers clustering algorithm.

    K-centers is essentially an outlier detection algorithm. It
    iteratively searches out the point that is most distant from all
    existing cluster centers, and adds it as a new cluster centers.
    Its worst-case runtime is O(kn), where k is the number of cluster
    centers and n is the number of observations.

    This method can be used in MPI mode, where `traj` is assumed to be
    only a subset of the data in a SIMD execution environment. As a
    consequence, some inter-process communication is required. The user
    is responsible for partitioning the data in `traj` appropriately
    across the workers and for assembling the results correctly.

    Parameters
    ----------
        traj : array-like
            The data to cluster with kcenters.
        distance_method : callable
            A callable that takes two arguments: an array of shape
            `traj.shape` and and array of shape `traj.shape[1:]`, and
            returns an array of shape `traj.shape[0]`, representing the
            'distance' between each element of the `traj` and a proposed
            cluster center.
        n_clusters : int (default=np.inf)
            Stop finding new cluster centers when the number of clusters
            reaches this value.
        dist_cutoff : float (default=0)
            Stop finding new cluster centers when the maximum minimum
            distance between any point and a cluster center reaches this
            value.
        init_centers : array-like, shape=(n_centers, n_features)
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
            and center indices for this function. In MPI mode, distances
            and assignments are partitioned by node, and center indices
            take the form (node, index). In regular mode, distances and
            assignments are for all frames and center indices are just
            positions.

    References
    ----------
    .. [1] Gonzalez, T. F. Clustering to minimize the maximum intercluster
    distance. Theoretical Computer Science 38, 293–306 (1985).
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

    min_max_dist = np.inf

    if random_first_center:
        raise NotImplementedError(
            "We haven't implemented kcenters 'random_first_center' yet.")

    if init_centers is None:
        ctr_inds = []
        centers = []
        assignments = np.full(len(traj), -1, dtype=int)
        distances = np.full(len(traj), np.inf, dtype=float)
    else:
        centers = [c for c in init_centers]
        logger.info("Updating assignments to previous cluster centers")
        assignments, distances = util.assign_to_nearest_center(
            traj, centers, distance_method)
        ctr_inds = list(
            util.find_cluster_centers(assignments, distances))

    if mpi_mode:
        iteration = _kcenters_iteration_mpi
    else:
        iteration = _kcenters_iteration

    while (len(ctr_inds) < n_clusters) and (min_max_dist > dist_cutoff):

        new_center, min_max_dist, distances, assignments, center_inds = \
            iteration(traj, distance_method, distances, assignments, ctr_inds)
        centers.append(new_center)

        if mpi.MPI_RANK == 0:
            logger.info(
                "Center %s gives max dist of %.6f (stopping @ %.6f).",
                len(center_inds), min_max_dist, dist_cutoff)

    return util.ClusterResult(
        center_indices=ctr_inds,
        assignments=assignments,
        distances=distances,
        centers=centers)


def _kcenters_iteration_mpi(traj, distance_method, distances, assignments,
                            center_inds):
    """The core inner loop of the kcenters iteration protocol. This can
    be used to start and stop doing kcenters (for example to save
    frequently or do checkpointing).
    """

    assert len(traj) == len(distances)
    assert len(traj) == len(assignments)
    assert np.issubdtype(type(assignments[0]), np.integer)

    if len(center_inds) == 0:
        new_cluster_center_index = 0
        new_cluster_center_owner = 0
        min_max_dist = np.inf
    else:
        with log.timed("Gathered distances in %.2f sec", logger.debug):
            # this could likely be accomplished with mpi.reduce instead...
            dist_locs = np.array(
                mpi.MPI.COMM_WORLD.allgather(np.argmax(distances)))
            dist_vals = np.array(
                mpi.MPI.COMM_WORLD.allgather(np.max(distances)))

        new_cluster_center_owner = np.argmax(dist_vals)
        new_cluster_center_index = dist_locs[new_cluster_center_owner]

        min_max_dist = np.max(dist_vals)

    with log.timed("Distributed cluster ctr in %.2f sec",
                   log_func=logger.info):
        new_center = mpi.ops.distribute_frame(
            data=traj,
            world_index=new_cluster_center_index,
            owner_rank=new_cluster_center_owner)

    with log.timed("Computed distance in %.2f sec", log_func=logger.info):
        new_dists = distance_method(traj, new_center)
    assert len(distances.shape) == len(new_dists.shape)

    inds = (new_dists < distances)

    distances[inds] = new_dists[inds]
    assignments[inds] = len(center_inds)

    center_inds.append(
        (new_cluster_center_owner, new_cluster_center_index))

    return new_center, min_max_dist, distances, assignments, center_inds
