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

    The original algorithm and optimality guarantees are described in
    [1]_.

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
    mpi_mode : bool, default=None
        Use the MPI version of the algorithm. This assumes that each node
        in the MPI swarm owns its own data. If None, it is determined
        automatically.

    References
    ----------
    .. [1] Gonzalez, T. F. Clustering to minimize the maximum
        intercluster distance. Theoretical Computer Science 38, 293–306
        (1985).
    """

    def __init__(
            self, metric, n_clusters=None, cluster_radius=None,
            random_first_center=False, random_state=None, mpi_mode=None):

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.metric = util._get_distance_method(metric)

        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

        self.random_state = check_random_state(random_state)
        self.mpi_mode = mpi.size() != 1 if mpi_mode is None else mpi_mode

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

        t0 = time.perf_counter()

        self.result_ = kcenters(
            X,
            distance_method=self.metric,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            init_centers=init_centers,
            random_first_center=self.random_first_center,
            mpi_mode=self.mpi_mode)

        self.runtime_ = time.perf_counter() - t0
        return self


def kcenters_mpi(*args, **kwargs):
    kwargs.pop('mpi_mode', None)
    return kcenters(*args, mpi_mode=True, **kwargs)


def kcenters(traj, distance_method, n_clusters=np.inf, dist_cutoff=0,
             init_centers=None, random_first_center=False,
             use_triangle_inequality=False, mpi_mode=False):
    """Function implementation of the k-centers clustering algorithm.

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

    The original algorithm and optimality guarantees are described in
    [2]_.

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
    use_triangle_inequality : bool, default=False
        Use the fact that the the new center's current distance must be
        greater than half than its nearest intercluster distance to avoid
        recomputing some distances. This optimization was developed in
        ref [3]_.

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
    .. [2] Gonzalez, T. F. Clustering to minimize the maximum intercluster
        distance. Theoretical Computer Science 38, 293–306 (1985).
    .. [3] Zhao, Y., Sheong, F. K., Sun, J., Sander, P. & Huang, X. A fast
        parallel clustering algorithm for molecular simulation trajectories.
        J. Comput. Chem. 34, 95–104 (2013).
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
        kwargs = {'centers': centers}
    else:
        iteration = _kcenters_iteration
        kwargs = {}

    maxdist = (mpi.ops.striped_array_max(distances) if mpi_mode
               else distances.max())
    while (len(ctr_inds) < n_clusters) and (maxdist > dist_cutoff):

        new_center, distances, assignments, center_inds = \
            iteration(traj, distance_method, distances, assignments, ctr_inds,
                      use_triangle_inequality=use_triangle_inequality,
                      **kwargs)

        centers.append(new_center)
        maxdist = (mpi.ops.striped_array_max(distances) if mpi_mode
                   else distances.max())

        if mpi.rank() == 0:
            logger.info(
                "Center %s gives max dist of %.6f (stopping @ d=%.6f/n=%s).",
                len(center_inds), maxdist, dist_cutoff, n_clusters)

    logger.info("Terminated k-centers with n=%s and d=%0.6f.",
                len(center_inds), maxdist,)

    return util.ClusterResult(
        center_indices=ctr_inds,
        assignments=assignments,
        distances=distances,
        centers=centers)


def _kcenters_iteration(
        traj, distance_method, distances, assignments, center_inds,
        use_triangle_inequality=False):
    """Core inner loop for kcenters centers discovery.

    Parameters
    ----------
    traj : md.Trajectory or np.ndarray
        The data to cluster with kcenters.
    distance_method : callable(X, y)
        Distance function to use to compute distances between a dataset
        (X) and a single point (y)
    distances : np.ndarray
        The current distance between each point and its nearest cluster
        center
    assignments : np.ndarray
        The assignment of each point to a cluster center.
    center_inds : list
        The position of each center in ``traj``.

    Returns
    -------
    new_center : np.ndarray or md.Trajectory
        Data representing the new center chosen by this iteration of kcenters
    distances : np.ndarray
        Distances between each point and its nearest center, after the
        inclusion of ``new_center``
    assignments : np.ndarray
        Assignment of each poitn to its nearest center, after the inclusion
        of ``new_center``
    center_inds : list
        The location of each center (including ``new_center``) in the
        dataset.
    """

    assert len(traj) == len(distances)
    assert len(traj) == len(assignments)
    assert np.issubdtype(type(assignments[0]), np.integer)

    new_center_index = np.argmax(distances)
    new_center = traj[new_center_index]

    logger.debug("Chose frame %s as new center", new_center_index)

    if use_triangle_inequality and np.all(assignments >= 0):
        cc_dists = distance_method(traj[center_inds], new_center)
        recompute_dists = distances > (cc_dists[assignments] / 2)

        logger.debug("Recomputing %s of %s distances",
                     np.count_nonzero(recompute_dists), len(recompute_dists))

        dist = distances.copy()
        dist[recompute_dists] = distance_method(
            traj[recompute_dists], new_center)
    else:
        dist = distance_method(traj, new_center)

    # scipy distance metrics return shape (n, 1) instead of (n), which
    # causes breakage here.
    assert len(dist.shape) == len(distances.shape)

    inds = (dist < distances)
    distances[inds] = dist[inds]
    assignments[inds] = len(center_inds)

    center_inds.append(new_center_index)
    new_center_index = np.argmax(distances)

    return new_center, distances, assignments, center_inds


def _kcenters_iteration_mpi(
        traj, distance_method, distances, assignments, center_inds,
        centers, use_triangle_inequality=False):
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
    else:
        with log.timed("Gathered distances in %.2f sec", logger.debug):
            # this could likely be accomplished with mpi.reduce instead...
            dist_locs = np.array(
                mpi.comm.allgather(np.argmax(distances)))
            dist_vals = np.array(
                mpi.comm.allgather(np.max(distances)))

        new_cluster_center_owner = np.argmax(dist_vals)
        new_cluster_center_index = dist_locs[new_cluster_center_owner]

    logger.debug("Chose frame %s (node %s) as new center",
                 new_cluster_center_index, new_cluster_center_owner)

    with log.timed("Distributed cluster ctr in %.2f sec",
                   log_func=logger.info):
        new_center = mpi.ops.distribute_frame(
            data=traj,
            world_index=new_cluster_center_index,
            owner_rank=new_cluster_center_owner)

    with log.timed("Computed distance in %.2f sec", log_func=logger.info):
        if use_triangle_inequality and np.all(assignments >= 0):
            if hasattr(centers[0], 'xyz'):
                cc_dists = np.array([distance_method(c, new_center).squeeze()
                                     for c in centers])
            else:
                cc_dists = distance_method(np.array(centers), new_center)
            recompute_dists = (distances > (cc_dists[assignments] / 2))
            logger.debug(
                "Recomputing %s of %s distances",
                np.count_nonzero(recompute_dists), len(recompute_dists))

            new_dists = distances.copy()
            new_dists[recompute_dists] = distance_method(
                traj[recompute_dists], new_center)
        else:
            new_dists = distance_method(traj, new_center)

    assert len(distances.shape) == len(new_dists.shape)

    inds = (new_dists < distances)

    distances[inds] = new_dists[inds]
    assignments[inds] = len(center_inds)

    center_inds.append(
        (new_cluster_center_owner, new_cluster_center_index))

    return new_center, distances, assignments, center_inds
