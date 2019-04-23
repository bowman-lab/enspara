import time
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from . import kcenters
from . import kmedoids
from . import util

from ..exception import ImproperlyConfigured
from .. import mpi

logger = logging.getLogger(__name__)


class KHybrid(BaseEstimator, ClusterMixin, util.MolecularClusterMixin):
    """Sklearn-style object for khybrid clustering.

    KHybrid clustering uses the k-centers protocol to define cluster
    centers and the kmedoids protocol to refine the clustering.

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
    kmedoids_updates : it, default=None
        Number of rounds of kmedoids to run.
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
    .. [1] Beauchamp, K. A. et al. MSMBuilder2: Modeling Conformational
    Dynamics at the Picosecond to Millisecond Scale. J. Chem. Theory
    Comput. 7, 3412–3419 (2011).
    """

    def __init__(self, metric, n_clusters=None, cluster_radius=None,
                 kmedoids_updates=5, random_first_center=False,
                 random_state=None, mpi_mode=None):

        if n_clusters is None and cluster_radius is None:
            raise ImproperlyConfigured("Either n_clusters or cluster_radius "
                                       "is required for KHybrid clustering")

        self.kmedoids_updates = kmedoids_updates
        self.n_clusters = n_clusters
        self.cluster_radius = cluster_radius
        self.random_first_center = random_first_center

        self.metric = util._get_distance_method(metric)
        self.random_state = check_random_state(random_state)
        self.mpi_mode = mpi_mode if mpi_mode is not None else mpi.size() != 1

    def fit(self, X, init_centers=None):
        """Takes trajectories, X, and performs KHybrid clustering.
        Optionally continues clustering from an initial set of cluster
        centers.

        Parameters
        ----------
        X : array-like, shape=(n_observations, n_features(, n_atoms))
            Data to cluster.
        """

        t0 = time.perf_counter()

        self.result_ = hybrid(
            X, self.metric,
            n_iters=self.kmedoids_updates,
            n_clusters=self.n_clusters,
            dist_cutoff=self.cluster_radius,
            random_first_center=self.random_first_center,
            init_centers=init_centers,
            random_state=self.random_state,
            mpi_mode=self.mpi_mode)

        self.runtime_ = time.perf_counter() - t0

        return self


def hybrid(
        X, distance_method, n_iters=5, n_clusters=np.inf,
        dist_cutoff=0, random_first_center=False,
        init_centers=None, random_state=None, mpi_mode=False):

    distance_method = util._get_distance_method(distance_method)

    result = kcenters.kcenters(
        X, distance_method, n_clusters=n_clusters, dist_cutoff=dist_cutoff,
        init_centers=init_centers, random_first_center=random_first_center,
        mpi_mode=mpi_mode)

    cluster_center_inds, assignments, distances, centers = (
        result.center_indices, result.assignments, result.distances,
        result.centers)

    for i in range(n_iters):
        cluster_center_inds, distances, assignments, centers = \
            kmedoids._kmedoids_pam_update(
                X, distance_method,
                cluster_center_inds, assignments, distances,
                random_state=random_state)

        logger.info("KMedoids update %s of %s", i, n_iters)

    return util.ClusterResult(
        center_indices=cluster_center_inds,
        assignments=assignments,
        distances=distances,
        centers=centers)
