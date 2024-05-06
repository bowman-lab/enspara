import time
import logging

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from enspara.ra import ra

from .. import mpi
from .. import exception
from ..exception import ImproperlyConfigured

from ..util.log import timed

from . import util
# from enspara.cluster.util import *

try:
    from ..apps.cluster import write_assignments_and_distances_with_reassign, \
    write_centers, write_centers_indices
except:
    pass

logger = logging.getLogger(__name__)


class KMedoids(BaseEstimator, ClusterMixin, util.MolecularClusterMixin):
    """SKlearn-style object for kmedoids clustering.

    K-Medoids is a clustering algorithm similar to the k-means algorithm
    but the center of each cluster is required to actually be an
    observation in the input data.

    Parameters
    ----------
    metric : required
        Distance metric used while comparing data points.
    n_clusters : int, default=None
        The number of clusters to build using kmedoids. Only used if kmedoids
        is run without initial assignments, distances, or cluster_center_inds.
    n_iters : int, default=5
        Number of rounds of new proposed centers to run.

    Returns
    -------
    result : ClusterResult
        Subclass of NamedTuple containing assignments, distances,
        and center indices for this function.
    """

    def __init__(
            self, metric, n_clusters=None, n_iters=5, args=None, lengths=None):
        
        self.metric = util._get_distance_method(metric)

        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.args = args
        self.lengths = lengths

    def fit(self, X, assignments=None, distances=None,
            cluster_center_inds=None, X_lengths=None, args=None):
        """Takes trajectories, X, and performs KMedoids clustering.
        Automatically determines whether or not to use the MPI version of this
        algorithm. Can start from scratch or perform a warm start using inital
        assignments, distances, and cluster_center_inds. In mpi mode, the warm
        start requires initial assignments, distances, and cluster centers to be
        supplied. If not in mpi mode, can start with either just
        cluster_center_inds, or just assignments and distances, or all.

        Parameters
        ----------
        X : array-like, shape=(n_observations, n_features(, n_atoms))
            Data to cluster. In mpi mode, the user is responible for
            pre-partitioning this data across nodes.
        cluster_center_inds : 
            list, [(global_traj_id, frame_id), ...] or [index, ...], default=None
            A list of the locations of center indices with respect to all data
            not just the data on a single MPI rank.
        assignments : ndarray, shape=(X.shape[0],), default=None
            Array indicating the assignment of each frame in `X` to a
            cluster center.
        distances : ndarray, shape=(X.shape[0],), default=None
            Array giving the distance between this observation/frame and the
            relevant cluster center.
        X_lengths : list, [traj1_length, traj2_length, ...], default=None
            List of the lengths of all trajectories with respect to all data
            not just the data on a single MPI rank.
        """

        t0 = time.perf_counter()

        self.result_ = kmedoids(
            X,
            distance_method=self.metric,
            n_clusters=self.n_clusters,
            n_iters=self.n_iters,
            assignments=assignments,
            distances=distances,
            cluster_center_inds=cluster_center_inds,
            X_lengths=X_lengths, args=args)

        self.runtime_ = time.perf_counter() - t0
        return self


def kmedoids(X, distance_method, n_clusters=None, n_iters=5, assignments=None,
             distances=None, cluster_center_inds=None, proposals=None,
             X_lengths=None, args=None, lengths=None, random_state=None):
    """K-Medoids clustering.

    K-Medoids is a clustering algorithm similar to the k-means algorithm
    but the center of each cluster is required to actually be an
    observation in the input data.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, ``*``)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    distance_method : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    n_clusters : int, default=None
        Number of kmedoids clusters. Only used if cluster_center_inds are
        not supplied / can't be inferred from assignments and distances.
    n_iters : int, default=5
        Number of rounds of new proposed centers to run.
    assignments : ndarray, shape=(X.shape[0],), default=None
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],), default=None
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    cluster_center_inds :
        list, [[global_traj_id, frame_id], ...] or [index, ...], default=None
        A list of the locations of center indices with respect to all data
        not just the data on a single MPI rank.
    proposals : array-like, default=None
        If specified, this list is a list of indices to propose as a
        center (rather than choosing randomly).
    X_lengths : list, [traj1_length, traj2_length, ...], default=None
        List of the lengths of all trajectories with respect to all data
        not just the data on a single MPI rank.
    random_state : int, default=None
        Random state to fix RNG with.

    Returns
    -------
    result : ClusterResult
        Subclass of NamedTuple containing assignments, distances,
        and center indices for this function.
    """

    if cluster_center_inds is not None:
        if hasattr(cluster_center_inds[0], '__len__') and X_lengths==None:
            raise ImproperlyConfigured(
            "If cluster_center_inds is given as [[global_traj_id, frame_id],...]"
            "then X_lengths also needs to be supplied")

    if cluster_center_inds is None and n_clusters is None:
        if mpi.size() > 1:
            raise ImproperlyConfigured(
            "Must provide n_clusters or cluster_center_inds, assignments,"
            "and distances for KMedoids in MPI mode.")
        elif assignments is None and distances is None:
            raise ImproperlyConfigured(
            "Must provide n_clusters or cluster_center_inds or "
            " (assignments and distances) for KMedoids")

    distance_method = util._get_distance_method(distance_method)

    n_frames = len(X)

    if mpi.size() > 1:
        assignments, distances, cluster_center_inds = \
         _kmedoids_inputs_tree_mpi(X, distance_method, n_clusters, assignments,
                               distances, cluster_center_inds, X_lengths,
                               random_state=random_state) 
        
        #Check that the cluster_center_inds on this ranks corresponed to
        # distances with value 0.
        local_ctr_inds = [pair[1] for pair in cluster_center_inds \
                          if pair[0] == mpi.rank()]
        assert np.all(distances[local_ctr_inds] < 0.001)

    else:
        assignments, distances, cluster_center_inds = \
            _kmedoids_inputs_tree(X, distance_method, n_clusters, assignments,
                                  distances, cluster_center_inds, X_lengths,
                                  random_state=random_state)
        ctr_ids = util.find_cluster_centers(assignments, distances)
        
        #Should be all 0s, but machine precision issues means they might
        # be very close to 0 but not eactly 0.
        assert np.all(distances[cluster_center_inds] < 0.001)

    return _kmedoids_iterations(
               X, distance_method, n_iters, cluster_center_inds,
               assignments, distances, proposals=proposals, args=args, lengths=lengths,
               random_state=random_state)

def _kmedoids_inputs_tree_mpi(X, distance_method, n_clusters, assignments,
                              distances, cluster_center_inds, X_lengths,
                              random_state=None):
    """Helper function to process K-Medoids clustering inputs in mpi mode.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, ``*``)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    distance_method : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    n_clusters : int
        Number of kmedoids clusters. Only used if cluster_center_inds are
        not supplied / can't be inferred from assignments and distances.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    cluster_center_inds :
        list, [(global_traj_id, frame_id), ...] or [index, ...]
        A list of the locations of center indices with respect to all data
        not just the data on a single MPI rank.
    X_lengths : list, [traj1_length, traj2_length, ...]
        List of the lengths of all trajectories with respect to all data
        not just the data on a single MPI rank.
    random_state : int, default = None
        Random state to fix RNG with.

    Returns
    -------
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    cluster_center_inds : list, [(owner_rank, world_index), ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    """
   
    # If we're not given warm start, we need to randomly generate
    # cluster_center_inds by communicating across ranks. Then, we
    # can obtain center coordinates and calculate assignments and distances
    # on each rank
    if (cluster_center_inds is None and distances is None
       and assignments is None):

        for i in range(n_clusters):
            r, idx = mpi.ops.randind(np.arange(X), check_random_state(random_state))
            cluster_center_inds.append((r, idx))
        
        medoid_coords = []
        assert len(cluster_center_inds[0]) == 2
        for center_idx, (rank, frame_idx) in enumerate(cluster_center_inds):
            assert rank < mpi.size()
            new_center = mpi.ops.distribute_frame(
                data=X, owner_rank=rank, world_index=frame_idx)
            medoid_coords.append(new_center)

        assignments, distances = util.assign_to_nearest_center(
                X, medoid_coords, distance_method)
        
    # If we are given a warm start, we have to translate cluster_center_inds
    # into the form that is appropriate for MPI communication
    elif (cluster_center_inds is not None and distances is not None
         and assignments is not None):
        cluster_center_inds = ctr_ids_mpi(cluster_center_inds, X_lengths)

    else:
        raise ImproperlyConfigured(
            "For KMedoids, MPI mode can start from scratch without "
            "assignments, distances, or cluster_center_inds. "
            "Or, it requires that all are supplied.")
    
    return assignments, distances, cluster_center_inds

def _kmedoids_inputs_tree(
        X, distance_method, n_clusters, assignments, distances,
        cluster_center_inds, X_lengths, random_state=None):
    """Helper function to process K-Medoids clustering inputs in mpi mode.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, ``*``)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    distance_method : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    n_clusters : int
        Number of kmedoids clusters. Only used if cluster_center_inds are
        not supplied / can't be inferred from assignments and distances.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    cluster_center_inds :
        list, [(global_traj_id, frame_id), ...] or [index, ...]
        A list of the locations of center indices with respect to all data
        not just the data on a single MPI rank.
    X_lengths : list, [traj1_length, traj2_length, ...]
        List of the lengths of all trajectories with respect to all data
        not just the data on a single MPI rank.
    random_state : int, default = None
        Random state to fix RNG with.

    Returns
    -------
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    cluster_center_inds : list, [index, ...]
        A list of the locations of center indices.
    """

    rng = np.random.default_rng(seed=random_state)

    if ((assignments is not None and distances is None) or 
        (assignments is None and distances is not None)):
        raise ImproperlyConfigured(
            "Assignments and distances need to both be supplied, "
            "or neither supplied.")

    # If no cluster center indices were given, we need to infer them
    # from assignments and distances, or randomly generate them
    if cluster_center_inds is None:
        if assignments is not None and distances is not None:
            cluster_center_inds = \
                util.find_cluster_centers(assignments,distances)
        else:
            # for short lists, np.random.random_integers sometimes forgets
            # to assign something to each cluster. This will simply repeat
            # the assignments if that is the case.
            cluster_center_inds = np.array([])
            while len(np.unique(cluster_center_inds)) < n_clusters:
                cluster_center_inds = \
                    rng.integers(0,len(X),n_clusters)
    
    # If cluster_center_inds is given as [(trj id, frame id), ...]
    elif hasattr(cluster_center_inds[0], '__len__'):
        cluster_center_inds = [sum(X_lengths[:cluster_center_inds[i][0]]) \
                               + cluster_center_inds[i][1] for i in \
                               np.arange(len(cluster_center_inds))] 

    # Now we need to make sure we have assignments and distances
    if assignments is None and distances is None:
        assignments, distances = util.assign_to_nearest_center(
                X, X[cluster_center_inds], distance_method)

    return assignments, distances, cluster_center_inds

def ctr_ids_mpi(cluster_center_inds, lengths):
    """Map cluster_center_inds to MPI compatible format
   
    Parameters
    ----------
    cluster_center_inds :
        list, [(global_traj_id, frame_id), ...] or [index, ...]
        A list of the locations of center indices with respect to all data
        not just the data on a single MPI rank.
    X_lengths : list, [traj1_length, traj2_length, ...]
        List of the lengths of all trajectories with respect to all data
        not just the data on a single MPI rank.

    Returns
    -------
    updated_ctr_inds : list, [(rank, index), ...]
        List of cluster center indices in format expected for MPI mode.
"""

    num_procs = mpi.size()
    updated_ctr_inds = []
    global_inds = ra.RaggedArray(np.arange(sum(lengths)),lengths=lengths)

    if not hasattr(cluster_center_inds[0], '__len__'):
        # Convert from [global_frame_ind, ...] to 
        # [[global_traj_id, local_frame_id],...]
        cluster_center_inds = [[np.where(global_inds == c)[0][0], \
                           np.where(global_inds == c)[1][0]] for c in \
                           cluster_center_inds]

    # Converting from [[global_traj_id, local_frame_id],...] to 
    # [(mpi_rank, local_frame_ind), ...]
    for pair in cluster_center_inds:
        global_traj_id, frame_id = pair
        mpi_rank = global_traj_id % num_procs
        trajs_owned = global_inds[np.arange(len(lengths))[mpi_rank::num_procs]]
        trajs_owned_local_inds = \
            ra.RaggedArray(np.arange(sum(trajs_owned.lengths)),
                           lengths=trajs_owned.lengths)
        local_trj_id = int(global_traj_id/num_procs)
        concat_idx = trajs_owned_local_inds[local_trj_id][frame_id]
        updated_ctr_inds.append((mpi_rank,concat_idx))

    return updated_ctr_inds

def _kmedoids_iterations(
        X, distance_method, n_iters, cluster_center_inds,
        assignments, distances, proposals=None, args=None, 
        lengths=None, random_state=None):
    """Inner loop performing kmedoids updates.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, *)
        Data to cluster. The user is responsible for pre-partitioning
        this data across nodes.
    disance_method : callable
        Function that takes a parameter like `X` and a single frame
        of `X` (_i.e._ X.shape[1:]).
    n_iters : int
        Number of rounds of new proposed centers to run.
    cluster_center_inds : list, [(rank, index), ...] if MPI or [index, ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    proposals : array-like, default=None
        If specified, this list is a list of indices to propose as a
        center (rather than choosing randomly).
    random_state : int, default = None
        Random state to fix RNG with.

    Returns
    -------
    result : ClusterResult
        Subclass of NamedTuple containing assignments, distances,
        and center indices for this function.
    """

    for i in range(n_iters):
        cluster_center_inds, distances, assignments, centers = \
            _kmedoids_pam_update(X, distance_method, cluster_center_inds,
                                 assignments, distances, proposals=proposals,
                                 random_state=random_state)
        result = util.ClusterResult(
            center_indices=cluster_center_inds,
            assignments=assignments,
            distances=distances,
            centers=centers)

        if args != None and args.save_intermediates:
            #if on the last iteration, about to save anyways...
            int_result = result.partition(lengths)
            int_indcs, int_assigs, int_dists, int_centers = int_result

            if i != n_iters -1:
                with timed("Wrote center indices in %.2f sec.", logger.info):
                    util.write_centers_indices(
                        args.center_indices,
                        [(t, f * args.subsample) for t, f in int_indcs],
                        intermediate_n=f'kmedoids-{i}')
                with timed("Wrote center structures in %.2f sec.", logger.info):
                    util.write_centers(int_result, args, intermediate_n=f'kmedoids-{i}')
                util.write_assignments_and_distances_with_reassign(int_result, args, 
                    intermediate_n=f'kmedoids-{i}')
        logger.info("KMedoids update %s", i)

    return result

def _msq(x):
    return mpi.ops.striped_array_mean(np.square(x))


def _propose_new_center_amongst(X, state_inds, mpi_mode, random_state):
    """Propose a new center amongst a list of indices.

    Parameters
    ----------
    X : array-like, shape=(n_observations, n_features, *)
        Data from which to propose the new center.
    state_inds : array-like
        The indices (in X) from which to propose new centers.
    mpi_mode : boolean
        Propose a center in the form (rank, local index) rather than in
        the form of a single index.
    random_state : numpy.RandomState
        The state of the RNG to use when drawing new random values.
    """
    random_state = check_random_state(random_state)

    # TODO: make it impossible to choose the current center
    if mpi_mode:
        r, idx = mpi.ops.randind(state_inds, random_state)
        if mpi.rank() == r:
            i = mpi.comm.bcast(state_inds[idx], root=r)
        else:
            i = mpi.comm.bcast(None, root=r)
        proposed_center = mpi.ops.distribute_frame(
            data=X, owner_rank=r, world_index=i)
        proposed_center_ind = (r, i)

        logger.debug(
            "Proposing new center %s, at %s.",
            proposed_center_ind, (r, idx))
    else:
        proposed_center_ind = random_state.choice(state_inds)
        proposed_center = X[proposed_center_ind]

    return proposed_center, proposed_center_ind


def _kmedoids_pam_update(
        X, metric, medoid_inds, assignments, distances, proposals=None,
        cost=_msq, random_state=None):
    """Compute a kmedoids update using Partitioning Around Medoids (PAM)

    PAM iteratively proposes a new cluster center from among the points
    assigned to a cluster center, recomputes the cost function, and
    accepts the proposal if cost goes down. Its time complexity is
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
    medoid_inds : list, [(rank, index), ...] if MPI or [index, ...]
        A list of the locations of center indices in terms of the rank
        of the node that owns them and the index within that world.
    assignments : ndarray, shape=(X.shape[0],)
        Array indicating the assignment of each frame in `X` to a
        cluster center.
    distances : ndarray, shape=(X.shape[0],)
        Array giving the distance between this observation/frame and the
        relevant cluster center.
    proposals : array-like, default=None
        If specified, this list is a list of indices to propose as a
        center (rather than choosing randomly).
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
    updated_centers : list
        List of center coordinates (n_atoms, 3) or (n_features,) after
        kmedoids updates have been run
    """

    assert np.issubdtype(type(assignments[0]), np.integer)
    assert len(assignments) == len(X)
    assert len(distances) == len(X)

    random_state = check_random_state(random_state)

    if proposals is not None:
        logger.debug("Got proposals, won't randomly propose.")
        if len(proposals) != len(medoid_inds):
            raise exception.DataInvalid(
                "Length of 'proposals' didn't match length of 'medoid_inds' "
                "({} != {}).".format(len(proposals), len(medoid_inds)))
        if (hasattr(proposals[0], '__len__') !=
                hasattr(medoid_inds[0], '__len__')):
            raise exception.DataInvalid(
                "Depth of 'proposals' didn't match 'medoid_inds' "
                "(proposals[0] == {}, whereas medoid_inds[0] == {})".format(
                    proposals[0], medoid_inds[0]))

    # first we build a list of the actual coordinates of the cluster centers
    # this list will be updated as we go; this is primarily because we want
    # to limit the amount of communication that happens when we're running
    # MPI mode.
    medoid_coords = []
    if hasattr(medoid_inds[0], '__len__'):
        assert len(medoid_inds[0]) == 2
        for center_idx, (rank, frame_idx) in enumerate(medoid_inds):
            assert rank < mpi.size()
            new_center = mpi.ops.distribute_frame(
                data=X, owner_rank=rank, world_index=frame_idx)
            medoid_coords.append(new_center)
    else:
        medoid_coords = [X[i] for i in medoid_inds]

    acceptances = 0
    for cid in range(len(medoid_inds)):
        state_inds = np.where(assignments == cid)[0]

        # first, we propose a new center. This works a bit differently
        # if we're running with MPI, because we want to make a choice
        # that uniformly distributed across any node.
        if proposals is None:
            proposed_center, proposed_center_ind = _propose_new_center_amongst(
                X, state_inds,
                mpi_mode=hasattr(medoid_inds[0], '__len__'),
                random_state=random_state)
        else:
            proposed_center_ind = proposals[cid]
            if hasattr(proposed_center_ind, '__len__'):
                proposed_center = mpi.ops.distribute_frame(
                    data=X, owner_rank=proposed_center_ind[0],
                    world_index=proposed_center_ind[1])
            else:
                proposed_center = X[proposed_center_ind]

        logger.debug("Proposed new medoid (%s -> %s) for k=%s",
                     medoid_inds[cid], proposed_center_ind, cid)

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

        with timed("Recomputed nearest medoid for {n} points in %.2f sec."
                   .format(n=np.count_nonzero(dst_up_assig_this)),
                   logger.debug):
            ambig_assigs, ambig_dists = util.assign_to_nearest_center(
                X[dst_up_assig_this], new_medoids, metric)

        new_assig[dst_up_assig_this] = ambig_assigs
        new_dist[dst_up_assig_this] = ambig_dists

        # every element of new_dist and new_assig should have been touched
        assert np.all(new_assig >= 0)
        assert np.all(new_dist >= 0)


        # Added timing to cost computation
        with timed("Computed costs points in %.2f sec.",
                   logger.debug):
            old_cost = cost(distances)
            new_cost = cost(new_dist)

        if new_cost < old_cost:
            logger.debug(
                "Accepted proposed center for k=%s: cost %.5f -> %.5f).",
                cid, old_cost, new_cost)
            distances, assignments = new_dist, new_assig
            medoid_coords = new_medoids
            medoid_inds[cid] = proposed_center_ind
            acceptances += 1
        else:
            logger.debug(
                "Rejected proposed center for k=%s: cost %.5f -> %.5f).",
                cid, old_cost, new_cost)

    logger.info("Kmedoid sweep reduced cost to %.7f (%.2f%% acceptance)",
                min(old_cost, new_cost), acceptances / len(medoid_inds) * 100)

    return medoid_inds, distances, assignments, medoid_coords
