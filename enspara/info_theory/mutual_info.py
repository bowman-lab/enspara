# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging
import warnings
import ctypes

import multiprocessing as mp
from functools import partial

import numpy as np

from .. import exception
from ..util import array as ra

from . import entropy
from . import libinfo


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_features_states(states, n_states):
    n_features = len(n_states)

    if len(states[0][0]) != n_features:
        raise exception.DataInvalid(
            ("The number-of-states vector's length ({s}) didn't match the "
             "width of state assignments array with shape {a}.")
            .format(s=len(n_states), a=len(states[0][0])))

    if not all(len(t[0]) == len(states[0][0]) for t in states):
        raise exception.DataInvalid(
            ("The number of features differs between trajectories. "
             "Numbers of features were: {l}.").
            format(l=[len(t[0]) for t in states]))


def nmi_apc_matrix(assignments, n_states, n_procs=None):
    """Compute the normalized mutual information-average product
    correlation across trajectories of assigned states, messages,
    sequences, etc.

    Parameters
    ----------
    assignments_a : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    assignments_b : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    n_a_states_list : array, shape(n_features_a,)
        Number of possible states for each feature in `states_a`
    n_b_states_list : array, shape=(n_features_b,)
        Number of possible states for each feature in `states_b`

    Returns
    -------
    nmi_apc : float
        The NNI-APC between each pair of states in assignments_a and
        assignments_b

    Notes
    -----
    This function implements the NMI-APC metric proposed by Lopez et al
    for assessing sequence covariation. The equation is a combination of
    normalized mutual entropy (NMI) and average product correlation
    (APC), both of which are used for sequence covariation. The equation
    for NMI-APC is

    NMI-APC(M_i, M_j) = I(M_i, M_j) - APC(M_i, M_j) / H(M_i, M_j)

    where I is the mutual information and H is the shannon entropy of
    the joint distributions of random variables/distributions M_i and
    M_j. Supplimentary Note 1 in ref [1] is an excellent summary of this
    approach.

    See Also
    --------
    mi_matrix : computes the mutual information for a message.
    apc_matrix : computes the average product correlation for a set of
        assignments

    References
    ----------
    [1] Dunn, S.D., et al (2008) Bioinformatics 24 (3): 330--40.
    [2] Lopez, T., et al (2017) Nat. Struct. & Mol. Biol. 24: 726--33.
        doi:10.1038/nsmb.3440
    """

    # compute mutual information
    mi_arr = mi_matrix(assignments, assignments, n_states, n_states, n_procs)
    apc_arr = mi_to_apc(mi_arr)
    nmi_apc_arr = mi_arr - apc_arr

    H_marginal = np.diag(mi_arr)

    # compute the joint shannon entropies using MI and marginal entropies
    H_joint = np.zeros_like(nmi_apc_arr)
    for i in range(H_joint.shape[0]):
        for j in range(H_joint.shape[1]):
            H_joint[i, j] = mi_arr[i, j] - H_marginal[i] - H_marginal[j]

    # normalize NMI-APC by joint entropies.
    with warnings.catch_warnings():
        # suppress potential divide by zero
        warnings.simplefilter("ignore")
        nmi_apc_arr /= H_joint

    return nmi_apc_arr


def apc_matrix(assignments_a, assignments_b, n_states_a, n_states_b,
               n_procs=None):
    """Compute the average product correlation for  trajectories of
    assigned states, messages, sequences, etc.

    Parameters
    ----------
    assignments_a : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    assignments_b : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    n_a_states_list : array, shape(n_features_a,)
        Number of possible states for each feature in `states_a`
    n_b_states_list : array, shape=(n_features_b,)
        Number of possible states for each feature in `states_b`

    Returns
    -------
    apc : float
        The average product correlation between each pair of states in
        assignments_a and assignments_b

    See Also
    --------
    mi_matrix : computes the mutual information for sets of
        messages/assignments/sequences.

    References
    ----------
    [1] Dunn, S.D., et al (2008) Bioinformatics 24 (3): 330--40.
    [2] Lopez, T., et al (2017) Nat. Struct. & Mol. Biol. 24: 726--33.
        doi:10.1038/nsmb.3440
    """

    mi_arr = mi_matrix(assignments_a, assignments_b, n_states_a, n_states_b,
                       n_procs)
    apc_matrix = mi_to_apc(mi_arr)

    return apc_matrix


def deconvolute_network(G_obs):
    """Compute the deconvolution of a given network. This method
    attempts to estimate the direct network given a network that is a
    combination of direct and indirect effects. For example, if A is
    correlated with B and B is correlated with C, then A and C will also
    be correlated.

    Specifically, this method solves for G_dir given G_obs:

    G_obs = G_dir + G_dir^2 + G_dir^3 + ...
          = G_dir * (I - G_dir)^-1

    Parameters
    ----------
    G_obs : ndarray, shape=(n, n)
        Weight matrix for the observed correlations in the network

    Returns
    -------
    G_dir : ndarray, shape=(n, n)
        The direct correlations inferred from G_obs

    References
    ----------
    [1] Feizi, S., et al (2013) Nat. Biotechnol 31 (8): 726--33.
    """

    from numpy.linalg import eig, inv

    v, w = eig(G_obs)
    v_dir = v / (1 + v)
    sig_dir = np.diagflat(v_dir)
    G_dir = np.matmul(np.matmul(w, sig_dir), inv(w))

    return G_dir

def mi_to_apc(mi_arr):
    """Given a mutual information matrix, compute the average product
    correlation.

    Parameters
    ----------
    mi_arr : ndarray, shape=(n_features, n_features)
        Mutual information matrix of which to compute the APC.

    Returns
    -------
    apc_matrix : ndarray, shape=(n_features, n_features)

    Notes
    -----
    The equation for APC given MI is

    APC(M_i, M_j) = Σr I(M_i, M_r)*I(M_j, M_r) ; r ∈ [0, n_features]
    """

    if len(mi_arr.shape) != 2:
        raise exception.DataInvalid(
            'MI arrays must be 2D. Got %s.' % len(mi_arr.shape))
    if mi_arr.shape[0] != mi_arr.shape[1]:
        raise exception.DataInvalid(
            "Only square MI arrays' APC can be computed. Shape was %s."
            % mi_arr.shape)
    if not np.all(mi_arr.T == mi_arr):
        raise exception.DataInvalid(
            "Only symmetric MI arrays' APC can be computed")

    return np.matmul(mi_arr, mi_arr) / np.prod(mi_arr.shape)


def mi_matrix(assignments_a, assignments_b, n_states_a, n_states_b,
              compute_diagonal=True, n_procs=None):
    """Compute the all-to-all matrix of mutual information across
    trajectories of assigned states.

    Parameters
    ----------
    assignments_a : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    assignments_b : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    n_a_states_list : array, shape(n_features_a,)
        Number of possible states for each feature in `states_a`
    n_b_states_list : array, shape=(n_features_b,)
        Number of possible states for each feature in `states_b`
    compute_diagonal: bool, default=True
        Compute the diagonal of the MI matrix, which is the Shannon
        entropy of the univariate distribution (i.e. the feature
        itself).
    n_procs : int, default=1
        Number of cores to parallelize this computation across

    Returns
    -------
    mi : np.ndarray, shape=(n_features, n_features)
        Array where cell i, j is the mutual information between the ith
        feature of assignments_a and the jth feature of assignments_b of
        the mutual information between trajectories a and b for each
        feature.
    """

    n_features = assignments_a[0].shape[1]

    c_dtype = ctypes.c_int
    np_dtype = np.int32

    assignments_a = _end_to_end_concat(assignments_a)
    sa_a = _make_shared_array(assignments_a, c_dtype)
    logger.debug("Allocated shared-memory array of size %s", len(sa_a))

    if assignments_a is not assignments_b:
        assignments_b = _end_to_end_concat(assignments_b)
        sa_b = _make_shared_array(assignments_b, c_dtype)
        logger.debug(
            "Detected that assignments_a is not assignments_b, creating "
            "second shared-memory array of size %s", len(sa_a))
    else:
        logger.debug(
            "Detected that assignments_a is the same as assignments_a; "
            "not allocating a second shared-memory array")
        sa_b = sa_a

    mi = np.zeros((n_features, n_features))
    mi_calc_indices = np.triu_indices_from(
        mi, k=(0 if compute_diagonal else 1))

    # initializer function shares shared data with everybody.
    def _init(arr_a_, arr_b_):
        global arr_a, arr_b
        arr_a, arr_b = arr_a_, arr_b_

    with mp.Pool(processes=n_procs, initializer=_init,
                 initargs=(sa_a, sa_b)) as p:

        # partial function to repeat the number of states and shape info
        mi_func_repeated_part = partial(
            _mi_parallel_cell,
            n_states_a=n_states_a, n_states_b=n_states_b,
            shape_a=assignments_a.shape,
            shape_b=assignments_b.shape,
            np_dtype=np_dtype)

        mi_list = p.starmap(mi_func_repeated_part, zip(*mi_calc_indices))

    mi[mi_calc_indices] = mi_list
    mi += mi.T

    return mi


def mi_matrix_serial(states_a_list, states_b_list, n_a_states, n_b_states):
    n_traj = len(states_a_list)
    n_features = states_a_list[0].shape[1]
    mi = np.zeros((n_features, n_features))

    for i in range(n_features):
        logger.debug(i, "/", n_features)
        for j in range(i+1, n_features):
            jc = joint_counts(
                states_a_list[0][:, i], states_b_list[0][:, j],
                n_a_states[i], n_b_states[j])
            for k in range(1, n_traj):
                jc += joint_counts(
                    states_a_list[k][:, i], states_b_list[k][:, j],
                    n_a_states[i], n_b_states[j])
            mi[i, j] = mutual_information(jc)
            min_num_states = np.min([n_a_states[i], n_b_states[j]])
            mi[i, j] /= np.log(min_num_states)
            mi[j, i] = mi[i, j]

    return mi


def _end_to_end_concat(trjlist):
    """Concatenate all trajectories in an end-to-end way, to create rows
    that are a single feature across all timepoints.
    """

    if hasattr(trjlist, '_data'):
        return trjlist._data

    elif hasattr(trjlist, 'dtype'):
        return trjlist.reshape(-1, trjlist.shape[-1])

    else:
        warnings.warn(
            "Computing mutual information is coercing list of arrays "
            "into a ragged array, doubling memory usage.",
            ResourceWarning)

        rag = ra.RaggedArray(trjlist)

        return rag._data


def _make_shared_array(in_arr, dtype):

    if not np.issubdtype(in_arr.dtype, np.integer):
        raise exception.DataInvalid(
            "Given array (type '%s') is an integral type (e.g. int32). "
            "Mutual information calculations require discretized state "
            "trajectories." % in_arr.dtype)

    arr = mp.Array(dtype, in_arr.size, lock=False)
    arr[:] = in_arr.flatten()

    return arr


def _mi_parallel_cell(feature_id_a, feature_id_b, n_states_a, n_states_b,
                      shape_a, shape_b, np_dtype):

    assigs_a = np.frombuffer(arr_a, dtype=np_dtype).reshape(shape_a)
    assigs_b = np.frombuffer(arr_b, dtype=np_dtype).reshape(shape_b)

    trj_a = assigs_a[:, feature_id_a]
    trj_b = assigs_b[:, feature_id_b]

    n_a = n_states_a[feature_id_a]
    n_b = n_states_b[feature_id_b]

    jc = joint_counts(trj_a, trj_b, n_a, n_b)

    assert not np.any(np.isnan(jc))

    mi = mutual_information(jc) / np.log(min([n_a, n_b]))

    return mi


def joint_counts(state_traj_1, state_traj_2,
                 n_states_1=None, n_states_2=None):
    """Compute the matrix H, where H[i, j] is the number of times t
    where trajectory_1[t] == i and trajectory[t] == j.

    Parameters
    ----------
    state_traj_1 : array-like, dtype=int, shape=(n_observations,)
        List of assignments to discrete states for trajectory 1.
    state_traj_2 : array-like, dtype=int, shape=(n_observations,)
        List of assignments to discrete states for trajectory 2.
    n_states_1 : int, optional
        Number of total possible states in state_traj_1. If unspecified,
        taken to be max(state_traj_1)+1.
    n_states_2 : int, optional
        Number of total possible states in state_traj_2 If unspecified,
        taken to be max(state_traj_2)+1.
    """

    if n_states_1 is None:
        n_states_1 = state_traj_1.max()+1
    if n_states_2 is None:
        n_states_2 = state_traj_2.max()+1

    H = libinfo.bincount2d(
        state_traj_1.astype('int'), state_traj_2.astype('int'),
        n_states_1, n_states_2)

    return H


def mutual_information(joint_counts):
    """Compute the mutual information of a joint counts matrix.

    Parameters
    ----------
    joint_counts : ndarray, dtype=int, shape=(n_states, n_states)
        Matrix where the cell (i, j) represents the number of times the
        combination of state i and state j were observed concurrently.

    Returns
    -------
    mutual_information : float
        The mutual information of the joint counts matrix
    """

    counts_axis_1 = joint_counts.sum(axis=1)
    counts_axis_2 = joint_counts.sum(axis=0)

    p1 = counts_axis_1/counts_axis_1.sum()
    p2 = counts_axis_2/counts_axis_2.sum()
    joint_p = joint_counts.flatten()/joint_counts.sum()

    h1 = entropy.shannon_entropy(p1)
    h2 = entropy.shannon_entropy(p2)
    joint_h = entropy.shannon_entropy(joint_p)

    return h1+h2-joint_h
