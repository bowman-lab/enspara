# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

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


def mi_to_nmi_apc(mutual_information, H_marginal=None):
    """Compute the normalized mutual information-average product
    correlation given a mutual information matrix.

    Parameters
    ----------
    mutual_information : array, shape=(n_features, n_features)
        Mutual information array
    H_marginal: array, shape=(n_features), default=None
        The marginal shannon entropy of each feature. If None (the
        default), the diagonal of the mutual information matrix is
        assumed to be the marginal entropies, as computed by
        enspara.info_theory.mutual_information when compute_diagonal is
        True.

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

    _validate_mutual_information_matrix(mutual_information)

    # compute mutual information
    apc_arr = mi_to_apc(mutual_information)
    nmi = mi_to_nmi(mutual_information, H_marginal)

    with warnings.catch_warnings():
        # zeros in the NMI matix cause nans
        warnings.simplefilter("ignore")

        # the NMI computes the MI/joint_H, thus NMI^-1 * MI = joint_H.
        H_joint = (nmi ** -1) * mutual_information

    nmi_apc_arr = mutual_information - apc_arr

    # normalize NMI-APC by joint entropies.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # suppress potential divide by zero
        nmi_apc_arr /= H_joint

    nmi_apc_arr[np.isnan(nmi_apc_arr)] = 0

    return nmi_apc_arr


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


def mi_to_nmi(mutual_information, H_marginal=None):
    """Given a mutual information matrix, compute the normalized mutual
    information, which is given by:

    NMI(M_i, M_j) = I(M_i, M_j) / H(M_i, M_j)

    where I is the mutual information function and H is the shannon
    entropy of the joint distribution of M_i and M_j.

    Parameters
    ----------
    mutual_information : ndarray, shape=(n_features, n_features)
        Mutual information matrix.information
    H_marginal : array-like, shape=(n_features)
        The marginal entropies of each variable. If None, values will be
        inferred from the diagonal of `mutual_information`.

    Returns
    -------
    nmi : ndarray, shape=(n_features, n_features)
        The normalized mutual information matrix
    """

    _validate_mutual_information_matrix(mutual_information)

    if H_marginal is None:
        H_marginal = np.diag(mutual_information)
    if np.any(H_marginal == 0):
        warnings.warn(
            'H_marginal contains zero entries. This may lead to '
            'negative information.')

    if len(H_marginal) != len(mutual_information):
        raise exception.DataInvalid(
            "H_marginal must be the same length as the mutual "
            "information matrix. Got %s and %s." %
            (len(H_marginal), len(mutual_information)))

    if np.all(H_marginal == 0) or np.any(np.isnan(H_marginal)):
        raise exception.DataInvalid(
            'The mutual information matrix must have non-zero entries '
            'and cannot contain any nan values. Found %s zero entries '
            'and %s nan entries.' % (
                np.count_nonzero(H_marginal == 0),
                np.count_nonzero(np.isnan(H_marginal))))

    # if we got H_marginal as an argument, we'll fill it in for
    # simplicity's sake, but we don't want to modify the array the user
    # gave in place, so we copy
    mutual_information = mutual_information.copy()
    mutual_information[np.diag_indices_from(mutual_information)] = H_marginal

    # compute the joint shannon entropies using MI and marginal entropies
    H_joint = np.zeros_like(mutual_information)
    for i in range(len(H_joint)):
        for j in range(len(H_joint)):
            H_joint[i, j] = (
                H_marginal[i] + H_marginal[j] -
                mutual_information[i, j])

    nmi = mutual_information / H_joint

    # all diagonal entries should be 1
    np.fill_diagonal(nmi, 1)

    # nans introduced by H_joint == 0 should be 0
    nmi[np.isnan(nmi)] = 0

    return nmi


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

    Interestingly, this is the same as (MI/n)^2, where n is the number
    of rows or columns in the matrix.

    See Also
    --------
    enspara.info_theory.deconvolute_network : computes a similar
        quantity, but using MI^3, MI^4, ... too.

    References
    ----------
    [1] Dunn, S.D., et al (2008) Bioinformatics 24 (3): 330--40.
    """

    _validate_mutual_information_matrix(mi_arr)

    return np.matmul(mi_arr, mi_arr) / (len(mi_arr) * len(mi_arr))


def mi_matrix(assignments_a, assignments_b, n_states_a,
              n_states_b, compute_diagonal=True, n_procs=None):
    """Compute the all-to-all matrix of mutual information across
    trajectories of assigned states.

    Parameters
    ----------
    assignments_a : array-like, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    assignments_b : array-like, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    n_states_a : int or array, shape(n_features_a,)
        Number of possible states for each feature in `states_a`. If an
        integer is given, it is assumed to apply to all features.
    n_states_b : int or array, shape=(n_features_b,)
        As `n_states_a`, but for `assignments_b`
    compute_diagonal: bool, default=True
        Compute the diagonal of the MI matrix, which is the Shannon
        entropy of the univariate distribution (i.e. the feature
        itself).
    n_procs : int, default=None
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

    if not hasattr(n_states_a, '__len__'):
        n_states_a = [n_states_a] * n_features
    if not hasattr(n_states_b, '__len__'):
        n_states_b = [n_states_b] * n_features

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

    try:
        arr = mp.Array(dtype, in_arr.size, lock=False)
    except:
        logger.error(
            "Multiprocessing's Array failed to allocate an array of size "
            "%s of dtype %s. Typically this means /dev/shm or similar "
            "is full or too small.", in_arr.size, dtype)
        raise

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
        state_traj_1, state_traj_2,
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


def _validate_mutual_information_matrix(mi):
    """Check features of mutual information matrix:

    0. must be 2D
    1. must be square
    2. must be symmetric
    """

    if len(mi.shape) != 2:
        raise exception.DataInvalid(
            'MI arrays must be 2D. Got %s.' % len(mi.shape))

    if mi.shape[0] != mi.shape[1]:
        raise exception.DataInvalid(
            "Mutual information matrices must be square; got shape %s."
            % mi.shape)

    if not np.all(mi.T == mi):
        diffpos = np.where(mi.T != mi)
        raise exception.DataInvalid(
            "Mutual information matrices must be symmetric; found "
            "differences at %s positions." % len(diffpos[0]))
