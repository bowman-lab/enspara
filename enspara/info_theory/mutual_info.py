"""Module for high-level computations involving mutual information.

Includes such goodies as mutual information matrix calculation, wrappers
for array of joint count matrices calculations, MI with weights, and
normalized MI.
"""

import logging
import warnings

import numpy as np

from .. import exception
from . import libinfo


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mi_matrix(Xs, Ys, n_x, n_y, normalize=True):
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
        As `n_x`, but for `X`
    normalize : bool, default=True
        Normalize by channel capacity

    Returns
    -------
    mi : np.ndarray, shape=(n_features, n_features)
        Array where cell i, j is the mutual information between the ith
        feature of assignments_a and the jth feature of assignments_b of
        the mutual information between trajectories a and b for each
        feature.

    See Also
    --------
    channel_capacity_normalization, weighted_mi, joint_counts,
    mutual_information
    """

    jc = None
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        logger.debug("Starting joint-counts %s", i)
        jc_i = joint_counts(X, Y, np.max(n_x), np.max(n_y))

        if not hasattr(jc, 'shape'):
            jc = jc_i
        else:
            if jc.shape != jc_i.shape:
                raise exception.DataInvalid(("Trajectory %s gave a joint "
                    "counts matrix of shape %s where %s was expected. "
                    "Are you sure all your trajectories have the same "
                    "number of features?") % (i, jc_i.shape, jc.shape))
            jc += jc_i

    mi = mutual_information(jc)

    if normalize:
        mi = channel_capacity_normalization(mi, n_x, n_y)

    return mi


def weighted_mi(features, weights, normalize=True):
    """Compute a mutual information matrix using weighted observations.

    This function computes the mutual information of weighted samples by
    actually computing the marginal probability distributions P(x),
    P(y), and P(x, y) for each variable using the weights, rather than
    by computing a joint counts matrix.

    Parameters
    ----------
    features : np.ndarray, shape=(n_observations, n_features)
        Array of observations of multiple variables (features) between
        which to compute the pairwise mutual information.
    weights : np.ndarray, shape=(n_observations)
        Array containing a probability distribution across observations
        by which to weight each observation.
    normalize : bool, default=True
        Normalize by channel capacity (two in this case.)

    Returns
    -------
    mi : np.ndarray, shape=(n_features, n_features)
        Array where cell i, j is the mutual information between feature
        i and feature j. This array is symmmetic (i.e. mi.T == mi).
    """
    weights = np.array(weights, copy=True)

    assert len(features.shape) == 2
    assert len(weights.shape) == 1
    assert weights.shape[0] == features.shape[0]
    assert np.all(weights >= 0)
    assert np.sum(weights), 1

    if np.all(np.unique(features) != np.array([0, 1])):
        raise NotImplementedError(
            "Computing the weighted mutual information of non-binary "
            "variables is not yet supported.")

    mi_mtx = np.zeros((features.shape[1], features.shape[1]), dtype=np.float)

    for i in range(len(mi_mtx)):
        P_x = [weights[features[:, i] == 0].sum(),
               weights[features[:, i] == 1].sum()]
        for j in range(i, len(mi_mtx)):
            P_y = [weights[features[:, j] == 0].sum(),
                   weights[features[:, j] == 1].sum()]

            P_x_y = np.zeros((2, 2))
            for u in range(2):
                for v in range(2):
                    P_x_y[u, v] = weights[(features[:, i] == u) &
                                          (features[:, j] == v)].sum()

            mi = 0
            for u in range(len(P_x_y)):
                for v in range(len(P_x_y)):
                    if (P_x_y[u, v] != 0) and (P_x[u] != 0) and (P_y[v] != 0):
                        val = P_x_y[u, v] * np.log(P_x_y[u, v]/(P_x[u]*P_y[v]))
                        mi += val

            if np.isnan(mi):
                mi = 0

            mi_mtx[i, j] = mi

    mi_mtx += mi_mtx.T
    # the diagonal gets doubled by adding the transpose, reverse here
    mi_mtx[np.diag_indices_from(mi_mtx)] /= 2

    if normalize:
        mi_mtx = channel_capacity_normalization(mi_mtx, 2, 2)

    return mi_mtx


def mi_matrix_serial(states_a_list, states_b_list, n_a_states, n_b_states, normalize=True):
    """Compute the mutual information matrix in a serial fashion.

    Used mostly for testing.
    """
    n_traj = len(states_a_list)
    n_features = states_a_list[0].shape[1]
    mi = np.zeros((n_features, n_features))

    for i in range(n_features):
        logger.debug(i, "/", n_features)
        for j in range(i, n_features):
            jc = joint_counts(
                states_a_list[0][:, i], states_b_list[0][:, j],
                n_a_states[i], n_b_states[j])
            for k in range(1, n_traj):
                jc += joint_counts(
                    states_a_list[k][:, i], states_b_list[k][:, j],
                    n_a_states[i], n_b_states[j])

            mi[i, j] = mutual_information(jc)
            mi[j, i] = mi[i, j]

    if normalize:
        mi = channel_capacity_normalization(mi, n_a_states, n_b_states)

    return mi


def joint_counts(X, Y=None, n_x=None, n_y=None):
    """Compute the array of joint counts matrices between X and Y (or itself.)

    This function is thread-parallelized using OpenMP. The degree of
    parallelization can be controlled by the OMP_NUM_THREADS evironment
    variable.

    Parameters
    ----------
    X : np.ndarray, shape=(n_observations, n_features)
        List of assignments to discrete states for trajectory 1.
    Y : np.ndarray, shape=(n_observations, n_features), default=None
        As X, but can be left as None to indicate that joint counts are
        to be computed between X and itself.
    n_x : int, default=None
        Number of total possible states in X. If unspecified, taken to
        be max(X)+1.
    n_y : int, default=None
        Number of total possible states in Y. If unspecified, taken to be
        max(Y)+1.

    Returns
    -------
    jc : np.ndarray, shape=(n_features, n_features, n_x, n_y)
        Array of joint counts matrices, where the cell [x, y, i, j]
        holds the number of times features X and Y were found
        simultaneously in states i and j.
    """

    if len(X.shape) == 1:
        X = X[..., None]
    if Y is not None and len(Y.shape) == 1:
            Y = Y[..., None]

    if n_x is None:
        n_x = X.max()+1

    if Y is None:
        if n_y is not None:
            warnings.warn("n_y unused if Y is None.")
        jc = libinfo.matrix_bincount2d(X, X, n_x, n_x)
    else:
        if n_y is None:
            n_y = Y.max()+1

        if X.dtype != Y.dtype:
            warnings.warn(
                "Feature trajs (types %s and %s) being uptyped to match." %
                (X.dtype, Y.dtype), exception.PerformanceWarning)

            if X.dtype.itemsize > Y.dtype.itemsize:
                Y = Y.astype(X.dtype)
            else:
                X = X.astype(Y.dtype)

        jc = libinfo.matrix_bincount2d(X, Y, n_x, n_y)

    return jc


def mutual_information(jc):
    """Compute the mutual information of a joint counts matrix or matrix
    of joint counts matrices.

    Parameters
    ----------
    jc : ndarray, dtype=int, shape=(n_feat, n_feat, n_states, n_states)
        Array where the cell (i, j, u, v) represents the number of times
        feature i was seen in state u and feature j was seein in state v.

    Returns
    -------
    mutual_information : np.ndarray, shape=(n_feat, n_feat)
        The mutual information of the joint counts matrix
    """

    jc = _validate_joint_counts_matrix(jc)

    # sum across all possibilities
    n_obs_a_i = jc.sum(axis=-1)
    n_obs_b_i = jc.sum(axis=-2)

    P_a = n_obs_a_i / n_obs_a_i.sum(axis=-1)[..., None]
    P_b = n_obs_b_i / n_obs_b_i.sum(axis=-1)[..., None]

    n_obs = n_obs_a_i.sum(axis=-1)
    P_a_b = jc / n_obs[..., None, None]

    mi = np.zeros(shape=jc.shape[0:2])
    for i in range(jc.shape[0]):
        for j in range(jc.shape[1]):
            P_x_y = P_a_b[i, j]
            P_x = P_a[i, j]
            P_y = P_b[i, j]

            for u in range(P_x_y.shape[0]):
                for v in range(P_x_y.shape[1]):
                    unddef = ((P_x_y[u, v] == 0) or
                              (P_x[u] == 0) or
                              (P_y[v] == 0))
                    if not unddef:
                        mi[i, j] += (P_x_y[u, v] *
                                     np.log(P_x_y[u, v]/(P_x[u]*P_y[v])))

    return mi


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


def channel_capacity_normalization(mi, n_x, n_y):
    """Normalize an MI matrix by the channel capacity of each feature pair.

    The channel capacity is a information-theoretic quantity that measures the maximum amount of information that can be reliably transmitted along a channel. In our simple case, this is the log of the number of states.

    Parameters
    ----------
    mi : np.ndarray, shape=(n_features_a, n_features_b)
        Mutual information matrix
    n_x : np.ndarray or int, shape(n_features_a)
        Vector with element i representing the number of states
        feature_a i takes.
    n_y : np.ndarray or int, shape(n_features_b)
        Vector with element i representing the number of states
        feature_b i takes.
    Returns
    -------
    cc_mi : np.ndarray, shape=(n_features_a, n_features_b)
        Mutual information matrix scaled by channel capacity.
    """
    mi = mi.copy()

    if not hasattr(n_x, '__len__'):
        n_x = [n_x]*mi.shape[0]
    if not hasattr(n_y, '__len__'):
        n_y = [n_y]*mi.shape[1]
    for i in range(mi.shape[0]):
        for j in range(mi.shape[1]):
            min_num_states = np.min([n_x[i], n_y[j]])
            mi[i, j] /= np.log(min_num_states)

    return mi


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


def _validate_joint_counts_matrix(jc):

    if len(jc.shape) == 2:
        raise exception.DataInvalid(
            ("Expected a 4D array of joint counts matrices, but got a 2D "
             " array. If your dataset is a single joint counts matrix, "
             "try `jc[None, None, ...]` to expand its dimensions."))
    if len(jc.shape) != 4:
        raise exception.DataInvalid(
            ("Expected a 4D array of joint counts matrices, but an array "
             "with shape %s.") % (jc.shape,))

    return jc


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
