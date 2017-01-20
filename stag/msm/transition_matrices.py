# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _assigns_to_counts_helper(
        assigns_1d, n_states=None, lag_time=1, sliding_window=True):
    # TODO: check trajectory is 1d array

    if n_states is None:
        n_states = assigns_1d.max() + 1

    if sliding_window:
        start_states = assigns_1d[:-lag_time:1]
        end_states = assigns_1d[lag_time::1]
    else:
        start_states = assigns_1d[:-lag_time:lag_time]
        end_states = assigns_1d[lag_time::lag_time]
    transitions = np.row_stack((start_states, end_states))
    counts = np.ones(transitions.shape[1], dtype=int)
    C = scipy.sparse.coo_matrix((counts, transitions),
                                shape=(n_states, n_states), dtype=int)

    return C.tolil()


def assigns_to_counts(
        assigns, n_states=None, lag_time=1, sliding_window=True):
    """Count transitions between states in a single trajectory.

    Parameters
    ----------
    assigns : array, shape=(traj_len, )
        A 2-D array where each row is a trajectory consisting of a
        sequence of state indices.
    n_states : int, default=None
        The number of states. This is useful for controlling the
        dimensions of the transition count matrix in cases where the
        input trajectory does not necessarily visit every state.
    lag_time : int, default=1
        The lag time (i.e. observation interval) for counting
        transitions.
    sliding_window : bool, default=True
        Whether to use a sliding window for counting transitions or to
        take every lag_time'th state.

    Returns
    -------
    C :  array, shape=(n_states, n_states)
        A transition count matrix.
    """

    n_traj = len(assigns)

    assigns = np.array([a[np.where(a != -1)] for a in assigns])

    if n_states is None:
        n_states = 0
        for i in range(n_traj):
            traj_n_states = assigns[i].max() + 1
            if traj_n_states > n_states:
                n_states = traj_n_states

    C = scipy.sparse.lil_matrix((n_states, n_states), dtype=int)
    for i in range(n_traj):
        traj_C = _assigns_to_counts_helper(
            assigns[i], n_states=n_states, lag_time=lag_time,
            sliding_window=sliding_window)
        C += traj_C

    return C


def _normalize_rows(C):
    """Normalize every row of a transition count matrix to obtain a
    transition probability matrix.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        A transition count matrix.

    Returns
    -------
    T : array, shape=(n_states, n_states)
        A row-normalized transition probability matrix.
    """

    n_states = C.shape[0]

    if scipy.sparse.isspmatrix(C):
        C_csr = scipy.sparse.csr_matrix(C).asfptype()
        weights = np.asarray(C_csr.sum(axis=1)).flatten()
        inv_weights = np.zeros(n_states)
        inv_weights[weights > 0] = 1.0 / weights[weights > 0]
        inv_weights = scipy.sparse.dia_matrix((inv_weights, 0),
                                              C_csr.shape).tocsr()
        T = inv_weights.dot(C_csr)
        T = type(C)(T)  # recast T to the input type
    else:
        weights = np.asarray(C.sum(axis=1)).flatten()
        inv_weights = np.zeros(n_states)
        inv_weights[weights > 0] = 1.0 / weights[weights > 0]
        T = C * inv_weights.reshape((n_states, 1))

    return T


def counts_to_probs(C, symmetrization=None):
    """Infer a transition probability matrix from a transition count
    matrix using the specified method to enforce microscopic
    reversibility.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        A transition count matrix.
    symmetrization : {None, 'transpose', 'mle'}
        Method to use to enforce microscopic reversibility.

    Returns
    -------
    T : array, shape=(n_states, n_states)
        A row-normalized transition probability matrix.
    """

    if symmetrization is None:
        T = _normalize_rows(C)
    elif symmetrization.lower() == "transpose":
        C_sym = C + C.T
        T = _normalize_rows(C_sym)
    elif symmetrization.lower() == "mle":
        raise NotImplementedError("MLE option not yet implemented")
    else:
        raise NotImplementedError(
            "Invalid symmetrization option %s in count_matrix_to_probabilities"
            % symmetrization)

    return T


def eigenspectra(T, n_eigs=None, left=True, maxiter=100000, tol=1E-30):
    """Compute the eigenvectors and eigenvalues of a transition
    probability matrix.

    Parameters
    ----------
    T : array, shape=(n_states, n_states)
        A transition probability matrix.
    n_eigs : int, optional
        The number of eigenvalues and eigenvectors to compute. If not
        speficied, all are computed.
    left: bool, default=False
        Compute the left eigenvalues rather than the right eigenvalues.
    maxiter : int, default=100000
        Limit the maximum number of iterations used by the sparse
        eigenvalue solver. (Used only for sparse matrices.)
    tol : float, default=1e-30
        Relative accuracy for eigenvalues (stopping criterion). (Used
        only for sparse matrices.)

    Returns
    -------
    vals, vecs : 2-tuple, (ndarray, ndarray)
        Eigenvalues and eigenvectors for this system, respectively.
    """

    if n_eigs is None:
        n_eigs = T.shape[0]
    elif n_eigs < 2:
        raise ValueError('n_eig must be greater than or equal to 2')

    # left eigenvectors input processing (?)
    T = T.T if left else T

    # performance improvement for small arrays; also prevents erroring
    # out when ndim - 2 <= n_eigs and T is sparse.
    if T.shape[0] < 1000 and scipy.sparse.issparse(T):
        T = T.toarray()

    if scipy.sparse.issparse(T):
        vals, vecs = scipy.sparse.linalg.eigs(
            T.tocsr(), n_eigs, which="LR", maxiter=maxiter, tol=tol)
    else:
        vals, vecs = scipy.linalg.eig(T)

    order = np.argsort(-np.real(vals))
    vals = vals[order]
    vecs = vecs[:, order]

    # normalize the first eigenvector to obtain the eq populations
    vecs[:, 0] /= vecs[:, 0].sum()

    vals = np.real(vals[:n_eigs])
    vecs = np.real(vecs[:, :n_eigs])

    return vals, vecs


def eq_probs(T, maxiter=100000, tol=1E-30):
    val, vec = eigenspectra(T, n_eigs=3, left=True, maxiter=maxiter, tol=tol)

    return vec[:, 0]
