"""The builders submodule is where all the methods that fit a transition
probability matrix and/or equilibrium probability distributions of an
MSM live. All the builders (i.e. anything in this module not prefixed
with an underscore) should be safe to pass to an MSM object as its
builder.
"""

import logging
import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from enspara import exception

from .transition_matrices import eq_probs
from .libmsm import _mle_prinz_dense

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mle(C, prior_counts=None, calculate_eq_probs=True):
    """Transform a counts matrix to a probability matrix using
    maximum-liklihood estimation (prinz) method.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to symmetrize
    prior_counts: int or array, shape=(n_states, n_states), default=None
        Number or matrix of pseudocounts to add to the transition counts
        matrix.
    calculate_eq_probs: bool, default=True
        Compute the equilibrium probability distribution of the output
        matrix T. This flag is provided for compatibility with other
        builders only, as it has no effect in MLE (and, in fact, emits a
        warning).

    Returns
    -------
    C : array, shape=(n_states, n_states)
        Transition counts matrix after the addition of pseudocounts.
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.
    eq_probs : array, shape=(n_states)
        Equilibrium probability distribution of `T`.

    See Also
    --------
    msmbuilder.msm.MarkovStateModel and
    msmbuilder._markovstatemodel._transmat_mle_prinz

    References
    ----------
    [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J Chem. Phys. 134.17 (2011): 174105.
    """

    C = _apply_prior_counts(C, prior_counts)

    sparsetype = np.array
    if scipy.sparse.issparse(C):
        sparsetype = type(C)
        C = C.todense()

    equilibrium = None
    if not calculate_eq_probs:
        warnings.warn('MLE method cannot suppress calculation of '
                      'equilibrium probabilities, since they are calculated '
                      'together.', category=RuntimeWarning)
        T, _ = _prinz_mle_py(C)
    else:
        T, equilibrium = _prinz_mle_py(C)

    C = sparsetype(C)
    T = sparsetype(T)

    return C, T, equilibrium


def transpose(C, prior_counts=None, calculate_eq_probs=True):
    """Transform a counts matrix to a probability matrix using the
    transpose method.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to symmetrize
    calculate_eq_probs: bool, default=True
        Compute the equilibrium probability distribution of the output
        matrix T. For transpose, this computation is cheap, but the flag
        is still supported for compatibility purposes.

    Returns
    -------
    C : array, shape=(n_states, n_states)
        Transition counts matrix after symmetrization.
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.
    eq_probs : array, shape=(n_states)
        Equilibrium probability distribution of `T`.
    """

    C = _apply_prior_counts(C, prior_counts)

    C_sym = C + C.T
    probs = _row_normalize(C_sym)

    # C + C.T changes the type of sparse matrices, so recast here.
    if type(C) is not type(probs):
        probs = type(C)(probs)
        C_sym = type(C)(C_sym)

    equilibrium = None
    if calculate_eq_probs:
        equilibrium = np.array(C_sym.sum(axis=1) / C_sym.sum()).flatten()

    return C_sym/2, probs, equilibrium


def normalize(C, prior_counts=None, calculate_eq_probs=True):
    """Transform a transition counts matrix to a transition probability
    matrix by row-normalizing it. This does not guarantee ergodicity or
    enforce equilibrium.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to normalize.
    calculate_eq_probs: bool, default=True
        Compute the equilibrium probability distribution of the output
        matrix T. This is useful because calculating the eq probs is
        expensive.

    Returns
    -------
    C : array, shape=(n_states, n_states)
        Transition counts matrix after symmetrization.
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.
    eq_probs : array, shape=(n_states)
        Equilibrium probability distribution of `T`.
    """

    C = _apply_prior_counts(C, prior_counts)

    probs = _row_normalize(C)

    equilibrium = None
    if calculate_eq_probs:
        equilibrium = eq_probs(probs)

    return C, probs, equilibrium


def _apply_prior_counts(C, prior_counts):
    """Apply prior_counts to counts matrix C
    """

    if prior_counts is not None:
        try:
            C = C + prior_counts
        except NotImplementedError:
            C = np.array(C.todense()) + prior_counts

    return C


def _row_normalize(C):
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
        C = np.array(C)
        weights = np.asarray(C.sum(axis=1)).flatten()
        inv_weights = np.zeros(n_states)
        inv_weights[weights > 0] = 1.0 / weights[weights > 0]
        T = C * inv_weights.reshape((n_states, 1))

    return T


def _prinz_mle(C, *args, **kwargs):

    if scipy.sparse.issparse(C):
        assert False
    else:
        return _mle_prinz_dense(C, *args, **kwargs)


def _prinz_mle_py(C, tol=1e-10, max_iter=10**5):
    """Fit a transition probability using the detailed balance-enforced
    maximum-liklihood estimation (Prinz) method.

    This method is not numpy vectorized or written in C and is very
    slow for large counts matrices.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to symmetrize
    tol: float, default=1e-10
        The log-likelihood change at which the iterative method is
        considered to have been converged
    max_iter: int, default=10**5
        The maximum number of allowed iterations. If this this number of
        iterations is reached, calculation will stop and a warning will
        be emitted.

    Returns
    -------
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.

    References
    ----------
    [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J Chem. Phys. 134.17 (2011): 174105.
    """
    C = C.copy().astype(float)
    X = C + C.T

    X_rs = X.sum(axis=1)
    C_rs = C.sum(axis=1)

    assert np.all(X_rs > 0)
    assert np.all(C_rs > 0)

    oldlogl = 0
    for n_iter in range(max_iter):
        logl = 0

        for i in range(len(C)):
            tmp = X[i,i];
            denom = C_rs[i] - C[i, i];
            if (denom > 0):
                X[i, i] = C[i, i] * (X_rs[i] - X[i, i]) / denom;

            X_rs[i] = X_rs[i] + (X[i, i] - tmp);

            if (X[i, i] > 0):
                logl += C[i,i] * np.log(X[i, i] / X_rs[i]);

        for i in range(len(C) - 1):
            for j in range(i+1, len(C)):

                a = (C_rs[i] - C[i, j]) + (C_rs[j] - C[j, i])
                b = C_rs[i] * (X_rs[j] - X[i, j]) +\
                    C_rs[j] * (X_rs[i] - X[i, j]) -\
                    (C[i, j] + C[j, i]) * (X_rs[i] + X_rs[j] - 2*X[i, j])

                c = -(C[i, j] + C[j, i]) *\
                     (X_rs[i] - X[i, j]) *\
                     (X_rs[j] - X[i,j])

                assert c <= 0

#                 /* the new value */
                if (a == 0):
                    v = X[j, i];
                else:
                    v = (-b + np.sqrt((b**2) - (4*a*c))) / (2*a)

#                 /* update the row sums */
                X_rs[i] = X_rs[i] + (v - X[i, j])
                X_rs[j] = X_rs[j] + (v - X[j, i])

#                 /* add in the new value */
                X[i, j] = v
                X[j, i] = v

                if (X[i, j] > 0):
                    logl += (
                        ((C[i,j] * np.log(X[i, j]) / X_rs[i])) +
                        ((C[j,i] * np.log(X[j, i]) / X_rs[j])))


        if abs(logl - oldlogl) > tol:
            oldlogl = logl
        else:
            break

    if n_iter == max_iter - 1:
        warnings.warn(
            exception.ConvergenceWarning,
            "Prinz MLE did not converge after %s iterations." % n_iter)

    T = X / X.sum(axis=-1).reshape(len(X), 1)
    pi = X_rs / X_rs.sum()[..., None]

    assert np.all(T.sum(axis=1) == 1)
    assert np.sum(pi) == 1

    return T, pi
