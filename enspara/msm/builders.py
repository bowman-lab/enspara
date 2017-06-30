import logging
import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .transition_matrices import eq_probs

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

    try:
        with warnings.catch_warnings():
            # this is here to catch sklearn deprication warnings
            warnings.simplefilter("ignore")
            from msmbuilder.msm._markovstatemodel import \
                _transmat_mle_prinz as mle
    except ImportError:
        logger.error("To use MLE MSM fitting, MSMBuilder is required. "
                     "See http://msmbuilder.org.")
        raise

    # this is extremely wierd, since I actually expect anything taking
    # a counts matrix to take integers?
    C = C.astype('double')

    if prior_counts:
        C += prior_counts

    sparsetype = np.array
    if scipy.sparse.issparse(C):
        sparsetype = type(C)
        C = C.todense()

    equilibrium = None
    if not calculate_eq_probs:
        logger.warning('MLE method cannot suppress calculation of '
                       'equilibrium probabilities, since they are calculated '
                       'together.')
        T, _ = mle(C)
    else:
        T, equilibrium = mle(C)

    C = sparsetype(C)
    T = sparsetype(T)

    return C, T, equilibrium


def transpose(C, calculate_eq_probs=True):
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

    C_sym = C + C.T
    probs = row_normalize(C_sym)

    # C + C.T changes the type of sparse matrices, so recast here.
    if type(C) is not type(probs):
        probs = type(C)(probs)
        C_sym = type(C)(C_sym)

    equilibrium = None
    if calculate_eq_probs:
        equilibrium = np.array(np.sum(C_sym, axis=1) / np.sum(C_sym)).flatten()

    return C_sym/2, probs, equilibrium


def normalize(C, calculate_eq_probs=True):
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

    probs = row_normalize(C)

    equilibrium = None
    if calculate_eq_probs:
        equilibrium = eq_probs(probs)

    return C, probs, equilibrium


def row_normalize(C):
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
