import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .transition_matrices import eq_probs


def transpose(C):
    '''Transform a counts matrix to a probability matrix using the
    transpose method.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to symmetrize

    Returns
    -------
    C : array, shape=(n_states, n_states)
        Transition counts matrix after symmetrization.
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.
    eq_probs : array, shape=(n_states)
        Equilibrium probability distribution of `T`.
    '''

    C_sym = C + C.T
    probs = row_normalize(C_sym)

    # C + C.T changes the type of sparse matrices, so recast here.
    if type(C) is not type(probs):
        probs = type(C)(probs)
        C_sym = type(C)(C_sym)

    equilibrium = np.array(np.sum(C_sym, axis=1) / np.sum(C_sym)).flatten()

    return C_sym/2, probs, equilibrium


def normalize(C):
    '''Transform a transition counts matrix to a transition probability
    matrix by row-normalizing it. This does not guarantee ergodicity or
    enforce equilibrium.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to normalize.

    Returns
    -------
    C : array, shape=(n_states, n_states)
        Transition counts matrix after symmetrization.
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.
    eq_probs : array, shape=(n_states)
        Equilibrium probability distribution of `T`.
    '''

    probs = row_normalize(C)
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
