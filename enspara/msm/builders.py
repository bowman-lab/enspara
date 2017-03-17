import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def transpose(C):
    '''Transform a counts matrix to a probability matrix using the
    transpose method.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to symmetrize
    '''

    C_sym = C + C.T
    probs = normalize(C_sym)

    # C + C.T changes the type of sparse matrices, so recast here.
    if type(C) is not type(probs):
        probs = type(C)(probs)

    return probs


def normalize(C):
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
