from numpy.testing import assert_array_equal

import numpy as np
import scipy.sparse

from .transition_matrices import counts_to_probs


def test_counts_to_probs_ndarray():
    '''counts_to_probs accepts ndarrays.'''

    probs = counts_to_probs(np.array([[0, 2, 8],
                                      [4, 2, 4],
                                      [7, 3, 0]]))
    probs = np.round(probs.toarray(), decimals=1)

    expected = np.array(
        [[0,   0.2, 0.8],
         [0.4, 0.2, 0.4],
         [0.7, 0.3, 0]])

    assert_array_equal(
        probs,
        expected)


def test_counts_to_probs_sparse():
    '''counts_to_probs accepts lil sparse arrays'''

    probs = counts_to_probs(scipy.sparse.lil_matrix(
        [[0, 2, 8],
         [4, 2, 4],
         [7, 3, 0]]))

    expected = np.array(
        [[0,   0.2, 0.8],
         [0.4, 0.2, 0.4],
         [0.7, 0.3, 0]])

    probs = np.round(probs.toarray(), decimals=1)
    assert_array_equal(
        probs,
        expected)
