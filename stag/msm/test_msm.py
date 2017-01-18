from nose.tools import with_setup, assert_equal
from numpy.testing import assert_array_equal

import numpy as np
import scipy.sparse

from .transition_matrices import counts_to_probs, assigns_to_counts


def test_assigns_to_counts_negnums():
    '''counts_to_probs ignores -1 values
    '''

    in_m = np.array(
            [[0, 2,  0, -1],
             [1, 2, -1, -1],
             [1, 0,  0, 1]])

    counts = assigns_to_counts(in_m)

    expected = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 0, 0]])

    assert_array_equal(counts.toarray(), expected)


def test_counts_to_probs_types():
    '''counts_to_probs accepts & returns ndarrays, spmatrix subclasses.
    '''

    arr_types = [
        np.array, scipy.sparse.lil_matrix, scipy.sparse.csr_matrix,
        scipy.sparse.coo_matrix, scipy.sparse.csc_matrix,
        scipy.sparse.dia_matrix, scipy.sparse.dok_matrix
    ]

    for arr_type in arr_types:

        in_m = arr_type(
            [[0, 2, 8],
             [4, 2, 4],
             [7, 3, 0]])
        out_m = counts_to_probs(in_m)

        assert_equal(type(in_m), type(out_m))

        # cast to ndarray if necessary for comparison to correct result
        try:
            out_m = out_m.toarray()
        except AttributeError:
            pass

        out_m = np.round(out_m, decimals=1)

        expected = np.array(
            [[0,   0.2, 0.8],
             [0.4, 0.2, 0.4],
             [0.7, 0.3, 0]])

        assert_array_equal(
            out_m,
            expected)
