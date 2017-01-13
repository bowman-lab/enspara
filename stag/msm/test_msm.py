from numpy.testing import assert_array_equal

import numpy as np

from .transition_matrices import counts_to_probs


def test_counts_to_probs_ndarray():

    probs = counts_to_probs(np.array([[3, 2, 1],
                                      [1, 2, 3]]))

    expected = np.array()

    assert_array_equal(
        probs,
        expected)
