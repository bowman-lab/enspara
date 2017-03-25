from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_allclose

import numpy as np

from . import disorder


def test_trj_ord_disord_times_one_transition():

    transition_times = np.array([0.0, 0.5, 0.5, 1.0, 1.0, 0.5])

    result = disorder.traj_ord_disord_times(transition_times)

    expected = (1.25, 0.5, 0.1, 0.5)
    assert_equal(expected, result)


def test_trj_ord_disord_times_many_transition():

    transition_times = np.array(
        [[0.0, 0.5, 0.5, 1.0, 1.0, 0.5],
         [0.5, 1.0, 1.0, 1.5, 0.0, 0.5]])

    result = disorder.traj_ord_disord_times(transition_times)

    expected = (1.25, 0.5, 0.1, 0.5)
    assert_equal(expected, result)
