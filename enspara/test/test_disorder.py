from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

from ..util import array as ra
from ..cards import disorder


def test_transition_times():

    states = np.array([0, 0, 1, 1, 1, 2, 3, 3])
    transitions = disorder.transitions(states)
    assert_array_equal([1, 4, 5], transitions)


def test_transition_times_multidim():

    states = np.array(
        [[0, 0, 1, 1, 1, 2, 3, 3],
         [0, 0, 1, 1, 1, 2, 2, 2]])
    transitions = disorder.transitions(states)

    assert_array_equal([1, 4, 5], transitions[0])
    assert_array_equal([1, 4], transitions[1])


def test_transition_times_ragged():

    states = ra.RaggedArray(
        [[0, 0, 1, 1, 1, 2, 3, 3],
         [0, 0, 1, 1, 1]])
    transitions = disorder.transitions(states)

    assert_array_equal([1, 4, 5], transitions[0])
    assert_array_equal([1], transitions[1])


def test_trj_ord_disord_times_one_transition():

    transition_times = np.array([0.0, 0.5, 0.5, 1.0, 1.0, 0.5])

    result = disorder.traj_ord_disord_times(transition_times)

    expected = (1.25, 0.5, 0.1, 0.5)
    assert_equal(expected, result)


# def test_trj_ord_disord_times_many_transition():

#     transition_times = np.array([[3, 10, 15],
#                                  [5, 10, 11]])

#     result = disorder.traj_ord_disord_times(transition_times)

#     expected = (1.25, 0.5, 0.1, 0.5)
#     assert_equal(expected, result)
