import numpy as np

from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_allclose

from enspara import cards
from enspara import exception
from enspara.util import array as ra


def test_check_feature_size():

    states_same = [
        np.array([[0, 0, 0],
                  [0, 0, 0]]),
        np.array([[0, 0, 0],
                  [0, 0, 0]])]

    cards.check_features_states(states_same, [2, 2, 2])

    with assert_raises(exception.DataInvalid):
        cards.check_features_states(states_same, [2, 2])

    states_different = [
        np.array([[0, 0, 0],
                  [0, 0, 0]]),
        np.array([[0, 0, 0]])]

    cards.check_features_states(states_different, [2, 2, 2])

    states_different_features = [
        np.array([[0, 0],
                  [0, 0]]),
        np.array([[0, 0, 0],
                  [0, 0, 0]])]

    with assert_raises(exception.DataInvalid):
        cards.check_features_states(states_different_features, [3])


def test_mi_matrix_ra():

    from numpy import random

    n_frames = 10000
    n_features = 5

    n_states = np.random.randint(1, 5, (n_frames, n_features))

    a = ra.RaggedArray(array=n_states, lengths=[1000, 2000, 5000, 2000])

    n_states = [5] * n_features

    mi = cards.mi_matrix(a, a, n_states, n_states)

    assert_allclose(mi, 0, atol=1e-3)
