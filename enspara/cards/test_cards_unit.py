import numpy as np

from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_allclose

from enspara import cards
from enspara import exception


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
