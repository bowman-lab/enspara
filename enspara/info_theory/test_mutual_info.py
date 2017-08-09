from nose.tools import assert_equal, assert_raises
from numpy.testing import (assert_array_equal, assert_allclose,
                           assert_almost_equal)

import numpy as np

from enspara import cards
from enspara import exception

from enspara.util import array as ra
from enspara.info_theory import mutual_info


def zero_mi_np():
    n_trjs = 3
    n_frames = 10000
    n_features = 5

    data = np.random.randint(1, 5, (n_trjs, n_frames, n_features))
    n_states = [5] * n_features

    return data, n_states


def nonzero_mi_np():
    a, n_states = zero_mi_np()
    a[:, :, -2] = a[:, :, -1]
    return a, n_states


def zero_mi_ra():
    data, n_states = zero_mi_np()
    a = ra.RaggedArray(array=data[0], lengths=[1000, 2000, 5000, 2000])
    return a, n_states


def zero_mi_list():
    a, n_states = zero_mi_ra()
    l = [row for row in a]
    return l, n_states


def nonzero_mi_ra():
    data, n_states = nonzero_mi_np()
    a = ra.RaggedArray(array=data[0], lengths=[1000, 2000, 5000, 2000])
    return a, n_states


def nonzero_mi_list():
    a, n_states = nonzero_mi_ra()
    l = [row for row in a]
    return l, n_states


def test_check_feature_size():

    states_same = [
        np.array([[0, 0, 0],
                  [0, 0, 0]]),
        np.array([[0, 0, 0],
                  [0, 0, 0]])]

    mutual_info.check_features_states(states_same, [2, 2, 2])

    with assert_raises(exception.DataInvalid):
        mutual_info.check_features_states(states_same, [2, 2])

    states_different = [
        np.array([[0, 0, 0],
                  [0, 0, 0]]),
        np.array([[0, 0, 0]])]

    mutual_info.check_features_states(states_different, [2, 2, 2])

    states_different_features = [
        np.array([[0, 0],
                  [0, 0]]),
        np.array([[0, 0, 0],
                  [0, 0, 0]])]

    with assert_raises(exception.DataInvalid):
        mutual_info.check_features_states(states_different_features, [3])


def test_symmetrical_mi_zero():

    zero_mi_funcs = [zero_mi_np, zero_mi_ra, zero_mi_list]

    for a, n_states in (f() for f in zero_mi_funcs):
        mi = mutual_info.mi_matrix(a, a, n_states, n_states)
        assert_allclose(mi, 0, atol=1e-3)


def test_symmetrical_mi_nonzero():

    nonzero_mi_funcs = [nonzero_mi_np, nonzero_mi_ra, nonzero_mi_list]
    for a, n_states in (f() for f in nonzero_mi_funcs):

        mi = mutual_info.mi_matrix(a, a, n_states, n_states)

        assert_almost_equal(mi[-1, -2], 0.86114, decimal=3)
        mi[-1, -2] = mi[-2, -1] = 0

        assert_allclose(mi, 0, atol=1e-3)


def test_joint_count_binning():

    trj1 = np.array([1]*3 + [2]*6 + [1]*6)
    trj2 = np.array([1]*9 + [0]*3 + [2]*3)

    expected_jc = np.array([[0, 0, 0],
                            [3, 3, 3],
                            [0, 6, 0]])

    jc = mutual_info.joint_counts(trj1, trj2)
    assert_equal(jc.dtype, 'int')
    assert_array_equal(jc, expected_jc)

    jc = mutual_info.joint_counts(trj1, trj2, 3, 3)
    assert_equal(jc.dtype, 'int')
    assert_array_equal(jc, expected_jc)
