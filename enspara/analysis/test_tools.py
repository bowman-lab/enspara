import unittest

import numpy as np

from nose.tools import assert_raises, assert_equals, assert_is
from numpy.testing import (
    assert_array_equal, assert_array_almost_equal, assert_almost_equal)

from .msm_tools import (
    KL_divergence, state_relative_entropy, Q_from_assignments,
    relative_entropy)



class Test_Tools(unittest.TestCase):

    def test_KL_divergence(self):
        # reference distributions
        P_test = np.array(
            [
                [0.5, 0.5, 0],
                [0.25, 0.25, 0.5],
                [0, 0.25, 0.75]])

        # divergant distributions
        Q_test = np.array(
            [
                [0.25, 0.25, 0.5],
                [0.25, 0.25, 0.5],
                [0.1, 0.65, 0.25]])

        # True divergences
        true_divergences_base_2 = np.array([1., 0.0, 0.84409397])
        true_divergences_base_e = np.array([0.6931472, 0.0, 0.58508136])
        true_divergences_base_10 = np.array([0.3010299957, 0.0, 0.25409760])

        # Test full matrix distributions with different bases
        test_divergences_base_2 = KL_divergence(P_test, Q_test)
        test_divergences_base_e = KL_divergence(P_test, Q_test, base=np.e)
        test_divergences_base_10 = KL_divergence(P_test, Q_test, base=10)

        assert_array_almost_equal(
            true_divergences_base_2, test_divergences_base_2, 7)
        assert_array_almost_equal(
            true_divergences_base_e, test_divergences_base_e, 7)
        assert_array_almost_equal(
            true_divergences_base_10, test_divergences_base_10, 7)

        # Test individual distributions
        test_divergences_base_2_r0 = KL_divergence(P_test[0], Q_test[0])
        test_divergences_base_2_r1 = KL_divergence(P_test[1], Q_test[1])
        test_divergences_base_2_r2 = KL_divergence(P_test[2], Q_test[2])

        test_divergences_base_e_r0 = KL_divergence(P_test[0], Q_test[0], base=np.e)
        test_divergences_base_e_r1 = KL_divergence(P_test[1], Q_test[1], base=np.e)
        test_divergences_base_e_r2 = KL_divergence(P_test[2], Q_test[2], base=np.e)

        test_divergences_base_10_r0 = KL_divergence(P_test[0], Q_test[0], base=10)
        test_divergences_base_10_r1 = KL_divergence(P_test[1], Q_test[1], base=10)
        test_divergences_base_10_r2 = KL_divergence(P_test[2], Q_test[2], base=10)

        assert_almost_equal(
            true_divergences_base_2[0], test_divergences_base_2_r0, 7)
        assert_almost_equal(
            true_divergences_base_2[1], test_divergences_base_2_r1, 7)
        assert_almost_equal(
            true_divergences_base_2[2], test_divergences_base_2_r2, 7)

        assert_almost_equal(
            true_divergences_base_e[0], test_divergences_base_e_r0, 7)
        assert_almost_equal(
            true_divergences_base_e[1], test_divergences_base_e_r1, 7)
        assert_almost_equal(
            true_divergences_base_e[2], test_divergences_base_e_r2, 7)

        assert_almost_equal(
            true_divergences_base_10[0], test_divergences_base_10_r0, 7)
        assert_almost_equal(
            true_divergences_base_10[1], test_divergences_base_10_r1, 7)
        assert_almost_equal(
            true_divergences_base_10[2], test_divergences_base_10_r2, 7)
