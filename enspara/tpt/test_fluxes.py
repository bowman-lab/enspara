import unittest

import numpy as np

from nose.tools import assert_raises, assert_equals, assert_is
from numpy.testing import assert_array_equal

from .committors import committors
from .fluxes import fluxes

Tij = np.array(
    [
        [0.5, 0.4, 0.1],
        [0.25, 0.5, 0.25],
        [0.1, 0.5, 0.4]])


class Test_Fluxes(unittest.TestCase):

    def test_committors(self):
        true_committors = np.array([0, 0.5, 1.])

        for_committors = committors(Tij, 0, 2)
        assert_array_equal(for_committors, true_committors)

        for_committors = committors(Tij, [0], [2])
        assert_array_equal(for_committors, true_committors)

    def test_fluxes(self):
        Tij = np.array(
            [
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5]])
        pops = np.zeros(3) + (1/3.)

        # true fluxes
        true_fluxes = np.zeros((3,3))
        true_fluxes[0,1] = 1/12.
        true_fluxes[1,2] = 1/12.
        true_fluxes = np.around(true_fluxes, 5)

        # test fluxes
        calc_fluxes = np.around(fluxes(Tij, 0, 2, populations=pops), 5)
        assert_array_equal(calc_fluxes, true_fluxes)

        # test fluxes without pops
        calc_fluxes = np.around(fluxes(Tij, 0, 2), 5)
        assert_array_equal(calc_fluxes, true_fluxes)
