import unittest

import numpy as np

from nose.tools import assert_raises, assert_equals, assert_is
from numpy.testing import assert_array_equal

from .committors import committors
from .fluxes import fluxes



class Test_Fluxes(unittest.TestCase):

    def test_committors(self):
        Tij1 = np.array(
            [
                [0.5, 0.4, 0.1],
                [0.25, 0.5, 0.25],
                [0.1, 0.5, 0.4]])

        true_committors = np.array([0, 0.5, 1.])

        for_committors = committors(Tij1, 0, 2)
        assert_array_equal(for_committors, true_committors)

        for_committors = committors(Tij1, [0], [2])
        assert_array_equal(for_committors, true_committors)

        Tij2 = np.array(
            [
                [0.5, 0.4, 0.1, 0.],
                [0.25, 0.5, 0.2, 0.05],
                [0.1, 0.15, 0.5, 0.25],
                [0., 0.1, 0.4, 0.5]])

        true_committors = np.array([0, 0.34091, 0.60227, 1.])

        for_committors = np.around(committors(Tij2, 0, 3), 5)
        assert_array_equal(for_committors, true_committors)

        for_committors = committors(Tij2, [0,2], [3])
        assert_array_equal(for_committors, np.array([0, 0.1, 0, 1.0]))

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
