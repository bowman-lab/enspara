import unittest

import numpy as np

from nose.tools import assert_raises, assert_equals, assert_is
from numpy.testing import assert_array_equal

from .committors import committors

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
