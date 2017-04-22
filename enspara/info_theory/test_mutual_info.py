from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

from . import mutual_info


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
