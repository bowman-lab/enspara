import warnings

import numpy as np
import mdtraj as md

from nose.tools import assert_equal, assert_is
from numpy.testing import assert_array_equal

from enspara.util.preprocessing import ResidueTypeScaler
from enspara import exception
from enspara.test.util import get_fn


def test_residue_scaler():

    t = md.load(get_fn('frame0.h5'))
    s = ResidueTypeScaler(np.mean, top=t.top)

    a = np.array([[2,  1, 4],
                  [2, -4, 4],
                  [2,  1, 4],
                  [2,  1, 9],
                  [2,  1, 9],]).astype(float)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert_is(type(s.fit(a)), ResidueTypeScaler)

        assert_equal(len(w), 1)
        assert_is(w[-1].category, exception.SuspiciousDataWarning)

    expected = [
        [0.5,  1, 1],
        [0.5, -4, 1],
        [0.5,  1, 1],
        [0.5,  1, 2.25],
        [0.5,  1, 2.25],]

    assert_array_equal(
        s.transform(a),
        expected)
