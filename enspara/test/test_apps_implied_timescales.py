import os

import numpy as np

from nose.tools import assert_raises, assert_equals
from numpy.testing import assert_array_equal

from .. import exception
from ..apps import implied_timescales

from .util import get_fn

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'cards_data')
TRJ_PATH = os.path.join(TEST_DATA_DIR, "trj0.xtc")


def test_process_units():

    with assert_raises(exception.ImproperlyConfigured):
        implied_timescales.process_units(timestep=10, infer_timestep=TRJ_PATH)

    assert_equals(
        implied_timescales.process_units(timestep=10),
        (10, 'ns'))

    assert_equals(
        implied_timescales.process_units(None, None),
        (1, 'frames'))

    assert_equals(
        implied_timescales.process_units(),
        (1, 'frames'))

    assert_equals(
        implied_timescales.process_units(infer_timestep=TRJ_PATH),
        (100, 'ns'))

    print(get_fn('frame0.h5'))

    assert_equals(
        implied_timescales.process_units(infer_timestep=get_fn('frame0.h5')),
        (1000, 'ns'))

    assert_equals(
        implied_timescales.process_units(infer_timestep=get_fn('frame0.xtc')),
        (1000, 'ns'))


def test_prior_counts():

    from enspara.msm.builders import normalize

    C = np.array([[7, 1, 3, 1],
                  [1, 8, 3, 1],
                  [0, 7, 9, 2],
                  [0, 2, 3, 4]])

    C_a, T_a, eq_a = implied_timescales.prior_counts(C)
    C_b, T_b, eq_b = normalize(C, prior_counts=1/len(C))

    assert_array_equal(C_a, C_b)
    assert_array_equal(T_a, T_b)
    assert_array_equal(eq_a, eq_b)
