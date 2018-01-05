import os

from mdtraj.testing import get_fn

from nose.tools import assert_raises, assert_equals

from .. import exception
from ..apps import implied_timescales

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
