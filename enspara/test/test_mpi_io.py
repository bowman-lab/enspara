import tempfile
import random

import numpy as np
from numpy.testing import assert_array_equal

from nose.plugins.attrib import attr


from .. import mpi, ra


@attr('mpi')
def test_parallel_h5_read():

    full_arr = ra.RaggedArray([
        np.random.random(size=(ra_len, 11))
        for ra_len in [random.randint(3, 17) for i in range(mpi.size() * 3)]
    ])

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        ra.save(f.name, full_arr)
        global_lengths, local_arr = mpi.io.load_h5_as_striped(f.name)

    assert_array_equal(local_arr,
                       full_arr[mpi.rank()::mpi.size()]._data)


@attr('mpi')
def test_parallel_h5_read_strided():

    full_arr = ra.RaggedArray([
        np.random.random(size=(ra_len, 11))
        for ra_len in [random.randint(3, 17) for i in range(mpi.size() * 3)]
    ])

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        ra.save(f.name, full_arr)
        global_lengths, local_arr = mpi.io.load_h5_as_striped(f.name, stride=3)

    assert_array_equal(local_arr,
                       full_arr[mpi.rank()::mpi.size(), ::3]._data)
