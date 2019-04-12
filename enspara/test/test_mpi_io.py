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
        for ra_len in [random.randint(3, 17) for i in range(mpi.MPI_SIZE * 3)]
    ])

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        ra.save(f.name, full_arr)
        global_lengths, local_arr = mpi.io.load_h5_as_striped(f.name)

    for i, subarr in enumerate(full_arr[mpi.MPI_RANK::mpi.MPI_SIZE]):
        assert_array_equal(subarr, local_arr[i])
