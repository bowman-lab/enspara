import numpy as np
from numpy.testing import assert_array_equal

import mdtraj as md
from mdtraj.testing import get_fn

from nose.tools import assert_is
from nose.plugins.attrib import attr

from .. import mpi


@attr('mpi')
def test_mpi_distribute_frame_ndarray():

    data = np.arange(10*100*3).reshape(10, 100, 3)

    d = mpi.ops.distribute_frame(data, 7, mpi.MPI_SIZE-1)

    assert_array_equal(d, data[7])
    assert_is(type(d), type(data))


@attr('mpi')
def test_mpi_distribute_frame_mdtraj():

    data = md.load(get_fn('frame0.h5'))

    d = mpi.ops.distribute_frame(data, 7, mpi.MPI_SIZE-1)

    assert_array_equal(d.xyz, data[7].xyz)
    assert_is(type(d), type(data))


@attr('mpi')
def test_mpi_assemble_striped_array():

    a = np.arange(77) + 1

    b = mpi.ops.assemble_striped_array(a[::mpi.MPI_SIZE])

    assert_array_equal(a, b)
