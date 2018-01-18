import numpy as np
from numpy.testing import assert_array_equal

import mdtraj as md
from mdtraj.testing import get_fn

from nose.tools import assert_is

from .. import mpi


def test_mpi_distribute_frame_ndarray():

    from ..mpi import MPI
    data = np.arange(10*100*3).reshape(10, 100, 3)

    d = mpi.distribute_frame(data, 7, MPI.COMM_WORLD.Get_size()-1)

    assert_array_equal(d, data[7])
    assert_is(type(d), type(data))


def test_mpi_distribute_frame_mdtraj():

    from ..mpi import MPI
    data = md.load(get_fn('frame0.h5'))

    d = mpi.distribute_frame(data, 7, MPI.COMM_WORLD.Get_size()-1)

    assert_array_equal(d.xyz, data[7].xyz)
    assert_is(type(d), type(data))
