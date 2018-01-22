import numpy as np
from numpy.testing import assert_array_equal

import mdtraj as md
from mdtraj.testing import get_fn

from nose.tools import assert_is, assert_almost_equal, assert_equal
from nose.plugins.attrib import attr

from .. import mpi


@attr('mpi')
def test_mpi_mean():

    a = np.zeros((10,))
    assert_equal(a.mean(), mpi.ops.mean(a))

    a = np.ones((10,))
    assert_equal(a.mean(), mpi.ops.mean(a))

    a = np.arange(10)
    assert_equal(a.mean(), mpi.ops.mean(a))

    a = np.square(np.arange(10)) - 25
    assert_equal(a.mean(), mpi.ops.mean(a))


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

    b = mpi.ops.assemble_striped_array(a[mpi.MPI_RANK::mpi.MPI_SIZE])

    assert_array_equal(a, b)


@attr('mpi')
def test_mpi_np_choice():

    a = np.arange(17)

    hits = []

    for i in range(100):
        r, o = mpi.ops.np_choice(a[mpi.MPI_RANK::mpi.MPI_SIZE])

        hits.append(mpi.ops.convert_local_indices(
            [(r, o)],
            [len(a[r::mpi.MPI_SIZE]) for r in range(mpi.MPI_SIZE)])[0])

    distro = np.bincount(hits)
    assert_almost_equal(distro.mean(), (i+1)/len(a))


@attr('mpi')
def test_mpi_np_choice_same_as_np():

    a = np.arange(17)

    for seed in range(100):
        r, o = mpi.ops.np_choice(
            a[mpi.MPI_RANK::mpi.MPI_SIZE],
            random_state=seed)

        assert_equal(
            np.random.RandomState(seed).choice(a),
            a[r::mpi.MPI_SIZE][o])


@attr('mpi')
def test_mpi_np_choice_uniform():

    a = np.arange(17)

    hits = []

    for i in range(100):
        r, o = mpi.ops.np_choice(
            a[mpi.MPI_RANK::mpi.MPI_SIZE],
            random_state=0)

        hits.append(mpi.ops.convert_local_indices(
            [(r, o)],
            [len(a[r::mpi.MPI_SIZE]) for r in range(mpi.MPI_SIZE)])[0])

    distro = np.bincount(hits)
    assert_equal(distro[np.argmax(distro)], i+1)
