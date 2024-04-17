import numpy as np
from numpy.testing import assert_array_equal
import pytest

import mdtraj as md

from .util import get_fn
from .. import exception
from .. import mpi


@pytest.mark.mpi
def test_mpi_mean():

    a = np.zeros((10,))
    assert a.mean() == mpi.ops.striped_array_mean(a)

    a = np.ones((10,))
    assert a.mean() == mpi.ops.striped_array_mean(a)

    a = np.arange(10)
    assert a.mean() == mpi.ops.striped_array_mean(a)

    a = np.square(np.arange(10)) - 25
    assert a.mean() == mpi.ops.striped_array_mean(a)


@pytest.mark.mpi
def test_mpi_max():

    endpt = 5 * (mpi.rank() + 2)
    #Need to explicitly define the expected max as enpt varies across processes.
    #MPI.size gives us the total number of processes, rank is processes - 1 (0 indexed)
    expected_max = 5 * ( (mpi.size() - 1) + 2) - 1
    #Needs to be -1 different from endpt calc since np.arange excludes endpt

    #Give each process a different array
    a = np.arange(endpt)
    np.random.shuffle(a)

    #Should find the global max.
    assert expected_max == mpi.ops.striped_array_max(a)

    a = -np.arange(5 * (mpi.rank() + 1))
    assert 0 == mpi.ops.striped_array_max(a)


@pytest.mark.mpi
def test_mpi_distribute_frame_ndarray():

    data = np.arange(10*100*3).reshape(10, 100, 3)

    d = mpi.ops.distribute_frame(data, 7, mpi.size()-1)

    assert_array_equal(d, data[7])
    assert type(d) is type(data)


@pytest.mark.mpi
def test_mpi_distribute_frame_mdtraj():

    data = md.load(get_fn('frame0.h5'))

    d = mpi.ops.distribute_frame(data, 7, mpi.size()-1)

    assert_array_equal(d.xyz, data[7].xyz)
    assert type(d) is type(data)


@pytest.mark.mpi
def test_mpi_assemble_striped_array():

    a = np.arange(77) + 1

    b = mpi.ops.assemble_striped_array(a[mpi.rank()::mpi.size()])

    assert_array_equal(a, b)


@pytest.mark.mpi
def test_mpi_randind():

    a = np.arange(17)

    hits = []

    for i in range(100):
        r, o = mpi.ops.randind(a[mpi.rank()::mpi.size()])

        hits.append(mpi.ops.convert_local_indices(
            [(r, o)],
            [len(a[r::mpi.size()]) for r in range(mpi.size())])[0])

    distro = np.bincount(hits)
    assert pytest.approx(distro.mean(), abs=1e-7) == (i+1)/len(a)


@pytest.mark.mpi
def test_mpi_randind_few_options():

    # test an array that is only on rank 0
    a = np.array([7])

    r, o = mpi.ops.randind(a[mpi.rank()::mpi.size()])

    assert r == 0
    assert o == 0

    # test an array that is only on rank 1
    if mpi.size() > 1:
        if mpi.rank() == 1:
            r, o = mpi.ops.randind(a)
        else:
            r, o = mpi.ops.randind(np.array([]))

        assert r == 1
        assert o == 0

    with pytest.raises(exception.DataInvalid):
        # test on an empty array
        a = np.array([])

        r, o = mpi.ops.randind(a)



@pytest.mark.mpi
def test_mpi_randind_same_as_np():

    a = np.arange(17)

    for seed in range(100):
        r, o = mpi.ops.randind(
            a[mpi.rank()::mpi.size()],
            random_state=seed)

        assert (
            np.random.RandomState(seed).choice(a) ==
            a[r::mpi.size()][o])


@pytest.mark.mpi
def test_mpi_randind_uniform():

    a = np.arange(17)

    hits = []

    for i in range(100):
        r, o = mpi.ops.randind(
            a[mpi.rank()::mpi.size()],
            random_state=0)

        hits.append(mpi.ops.convert_local_indices(
            [(r, o)],
            [len(a[r::mpi.size()]) for r in range(mpi.size())])[0])

    distro = np.bincount(hits)
    assert distro[np.argmax(distro)] == i+1
