import os
import tempfile
import shutil
import pickle
import warnings

import mdtraj as md
from mdtraj.testing import get_fn

from nose.tools import assert_equal
from nose.plugins.attrib import attr

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from ..apps import rmsd_cluster_mpi
from ..cluster.util import assign_to_nearest_center
from ..util import array as ra
from ..mpi import MPI, MPI_RANK, MPI_SIZE

from .util import fix_np_rng

TEST_DIR = os.path.dirname(__file__)


def runhelper(args, expected_size, expect_reassignment=True):

    if MPI_RANK == 0:
        td = tempfile.mkdtemp()
    else:
        td = None
    td = MPI.COMM_WORLD.bcast(td, root=0)

    fnames = {
        'center-inds': td+'/center-inds.pkl',
        'center-structs': td+'/center-structs.pkl',
        'distances': td+'/distances.h5',
        'assignments': td+'/assignments.h5',
    }

    try:
        rmsd_cluster_mpi.main([
            '',  # req'd because arg[0] is expected to be program name
            '--distances', fnames['distances'],
            '--assignments', fnames['assignments'],
            '--center-indices', fnames['center-inds'],
            '--center-structures', fnames['center-structs']] + args)
        MPI.COMM_WORLD.Barrier()

        if expect_reassignment:
            assert os.path.isfile(fnames['assignments']), \
                "Couldn't find %s. Dir contained: %s" % (
                    fnames['assignments'],
                    os.listdir(os.path.dirname(fnames['assignments'])))

            assigns = ra.load(fnames['assignments'])
            if type(assigns) is ra.RaggedArray:
                assert_equal(len(assigns), expected_size[0])
                assert_equal(assigns._data.dtype, np.int)
                assert_array_equal(assigns.lengths, expected_size[1])
            else:
                assert_equal(assigns.shape, expected_size)
                assert_equal(assigns.dtype, np.int)

            distfile = fnames['distances']
            assert os.path.isfile(distfile), \
                "Couldn't find %s. Dir contained: %s" % (
                distfile, os.listdir(os.path.dirname(distfile)))
            dists = ra.load(distfile)
        else:
            assert not os.path.isfile(fnames['assignments'])
            assert not os.path.isfile(fnames['distances'])

        ctrindfile = fnames['center-inds']
        assert os.path.isfile(ctrindfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrindfile, os.listdir(os.path.dirname(ctrindfile)))
        with open(ctrindfile, 'rb') as f:
            center_inds = pickle.load(f)

        ctrstructfile = fnames['center-structs']
        assert os.path.isfile(ctrstructfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrstructfile, os.listdir(os.path.dirname(ctrstructfile)))
        with open(ctrstructfile, 'rb') as f:
            center_structs = pickle.load(f)

    finally:
        MPI.COMM_WORLD.Barrier()
        if MPI_RANK == 0:
            shutil.rmtree(td)

    return assigns, dists, center_inds, center_structs


@fix_np_rng()
@attr('mpi')
def test_rmsd_cluster_mpi_basic():

    expected_size = (2, 501)

    TRJFILE = get_fn('frame0.xtc')
    TOPFILE = get_fn('native.pdb')
    SELECTION = '(name N or name C or name CA or name H or name O)'

    with tempfile.TemporaryDirectory() as tdname:

        shutil.copy(TRJFILE, os.path.join(tdname, 'frame0.xtc'))
        shutil.copy(TRJFILE, os.path.join(tdname, 'frame1.xtc'))

        tdname = MPI.COMM_WORLD.bcast(tdname, root=0)

        print('rank', MPI.COMM_WORLD.Get_rank())
        MPI.COMM_WORLD.Barrier()

        a, d, i, s = runhelper([
            '--trajectories', os.path.join(tdname, 'frame?.xtc'),
            '--topology', TOPFILE,
            '--cluster-radii', '0.1',
            '--selection', SELECTION,
            '--kmedoids-iters', 0,
            ],
            expected_size=expected_size)

    a = a.flatten()
    d = d.flatten()

    trj = md.load(TRJFILE, top=TOPFILE)
    trj_sele = trj.atom_slice(trj.top.select(SELECTION))

    # expected_i = [(1, 194), (1, 40), (0, 430), (1, 420)]
    expected_i = [[  0,   0], [  0,  42], [  0, 430], [  0, 319]]
    assert_array_equal(i, expected_i)

    expected_s = md.join([trj[i[1]] for i in expected_i])
    assert_array_equal(
        expected_s.xyz,
        md.join(s).xyz)

    expect_a, expect_d = assign_to_nearest_center(
        md.join([trj_sele]*2),
        md.join([trj_sele[i[1]] for i in expected_i]), md.rmsd)

    assert_array_equal(expect_a, a)
    assert_allclose(expect_d, d, atol=1e-4)


@attr('mpi')
def test_rmsd_cluster_mpi_subsample():

    TRJFILE = get_fn('frame0.xtc')
    TOPFILE = get_fn('native.pdb')
    SELECTION = '(name N or name C or name CA or name H or name O)'
    SUBSAMPLE_FACTOR = 3

    expected_size = (5, (np.ceil(501 / 3),)*5)

    with tempfile.TemporaryDirectory() as tdname:

        tdname = MPI.COMM_WORLD.bcast(tdname, root=0)

        for i in range(expected_size[0]):
            shutil.copy(TRJFILE, os.path.join(tdname, 'frame%s.xtc' % i))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            a, d, i, s = runhelper([
                '--trajectories', os.path.join(tdname, 'frame?.xtc'),
                '--topology', TOPFILE,
                '--cluster-radii', '0.1',
                '--subsample', str(SUBSAMPLE_FACTOR),
                '--selection', SELECTION,
                '--random-state', str(2),
                '--kmedoids-iters', str(1),
                ],
                expected_size=expected_size)

    a = a.flatten()
    d = d.flatten()

    trj = md.load(TRJFILE, top=TOPFILE)
    trj_sele = trj.atom_slice(trj.top.select(SELECTION))

    if MPI_SIZE == 1:
        expected_i = [[ 1,  3], [ 0, 45], [ 0, 24], [ 0, 66]]
    elif MPI_SIZE == 2:
        expected_i = [[  0,   0], [  1,  21], [  0, 105], [  0,  60]]
    else:
        raise NotImplementedError(
            "We dont know what the right answer to this test is with "
            "MPI size %s" % MPI_SIZE)

    assert_array_equal(i, expected_i)

    expected_s = md.join([trj[i[1]] for i in expected_i])
    assert_array_equal(
        expected_s.xyz,
        md.join(s).xyz)

    expect_a, expect_d = assign_to_nearest_center(
        md.join([trj_sele]*expected_size[0]),
        md.join([trj_sele[i[1]] for i in expected_i]), md.rmsd)

    assert_array_equal(expect_a[::SUBSAMPLE_FACTOR], a)
    assert_allclose(expect_d[::SUBSAMPLE_FACTOR], d, atol=1e-4)
