import os
import tempfile
import shutil
import pickle
import warnings

import mdtraj as md

from nose.tools import assert_equal
from nose.plugins.attrib import attr

from sklearn.datasets import make_blobs

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from ..import mpi
from ..apps import cluster
from ..cluster import util, kcenters
from ..util import array as ra

from .util import fix_np_rng

TEST_DIR = os.path.dirname(__file__)


def runhelper(args, expected_size, expect_reassignment=True,
              centers_format='pkl'):

    if mpi.rank() == 0:
        td = tempfile.mkdtemp()
    else:
        td = None
    td = mpi.comm.bcast(td, root=0)

    fnames = {
        'center-inds': td + '/center-inds.npy',
        'center-structs': td + 'center-features.%s' % centers_format,
        'distances': td + '/distances.h5',
        'assignments': td + '/assignments.h5',
    }

    try:

        cluster.main([
            '',  # req'd because arg[0] is expected to be program name
            '--distances', fnames['distances'],
            '--assignments', fnames['assignments'],
            '--center-indices', fnames['center-inds'],
            '--center-features', fnames['center-structs']] + args)
        mpi.comm.Barrier()

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
            assigns, dists = None, None
            assert not os.path.isfile(fnames['assignments'])
            assert not os.path.isfile(fnames['distances'])

        ctrindfile = fnames['center-inds']
        assert os.path.isfile(ctrindfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrindfile, os.listdir(os.path.dirname(ctrindfile)))
        with open(ctrindfile, 'rb') as f:
            center_inds = np.load(f)

        ctrstructfile = fnames['center-structs']
        assert os.path.isfile(ctrstructfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrstructfile, os.listdir(os.path.dirname(ctrstructfile)))
        try:
            with open(ctrstructfile, 'rb') as f:
                center_structs = pickle.load(f)
        except pickle.UnpicklingError:
            center_structs = np.load(ctrstructfile)
    finally:
        mpi.comm.Barrier()
        if mpi.rank() == 0:
            shutil.rmtree(td)

    return assigns, dists, center_inds, center_structs


@attr('mpi')
def test_rmsd_kcenters_mpi():

    TRJFILE = os.path.join(os.path.dirname(__file__), 'data', 'frame0.xtc')
    TOPFILE = os.path.join(os.path.dirname(__file__), 'data', 'native.pdb')
    SELECTION = '(name N or name C or name CA or name H or name O)'

    expected_size = (5, 501)

    with tempfile.TemporaryDirectory() as tdname:

        tdname = mpi.comm.bcast(tdname, root=0)

        for i in range(expected_size[0]):
            shutil.copy(TRJFILE, os.path.join(tdname, 'frame%s.xtc' % i))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            a, d, inds, s = runhelper([
                '--trajectories', os.path.join(tdname, 'frame?.xtc'),
                '--topology', TOPFILE,
                '--cluster-number', '4',
                # '--subsample', str(SUBSAMPLE_FACTOR),
                '--atoms', SELECTION,
                '--algorithm', 'kcenters'],
                expected_size=expected_size,
                expect_reassignment=True)

    trj = md.load(TRJFILE, top=TOPFILE)
    trj_sele = trj.atom_slice(trj.top.select(SELECTION))

    result = kcenters.kcenters(trj_sele, 'rmsd', n_clusters=4, mpi_mode=False)

    assert_array_equal(result.center_indices, inds[:, 1])
    assert_array_equal(result.distances, d[0])
    assert_array_equal(result.assignments, a[0])

    expected_s = md.join([trj[i[1]] for i in inds])

    assert_array_equal(expected_s.xyz, md.join(s).xyz)


@attr('mpi')
def test_rmsd_kcenters_mpi_subsample():

    TRJFILE = os.path.join(os.path.dirname(__file__), 'data', 'frame0.xtc')
    TOPFILE = os.path.join(os.path.dirname(__file__), 'data', 'native.pdb')
    SELECTION = '(name N or name C or name CA or name H or name O)'
    SUBSAMPLE_FACTOR = 3

    expected_size = (5, np.ceil(501 / SUBSAMPLE_FACTOR))

    with tempfile.TemporaryDirectory() as tdname:

        tdname = mpi.comm.bcast(tdname, root=0)

        for i in range(expected_size[0]):
            shutil.copy(TRJFILE, os.path.join(tdname, 'frame%s.xtc' % i))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            a, d, inds, s = runhelper([
                '--trajectories', os.path.join(tdname, 'frame?.xtc'),
                '--topology', TOPFILE,
                '--cluster-number', '4',
                '--subsample', str(SUBSAMPLE_FACTOR),
                '--atoms', SELECTION,
                '--algorithm', 'kcenters'],
                expected_size=expected_size,
                expect_reassignment=False)

    trj = md.load(TRJFILE, top=TOPFILE)
    trj_sele = trj.atom_slice(trj.top.select(SELECTION))

    result = kcenters.kcenters(trj_sele[::SUBSAMPLE_FACTOR],
                               'rmsd', n_clusters=4, mpi_mode=False)

    expected_indices = [np.argmin(md.rmsd(trj_sele, c))
                        for c in result.centers]
    assert_array_equal(expected_indices, inds[:, 1])

    expected_s = md.join([trj[i[1]] for i in inds])
    assert_array_equal(expected_s.xyz, md.join(s).xyz)


@fix_np_rng()
@attr('mpi')
def test_rmsd_khybrid_mpi_basic():

    expected_size = (2, 501)

    TRJFILE = os.path.join(os.path.dirname(__file__), 'data', 'frame0.xtc')
    TOPFILE = os.path.join(os.path.dirname(__file__), 'data', 'native.pdb')
    SELECTION = '(name N or name C or name CA or name H or name O)'

    with tempfile.TemporaryDirectory() as tdname:

        shutil.copy(TRJFILE, os.path.join(tdname, 'frame0.xtc'))
        shutil.copy(TRJFILE, os.path.join(tdname, 'frame1.xtc'))

        tdname = mpi.comm.bcast(tdname, root=0)

        mpi.comm.Barrier()

        a, d, idx, s = runhelper([
            '--trajectories', os.path.join(tdname, 'frame?.xtc'),
            '--topology', TOPFILE,
            '--cluster-radius', '0.095',
            '--atoms', SELECTION,
            '--algorithm', 'khybrid'],
            expected_size=expected_size)

    a = a.flatten()
    d = d.flatten()

    trj = md.load(TRJFILE, top=TOPFILE)
    trj_sele = trj.atom_slice(trj.top.select(SELECTION))

    expected_s = md.join([trj[i[1]] for i in idx])
    expected_i = [[0, 0],
                  [0, 55],
                  [1, 102],
                  [1, 196]]

    assert_array_equal(idx, expected_i)

    expected_s = md.join([trj[i[1]] for i in idx])
    assert_array_equal(
        expected_s.xyz,
        md.join(s).xyz)

    expect_a, expect_d = util.assign_to_nearest_center(
        md.join([trj_sele] * 2),
        md.join([trj_sele[i[1]] for i in idx]), md.rmsd)

    assert_array_equal(expect_a, a)
    assert_allclose(expect_d, d, atol=1e-4)


@attr('mpi')
def test_rmsd_khybrid_mpi_subsample():

    TRJFILE = os.path.join(os.path.dirname(__file__), 'data', 'frame0.xtc')
    TOPFILE = os.path.join(os.path.dirname(__file__), 'data', 'native.pdb')
    SELECTION = '(name N or name C or name CA or name H or name O)'
    SUBSAMPLE_FACTOR = 3

    expected_size = (5, (np.ceil(501 / 3),) * 5)

    with tempfile.TemporaryDirectory() as tdname:

        tdname = mpi.comm.bcast(tdname, root=0)

        for i in range(expected_size[0]):
            shutil.copy(TRJFILE, os.path.join(tdname, 'frame%s.xtc' % i))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            a, d, inds, s = runhelper([
                '--trajectories', os.path.join(tdname, 'frame?.xtc'),
                '--topology', TOPFILE,
                '--cluster-radius', '0.1',
                '--subsample', str(SUBSAMPLE_FACTOR),
                '--atoms', SELECTION,
                '--algorithm', 'khybrid'],
                expected_size=expected_size,
                expect_reassignment=False)

    trj = md.load(TRJFILE, top=TOPFILE)
    expected_s = md.join([trj[i[1]] for i in inds])
    assert_array_equal(expected_s.xyz, md.join(s).xyz)


@fix_np_rng(5)
@attr('mpi')
def test_feature_khybrid_mpi_basic():
    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=0)

    try:
        if mpi.rank() == 0:
            td = tempfile.TemporaryDirectory()
            a = ra.RaggedArray(array=X, lengths=[50, 30, 20])

            for row_i in range(len(a.lengths)):
                pathname = os.path.join(td.name, "%s.npy" % row_i)
                np.save(pathname, a[row_i])
        else:
            td = None

        tdname = mpi.comm.bcast(td.name if mpi.rank() == 0 else None,
                                root=0)
        a, d, inds, s = runhelper([
            '--features', os.path.join(tdname, '*.npy'),
            '--cluster-number', '3',
            '--algorithm', 'khybrid',
            '--cluster-distance', 'euclidean'],
            expected_size=expected_size,
            centers_format='npy')

        assert_equal(len(inds), 3)
    finally:
        if mpi.rank() == 0:
            td.cleanup()

        mpi.comm.Barrier()

    y_ra = ra.RaggedArray(y, a.lengths)
    for cid in range(len(inds)):
        iis = ra.where(y_ra == cid)
        i = (iis[0][0], iis[1][0])
        assert np.all(a[i] == a[iis])


@fix_np_rng(5)
@attr('mpi')
def test_feature_khybrid_mpi_h5():
    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=0)

    try:
        if mpi.rank() == 0:
            td = tempfile.TemporaryDirectory()
            a = ra.RaggedArray(array=X, lengths=[50, 30, 20])

            for row_i in range(len(a.lengths)):
                pathname = os.path.join(td.name, "data.h5")
                ra.save(pathname, a)
        else:
            td = None

        pathname = mpi.comm.bcast(pathname if mpi.rank() == 0 else None,
                                  root=0)
        a, d, inds, s = runhelper([
            '--features', pathname,
            '--cluster-number', '3',
            '--algorithm', 'khybrid',
            '--cluster-distance', 'manhattan'],
            expected_size=expected_size,
            centers_format='npy')

        assert_equal(len(inds), 3)
    finally:
        if mpi.rank() == 0:
            td.cleanup()

        mpi.comm.Barrier()

    y_ra = ra.RaggedArray(y, a.lengths)
    for cid in range(len(inds)):
        iis = ra.where(y_ra == cid)
        i = (iis[0][0], iis[1][0])
        assert np.all(a[i] == a[iis])

