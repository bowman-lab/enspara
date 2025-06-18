import os
import tempfile
import shutil
import pickle
import warnings
import pytest

import mdtraj as md

from sklearn.datasets import make_blobs

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from ..import mpi
from ..apps import cluster
from ..cluster import util, kcenters, kmedoids
from enspara import ra

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
                assert len(assigns) == expected_size[0]
                assert assigns._data.dtype == int
                assert_array_equal(assigns.lengths, expected_size[1])
            else:
                assert assigns.shape == expected_size
                assert assigns.dtype == int

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


@pytest.mark.mpi
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


@pytest.mark.mpi
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
                '--algorithm', 'kcenters',
                '--no-reassign'],
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
@pytest.mark.mpi
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
    full_traj = md.join([trj_sele] * 2)

    assert a.shape == (1002,)
    assert d.shape == (1002,)
    assert len(idx) == len(s)
    assert len(s) == 4

    selection_indices = trj.top.select(SELECTION)

    for i, (traj_i, frame_i) in enumerate(idx):
        center = trj_sele.slice([frame_i])
        returned_center = s[i].atom_slice(selection_indices)

        rmsd = md.rmsd(returned_center, center)[0]
        assert rmsd < 1e-3, f"Returned center structure {i} doesn't match frame {frame_i} (RMSD={rmsd})"

    selection_indices = trj.top.select(SELECTION)
    cluster_centers = md.join(s).atom_slice(selection_indices)
    expect_a, expect_d = util.assign_to_nearest_center(full_traj, cluster_centers, md.rmsd)

    assert_array_equal(expect_a, a)
    assert_allclose(expect_d, d, atol=1e-4)


@pytest.mark.mpi
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
                '--algorithm', 'khybrid',
                '--no-reassign'],
                expected_size=expected_size,
                expect_reassignment=False)

    trj = md.load(TRJFILE, top=TOPFILE)
    expected_s = md.join([trj[i[1]] for i in inds])
    assert_array_equal(expected_s.xyz, md.join(s).xyz)


@fix_np_rng(5)
@pytest.mark.mpi
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

        assert len(inds) == 3
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
@pytest.mark.mpi
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

        assert len(inds) == 3
    finally:
        if mpi.rank() == 0:
            td.cleanup()

        mpi.comm.Barrier()

    y_ra = ra.RaggedArray(y, a.lengths)
    for cid in range(len(inds)):
        iis = ra.where(y_ra == cid)
        i = (iis[0][0], iis[1][0])
        assert np.all(a[i] == a[iis])

@pytest.mark.mpi
def test_kmedoid_warm_start_mpi():

    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=3)

    with tempfile.TemporaryDirectory() as d:
        a = ra.RaggedArray(array=X, lengths=[50, 30, 20])

        pathnames = []
        for row_i in range(len(a.lengths)):
            pathname = os.path.join(d, "%s.npy" % row_i)
            np.save(pathname, a[row_i])
            pathnames.append(pathname)

        assignments, distances, inds, structs = runhelper([
            '--features', pathnames[0], pathnames[1], pathnames[2],
            '--cluster-number', '3',
            '--algorithm', 'kcenters',
            '--cluster-distance', 'euclidean'],
            expected_size=expected_size,
            centers_format='npy')

        pathname1 = os.path.join(d, "init_assignments.h5")
        ra.save(pathname1, assignments)
        pathname2 = os.path.join(d, "init_distances.h5")
        ra.save(pathname2, distances)
        pathname3 = os.path.join(d, "init_cluster_center_inds.npy")
        np.save(pathname3, inds)

        assignments2, distances2, inds2, structs2 = runhelper([
            '--features', pathnames[0], pathnames[1], pathnames[2],
            '--cluster-number', '3',
            '--algorithm', 'kmedoids',
            '--cluster-iterations', '1',
            '--cluster-distance', 'euclidean',
            '--init-assignments', pathname1,
            '--init-distances', pathname2,
            '--init-center-inds', pathname3],
            expected_size=expected_size,
            centers_format='npy')

    distances2 = np.concatenate(distances2)
    assignments2 = np.concatenate(assignments2)
    distances = np.concatenate(distances)
    assignments = np.concatenate(assignments)
    
    #Run kmedoids with output files from kcenters
    old_cost = kmedoids._msq(distances)
    new_cost = kmedoids._msq(distances2)
    #KMedoids should decrease the total distances
    assert new_cost < old_cost

    #Every cluster center chosen from kmedoids should be the cluster center
    #of the cluster it was assigned to at the end of KCenters
    #Only true when we do 1 iteration of KMedoids
    #This verifies we're really using cluster center info from KCenters
    cluster_center_inds2 = util.find_cluster_centers(assignments2, distances2)
    assert_array_equal(assignments[cluster_center_inds2],
                       np.arange(len(cluster_center_inds2)))
