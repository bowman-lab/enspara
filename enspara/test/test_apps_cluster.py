import os
import tempfile
import hashlib
import shutil

from datetime import datetime

from nose.tools import assert_equal, assert_raises

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.datasets import make_blobs

from .. import exception
from ..util import array as ra

from ..apps import cluster

TEST_DIR = os.path.dirname(__file__)
TRJFILE = os.path.join(os.path.dirname(__file__), 'data', 'frame0.xtc')
TOPFILE = os.path.join(os.path.dirname(__file__), 'data', 'native.pdb')


def runhelper(args, expected_size, algorithm='khybrid', expected_k=None,
              expect_reassignment=True, centers_format='pkl'):

    td = tempfile.mkdtemp(dir=os.getcwd())
    tf = hashlib.md5(str(datetime.now().timestamp())
                     .encode('utf-8')).hexdigest()[0:8]
    base = os.path.join(td, tf)

    fnames = {
        'distances': base + 'distances.h5',
        'assignments': base + 'assignments.h5',
        'center-features': base + 'center-features.%s' % centers_format,
    }

    try:
        argv = ['']
        for argname in ['distances', 'center-features', 'assignments']:
            if '--%s' % argname not in args:
                argv.extend(['--%s' % argname, fnames[argname]])

        argv.extend(args)
        cluster.main(argv)

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
                if expected_k is not None:
                    assert_array_equal(
                        np.unique(assigns._data),
                        np.arange(expected_k))
            else:
                assert_equal(assigns.shape, expected_size)
                assert_equal(assigns.dtype, np.int)
                if expected_k is not None:
                    assert_array_equal(
                        np.unique(assigns),
                        np.arange(expected_k))

            distfile = fnames['distances']
            assert os.path.isfile(distfile), \
                "Couldn't find %s. Dir contained: %s" % (
                distfile, os.listdir(os.path.dirname(distfile)))
            dists = ra.load(fnames['distances'])
        else:
            assert not os.path.isfile(fnames['assignments'])
            assert not os.path.isfile(fnames['distances'])
            dists = None
            assigns = None

        ctrsfile = fnames['center-features']
        assert os.path.isfile(ctrsfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrsfile, os.listdir(os.path.dirname(ctrsfile)))

    finally:
        shutil.rmtree(td)
        pass

    return dists, assigns


def test_rmsd_cluster_basic():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--cluster-radius', '0.1',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_basic_kcenters():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--cluster-radius', '0.1',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'kcenters'],
        expected_size=expected_size,
        algorithm='kcenters')


def test_rmsd_cluster_fixed_k_kcenters():

    expected_size = (2, 501)
    expected_k = 10

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--cluster-number', str(expected_k),
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'kcenters'],
        expected_size=expected_size,
        algorithm='kcenters',
        expected_k=expected_k)


def test_rmsd_cluster_broken_atoms():

    expected_size = (2, 501)

    with assert_raises(exception.ImproperlyConfigured):
        runhelper([
            '--trajectories', TRJFILE, TRJFILE,
            '--topology', TOPFILE,
            '--cluster-radius', '0.1',
            '--atoms', 'residue -1',
            '--algorithm', 'khybrid'],
            expected_size=expected_size)


def test_rmsd_cluster_selection():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--cluster-radius', '0.1',
        '--atoms', '(name N or name C or name CA)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_subsample():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--cluster-radius', '0.1',
        '--subsample', '4',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_multiprocess():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--cluster-radius', '0.1',
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_subsample_and_noreassign():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--atoms', '(name N or name C or name CA or name H or name O)',
        '--cluster-radius', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4',
        '--no-reassign'],
        expect_reassignment=False,
        expected_size=expected_size)


def test_rmsd_cluster_multitop():

    expected_size = (3, (501, 501, 5001))

    # trj is length 5001
    xtc2 = os.path.join(TEST_DIR, 'cards_data', 'trj0.xtc')
    top2 = os.path.join(TEST_DIR, 'cards_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--trajectories', xtc2,
        '--topology', TOPFILE,
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O) '
                   'and (residue 2)',
        '--cluster-radius', '0.1',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_multitop_multiselection():

    expected_size = (3, (501, 501, 5001))

    xtc2 = os.path.join(TEST_DIR, 'cards_data', 'trj0.xtc')
    top2 = os.path.join(TEST_DIR, 'cards_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--atoms', '(name N or name O) and (residue 2)',
        '--trajectories', xtc2,
        '--topology', top2,
        '--atoms', '(name CA) and (residue 3 or residue 4)',
        '--cluster-radius', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4'],
        expected_size=expected_size)

    # reverse the order. This will catch some cases where the first
    # selection works on both.
    runhelper([
        '--trajectories', xtc2,
        '--topology', top2,
        '--atoms', '(name CA) and (residue 3 or residue 4)',
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--atoms', '(name N or name O) and (residue 2)',
        '--cluster-radius', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4'],
        expected_size=(expected_size[0], expected_size[1][::-1]))


def test_rmsd_cluster_multitop_multiselection_noreassign():

    expected_size = (3, (501, 501, 5001))

    xtc2 = os.path.join(TEST_DIR, 'cards_data', 'trj0.xtc')
    top2 = os.path.join(TEST_DIR, 'cards_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--atoms', '(name N or name O) and (residue 2)',
        '--trajectories', xtc2,
        '--topology', top2,
        '--atoms', '(name CA) and (residue 3 or residue 4)',
        '--cluster-radius', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4',
        '--no-reassign'],
        expected_size=expected_size,
        expect_reassignment=False)

    # reverse the order. This will catch some cases where the first
    # selection works on both.
    runhelper([
        '--trajectories', xtc2,
        '--topology', top2,
        '--atoms', '(name CA) and (residue 3 or residue 4)',
        '--trajectories', TRJFILE, TRJFILE,
        '--topology', TOPFILE,
        '--atoms', '(name N or name O) and (residue 2)',
        '--cluster-radius', '0.1',
        '--algorithm', 'khybrid',
        '--subsample', '4',
        '--no-reassign'],
        expect_reassignment=False,
        expected_size=(expected_size[0], expected_size[1][::-1]))


def test_feature_cluster_basic():

    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=0)

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:

        a = ra.RaggedArray(array=X, lengths=[50, 30, 20])
        ra.save(f.name, a)

        with tempfile.NamedTemporaryFile(suffix='.npy') as ind_f:
            distances, assignments = runhelper([
                '--features', f.name,
                '--cluster-number', '3',
                '--algorithm', 'khybrid',
                '--cluster-distance', 'euclidean',
                '--center-indices', ind_f.name],
                expected_size=expected_size,
                centers_format='npy')

            center_indices = np.load(ind_f)
            assert_equal(len(center_indices), 3)

    y_ra = ra.RaggedArray(y, assignments.lengths)
    for cid in range(len(center_indices)):
        iis = ra.where(y_ra == cid)
        i = (iis[0][0], iis[1][0])
        assert np.all(assignments[i] == assignments[iis])


def test_feature_cluster_manhattan():

    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=0)

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:

        a = ra.RaggedArray(array=X, lengths=[50, 30, 20])
        ra.save(f.name, a)

        with tempfile.NamedTemporaryFile(suffix='.npy') as ind_f:
            distances, assignments = runhelper([
                '--features', f.name,
                '--cluster-number', '3',
                '--algorithm', 'khybrid',
                '--cluster-distance', 'manhattan',
                '--center-indices', ind_f.name],
                expected_size=expected_size,
                centers_format='npy')

            center_indices = np.load(ind_f)
            assert_equal(len(center_indices), 3)

    y_ra = ra.RaggedArray(y, assignments.lengths)
    for cid in range(len(center_indices)):
        iis = ra.where(y_ra == cid)
        i = (iis[0][0], iis[1][0])
        assert np.all(assignments[i] == assignments[iis])


def test_feature_cluster_radius_based_h5_input():

    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=3)

    with tempfile.NamedTemporaryFile(suffix='.h5') as f:

        a = ra.RaggedArray(array=X, lengths=[50, 30, 20])
        ra.save(f.name, a)

        distances, assignments = runhelper([
            '--features', f.name,
            '--cluster-radius', '3',
            '--algorithm', 'kcenters',
            '--cluster-distance', 'euclidean'],
            expected_size=expected_size,
            centers_format='npy')

        assert_equal(len(np.unique(assignments.flatten())), 11)


def reorder_assignments(assigs):
    """Rewrite an assignments array so that lower indices appear earlier.
    """

    assigs = assigs.copy()
    aids = np.unique(assigs)

    assig_is = {}
    order = np.argsort([np.where(assigs == aid)[0][0] for aid in aids])
    for aid in aids:
        assig_is[aid] = (assigs == aid)

    for i, aid in enumerate(order):
        assigs[assig_is[aid]] = i

    return assigs


def test_feature_cluster_number_khybrid_npy_input():

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

        distances, assignments = runhelper([
            '--features', pathnames[0], pathnames[1], pathnames[2],
            '--cluster-number', '3',
            '--algorithm', 'khybrid',
            '--cluster-distance', 'euclidean'],
            expected_size=expected_size,
            centers_format='npy')

    assert_array_equal(a.lengths, assignments.lengths)
    assert_array_equal(a.lengths, distances.lengths)

    y = reorder_assignments(y)
    assignments = reorder_assignments(assignments.flatten())

    assert_equal(len(np.unique(assignments)), 3)
    assert_array_equal(y, assignments)


def test_feature_cluster_number_kcenters_npy_input():

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

        distances, assignments = runhelper([
            '--features', pathnames[0], pathnames[1], pathnames[2],
            '--cluster-number', '3',
            '--algorithm', 'kcenters',
            '--cluster-distance', 'euclidean'],
            expected_size=expected_size,
            centers_format='npy')

    assert_array_equal(a.lengths, assignments.lengths)
    assert_array_equal(a.lengths, distances.lengths)

    y = reorder_assignments(y)
    assignments = reorder_assignments(assignments.flatten())

    assert_equal(len(np.unique(assignments)), 3)
    assert_array_equal(y, assignments)


def test_feature_cluster_number_kcenters_npy_input_iterations_flag_error():

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

        with assert_raises(exception.ImproperlyConfigured):
            _, _ = runhelper([
                '--features', pathnames[0], pathnames[1], pathnames[2],
                '--cluster-number', '3',
                '--cluster-iterations', '100',
                '--algorithm', 'kcenters',
                '--cluster-distance', 'euclidean'],
                expected_size=expected_size,
                centers_format='npy')


def test_feature_cluster_number_khybrid_npy_input_zero_iterations():

    expected_size = (3, (50, 30, 20))

    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, center_box=(0, 100),
        random_state=3)

    kc = cluster.KCenters('euclidean', n_clusters=3)
    kc.fit(X)

    with tempfile.TemporaryDirectory() as d:

        a = ra.RaggedArray(array=X, lengths=[50, 30, 20])

        pathnames = []
        for row_i in range(len(a.lengths)):
            pathname = os.path.join(d, "%s.npy" % row_i)
            np.save(pathname, a[row_i])
            pathnames.append(pathname)

        distances, assignments = runhelper([
            '--features', pathnames[0], pathnames[1], pathnames[2],
            '--cluster-number', '3',
            '--algorithm', 'khybrid',
            '--cluster-iterations', '0',
            '--cluster-distance', 'euclidean'],
            expected_size=expected_size,
            centers_format='npy')

    assert_array_equal(a.lengths, assignments.lengths)
    assert_array_equal(a.lengths, distances.lengths)

    y = reorder_assignments(y)
    assignments = reorder_assignments(assignments.flatten())

    assert_equal(len(np.unique(assignments)), 3)
    assert_array_equal(y, assignments)

    assert_array_equal(kc.distances_, distances.flatten())
