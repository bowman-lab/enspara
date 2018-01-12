from __future__ import print_function, division, absolute_import

import unittest
import warnings
import time
import os

from functools import wraps

import numpy as np
import mdtraj as md
from mdtraj.testing import get_fn

from nose.tools import (assert_raises, assert_less, assert_true, assert_is,
                        assert_equal)
from numpy.testing import assert_array_equal, assert_allclose

from ..cluster.hybrid import KHybrid, hybrid
from ..cluster import kcenters, kmedoids, util

from ..exception import DataInvalid, ImproperlyConfigured


class TestTrajClustering(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')

        self.trj = md.load(self.trj_fname, top=self.top_fname)

    def test_kcenters_object(self):
        # '''
        # KHybrid() clusterer should produce correct output.
        # '''
        N_CLUSTERS = 5
        CLUSTER_RADIUS = 0.1

        with assert_raises(ImproperlyConfigured):
            kcenters.KCenters(metric=md.rmsd)

        # Test n clusters
        clustering = kcenters.KCenters(
            metric=md.rmsd,
            n_clusters=N_CLUSTERS)

        clustering.fit(self.trj)

        assert hasattr(clustering, 'result_')
        assert len(np.unique(clustering.labels_)) == N_CLUSTERS, \
            clustering.labels_

        # Test dist_cutoff
        clustering = kcenters.KCenters(
            metric=md.rmsd,
            cluster_radius=CLUSTER_RADIUS)

        clustering.fit(self.trj)

        assert hasattr(clustering, 'result_')
        assert clustering.distances_.max() < CLUSTER_RADIUS, \
            clustering.distances_

        # Test n clusters and dist_cutoff
        clustering = kcenters.KCenters(
            metric=md.rmsd,
            n_clusters=N_CLUSTERS,
            cluster_radius=CLUSTER_RADIUS)

        clustering.fit(self.trj)

        # In this particular case, the clustering should cut off at 5 clusters
        # despite not reaching the distance cutoff
        assert hasattr(clustering, 'result_')
        assert len(np.unique(clustering.labels_)) == N_CLUSTERS, \
            clustering.labels_

    def test_khybrid_object(self):
        # '''
        # KHybrid() clusterer should produce correct output.
        # '''
        N_CLUSTERS = 5

        with assert_raises(ImproperlyConfigured):
            KHybrid(metric=md.rmsd, kmedoids_updates=1000)

        clustering = KHybrid(
            metric=md.rmsd,
            n_clusters=N_CLUSTERS,
            kmedoids_updates=1000)

        clustering.fit(self.trj)

        assert hasattr(clustering, 'result_')
        assert len(np.unique(clustering.labels_)) == N_CLUSTERS, \
            clustering.labels_

        self.assertAlmostEqual(
            np.average(clustering.distances_), 0.1, delta=0.02)
        self.assertAlmostEqual(
            np.std(clustering.distances_), 0.01875, delta=0.005)

        with self.assertRaises(DataInvalid):
            clustering.result_.partition([5, 10])

        presult = clustering.result_.partition([len(self.trj)-100, 100])

        self.assertTrue(np.all(
            presult.distances[0] == clustering.distances_[0:-100]))

        self.assertTrue(
            np.all(presult.assignments[0] == clustering.labels_[0:-100]))

        self.assertEqual(len(presult.center_indices[0]), 2)

    def test_kmedoids(self):
        '''
        Check that pure kmedoid clustering is behaving as expected on
        md.Trajectory objects.
        '''

        N_CLUSTERS = 5

        result = kmedoids.kmedoids(
            self.trj,
            distance_method='rmsd',
            n_clusters=N_CLUSTERS,
            n_iters=1000)

        # kcenters will always produce the same number of clusters on
        # this input data (unchanged by kmedoids updates)
        self.assertEqual(len(np.unique(result.assignments)), N_CLUSTERS)

        self.assertAlmostEqual(
            np.average(result.distances), 0.085, delta=0.01)
        self.assertAlmostEqual(
            np.std(result.distances), 0.019, delta=0.005)

    def test_hybrid(self):
        '''
        Clustering works on md.Trajectories.
        '''
        N_CLUSTERS = 5

        results = [hybrid(
            self.trj,
            distance_method='rmsd',
            init_centers=None,
            n_clusters=N_CLUSTERS,
            random_first_center=False,
            n_iters=10
            ) for i in range(10)]

        result = results[0]

        # kcenters will always produce the same number of clusters on
        # this input data (unchanged by kmedoids updates)
        assert len(np.unique(result.assignments)) == N_CLUSTERS

        # we do this here because hybrid seems to be more unstable than the
        # other two testing methods for some reason.
        all_dists = np.concatenate([r.distances for r in results])
        assert_less(
            abs(np.average(all_dists) - 0.085),
            0.012)

        assert_less(
            abs(np.std(result.distances) - 0.0187),
            0.005)

    def test_kcenters_maxdist(self):
        result = kcenters.kcenters(
            self.trj,
            distance_method='rmsd',
            init_centers=None,
            dist_cutoff=0.1,
            random_first_center=False
            )

        self.assertEqual(len(np.unique(result.assignments)), 17)

        # this is simlar to asserting the distribution of distances.
        # since KCenters is deterministic, this shouldn't ever change?
        self.assertAlmostEqual(np.average(result.distances),
                               0.074690734158752686)
        self.assertAlmostEqual(np.std(result.distances),
                               0.018754008455304401)

    def test_kcenters_nclust(self):
        N_CLUSTERS = 3

        result = kcenters.kcenters(
            self.trj,
            distance_method='rmsd',
            n_clusters=N_CLUSTERS,
            init_centers=None,
            random_first_center=False
            )

        self.assertEqual(len(np.unique(result.assignments)), N_CLUSTERS)

        # this is simlar to asserting the distribution of distances.
        # since KCenters is deterministic, this shouldn't ever change?
        self.assertAlmostEqual(np.average(result.distances),
                               0.10387578309920734)
        self.assertAlmostEqual(np.std(result.distances),
                               0.018355072790569946)

def test_kcenters_mpi_traj():
    from mpi4py import MPI
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_SIZE = MPI.COMM_WORLD.Get_size()

    trj = md.load(get_fn('frame0.h5'))

    data = trj[MPI_RANK::MPI_SIZE]

    r = kcenters.kcenters_mpi(data, md.rmsd, n_clusters=10)
    world_distances, world_assignments, world_ctr_inds = r

    mpi_assigs = np.empty((len(trj),), dtype=world_assignments.dtype)
    mpi_dists = np.empty((len(trj),), dtype=world_distances.dtype)
    mpi_ctr_inds = [(i*MPI_SIZE)+r for r, i in world_ctr_inds]

    try:
        np.save('assig%s.npy' % MPI_RANK, world_assignments)
        np.save('dists%s.npy' % MPI_RANK, world_distances)
        MPI.COMM_WORLD.Barrier()
        time.sleep(2)

        for i in range(MPI_SIZE):
            mpi_assigs[i::MPI_SIZE] = np.load('assig%s.npy' % i)
            mpi_dists[i::MPI_SIZE] = np.load('dists%s.npy' % i)
    finally:
        os.remove('assig%s.npy' % MPI_RANK)
        os.remove('dists%s.npy' % MPI_RANK)

    if MPI_RANK == 0:
        r = kcenters.kcenters(trj, md.rmsd, n_clusters=10)

        assert_array_equal(mpi_assigs, r.assignments)
        assert_array_equal(mpi_dists, r.distances)
        assert_array_equal(mpi_ctr_inds, r.center_indices)


def test_kcenters_mpi_numpy():
    from mpi4py import MPI
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_SIZE = MPI.COMM_WORLD.Get_size()

    trj = md.load(get_fn('frame0.h5')).xyz[:, :, 0].copy()

    data = trj[MPI_RANK::MPI_SIZE]

    def euc(X, x):
        return np.square(X -x).sum(axis=1)

    r = kcenters.kcenters_mpi(data, euc, n_clusters=10)
    world_distances, world_assignments, world_ctr_inds = r

    mpi_assigs = np.empty((len(trj),), dtype=world_assignments.dtype)
    mpi_dists = np.empty((len(trj),), dtype=world_distances.dtype)
    mpi_ctr_inds = [(i*MPI_SIZE)+r for r, i in world_ctr_inds]

    try:
        np.save('assig%s.npy' % MPI_RANK, world_assignments)
        np.save('dists%s.npy' % MPI_RANK, world_distances)
        MPI.COMM_WORLD.Barrier()
        time.sleep(2)

        for i in range(MPI_SIZE):
            mpi_assigs[i::MPI_SIZE] = np.load('assig%s.npy' % i)
            mpi_dists[i::MPI_SIZE] = np.load('dists%s.npy' % i)
    finally:
        os.remove('assig%s.npy' % MPI_RANK)
        os.remove('dists%s.npy' % MPI_RANK)

    if MPI_RANK == 0:
        r = kcenters.kcenters(trj, euc, n_clusters=10)

        assert_array_equal(mpi_assigs, r.assignments)
        assert_array_equal(mpi_dists, r.distances)
        assert_array_equal(mpi_ctr_inds, r.center_indices)


def fix_rng(seed=0):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):

            state = np.random.get_state()
            np.random.seed(seed)

            try:
                return f(*args, **kwargs)
            finally:
                np.random.set_state(state)

        return wrapper
    return decorator

@fix_rng()
def test_kmedoids_update_mpi_mdtraj():
    from mpi4py import MPI
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_SIZE = MPI.COMM_WORLD.Get_size()

    trj = md.load(get_fn('frame0.h5'))
    DIST_FUNC = md.rmsd

    data = trj[MPI_RANK::MPI_SIZE]

    r = kcenters.kcenters_mpi(data, DIST_FUNC, n_clusters=10)
    local_distances, local_assignments, local_ctr_inds = r

    r = kmedoids._kmedoids_update_mpi(
        data, DIST_FUNC, local_ctr_inds, local_assignments, local_distances)
    local_ctr_inds, local_assignments, local_distances = r

    mpi_ctr_inds = [(i*MPI_SIZE)+r for r, i in local_ctr_inds]

    assert_array_equal(
        mpi_ctr_inds,
        [  0,  44, 400, 105, 372, 327, 242, 346,  54, 295])

    true_assigs, true_dists = util.assign_to_nearest_center(
        trj, trj[mpi_ctr_inds], DIST_FUNC)

    assert_allclose(local_assignments, true_assigs[MPI_RANK::MPI_SIZE])
    assert_allclose(local_distances, true_dists[MPI_RANK::MPI_SIZE],
                    rtol=1e-06, atol=1e-03)


@fix_rng()
def test_kmedoids_update_mpi_numpy():
    from mpi4py import MPI
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_SIZE = MPI.COMM_WORLD.Get_size()

    trj = md.load(get_fn('frame0.h5')).xyz[:, :, 0].copy()

    data = trj[MPI_RANK::MPI_SIZE]

    def DIST_FUNC(X, x):
        return np.square(X -x).sum(axis=1)

    r = kcenters.kcenters_mpi(data, DIST_FUNC, n_clusters=10)
    local_distances, local_assignments, local_ctr_inds = r

    r = kmedoids._kmedoids_update_mpi(
        data, DIST_FUNC, local_ctr_inds, local_assignments, local_distances)
    local_ctr_inds, local_assignments, local_distances = r

    mpi_ctr_inds = [(i*MPI_SIZE)+r for r, i in local_ctr_inds]

    assert_array_equal(
        mpi_ctr_inds,
        [  0, 230, 262, 354,  55, 432, 211, 142, 325, 366])

    true_assigs, true_dists = util.assign_to_nearest_center(
        trj, trj[mpi_ctr_inds], DIST_FUNC)

    assert_allclose(local_assignments, true_assigs[MPI_RANK::MPI_SIZE])
    assert_allclose(local_distances, true_dists[MPI_RANK::MPI_SIZE],
                    rtol=1e-06, atol=1e-03)



class TestNumpyClustering(unittest.TestCase):

    generators = [(1, 1), (10, 10), (0, 20)]

    def setUp(self):

        g1 = self.generators[0]
        s1_x_coords = np.random.normal(loc=g1[0], scale=1, size=20)
        s1_y_coords = np.random.normal(loc=g1[1], scale=1, size=20)
        s1_xy_coords = np.zeros((20, 2))
        s1_xy_coords[:, 0] = s1_x_coords
        s1_xy_coords[:, 1] = s1_y_coords

        g2 = self.generators[1]
        s2_x_coords = np.random.normal(loc=g2[0], scale=1, size=20)
        s2_y_coords = np.random.normal(loc=g2[1], scale=1, size=20)
        s2_xy_coords = np.zeros((20, 2))
        s2_xy_coords[:, 0] = s2_x_coords
        s2_xy_coords[:, 1] = s2_y_coords

        g3 = self.generators[2]
        s3_x_coords = np.random.normal(loc=g3[0], scale=1, size=20)
        s3_y_coords = np.random.normal(loc=g3[1], scale=1, size=20)
        s3_xy_coords = np.zeros((20, 2))
        s3_xy_coords[:, 0] = s3_x_coords
        s3_xy_coords[:, 1] = s3_y_coords

        self.all_x_coords = np.concatenate(
            (s1_x_coords, s2_x_coords, s3_x_coords))
        self.all_y_coords = np.concatenate(
            (s1_y_coords, s2_y_coords, s3_y_coords))

        self.traj_lst = [s1_xy_coords, s2_xy_coords, s3_xy_coords]

    def test_predict(self):

        from ..cluster.util import ClusterResult

        centers = np.array(self.generators, dtype='float64')

        result = ClusterResult(
            centers=centers,
            assignments=None,
            distances=None,
            center_indices=None)

        clust = kcenters.KCenters('euclidean', cluster_radius=2)

        clust.result_ = result

        predict_result = clust.predict(np.concatenate(self.traj_lst))

        assert_array_equal(
            predict_result.assignments,
            [0]*20 + [1]*20 + [2]*20)

        assert_true(np.all(predict_result.distances < 4))

        assert_array_equal(
            np.argmin(predict_result.distances[0:20]),
            predict_result.center_indices[0])

        assert_is(predict_result.centers, centers)

    def test_kcenters_hot_start(self):

        clust = kcenters.KCenters('euclidean', cluster_radius=6)

        clust.fit(
            X=np.concatenate(self.traj_lst),
            init_centers=np.array(self.generators[0:2], dtype=float))

        print(clust.result_.center_indices, len(clust.result_.center_indices))

        assert_equal(len(clust.result_.center_indices), 3)
        assert_equal(len(np.unique(clust.result_.center_indices)),
                     np.max(clust.result_.assignments)+1)

        # because two centers were generators, only one center
        # should actually be a frame
        assert_equal(len(np.where(clust.result_.distances == 0)), 1)

    def test_numpy_hybrid(self):
        N_CLUSTERS = 3

        result = hybrid(
            np.concatenate(self.traj_lst),
            distance_method='euclidean',
            n_clusters=N_CLUSTERS,
            dist_cutoff=None,
            n_iters=100,
            random_first_center=False)

        assert len(result.center_indices) == N_CLUSTERS

        centers = util.find_cluster_centers(
            result.assignments, result.distances)
        self.check_generators(
            np.concatenate(self.traj_lst)[centers], distance=4.0)

    def test_numpy_kcenters(self):
        result = kcenters.kcenters(
            np.concatenate(self.traj_lst),
            distance_method='euclidean',
            n_clusters=3,
            dist_cutoff=2,
            init_centers=None,
            random_first_center=False)

        centers = util.find_cluster_centers(
            result.assignments, result.distances)
        self.check_generators(
            np.concatenate(self.traj_lst)[centers], distance=4.0)

    def test_numpy_kmedoids(self):
        N_CLUSTERS = 3

        result = kmedoids.kmedoids(
            np.concatenate(self.traj_lst),
            distance_method='euclidean',
            n_clusters=N_CLUSTERS,
            n_iters=1000)

        assert len(np.unique(result.assignments)) == N_CLUSTERS
        assert len(result.center_indices) == N_CLUSTERS

        centers = util.find_cluster_centers(
            result.assignments, result.distances)
        self.check_generators(
            np.concatenate(self.traj_lst)[centers], distance=4.0)

    def check_generators(self, centers, distance):

        import matplotlib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # req'd for some environments (esp. macOS).
            matplotlib.use('TkAgg')

        try:
            for c in centers:
                mindist = min([np.linalg.norm(c-g) for g in self.generators])
                self.assertLess(
                    mindist, distance,
                    "Expected center {c} to be less than 2 away frome one of"
                    "{g} (was {d}).".
                    format(c=c, g=self.generators, d=mindist))

            for g in self.generators:
                mindist = min([np.linalg.norm(c-g) for c in centers])
                self.assertLess(
                    mindist, distance,
                    "Expected generator {g} to be less than 2 away frome one"
                    "of {c} (was {d}).".
                    format(c=c, g=self.generators, d=mindist))
        except AssertionError:
            x_centers = [c[0] for c in centers]
            y_centers = [c[1] for c in centers]

            from pylab import scatter, show

            scatter(self.all_x_coords, self.all_y_coords, s=40, c='k')
            scatter(x_centers, y_centers, s=40, c='y')
            show()
            raise


def test_mpi_distribute_frame_ndarray():

    from mpi4py import MPI
    data = np.arange(10*100*3).reshape(10, 100, 3)

    d = util.mpi_distribute_frame(data, 7, MPI.COMM_WORLD.Get_size()-1)

    assert_array_equal(d, data[7])
    assert_is(type(d), type(data))


def test_mpi_distribute_frame_mdtraj():

    from mpi4py import MPI
    data = md.load(get_fn('frame0.h5'))

    d = util.mpi_distribute_frame(data, 7, MPI.COMM_WORLD.Get_size()-1)

    assert_array_equal(d.xyz, data[7].xyz)
    assert_is(type(d), type(data))


if __name__ == '__main__':
    unittest.main()
