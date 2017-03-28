from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
import mdtraj as md
from mdtraj.testing import get_fn

from nose.tools import assert_raises, assert_less

from . import save_states

from .hybrid import KHybrid, hybrid
from .kcenters import KCenters, kcenters
from .kmedoids import kmedoids
from .util import find_cluster_centers

from ..exception import DataInvalid, ImproperlyConfigured

import matplotlib
matplotlib.use('TkAgg')  # req'd for some environments (esp. macOS).


class TestTrajClustering(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')

        self.trj = md.load(self.trj_fname, top=self.top_fname)

    def test_centers_object(self):
        # '''
        # KHybrid() clusterer should produce correct output.
        # '''
        N_CLUSTERS = 5
        CLUSTER_RADIUS = 0.1

        with assert_raises(ImproperlyConfigured):
            KCenters(metric=md.rmsd)

        # Test n clusters
        clustering = KCenters(
            metric=md.rmsd,
            n_clusters=N_CLUSTERS)

        clustering.fit(self.trj)

        assert hasattr(clustering, 'result_')
        assert len(np.unique(clustering.labels_)) == N_CLUSTERS, \
            clustering.labels_

        # Test dist_cutoff
        clustering = KCenters(
            metric=md.rmsd,
            cluster_radius=CLUSTER_RADIUS)

        clustering.fit(self.trj)

        assert hasattr(clustering, 'result_')
        assert clustering.distances_.max() < CLUSTER_RADIUS, \
            clustering.distances_

        # Test n clusters and dist_cutoff
        clustering = KCenters(
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

        result = kmedoids(
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
        Check that hybrid clustering is behaving as expected on
        md.Trajectory objects.
        '''
        N_CLUSTERS = 5

        results = [hybrid(
            self.trj,
            distance_method='rmsd',
            init_cluster_centers=None,
            n_clusters=N_CLUSTERS,
            random_first_center=False,
            n_iters=100
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
            0.011)

        assert_less(
            abs(np.std(result.distances) - 0.0187),
            0.005)

    def test_kcenters_maxdist(self):
        result = kcenters(
            self.trj,
            distance_method='rmsd',
            init_cluster_centers=None,
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

        result = kcenters(
            self.trj,
            distance_method='rmsd',
            n_clusters=N_CLUSTERS,
            init_cluster_centers=None,
            random_first_center=False
            )

        self.assertEqual(len(np.unique(result.assignments)), N_CLUSTERS)

        # this is simlar to asserting the distribution of distances.
        # since KCenters is deterministic, this shouldn't ever change?
        self.assertAlmostEqual(np.average(result.distances),
                               0.10387578309920734)
        self.assertAlmostEqual(np.std(result.distances),
                               0.018355072790569946)


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

    def test_hybrid(self):
        N_CLUSTERS = 3

        result = hybrid(
            np.concatenate(self.traj_lst),
            distance_method='euclidean',
            n_clusters=N_CLUSTERS,
            dist_cutoff=None,
            n_iters=100,
            random_first_center=False)

        assert len(result.center_indices) == N_CLUSTERS

        centers = np.concatenate(find_cluster_centers(
            np.concatenate(self.traj_lst), result.distances))

        self.check_generators(centers, distance=4.0)

    def test_kcenters(self):
        result = kcenters(
            np.concatenate(self.traj_lst),
            distance_method='euclidean',
            n_clusters=3,
            dist_cutoff=2,
            init_cluster_centers=None,
            random_first_center=False)

        centers = np.concatenate(find_cluster_centers(
            np.concatenate(self.traj_lst), result.distances))

        self.check_generators(centers, distance=4.0)

    def test_kmedoids(self):
        N_CLUSTERS = 3

        result = kmedoids(
            np.concatenate(self.traj_lst),
            distance_method='euclidean',
            n_clusters=N_CLUSTERS,
            n_iters=10000)

        assert len(np.unique(result.assignments)) == N_CLUSTERS
        assert len(result.center_indices) == N_CLUSTERS

        centers = np.concatenate(find_cluster_centers(
            np.concatenate(self.traj_lst), result.distances))

        self.check_generators(centers, distance=2.0)

    def check_generators(self, centers, distance):

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


class TestSaveStates(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')

    def test_unique_state_extraction(self):
        '''
        Check to makes sure we get the unique states from the trajectory
        correctly
        '''

        states = [0, 1, 2, 3, 4]
        assignments = np.random.choice(states, (100000))

        self.assertTrue(
            all(save_states.unique_states(assignments) == states))

        states = [-1, 0, 1, 2, 3, 4]
        assignments = np.random.choice(states, (100000))

        self.assertTrue(
            all(save_states.unique_states(assignments) == states[1:]))


class TestUtils(unittest.TestCase):

    def test_find_cluster_centers(self):

        N_TRJS = 20
        many_trjs = [md.load(get_fn('frame0.xtc'), top=get_fn('native.pdb'))
                     for i in range(N_TRJS)]

        distances = np.ones((N_TRJS, len(many_trjs[0])))

        center_inds = [(0, 0), (5, 2), (15, 300)]

        for ind in center_inds:
            distances[center_inds[0], center_inds[1]] = 0

        centers = find_cluster_centers(many_trjs, distances)

        # we should get back the same number of centers as there are zeroes
        # in the distances matrix
        self.assertEqual(len(centers), np.count_nonzero(distances == 0))

        for indx in center_inds:
            expected_xyz = many_trjs[indx[0]][indx[1]].xyz
            self.assertTrue(np.any(
                expected_xyz == np.array([c.xyz for c in centers])))


if __name__ == '__main__':
    unittest.main()
