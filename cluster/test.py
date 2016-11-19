import unittest

import os
import tempfile

import numpy as np
import mdtraj as md
from mdtraj.testing import get_fn

import stag.cluster as cluster
import stag.cluster.save_states as save_states

import matplotlib
matplotlib.use('TkAgg')  # req'd for some environments.


class TestTrajClustering(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')

        self.trj = md.load(self.trj_fname, top=self.top_fname)

    def test_kmedoids(self):
        N_CLUSTERS = 5

        result = cluster.kmedoids(
            [self.trj],
            metric='rmsd',
            n_clusters=N_CLUSTERS,
            delete_trjs=False,
            n_iters=1000,
            output=open(os.devnull, 'w')
            )

        # kcenters will always produce the same number of clusters on
        # this input data (unchanged by kmedoids updates)
        self.assertEqual(len(np.unique(result.assignments)), N_CLUSTERS)

        self.assertAlmostEqual(
            np.average(result.distances), 0.083686112175266184, delta=0.005)
        self.assertAlmostEqual(
            np.std(result.distances), 0.018754008455304401, delta=0.005)

    def test_hybrid(self):
        N_CLUSTERS = 5

        result = cluster.hybrid(
            [self.trj],
            metric='rmsd',
            init_cluster_centers=None,
            n_clusters=N_CLUSTERS,
            random_first_center=False,
            delete_trjs=False,
            n_iters=100,
            output=open(os.devnull, 'w')
            )

        # kcenters will always produce the same number of clusters on
        # this input data (unchanged by kmedoids updates)
        self.assertEqual(len(np.unique(result.assignments)), N_CLUSTERS)

        self.assertAlmostEqual(
            np.average(result.distances), 0.083686112175266184, delta=0.005)
        self.assertAlmostEqual(
            np.std(result.distances), 0.018754008455304401, delta=0.005)

    def test_kcenters_maxdist(self):
        result = cluster.kcenters(
            [self.trj],
            metric='rmsd',
            init_cluster_centers=None,
            dist_cutoff=0.1,
            random_first_center=False,
            delete_trjs=False,
            output=open(os.devnull, 'w')
            )

        self.assertEqual(len(np.unique(result.assignments)), 17)

        # this is simlar to asserting the distribution of distances.
        # since KCenters is deterministic, this shouldn't ever change?
        self.assertAlmostEqual(np.average(result.distances), 0.074690734158752686)
        self.assertAlmostEqual(np.std(result.distances), 0.018754008455304401)

    def test_kcenters_nclust(self):
        N_CLUSTERS = 3

        result = cluster.kcenters(
            [self.trj],
            metric='rmsd',
            n_clusters=N_CLUSTERS,
            init_cluster_centers=None,
            random_first_center=False,
            delete_trjs=False,
            output=open(os.devnull, 'w')
            )

        self.assertEqual(len(np.unique(result.assignments)), N_CLUSTERS)

        # this is simlar to asserting the distribution of distances.
        # since KCenters is deterministic, this shouldn't ever change?
        self.assertAlmostEqual(np.average(result.distances), 0.10387578309920734)
        self.assertAlmostEqual(np.std(result.distances), 0.018355072790569946)

    def test_delete_trjs(self):
        N_TRJS = 20
        many_trjs = [md.load(self.trj_fname, top=self.top_fname)
                     for i in range(N_TRJS)]
        self.assertEqual(len(many_trjs), N_TRJS)

        cluster.kcenters(
            many_trjs,
            metric='rmsd',
            dist_cutoff=0.1,
            init_cluster_centers=None,
            random_first_center=False,
            delete_trjs=True,
            output=open(os.devnull, 'w')
            )

        self.assertTrue(all([t is None for t in many_trjs]))


class TestNumpyClustering(unittest.TestCase):

    generators = [(1, 1), (10, 10), (0, 20)]

    def setUp(self):

        g1 = self.generators[0]
        s1_x_coords = np.random.normal(loc=g1[0], scale=1, size=10)
        s1_y_coords = np.random.normal(loc=g1[1], scale=1, size=10)
        s1_xy_coords = np.zeros((10, 2))
        s1_xy_coords[:, 0] = s1_x_coords
        s1_xy_coords[:, 1] = s1_y_coords

        g2 = self.generators[1]
        s2_x_coords = np.random.normal(loc=g2[0], scale=1, size=10)
        s2_y_coords = np.random.normal(loc=g2[1], scale=1, size=10)
        s2_xy_coords = np.zeros((10, 2))
        s2_xy_coords[:, 0] = s2_x_coords
        s2_xy_coords[:, 1] = s2_y_coords

        g3 = self.generators[2]
        s3_x_coords = np.random.normal(loc=g3[0], scale=1, size=10)
        s3_y_coords = np.random.normal(loc=g3[1], scale=1, size=10)
        s3_xy_coords = np.zeros((10, 2))
        s3_xy_coords[:, 0] = s3_x_coords
        s3_xy_coords[:, 1] = s3_y_coords

        self.all_x_coords = np.concatenate(
            (s1_x_coords, s2_x_coords, s3_x_coords))
        self.all_y_coords = np.concatenate(
            (s1_y_coords, s2_y_coords, s3_y_coords))

        self.traj_lst = [s1_xy_coords, s2_xy_coords, s3_xy_coords]

    def test_hybrid(self):
        result = cluster.hybrid(
            self.traj_lst,
            metric='euclidean',
            n_clusters=3,
            dist_cutoff=None,
            n_iters=100,
            random_first_center=False,
            delete_trjs=False,
            output=open(os.devnull, 'w'))

        centers = cluster.utils.find_cluster_centers(
            self.traj_lst, result.distances)

        self.check_generators(centers, distance=2.0)

    def test_kcenters(self):
        result = cluster.kcenters(
            self.traj_lst,
            metric='euclidean',
            n_clusters=3,
            dist_cutoff=2,
            init_cluster_centers=None,
            random_first_center=False,
            delete_trjs=False,
            output=open(os.devnull, 'w'))

        centers = cluster.utils.find_cluster_centers(
            self.traj_lst, result.distances)

        self.check_generators(centers, distance=4.0)

    def test_kmedoids(self):
        N_CLUSTERS = 3

        result = cluster.kmedoids(
            self.traj_lst,
            metric='euclidean',
            n_clusters=N_CLUSTERS,
            delete_trjs=False,
            n_iters=10000,
            output=open(os.devnull, 'w'))

        assert len(np.unique(result.assignments)) == N_CLUSTERS
        centers = cluster.utils.find_cluster_centers(
            self.traj_lst, result.distances)

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

    def test_partition_indices(self):

        indices = [0, 10, 15, 37, 100]
        trj_lens = [10, 20, 100]

        partit_indices = cluster.utils._partition_indices(indices, trj_lens)

        self.assertEqual(
            partit_indices,
            [(0, 0), (1, 0), (1, 5), (2, 7), (2, 70)])

    def test_find_cluster_centers(self):

        N_TRJS = 20
        many_trjs = [md.load(get_fn('frame0.xtc'), top=get_fn('native.pdb'))
                     for i in range(N_TRJS)]

        distances = np.ones((N_TRJS, len(many_trjs[0])))

        center_inds = [(0, 0), (5, 2), (15, 300)]

        for ind in center_inds:
            distances[center_inds[0], center_inds[1]] = 0

        centers = cluster.utils.find_cluster_centers(many_trjs, distances)

        # we should get back the same number of centers as there are zeroes
        # in the distances matrix
        self.assertEqual(len(centers), np.count_nonzero(distances == 0))

        for indx in center_inds:
            expected_xyz = many_trjs[indx[0]][indx[1]].xyz
            self.assertTrue(np.any(
                expected_xyz == np.array([c.xyz for c in centers])))


if __name__ == '__main__':
    unittest.main()
