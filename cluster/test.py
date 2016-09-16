import unittest

import numpy as np
import stag.cluster as cluster
import os

import matplotlib
matplotlib.use('TkAgg')
from pylab import *


class TestStringMethods(unittest.TestCase):

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

    # def test_libdistance(self):

    #     centers, assigns, dists = cluster.kmedoids(
    #         self.traj_lst,
    #         metric='euclidean',
    #         n_clusters=3,
    #         delete_trjs=False,
    #         verbose=True,
    #         n_iters=1)

    def test_hybrid(self):
        assigns, dists = cluster.hybrid(
            self.traj_lst,
            metric='euclidean',
            n_clusters=3,
            dist_cutoff=None,
            n_iters=100,
            random_first_center=False,
            delete_trjs=False,
            output=open(os.devnull, 'w'))

        centers = cluster.utils.find_cluster_centers(self.traj_lst, dists)

        try:
            self.check_generators(centers, distance=2.0)
        except AssertionError:
            x_centers = [c[0] for c in centers]
            y_centers = [c[1] for c in centers]

            scatter(self.all_x_coords, self.all_y_coords, s=40, c='k')
            scatter(x_centers, y_centers, s=40, c='y')
            show()
            raise

    def test_kcenters(self):
        assigns, dists = cluster.kcenters(
            self.traj_lst,
            metric='euclidean',
            n_clusters=3,
            dist_cutoff=2,
            cluster_centers=None,
            random_first_center=False,
            delete_trjs=False,
            output=open(os.devnull, 'w'))

        centers = cluster.utils.find_cluster_centers(self.traj_lst, dists)

        try:
            self.check_generators(centers, distance=4.0)
        except AssertionError:
            x_centers = [c[0] for c in centers]
            y_centers = [c[1] for c in centers]

            scatter(self.all_x_coords, self.all_y_coords, s=40, c='k')
            scatter(x_centers, y_centers, s=40, c='y')
            show()
            raise

    def test_kmedoids(self):

        assigns, dists = cluster.kmedoids(
            self.traj_lst,
            metric='euclidean',
            n_clusters=3,
            delete_trjs=False,
            n_iters=10000,
            output=open(os.devnull, 'w'))

        centers = cluster.utils.find_cluster_centers(self.traj_lst, dists)

        try:
            self.check_generators(centers, distance=2.0)
        except AssertionError:
            x_centers = [c[0] for c in centers]
            y_centers = [c[1] for c in centers]

            scatter(self.all_x_coords, self.all_y_coords, s=40, c='k')
            scatter(x_centers, y_centers, s=40, c='y')
            show()
            raise

    def check_generators(self, centers, distance):

        for c in centers:
            mindist = min([np.linalg.norm(c-g) for g in self.generators])
            self.assertLess(
                mindist, distance,
                "Expected center {c} to be less than 2 away frome one of {g} "
                "(was {d}).".
                format(c=c, g=self.generators, d=mindist))

        for g in self.generators:
            mindist = min([np.linalg.norm(c-g) for c in centers])
            self.assertLess(
                mindist, distance,
                "Expected generator {g} to be less than 2 away frome one of "
                "{c} (was {d}).".
                format(c=c, g=self.generators, d=mindist))


if __name__ == '__main__':
    unittest.main()
