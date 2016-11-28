import unittest

import numpy as np

from stag.geometry.euc_dist import euclidean_distance
from sklearn.metrics.pairwise import sk_euclidean_distances


class TestTrajClustering(unittest.TestCase):

    def test_euclidean_distance(self):

        X = np.array([np.arange(i, i+10) for i in range(10)])
        Y = X[4]

        self.assertTrue(np.all(
            euclidean_distance(X, Y) ==
            sk_euclidean_distances(X, Y.reshape(1, -1))))


if __name__ == '__main__':
    unittest.main()
