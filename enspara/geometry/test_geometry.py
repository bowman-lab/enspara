from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances \
    as sk_euclidean_distances


class TestTrajClustering(unittest.TestCase):

    def test_euclidean_distance(self):
        '''
        Verify that the output of enspara.geometry.euclidean_distance is the same
        (albeit with a different shape) as the sklearn euclidean_distances.
        '''

        X = np.array([np.arange(i, i+10) for i in range(10)])
        Y = X[4]

        self.assertTrue(np.allclose(
            euclidean_distance(X, Y),
            sk_euclidean_distances(X, Y.reshape(1, -1)).flatten()))


if __name__ == '__main__':
    unittest.main()
