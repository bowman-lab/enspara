import numpy as np
from scipy.spatial.distance import cdist

from nose.tools import assert_raises
from numpy.testing import assert_array_equal

from enspara import exception
from enspara.geometry import libdist


def test_manhattan_distance():

    X = np.array([[ 1, 1],
                  [ 2, 2],
                  [ 3, 3],
                  [-1, 3]])
    y = np.array([0, 0])

    with assert_raises(exception.DataInvalid):
        libdist.manhattan(X, y.reshape(1, -1))

    with assert_raises(exception.DataInvalid):
        libdist.manhattan(X.reshape(1, -1), y)

    with assert_raises(exception.DataInvalid):
        libdist.manhattan(X.flatten(), y)

    with assert_raises(exception.DataInvalid):
        libdist.manhattan(X, y[1:])

    d = libdist.manhattan(X, y)

    assert_array_equal(
        d, cdist(X, y.reshape(1, -1), metric='cityblock').flatten())


def test_euclidean_distance():

    X = np.array([[ 1, 1],
                  [ 2, 2],
                  [ 3, 3],
                  [-1, 3]])
    y = np.array([0, 0])

    with assert_raises(exception.DataInvalid):
        libdist.euclidean(X, y.reshape(1, -1))

    with assert_raises(exception.DataInvalid):
        libdist.euclidean(X.reshape(1, -1), y)

    with assert_raises(exception.DataInvalid):
        libdist.euclidean(X.flatten(), y)

    with assert_raises(exception.DataInvalid):
        libdist.euclidean(X, y[1:])

    d = libdist.euclidean(X, y)

    assert_array_equal(d, cdist(X, y.reshape(1, -1)).flatten())


def test_euclidean_distance_noalloc():

    X = np.array([[ 1, 1],
                  [ 2, 2],
                  [ 3, 3],
                  [-1, 3]])
    y = np.array([0, 0])

    with assert_raises(exception.DataInvalid):
        d = libdist.euclidean(
            X, y,
            out=np.empty(shape=(X.shape[0]), dtype='int'))

    with assert_raises(exception.DataInvalid):
        d = libdist.euclidean(
            X, y,
            out=np.empty(shape=(X.shape[0]-1)))

    d = libdist.euclidean(
        X, y, out=
        np.empty(shape=(X.shape[0]), dtype='float64'))

    assert_array_equal(
        d,
        cdist(X, y.reshape(1, -1)).flatten())
