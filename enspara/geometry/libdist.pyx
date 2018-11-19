import numpy as np
from cython.parallel import prange

from enspara import exception

cimport cython
cimport numpy as np

ctypedef fused FLOAT_TYPE_T:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

ctypedef fused INTEGRAL_TYPE_T:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

cdef extern from "math.h" nogil:
    double sqrt(double x)
    double fabs(double x)
    float fabs(float x)

def _check_is_2d(X):
    if len(X.shape) != 2:
        raise exception.DataInvalid(
            "Data array dimension must be two, got shape %s." %
            str(X.shape))

def _check_is_1d(x):
    if len(x.shape) != 1:
        raise exception.DataInvalid(
            "Target point dimension must be one, got shape %s." %
            str(x.shape))

def _prepare_for_2d_to_1d_distance(X, y, out):

    _check_is_2d(X)
    _check_is_1d(y)
    if X.shape[1] != y.shape[0]:
        raise exception.DataInvalid(
            ("Target data point dimension (%s) must match data " +
             "array dimension (%s)") % (y.shape[0], X.shape[1]))

    # if `out` isn't provided, allocate it.
    # if `out` is provided, check it for appropriateness
    if out is None:
        out = np.zeros((X.shape[0]), dtype=np.float64)
    else:
        # precision problems happen if out is less than 64-bit
        if out.dtype != np.float64:
            raise exception.DataInvalid(
                "In-place output array must be np.float64, got '%s'."
                % out.dtype)
        if out.shape[0] != X.shape[0]:
            raise exception.DataInvalid(
                ("In-place output array dimension (%s) must match number of "
                 "samples in data array (%s)") % (out.shape[0], X.shape[0]))
        if len(out.shape) != 1:
            raise exception.DataInvalid(
                "In-place output array must be one-dimensional, "
                "got shape %s" %
                out.shape)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def _hamming(np.ndarray[INTEGRAL_TYPE_T, ndim=2] X,
             np.ndarray[INTEGRAL_TYPE_T, ndim=1] y,
             np.ndarray[np.float64_t, ndim=1] out):

    cdef long n_samples = len(out)
    cdef long n_features = len(y)
    assert len(out) == X.shape[0], "Size of output array didn't match number of observations in X"
    assert n_features == X.shape[1], "Number of features between X and y didn't match."

    cdef long i, j = 0

    for i in prange(n_samples, nogil=True):
        out[i] = 0
        for j in range(n_features):
            if y[j] != X[i, j]:
                out[i] += 1
        out[i] /= n_features

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def _manhattan(np.ndarray[FLOAT_TYPE_T, ndim=2] X,
               np.ndarray[FLOAT_TYPE_T, ndim=1] y,
               np.ndarray[np.float64_t, ndim=1] out):

    cdef long n_samples = len(out)
    cdef long n_features = len(y)
    assert len(out) == X.shape[0]
    assert n_features == X.shape[1]

    cdef long i, j = 0
    for i in prange(n_samples, nogil=True):
        out[i] = 0

    for i in prange(n_samples, nogil=True):
        for j in range(n_features):
            out[i] += fabs(X[i, j] - y[j])

    return out.reshape(-1, 1)


@cython.boundscheck(False)
@cython.wraparound(False)
def _euclidean(np.ndarray[FLOAT_TYPE_T, ndim=2] X,
               np.ndarray[FLOAT_TYPE_T, ndim=1] y,
               np.ndarray[np.float64_t, ndim=1] out):

    cdef long n_samples = len(out)
    cdef long n_features = len(y)
    assert len(out) == X.shape[0]
    assert n_features == X.shape[1]

    cdef long i, j = 0

    # zero out output array; this is fast compared to the actual
    # computation, so we always do it.
    for i in prange(n_samples, nogil=True):
        out[i] = 0

    for i in prange(n_samples, nogil=True):
        for j in range(n_features):
            out[i] += (X[i, j] - y[j])**2

    for i in prange(n_samples, nogil=True):
        out[i] = sqrt(out[i])

    return out.reshape(-1, 1)


def euclidean(X, y, out=None):
    """Compute the euclidean distance between a point, `y`, and a group
    of points `X`. Uses thread-parallelism with OpenMP.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features)
        The group of points for which to compute the distance from `y`.
    y: array, shape=(n_features)
        The point, for all rows in `X`, to compute the distance to.
    out: array, shape=(n_samples), default=None
        If provided, the array to place the distances in. If not provided,
        an array will be allocated for you.
    """
    out = _prepare_for_2d_to_1d_distance(X, y, out)
    _euclidean(X, y, out)
    return out

def manhattan(X, y, out=None):
    """Compute the Manhattan distance between a point `y` and a group of
    points `X`. Thread-parallized using OpenMP.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features)
        The group of points for which to compute the distance from `y`.
    y: array, shape=(n_features)
        The point, for all rows in `X`, to compute the distance to.
    out: array, shape=(n_samples), default=None
        If provided, the array to place the distances in. If not provided,
        an array will be allocated for you.
    """

    out = _prepare_for_2d_to_1d_distance(X, y, out)
    _manhattan(X, y, out)
    return out


def hamming(X, y, out=None):
    """Compute the Hamming distance between a point `y` and a group of
    points `X`. Thread-parallized using OpenMP.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features)
        The group of points for which to compute the distance from `y`.
    y: array, shape=(n_features)
        The point, for all rows in `X`, to compute the distance to.
    out: array, shape=(n_samples), default=None
        If provided, the array to place the distances in. If not provided,
        an array will be allocated for you.
    """

    out = _prepare_for_2d_to_1d_distance(X, y, out)
    _hamming(X, y, out)
    return out
