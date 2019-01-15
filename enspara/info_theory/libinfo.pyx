import numpy as np
from cython.parallel import prange

cimport numpy as np
cimport cython


ctypedef fused INTEGRAL_2D_ARRAY:
    np.ndarray[np.int8_t, ndim=2]
    np.ndarray[np.int16_t, ndim=2]
    np.ndarray[np.int32_t, ndim=2]
    np.ndarray[np.int64_t, ndim=2]
    np.ndarray[np.uint8_t, ndim=2]
    np.ndarray[np.uint16_t, ndim=2]
    np.ndarray[np.uint32_t, ndim=2]
    np.ndarray[np.uint64_t, ndim=2]

ctypedef fused INTEGRAL_1D_ARRAY:
    np.ndarray[np.int8_t, ndim=1]
    np.ndarray[np.int16_t, ndim=1]
    np.ndarray[np.int32_t, ndim=1]
    np.ndarray[np.int64_t, ndim=1]
    np.ndarray[np.uint8_t, ndim=1]
    np.ndarray[np.uint16_t, ndim=1]
    np.ndarray[np.uint32_t, ndim=1]
    np.ndarray[np.uint64_t, ndim=1]


@cython.boundscheck(False)
def bincount2d(
        INTEGRAL_1D_ARRAY a, INTEGRAL_1D_ARRAY b,
        int n_a, int n_b):

    cdef np.ndarray[np.uint32_t, ndim=2] H = np.zeros((n_a, n_b),
                                                      dtype=np.uint32)
    cdef unsigned int i, j, t

    assert a.shape[0] == b.shape[0]

    for t in range(a.shape[0]):
        i = a[t]
        j = b[t]
        H[i, j] += 1

    return H


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_bincount2d(
        INTEGRAL_2D_ARRAY a, INTEGRAL_2D_ARRAY b,
        int n_a, int n_b):

    # this guy is holding our joint counts, so this will top out at
    # ~4 billion timepoints
    assert a.shape[1] < 2**32, "No support for trajectories longer than 2^32"
    assert a.shape[0] == b.shape[0], 'Feature arrays a and b must match in length'
    assert a.max() < n_a, "States indices must be contiguous."
    assert b.max() < n_b, "States indices must be contiguous."

    cdef np.ndarray[np.uint32_t, ndim=4] jc = np.zeros(
        (a.shape[1], b.shape[1], n_a, n_b), dtype=np.uint32)

    cdef long a_row, b_row, i, j, t
    cdef long n_features = a.shape[1]

    for a_row in prange(a.shape[1], nogil=True):
        for b_row in range(b.shape[1]):
            for t in range(a.shape[0]):
                i = a[t, a_row]
                j = b[t, b_row]
                jc[a_row, b_row, i, j] += 1

    return jc
