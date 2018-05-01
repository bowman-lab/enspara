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

    cdef np.ndarray[np.uint32_t, ndim=2] H = np.zeros((n_a, n_b), dtype=np.uint32)
    cdef unsigned int i, j, t

    assert a.shape[0] == b.shape[0]
    
    for t in range(a.shape[0]):
        i = a[t]
        j = b[t]
        H[i, j] += 1

    return H


@cython.boundscheck(False)
@cython.wraparound(False) 
def matrix_bincount2d_symmetrical(INTEGRAL_2D_ARRAY a, int n):

    # this guy is holding our joint counts, so this will top out at
    # ~4 billion timepoints
    assert a.shape[1] < 2**32
    
    cdef np.ndarray[np.uint32_t, ndim=4] jc = np.zeros(
        (a.shape[1], a.shape[1], n, n), dtype=np.uint32)

    cdef long a_row, b_row, i, j, t, k = 0
    cdef long n_features = a.shape[1]
    cdef long n_coords = int(((n_features)*(n_features+1))/2)

    # assemble an array of half-matrix coordinates
    cdef np.ndarray[np.int32_t, ndim=2] coords = np.empty((n_coords, 2),
                                                          dtype=np.int32)

    k = 0
    for i in range(n_features):
        for j in range(i, n_features):
            coords[k, 0] = i
            coords[k, 1] = j
            k += 1
        
    k = 0

    for k in prange(n_coords, nogil=True):
        a_row = coords[k, 0]
        b_row = coords[k, 1]
        
        for t in range(a.shape[0]):
            i = a[t, a_row]
            j = a[t, b_row]
            jc[a_row, b_row, i, j] += 1
            jc[b_row, a_row, i, j] += 1

    return jc
