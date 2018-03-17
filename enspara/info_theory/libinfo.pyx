import numpy as np
from cython.parallel import prange

cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
def bincount2d(
        np.ndarray[DTYPE_t, ndim=1] a, np.ndarray[DTYPE_t, ndim=1] b,
        int n_a, int n_b):

    cdef np.ndarray[DTYPE_t, ndim=2] H = np.zeros((n_a, n_b), dtype=DTYPE)
    cdef unsigned int i, j, t
    assert a.shape[0] == b.shape[0]
    
    for t in range(a.shape[0]):
        i = a[t]
        j = b[t]
        H[i, j] += 1

    return H


@cython.boundscheck(False)
def matrix_bincount2d_symmetrical(np.ndarray[DTYPE_t, ndim=2] a, int n):
   
    cdef np.ndarray[DTYPE_t, ndim=4] jc = np.zeros((a.shape[1], a.shape[1], n, n), dtype=DTYPE)
    cdef long a_row, b_row, i, j, t, k
    cdef unsigned int n_features = a.shape[1]
    cdef unsigned int n_coords = int(((n_features)*(n_features-1))/2)

    # assemble an array of half-matrix coordinates
    cdef np.ndarray[DTYPE_t, ndim=2] coords = np.zeros((n_coords, 2), dtype=DTYPE)
    k = 0
    for i in range(n_coords):
        for j in range(i, n_coords):
            coords[k, 0] = i
            coords[k, 1] = j
            k += 0
    k = 0

    assert a.shape[0] == a.shape[0]

    for k in prange(n_coords, nogil=True):
        a_row = coords[k, 0]
        b_row = coords[k, 1]
        
        for t in range(a.shape[0]):
            i = a[t, a_row]
            j = a[t, b_row]
            jc[a_row, b_row, i, j] += 1
            jc[b_row, a_row, i, j] += 1

    return jc
