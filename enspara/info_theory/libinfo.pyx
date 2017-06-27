import numpy as np
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
