import warnings
import numpy as np

from enspara import exception

cimport cython
cimport numpy as np

cdef extern from "math.h" nogil:
    double sqrt(double x)
    double log10(double x)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _mle_prinz_dense(
        np.ndarray[np.float64_t, ndim=2] C,
        double tol=1e-10,
        long max_iter=10**5):
    cdef int n_states = len(C)
    cdef np.ndarray[np.float64_t, ndim=2] X = C + C.T

    cdef np.ndarray[np.float64_t, ndim=1] X_rs = X.sum(axis=1)
    cdef np.ndarray[np.float64_t, ndim=1] C_rs = C.sum(axis=1)

    cdef double logl, oldlogl = 0
    cdef long n_iter = 0

    assert np.all(X_rs > 0)
    assert np.all(C_rs > 0)

    cdef long i, j = 0
    cdef double tmp, a, b, c, v, denom = 0

    for n_iter in range(max_iter):
        logl = 0

        for i in range(n_states):
            tmp = X[i,i];
            denom = C_rs[i] - C[i, i];
            if (denom > 0):
                X[i, i] = C[i, i] * (X_rs[i] - X[i, i]) / denom;

            X_rs[i] = X_rs[i] + (X[i, i] - tmp);

            if (X[i, i] > 0):
                logl += C[i,i] * log10(X[i, i] / X_rs[i]);

        for i in range(n_states - 1):
            for j in range(i+1, n_states):

                a = (C_rs[i] - C[i, j]) + (C_rs[j] - C[j, i])
                b = C_rs[i] * (X_rs[j] - X[i, j]) +\
                    C_rs[j] * (X_rs[i] - X[i, j]) -\
                    (C[i, j] + C[j, i]) * (X_rs[i] + X_rs[j] - 2*X[i, j])

                c = -(C[i, j] + C[j, i]) *\
                     (X_rs[i] - X[i, j]) *\
                     (X_rs[j] - X[i,j])

                assert c <= 0

#                 /* the new value */
                if (a == 0):
                    v = X[j, i];
                else:
                    v = (-b + sqrt((b*b) - (4*a*c))) / (2*a)

#                 /* update the row sums */
                X_rs[i] = X_rs[i] + (v - X[i, j])
                X_rs[j] = X_rs[j] + (v - X[j, i])

#                 /* add in the new value */
                X[i, j] = v
                X[j, i] = v

                if (X[i, j] > 0):
                    logl += (
                        ((C[i,j] * log10(X[i, j]) / X_rs[i])) +
                        ((C[j,i] * log10(X[j, i]) / X_rs[j])))


        if abs(logl - oldlogl) > tol:
            oldlogl = logl
        else:
            break

    if n_iter == max_iter - 1:
        warnings.warn(
            exception.ConvergenceWarning,
            "Prinz MLE did not converge after %s iterations." % n_iter)

    T = X / X.sum(axis=-1).reshape(len(X), 1)
    pi = X_rs / X_rs.sum()

    assert np.allclose(T.sum(axis=1), 1, atol=1e-16), T.sum(axis=1)
    assert np.sum(pi) - 1 < 1e-14

    return T, pi
