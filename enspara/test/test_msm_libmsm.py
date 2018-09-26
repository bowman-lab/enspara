import numpy as np

from numpy.testing import assert_allclose

from enspara import exception
from enspara.msm.libmsm import _mle_prinz_dense


def prinz_mle_py(C, tol=1e-10, max_iter=10**5):
    """Fit a transition probability using the detailed balance-enforced
    maximum-liklihood estimation (Prinz) method.

    This method is not numpy vectorized or written in C and is very
    slow for large counts matrices.

    Parameters
    ----------
    C : array, shape=(n_states, n_states)
        The matrix to symmetrize
    tol: float, default=1e-10
        The log-likelihood change at which the iterative method is
        considered to have been converged
    max_iter: int, default=10**5
        The maximum number of allowed iterations. If this this number of
        iterations is reached, calculation will stop and a warning will
        be emitted.

    Returns
    -------
    T : array, shape=(n_states, n_states)
        Transition probabilities matrix derived from `C`.

    References
    ----------
    [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J Chem. Phys. 134.17 (2011): 174105.
    """
    C = C.copy().astype(float)
    X = C + C.T

    X_rs = X.sum(axis=1)
    C_rs = C.sum(axis=1)

    assert np.all(X_rs > 0)
    assert np.all(C_rs > 0)

    oldlogl = 0
    for n_iter in range(max_iter):
        logl = 0

        for i in range(len(C)):
            tmp = X[i,i];
            denom = C_rs[i] - C[i, i];
            if (denom > 0):
                X[i, i] = C[i, i] * (X_rs[i] - X[i, i]) / denom;

            X_rs[i] = X_rs[i] + (X[i, i] - tmp);

            if (X[i, i] > 0):
                logl += C[i,i] * np.log(X[i, i] / X_rs[i]);

        for i in range(len(C) - 1):
            for j in range(i+1, len(C)):

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
                    v = (-b + np.sqrt((b**2) - (4*a*c))) / (2*a)

#                 /* update the row sums */
                X_rs[i] = X_rs[i] + (v - X[i, j])
                X_rs[j] = X_rs[j] + (v - X[j, i])

#                 /* add in the new value */
                X[i, j] = v
                X[j, i] = v

                if (X[i, j] > 0):
                    logl += (
                        ((C[i,j] * np.log(X[i, j]) / X_rs[i])) +
                        ((C[j,i] * np.log(X[j, i]) / X_rs[j])))


        if abs(logl - oldlogl) > tol:
            oldlogl = logl
        else:
            break

    if n_iter == max_iter - 1:
        raise exception.DataInvalid(
            "Prinz MLE did not converge after %s iterations." % n_iter)

    T = X / X.sum(axis=-1).reshape(len(X), 1)
    pi = X_rs / X_rs.sum()[..., None]

    assert np.allclose(T.sum(axis=1), 1)
    assert np.sum(pi) - 1 < 1e-15, np.sum(pi) - 1

    return T, pi


def test_prinz_mle_pyx_py_agreement():
    for i in range(100):
        n_states = np.random.randint(2, 20)

        C = (np.random.poisson(lam=0.3, size=(n_states, n_states)) +
             np.diag(np.random.poisson(lam=10, size=(n_states,))))
        C = C.astype(float)
        C += float(1 / n_states)

        T_old, pi_old = prinz_mle_py(C)
        T_new, pi_new = _mle_prinz_dense(C)

        assert_allclose(T_old, T_new, atol=1e-5)
        assert_allclose(pi_old, pi_new, atol=1e-5)
