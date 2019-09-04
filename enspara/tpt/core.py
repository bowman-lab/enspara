# Author(s): Maxwell Zimmerman

"""
Core theorems for understanding transitions in MSMs. Calculation of
committor probabilities and mean first passage times (mfpts). These
are classic ideas in MSMs and are covered in the following reference.

References
----------
.. [1] Grinstead, C.M. and Snell, J.L., Introduction to Probability
       American Mathematical Society Providence (2006)
"""
from __future__ import print_function, division, absolute_import

import warnings

import numpy as np
import scipy.sparse

from ..msm.transition_matrices import eq_probs

__all__ = ['committors', 'mfpts']


def _I_m_Q(tprob, absorbing_states, n_states=None):
    """Calculates (I-Q) as is defined in ref [1]_. This is fundamental
    for calculating committors and mfpts.
    """
    # if no states are supplied, determine from tprob
    if n_states is None:
        n_states = len(tprob)
    # Calculate (I-Q)
    I_m_Q = np.eye(n_states) - tprob
    I_m_Q[:, absorbing_states] = 0.0
    I_m_Q[absorbing_states, :] = 0.0
    I_m_Q[absorbing_states, absorbing_states] = 1.0
    return I_m_Q


def committors(tprob, sources, sinks):
    """Get the forward committors of the reaction sources -> sinks.

    The forward committor probability, q+, for a state is the
    probability that it reaches a defined sink state(s) before it
    reaches a source state(s). The reverse committor, q-, is the
    probability of reaching the source state first; at equilibrium the
    forward and reverse committors are related by the following
    equation: :math:`q^+ = 1 - q^-`

    The forward committors are calculated by turning all sources and
    sinks into absorbing states and calculating the probability of
    reaching one set of aborbing states over the other, as covered in
    the above reference.

    Parameters
    ----------
    tprob : array-like, shape=(n_states, n_states)
        Transition probability matrix.
    sources : array-like, int
        The set of source (reactant) states.
    sinks : array-like, int
        The set of sink (product) states.

    Returns
    -------
    committors : np.ndarray
        The forward committors for the reaction sources -> sinks
    """

    # set the data structure for sources, sinks, and every state that we will
    # make absorbing
    sources = np.array(sources, dtype=int).reshape((-1, 1)).flatten()
    sinks = np.array(sinks, dtype=int).reshape((-1, 1)).flatten()
    all_absorbing = np.append(sources, sinks)

    if scipy.sparse.issparse(tprob):
        # required indexing operations are fast with LIL matrices
        tprob = tprob.tolil()

    n_states = tprob.shape[0]

    # R is the list of probabilities of going from a state to an absorbing one
    R = tprob[:, sinks]
    R[sinks] = 1.0
    R[sources] = 0.0

    # (I-Q)
    I_m_Q = _I_m_Q(tprob, all_absorbing, n_states=n_states)

    # solves for committors: committors = N*R, where N = (I-Q)^-1
    with warnings.catch_warnings():
        # ignore 'SparseEfficiencyWarning: spsolve requires A be CSC or CSR
        # matrix format'; TODO: is this because I_m_Q is dense?
        warnings.simplefilter("ignore")
        # solve for probability of landing in any of the sink states
        B = scipy.sparse.linalg.spsolve(I_m_Q, R)
        # reshape and sum over probabilities
        committors = B.reshape(n_states, sinks.shape[0]).sum(axis=1)
        # ensure sink states have correct sinkage
        committors[sinks] = 1.0

    return committors


def mfpts(tprob, sinks=None, populations=None, lagtime=1.):
    """Calclate the mean first passage times for all states in an MSM.
    Either all to all or to a set of sinks.

    Parameters
    ----------
    tprob : array, shape (n_states, n_states)
        Transition probability matrix.
    sinks : array_like, int (n_sinks, )
        The set of folded/product states.
    popualtions : array, shape (n_states, ), default = None
        List of equilibrium populations.
    lagtime : float, default = 1.0
        The lagtime to scale values by. If not specified (1.0), units
        are in lagtimes.

    Returns
    -------
    mfpts : np.ndarray
        The mean first passage times from all to all, or all to a set
        of sinks.
    """
    n_states = len(tprob)
    if populations is None:
        populations = eq_probs(tprob)

    # if there are no sink states, calculates the mfpts from all to all
    # usin the fundamental matrix, Z
    if sinks is None:
        # Fundamental matrix, Z, is calculated as (I - T - W)^-1 where I
        # is the identity matrix, T is the probabiiy matix, and each row
        # in W is the equilibrium populations
        W = np.array([populations] * n_states)
        Z = np.linalg.inv(np.eye(n_states) - tprob + W)
        mfpts = lagtime * (np.diag(Z) - Z) / W
    # if there are a set of sink states, calcuate average time t
    # absorption with the relationship: t = N*c, where N = (I-Q)^-1
    # and c is a row of 1's
    else:

        # reshape sinks
        sinks = np.array(sinks, dtype=int).reshape((-1, 1)).flatten()

        # calculates (I-Q) to solve t
        I_m_Q = _I_m_Q(tprob, sinks, n_states=n_states)

        # solve for t and multiply by lagtime
        c = np.ones(n_states)
        c[sinks] = 0
        mfpts = lagtime * np.linalg.solve(I_m_Q, c)
    return mfpts
