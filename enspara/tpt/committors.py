# Author(s): Maxwell Zimmerman 

"""
Generation of committor probabilities. The forward committor
probability, q+, for a state is the probability that it reaches a
defined sink state(s) before it reaches a source state(s). The reverse
committor, q-, is the probability of reaching the source state first; at
equilibrium the forward and reverse committors are related by the
following relation: q+ = 1 - q-

References
----------
.. [1] Grinstead, C.M. and Snell, J.L., Introduction to Probability
       American Mathematical Society Providence (2006)
"""
from __future__ import print_function, division, absolute_import
import numpy as np

__all__ = ['committors']

def committors(tprob, sources, sinks):
    """
    Get the forward committors of the reaction sources -> sinks.
    This is calculated by turning all sources and sinks into absorbing
    states and calculating the probability or reaching one set of
    aborbing states over the other, as covered in the above reference.

    Parameters
    ----------
    tprob : array, shape [n_states, n_states]
        Transition probability matrix.
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.

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

    n_states = tprob.shape[0]

    # R is the list of probabilities of going from a state to an absorbing one
    R = tprob[:, sinks]
    R[sinks] = 1.0
    R[sources] = 0.0

    # (I-Q)
    I_m_Q = np.eye(n_states) - tprob
    I_m_Q[:, all_absorbing] = 0
    I_m_Q[all_absorbing, :] = 0
    I_m_Q[all_absorbing, all_absorbing] = 1.0
    
    # solves for committors: committors = N*R, where N = (I-Q)^-1
    committors = np.linalg.solve(I_m_Q, R).flatten()

    return committors
