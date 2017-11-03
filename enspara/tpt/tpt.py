#author(s): Maxwell Zimmerman

"""
Functions for calculating metrics from transition path theory (TPT).
Included are, given a set of sources and sinks: 1) the flux through
states and 2) the density of each state.

References
----------
.. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path
       theory for Markov jump processes. Multiscale Model. Simul. 7,
       1192-1219 (2009).
"""
from __future__ import print_function, division, absolute_import

import numpy as np
from scipy import sparse

from . import committors
from ..msm.transition_matrices import eq_probs


__all__ = ['reactive_fluxes', 'net_fluxes', 'reactive_populations']


def _get_data_from_tprob(tprob, sources, sinks, populations):
    """A helper function for parsing data and returning relevant
       parameters for TPT analysis
    """

    sources = np.array(sources).reshape((-1,))
    sinks = np.array(sinks).reshape((-1,))
    # check to see if populations exist
    if populations is None:
        populations = eq_probs(tprob)

    n_states = len(populations)

    # check if committors exist
    forward_committors = committors(tprob, sources, sinks)

    # reverse committors if process is at equilibrium
    reverse_committors = 1 - forward_committors

    return populations, n_states, forward_committors, reverse_committors


def reactive_fluxes(tprob, sources, sinks, populations=None):
    """Computes the total flux along any edge in an MSM from a set of
    sources to sinks.

    Parameters
    ----------
    tprob : array, shape [n_states, n_states]
        Transition probability matrix.
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    populations : array, shape [n_states, ], optional, default: None
        Equilibrium populations of each state. If not provided, will
        recalculate from tprob.

    Returns
    -------
    fluxes : np.ndarray
        The flux through each edge in a MSM from a set of sources
        to sinks

    See Also
    --------
    """

    # parse data and obtain relevant parameters

    populations, n_states, forward_committors, reverse_committors = \
        _get_data_from_tprob(tprob, sources, sinks, populations)

    # fij = pi_i * q-_i * Tij * q+_j
    if sparse.issparse(tprob):
        fluxes = tprob.multiply((populations * reverse_committors)[:, None])\
                      .multiply(forward_committors)
        fluxes = fluxes.tolil()
    else:
        fluxes = \
            tprob * ((populations * reverse_committors)[:, None]) \
            * forward_committors

    fluxes[(np.arange(n_states), np.arange(n_states))] = np.zeros(n_states)

    return fluxes


def net_fluxes(tprob, sources, sinks, populations=None):
    """Computes the net fluxes along a given edge from a set of sources
    to sinks.

    Parameters
    ----------
    tprob : array, shape [n_states, n_states]
        Transition probability matrix.
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    populations : array, shape [n_states, ], optional, default: None
        Equilibrium populations of each state. If not provided, will
        recalculate from tprob.

    Returns
    -------
    net_fluxes : np.ndarray
        The flux through each edge in a MSM from a set of sources
        to sinks

    See Also
    --------
    """
    # calculate the probability flux through each edge
    fluxes = reactive_fluxes(tprob, sources, sinks, populations=populations)

    # get the net flux along each edge
    net_fluxes = fluxes - fluxes.T
    net_fluxes[np.where(net_fluxes < 0)] = 0
    return net_fluxes


def reactive_populations(tprob, sources, sinks, populations=None):
    """Compute the probability that a state is observed on a
    reactive trajectory.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : array, shape [n_states, n_states]
        Transition probability matrix.
    populations : array, shape [n_states, ], optional, default: None
        Equilibrium populations of each state. If not provided, will
        recalculate from tprob.

    Returns
    -------
    population : np.ndarray
        The probability a state is observed from A to B.

    See Also
    --------
    """
    # parse data and obtain relevant parameters
    populations, n_states, forward_committors, reverse_committors = \
        _get_data_from_tprob(tprob, sources, sinks, populations)

    # mR_i = pi_i * q+_i * q-_i
    densities = populations * forward_committors * reverse_committors
    populations = densities / np.sum(densities)

    return populations
