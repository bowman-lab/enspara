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

from . import committors
from ..msm.transition_matrices import eq_probs


__all__ = ['fluxes', 'net_fluxes']


def _get_data_from_tprob(tprob, sources, sinks, populations, forward_committors):
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
    if forward_committors is None:
        forward_committors = committors(tprob, sources, sinks)
    else:
        forward_committors = np.array(for_committors)
        if forward_committors.shape != (n_states,):
            raise ValueError(
                "Shape of committors %s should be %s" % \
                (str(forward_committors.shape), str((n_states,))))

    # reverse committors if process is at equilibrium
    reverse_committors = 1 - forward_committors

    return populations, n_states, forward_committors, reverse_committors


def fluxes(
        tprob, sources, sinks, populations=None,
        forward_committors=None):
    """
    Compute the transition path theory flux matrix.

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
    forward_committors : np.ndarray, optional
        The forward committors associated with `sources`, `sinks`, and
        `tprob`. If not provided, is calculated from scratch. If
        provided, `sources` and `sinks` are ignored.

    Returns
    -------
    state_fluxes : np.ndarray
        The flux through each edge in a MSM from a set of sources
        to sinks

    See Also
    --------
    """

    # parse data and obtain relevant parameters
    populations, n_states, forward_committors, reverse_committors = \
        _get_data_from_tprob(
            tprob, sources, sinks, populations, forward_committors)

    # fij = pi_i * q-_i * Tij * q+_j
    state_fluxes = \
        tprob * ((populations*reverse_committors)[:,None]) * forward_committors

    state_fluxes[(np.arange(n_states), np.arange(n_states))] = np.zeros(n_states)

    return state_fluxes


def densities(
        tprob, sources, sinks, populations=None,
        forward_committors=None):
    """
    Compute the transition path theory densities.

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
    forward_committors : np.ndarray, optional
        The forward committors associated with `sources`, `sinks`, and
        `tprob`. If not provided, is calculated from scratch. If
        provided, `sources` and `sinks` are ignored.

    Returns
    -------
    state_densities : np.ndarray
        The density of each node in an MSM from a set of sources to sinks.

    See Also
    --------
    """
    # parse data and obtain relevant parameters
    populations, n_states, forward_committors, reverse_committors = \
        _get_data_from_tprob(
            tprob, sources, sinks, populations, forward_committors)

    # mR_i = pi_i * q+_i * q-_i
    state_densities = populations * forward_committors * reverse_committors

    return state_densities
