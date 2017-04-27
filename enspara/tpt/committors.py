# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""
Functions for computing forward committors for an MSM. The forward
committor is defined for a set of sources and sink states, and for
each state, the forward committor is the probability that a walker
starting at that state will visit the sink state before the source
state.

These are some canonical references for TPT. Note that TPT
is really a specialization of ideas very familiar to the
mathematical study of Markov chains, and there are many
books, manuscripts in the mathematical literature that
cover the same concepts.

References
----------
.. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
       transition paths. J. Stat. Phys. 123, 503-523 (2006).
.. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
       Transition path theory for Markov jump processes.
       Multiscale Model. Simul. 7, 1192-1219 (2009).
.. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
       flux and folding pathways in network models of
       coarse-grained protein dynamics. J. Chem. Phys.
       130, 205102 (2009).
.. [4] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
       pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
       19011-19016.
"""
from __future__ import print_function, division, absolute_import
import numpy as np

from mdtraj.utils.six.moves import xrange

__all__ = ['committors', 'conditional_committors']

def committors(tprob, sources, sinks):
    """
    Get the forward committors of the reaction sources -> sinks.

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
    forward_committors : np.ndarray
        The forward committors for the reaction sources -> sinks

    References
    ----------
    .. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
           transition paths. J. Stat. Phys. 123, 503-523 (2006).
    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
           Transition path theory for Markov jump processes.
           Multiscale Model. Simul. 7, 1192-1219 (2009).
    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
           flux and folding pathways in network models of
           coarse-grained protein dynamics. J. Chem. Phys.
           130, 205102 (2009).
    .. [4] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
           pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
           19011-19016.
    """

    sources = np.array(sources, dtype=int).reshape((-1, 1))
    sinks = np.array(sinks, dtype=int).reshape((-1, 1))

    n_states = tprob.shape[0]

    # construct the committor problem
    lhs = np.eye(n_states) - tprob

    for a in sources:
        lhs[a, :] = 0.0  # np.zeros(n)
        lhs[:, a] = 0.0
        lhs[a, a] = 1.0

    for b in sinks:
        lhs[b, :] = 0.0  # np.zeros(n)
        lhs[:, b] = 0.0
        lhs[b, b] = 1.0

    ident_sinks = np.zeros(n_states)
    ident_sinks[sinks] = 1.0

    rhs = np.dot(tprob, ident_sinks)
    rhs[sources] = 0.0
    rhs[sinks] = 1.0

    forward_committors = np.linalg.solve(lhs, rhs)

    return forward_committors
