# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy
import scipy.sparse


def synthetic_trajectory(T, start_state, n_steps):
    """Simulate a single trajectory using kinetic Monte Carlo.

    Parameters
    ----------
    T : array, shape=(n_states, n_states)
        A row-normalized transition probability matrix.
    start_state : int
        State to start the trajectory from.
    n_steps : int
        Number of steps in the trajectory. This includes the starting state,
        so n_steps=2 would result in a trajectory consisting of the starting
        state and one additional state.

    Returns
    -------
    traj : array, shape=(n_steps, )
        A 1-D array containing a sequence of state indices (integers).
    """
    traj = -1*np.ones(n_steps, dtype=int)
    traj[0] = start_state
    states = T.shape[0]
    rng = np.random.default_rng()
    if scipy.sparse.isspmatrix(T):
        for i in range(n_steps-1):
            p = T[traj[i], :].toarray()[0]
            traj[i + 1] = rng.choice(states, 1, p=p)
    else:
        for i in range(n_steps - 1):
            p = T[traj[i], :]
            traj[i+1] = rng.choice(states, 1, p=p)
    return traj


def synthetic_ensemble(T, init_pops, n_steps, observable_per_state=None):
    """Simulate the time evolution of an ensemble.

    The time that elapses for each step is the lag time of the input transition
    probability matrix.

    If observable_per_state is specified, this is a 1-D array containing the
    population-weighted average observable as a function of time. Otherwise,
    this is a 2-D array where each row contains the populations of each state
    as a function of time.

    Parameters
    ----------
    T : ndarray, shape=(n_states, n_states)
        A row-normalized transition probability matrix.
    init_pops : array, shape=(n_states, )
        The initial probabilities of every state.
    n_steps : int
        Number of steps to advance the ensemble. This includes the starting
        populations, so n_steps=2 would result in a trajectory consisting of
        the starting state and one step forward in time.
    observable_per_state : array, shape=(n_states, ), default=None
        An array of floats representing some observable for each state.

    Returns
    -------
    out : array, shape=(n_steps, ...)
        An array representing the time evolution of an ensemble. If
        observable_per_state is specified, this is a 1-D array containing the
        population-weighted average observable as a function of time.
        Otherwise, this is a 2-D array where each row contains the populations
        of each state as a function of time.
    """

    # turn transitions probability matrix into linear operator
    if scipy.sparse.issparse(T):
        T_op = scipy.sparse.linalg.aslinearoperator(T.tocsr())
    else:
        T_op = scipy.sparse.linalg.aslinearoperator(T)

    p = init_pops.copy()
    if observable_per_state is not None:
        observations = [p.dot(observable_per_state)]
        for i in range(n_steps-1):
            p = T_op.rmatvec(p)
            observations.append(p.dot(observable_per_state))
    else:
        observations = [p]
        for i in range(n_steps-1):
            p = T_op.rmatvec(p)
            observations.append(p)

    observations = np.array(observations)

    return p, observations
