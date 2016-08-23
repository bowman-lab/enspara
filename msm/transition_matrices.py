# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy
import scipy.sparse

def trajectory_to_count_matrix(traj, n_states=None, lag_time=1, sliding_window=True):
    """Count transitions between states in a single trajectory.
    
    Parameters
    ----------
    traj : (N, ) array
        A 1-D array containing a sequence of state indices.
    n_states : int, default=None
        The number of states. This is useful for controlling the dimensions 
        of the transition count matrix in cases where the input trajectory 
        does not necessarily visit every state.
    lag_time : int, default=1
        The lag time (i.e. observation interval) for counting transitions.
    sliding_window : bool, default=True
        Whether to use a sliding window for counting transitions or to take 
        every lag_time'th state.
        
    Returns
    -------
    C : (n_states, n_states) array
        A transition count matrix.
    """
    
    # TODO: check trajectory is 1d array

    if n_states is None:
        n_states = traj.max() + 1
        
    if sliding_window:
        start_states = traj[:-lag_time:1]
        end_states = traj[lag_time::1]
    else:
        start_states = traj[:-lag_time:lag_time]
        end_states = traj[lag_time::lag_time]
    transitions = np.row_stack((start_states, end_states))
    counts = np.ones(transitions.shape[1], dtype=int)
    C = scipy.sparse.coo_matrix((counts, transitions), shape=(n_states,n_states))

    return C.to_lil()
    
def trajectories_to_count_matrix(trajs, n_states=None, lag_time=1, sliding_window=True):
    # trajectories is list of arrays (with possibly different lengths)
    return
    
def _normalize_rows(C):
    """Normalize every row of a transition count matrix to obtain a transition
    probability matrix.
    
    Parameters
    ----------
    C : (n_states, n_states) array
        A transition count matrix.
        
    Returns
    -------
    T : (n_states, n_states) array
        A row-normalized transition probability matrix.
    """
    
    n_states = C.shape[0]
    
    if scipy.sparse.isspmatrix(C):
        C_csr = scipy.sparse.csr_matrix(C).asfptype()
        weights = np.asarray(C_csr.sum(axis=1)).flatten()
        inv_weights = np.zeros(n_states)
        inv_weights[weights > 0] = 1.0 / weights[weights > 0]
        inv_weights = scipy.sparse.dia_matrix((inv_weights, 0), C_csr.shape).tocsr()
        T = inv_weights.dot(C_csr)
    else:
        weights = np.asarray(C.sum(axis=1)).flatten()
        inv_weights = np.zeros(n_states)
        inv_weights[weights > 0] = 1.0 / weights[weights > 0]
        T = C * inv_weights.reshape((n_states, 1))
        
    return T
    
def count_matrix_to_probabilities(C, symmetrization=None):
    """Infer a transition probability matrix from a transition count matrix
    using the specified method to enforce microscopic reversibility.
    
    Parameters
    ----------
    C : (n_states, n_states) array
        A transition count matrix.
    symmetrization : {None, 'transpose', 'mle'}
        Method to use to enforce microscopic reversibility.
        
    Returns
    -------
    T : (n_states, n_states) array
        A row-normalized transition probability matrix.
    """
    
    if symmetrization is None:
        T = _normalize_rows(C)
    elif symmetrization is "transpose":
        C_sym = C + C.T
        T = _normalize_rows(C_sym)
    elif symmetrization is "mle":
        print("MLE option not yet implemented")
        return
    else:
        print("Invalid symmetrization option in count_matrix_to_probabilities")
        return
        
    return T
    
