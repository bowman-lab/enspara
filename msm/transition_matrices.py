# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy
import scipy.sparse

def trajectory_to_count_matrix(traj, n_states=None, lag_time=1, sliding_window=True):
    # check trajectory is 1d array

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
    if symmetrization is None:
        T = _normalize_rows(C)
    elif symmetrization is "transpose":
        C_sym = C + C.T
        T = _normalize_rows(C_sym)
    elif symmetrization is "MLE":
        print("MLE option not yet implemented")
        return
    else:
        print("Invalid symmetrization option in count_matrix_to_probabilities")
        return
        
    return T
    

    
