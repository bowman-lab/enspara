# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:31:21 2016

@author: gbowman
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy
import scipy.sparse

def synthetic_trajectory(T, start_state, n_steps):
    traj = -1*np.ones(n_steps, dtype=int)
    traj[0] = start_state
    
    for i in range(n_steps-1):
        current_state = traj[i]
        if scipy.sparse.isspmatrix(T):
            p = T[current_state,:].toarray()
        else:
            p = T[current_state,:]
        new_state = np.where(scipy.random.multinomial(1, p) == 1)[0][0]
        traj[i+1] = new_state

    return traj

def synthetic_ensemble(T, init_pops, n_steps, observable_per_state=None):
    # return populations of all states if observable is None

    # turn transitions probability matrix into linear operator
    if scipy.sparse.issparse(T):
        T_op = scipy.sparse.linalg.aslinearoperator(T.tocsr())
    else:
        T_op = scipy.sparse.linalg.aslinearoperator(T)
        
    p = init_pops.copy()
    observations = []
    for i in range(n_steps):
        p = T_op.rmatvec(p)
        if observable_per_state is not None:
            observations.append(p.dot(observable_per_state))
        else:
            observations.append(p)
            
    observations = np.array(observations)
    
    return p, observations
    
    