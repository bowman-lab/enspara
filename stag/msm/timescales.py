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
import scipy.sparse.linalg

from .transition_matrices import *


def implied_timescales(
        assigns, lag_times, n_imp_times=None,
        sliding_window=True, trim=False,
        symmetrization=None, n_procs=1):
    # n_imp_times=None -> 10% number of states
    n_states = assigns.max() + 1

    if n_imp_times is None:
        n_imp_times = int(np.floor(n_states/10.0))+1
    if n_imp_times > n_states-1:  # -1 accounts for eq pops
        n_imp_times = n_states-1

    n_lag_times = len(lag_times)
    all_imp_times = np.zeros((n_lag_times*n_imp_times, 2))
    for i in range(n_lag_times):
        lag_time = lag_times[i]
        C = assigns_to_counts(
            assigns,
            n_states=n_states,
            lag_time=lag_time,
            sliding_window=sliding_window)
        T = counts_to_probs(C, symmetrization=symmetrization)
        e_vals, e_vecs = eigenspectra(
            T, n_eigs=n_imp_times+1)  # +1 accounts for eq pops
        imp_times = -lag_time / np.log(e_vals[1:])
        all_imp_times[i*n_imp_times:(i+1)*n_imp_times, 0] = lag_time
        all_imp_times[i*n_imp_times:(i+1)*n_imp_times, 1] = imp_times

    return all_imp_times
