# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import
import logging

import numpy as np

from sklearn.externals.joblib import Parallel, delayed

from .transition_matrices import assigns_to_counts, eigenspectrum, \
    trim_disconnected

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_imp_times(assigns, lag_time, n_states, n_times, method,
                   sliding_window, trim):
    """Embarassingly parallel part of the implied timescales plotting
    system. This function function computes an individual eigenspectrum
    for a specific lag time.
    """

    C = assigns_to_counts(
        assigns,
        max_n_states=n_states,
        lag_time=lag_time,
        sliding_window=sliding_window)

    if trim:
        mapping, C = trim_disconnected(C)

    _, T, _ = method(C)

    e_vals, e_vecs = eigenspectrum(
        T, n_eigs=n_times+1)  # +1 accounts for eq pops
    imp_times = -lag_time / np.log(e_vals[1:])

    return imp_times


def implied_timescales(
        assigns, lag_times, method, n_times=None,
        sliding_window=True, trim=False, n_procs=None):
    """Calculate the implied timescales across a range of lag times.

    Parameters
    ----------
    assigns : array, shape=(traj_len, )
        A 2-D array where each row is a trajectory consisting of a
        sequence of state indices.
    lag_times : list
        The lag times (i.e. observation intervals) for counting
        transitions. An eigenspectrum is calculated for each lag time in
        the list.
    method : function(C) -> C, T, p
        The function used to construct a transition probability matrix
        from assignments. Given a transition counts matrix, returns a
        symmetrized transition counts matrix, a transition probability
        matrix, and an equilibrium probability distribution.
    n_times : int, optional
        the number of implied timescales to calculate for each
        lag_time. If not specified, 10% of the number of states is used.
    trim : bool, default=False
        ignore states without transitions both in and out.
    sliding_window : bool, default=True
        Whether to use a sliding window for counting transitions or to
        take every lag_time'th state.
    n_procs : int, default=1,
        Parallelize this computation across this number of processes.

    Returns
    -------
    implied_times_list :  array, shape=(len(lag_times), n_times)
        A 2d array containing the eigenspectrum of each chosen lag time
        as a row.
    """

    # n_times=None -> 10% number of states
    n_states = assigns.max() + 1

    if n_times is None:
        n_times = int(np.floor(n_states/10.0))+1
    if n_times > n_states-1:  # -1 accounts for eq pops
        n_times = n_states-1

    implied_times_list = Parallel(n_jobs=n_procs)(
        delayed(calc_imp_times)(assigns, t, n_states, n_times, method,
                                sliding_window, trim) for t in lag_times)

    return np.array(implied_times_list)
