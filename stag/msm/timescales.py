# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import
import logging

import numpy as np

from .transition_matrices import counts_to_probs, assigns_to_counts, \
    eigenspectra, transpose

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def implied_timescales(
        assigns, lag_times, n_imp_times=None,
        sliding_window=True, trim=False,
        symmetrization=transpose, n_procs=None):
    """Calculate the implied timescales across a range of lag times.

    Parameters
    ----------
    assigns : array, shape=(traj_len, )
        A 2-D array where each row is a trajectory consisting of a sequence
        of state indices.
    lag_times : int
        The lag times (i.e. observation interval) for counting
        transitions. An eigenspectra is calculated for each lag time in
        the range [1, lag_times].
    n_imp_times : int, optional
        the number of implied timescales to calculate for each
        lag_time. If not specified, 10% of the number of states is used.
    trim : bool, default=False
        ignore states without transitions both in and out.
    symmetrization : { None, "transpose", "mle" }, default=None
        symmetrize the transitions matrix using this method, or don't,
        if None is provided.
    sliding_window : bool, default=True
        Whether to use a sliding window for counting transitions or to
        take every lag_time'th state.
    n_procs : int, default=1,
        Parallelize this computation across this number of processes.

    Returns
    -------
    all_imp_times :  array, shape=(n_states, n_states)
        A transition count matrix.
    """

    if n_procs is not None:
        logger.warning(
            "implied_timescales n_procs is currently unimplemented")

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
