# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

from . import entropy
from . import libinfo


def joint_counts(state_traj_1, state_traj_2,
                 n_states_1=None, n_states_2=None):
    """Compute the matrix H, where H[i, j] is the number of times t
    where trajectory_1[t] == i and trajectory[t] == j.

    Parameters
    ----------
    state_traj_1 : array-like, dtype=int
        List of assignments to discrete states for trajectory 1.
    state_traj_2 : array-like, dtype=int
        List of assignments to discrete states for trajectory 2.
    n_states_1 : int, optional
        Number of total possible states in state_traj_1. If unspecified,
        taken to be max(state_traj_1)+1.
    n_states_2 : int, optional
        Number of total possible states in state_traj_2 If unspecified,
        taken to be max(state_traj_2)+1.
    """

    if n_states_1 is None:
        n_states_1 = state_traj_1.max()+1
    if n_states_2 is None:
        n_states_2 = state_traj_2.max()+1

    H = libinfo.bincount2d(
        state_traj_1.astype('int'), state_traj_2.astype('int'),
        n_states_1, n_states_2)

    return H


def mutual_information(joint_counts):
    counts_axis_1 = joint_counts.sum(axis=1)
    counts_axis_2 = joint_counts.sum(axis=0)

    p1 = counts_axis_1/counts_axis_1.sum()
    p2 = counts_axis_2/counts_axis_2.sum()
    joint_p = joint_counts.flatten()/joint_counts.sum()

    h1 = entropy.shannon_entropy(p1)
    h2 = entropy.shannon_entropy(p2)
    joint_h = entropy.shannon_entropy(joint_p)

    return h1+h2-joint_h
