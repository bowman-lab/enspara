# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import numpy as np

from . import entropy


def joint_counts(state_traj_1, state_traj_2, n_states_1=None, n_states_2=None):
    if n_states_1 is None:
        n_states_1 = state_traj_1.max()+1
    if n_states_2 is None:
        n_states_2 = state_traj_2.max()+1

    joint_counts, x_edges, y_edges = np.histogram2d(state_traj_1, state_traj_2, bins=(n_states_1, n_states_2))

    return joint_counts


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
