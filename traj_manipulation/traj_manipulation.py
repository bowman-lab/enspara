# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

from copy import deepcopy

import mdtraj as md
import numpy as np


def sloopy_concatenate_trjs(traj_lst, delete_trjs=False):
    top = deepcopy(traj_lst[0].top)
    if delete_trjs:
        xyz = deepcopy(traj_lst[0].xyz)
        n_traj = len(traj_lst)
        traj_lst[0] = None
        for i in range(1, n_traj):
            xyz = np.concatenate([xyz, traj_lst[i].xyz])
            traj_lst[i] = None
    else:
        xyz = np.concatenate([t.xyz for t in traj_lst])

    return md.Trajectory(xyz, top)
