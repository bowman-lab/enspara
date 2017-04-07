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

from .. import geometry
from .. import info_theory
from . import disorder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cards(trajectories, buffer_width=15, n_procs=1):
    """Compute ordered, disordered and ordered-disordered mutual
    infrmation matrices for a set of trajectories.

    Parameters
    ----------
    trajectories: list
        List of trajectories to consider for the calculation
    buffer_width: int (default=15)
        The width of the no-man's land between rotameric bins. Angles
        in this range are not used in the calculation.
    n_procs: int (default=1)
        Number of cores to use for the parallel parts of the algorithm.

    Returns
    -------
    structural_mi: ndarrray, shape=(n_dihedrals, n_dihedrals)
        Matrix of MIs where (i,j) is the structural to structural
        communication between dihedrals i and j.
    disorder_mi: ndarray, shape=(n_dihedrals, n_dihedrals)
        Matrix of MIs where (i,j) is the disordered to disordered
        communication between dihedrals i and j.
    struct_to_disorder_mi: ndarray, shape=(n_dihedrals, n_dihedrals)
        Matrix of MIs where (i,j) is the structured to disordered
        communication between dihedrals i and j.
    disorder_to_struct_mi: ndarray, shape=(n_dihedrals, n_dihedrals)
        Matrix of MIs where (i,j) is the structured to disordered
        communication between dihedrals i and j.
    atom_inds: ndarray, shape=(n_dihedrals, 4)
        The atom indicies defining each dihedral

    References
    ----------
    [1] Singh, S., & Bowman, G. R. (2017). Quantifying Allosteric
        Communication via Correlations in Structure and Disorder.
        Biophysical Journal, 112(3), 498a.
    """

    n_traj = len(trajectories)

    logger.debug("Assigning to rotameric states")
    rotamer_trajs = []
    for i in range(n_traj):
        rots, inds, n_states = geometry.all_rotamers(trajectories[i],
                                                     buffer_width=buffer_width)
        rotamer_trajs.append(rots)
    atom_inds = inds
    rotamer_n_states = n_states

    logger.debug("identifying transition times")
    transition_times = []
    n_dihedrals = rotamer_trajs[0].shape[1]
    ordered_times = np.zeros((n_traj, n_dihedrals))
    n_ordered_times = np.zeros((n_traj, n_dihedrals))
    disordered_times = np.zeros((n_traj, n_dihedrals))
    n_disordered_times = np.zeros((n_traj, n_dihedrals))
    for i in range(n_traj):
        transition_times.append([])
        for j in range(n_dihedrals):
            tt = disorder.transitions(rotamer_trajs[i][:, j])
            transition_times[i].append(tt)
            (ordered_times[i, j], n_ordered_times[i, j],
             disordered_times[i, j], n_disordered_times[i, j]) = disorder.traj_ord_disord_times(tt)

    # get average ordered and idsordered times
    logger.debug("Calculating ordered/disordered times")
    mean_ordered_times = np.zeros(n_dihedrals)
    mean_disordered_times = np.zeros(n_dihedrals)
    for j in range(n_dihedrals):
        mean_ordered_times[j] = (
            ordered_times[:, j].dot(n_ordered_times[:, j]) /
            n_ordered_times[:, j].sum())
        mean_disordered_times[j] = disordered_times[:, j].dot(
            n_disordered_times[:, j])/n_disordered_times[:, j].sum()

    logger.debug("Assigning to disordered states")
    disordered_trajs = []
    for i in range(n_traj):
        traj_len = rotamer_trajs[i].shape[0]
        dis_traj = np.zeros((traj_len, n_dihedrals))
        for j in range(n_dihedrals):
            dis_traj[:, j] = disorder.create_disorder_traj(
                transition_times[i][j], traj_len, mean_ordered_times[j],
                mean_disordered_times[j])

        disordered_trajs.append(dis_traj)
    disorder_n_states = 2*np.ones(n_dihedrals, dtype='int')

    logger.debug("Calculating structural mutual information")
    structural_mi = mi_wrapper(
        rotamer_trajs, rotamer_trajs, rotamer_n_states, rotamer_n_states,
        n_procs=n_procs)

    logger.debug("Calculating disorder mutual information")
    disorder_mi = mi_wrapper(
        disordered_trajs, disordered_trajs, disorder_n_states,
        disorder_n_states, n_procs=n_procs)

    logger.debug("Calculating structure-disorder mutual information")
    struct_to_disorder_mi = mi_wrapper(
        rotamer_trajs, disordered_trajs, rotamer_n_states, disorder_n_states,
        n_procs=n_procs)

    logger.debug("Calculating disorder-structure mutual information")
    disorder_to_struct_mi = mi_wrapper(
        disordered_trajs, rotamer_trajs, disorder_n_states, rotamer_n_states,
        n_procs=n_procs)

    return structural_mi, disorder_mi, struct_to_disorder_mi, \
        disorder_to_struct_mi, atom_inds


def _mi_helper(i, states_a, states_b, n_a_states, n_b_states):
    n_traj = len(states_a)
    n_dihedrals = states_a[0].shape[1]
    mi = np.zeros(n_dihedrals)
    if i == n_dihedrals:
        return mi
    for j in range(i+1, n_dihedrals):
        jc = info_theory.joint_counts(
            states_a[0][:, i], states_b[0][:, j],
            n_a_states[i], n_b_states[j])
        for k in range(1, n_traj):
            jc += info_theory.joint_counts(
                states_a[k][:, i], states_b[k][:, j],
                n_a_states[i], n_b_states[j])
        mi[j] = info_theory.mutual_information(jc)
        min_num_states = np.min([n_a_states[i], n_b_states[j]])
        mi[j] /= np.log(min_num_states)

    return mi


def mi_wrapper(states_a, states_b, n_a_states, n_b_states, n_procs=1):
    n_dihedrals = states_a[0].shape[1]
    mi = np.zeros((n_dihedrals, n_dihedrals))

    mi = Parallel(n_jobs=n_procs, max_nbytes='1M')(
        delayed(_mi_helper)(i, states_a, states_b, n_a_states, n_b_states)
        for i in range(n_dihedrals))

    mi = np.array(mi)
    mi += mi.T

    return mi


def mi_wrapper_serial(states_a, states_b, n_a_states, n_b_states):
    n_traj = len(states_a)
    n_dihedrals = states_a[0].shape[1]
    mi = np.zeros((n_dihedrals, n_dihedrals))

    for i in range(n_dihedrals):
        logger.debug(i, "/", n_dihedrals)
        for j in range(i+1, n_dihedrals):
            jc = info_theory.joint_counts(
                states_a[0][:, i], states_b[0][:, j],
                n_a_states[i], n_b_states[j])
            for k in range(1, n_traj):
                jc += info_theory.joint_counts(
                    states_a[k][:, i], states_b[k][:, j],
                    n_a_states[i], n_b_states[j])
            mi[i, j] = info_theory.mutual_information(jc)
            min_num_states = np.min([n_a_states[i], n_b_states[j]])
            mi[i, j] /= np.log(min_num_states)
            mi[j, i] = mi[i, j]

    return mi
