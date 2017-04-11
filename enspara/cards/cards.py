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
    rotamer_trajs = [geometry.all_rotamers(t, buffer_width=buffer_width)[0]
                     for t in trajectories]
    _, atom_inds, rotamer_n_states = geometry.all_rotamers(
        trajectories[0], buffer_width=buffer_width)

    logger.debug("Calculating ordered/disordered times")
    n_dihedrals = rotamer_trajs[0].shape[1]
    transition_times, mean_ordered_times, mean_disordered_times = \
        transition_stats(rotamer_trajs)

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


def transition_stats(rotamer_trajs):

    n_traj = len(rotamer_trajs)

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

    trj_lengths = np.array([len(a) for a in rotamer_trajs])
    mean_ordered_times = aggregate_mean_times(
        ordered_times, n_ordered_times, trj_lengths)
    mean_disordered_times = aggregate_mean_times(
        disordered_times, n_disordered_times, trj_lengths)

    return transition_times, mean_ordered_times, mean_disordered_times


def aggregate_mean_times(times, n_times, weight):
    """Compute the mean transition time between a set of trajectories'
    mean transition times.

    Parameters
    ----------
    times : array, shape=(n_trajectories, n_dihedrals)
        Array of mean transition times for each trajectory and dihedral.
    n_times : array, shape=(n_trajectories, n_dihedrals)
        Array of numbers of transitions observed for each trajectory and
        dihedral.
    weight : array, shape=(n_trajectories,)
        Array of weights for each trajectory. Usually used to weight
        trajectories by their length. Any nonnegative weights can be used.

    Returns
    -------
    mean_times : np.ndarray, shape=(n_dihedrals,)
        Mean transition time across trajectories for each dihedral.
    """

    n_dihedrals = times.shape[1]
    mean_times = np.zeros(n_dihedrals)

    # we normalize by the maximum weight, such that the longest trajectory's
    # mean time is unchanged by the calculation.
    nl_weight = weight / np.sum(weight)

    # we suppress divide by zero errors here, since if we never see a
    # transition, the result of the divide by zero (a NaN) is an
    # acceptable representation of that time.
    with np.errstate(all='ignore'):
        for i in range(n_dihedrals):
            mean_times[i] = ((times[:, i] * nl_weight).sum())

    return mean_times


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
