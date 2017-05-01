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

from .. import exception
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
    trajectories: iterable
        Trajectories to consider for the calculation. Generators are
        accepted and can be used to mitigate memory usage.
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

    logger.debug("Assigning to rotameric states")

    # pull off the first trajectory so we can call all_rotamers to get
    # atom_inds and rotamer_n_states
    first_trj = next(trajectories)
    rotamer_trj, atom_inds, rotamer_n_states = geometry.all_rotamers(
        first_trj[0], buffer_width=buffer_width)

    # build the list of all of the rotamerized trajectories, starting
    # with the one we just calculated above.
    rotamer_trajs = [rotamer_trj]
    rotamer_trajs.extend(
        [geometry.all_rotamers(t, buffer_width=buffer_width)[0]
         for t in trajectories])

    disordered_trajs, disorder_n_states = assign_order_disorder(rotamer_trajs)

    logger.debug("Calculating structural mutual information")
    structural_mi = mi_matrix(
        rotamer_trajs, rotamer_trajs,
        rotamer_n_states, rotamer_n_states, n_procs=n_procs)

    logger.debug("Calculating disorder mutual information")
    disorder_mi = mi_matrix(
        disordered_trajs, disordered_trajs,
        disorder_n_states, disorder_n_states, n_procs=n_procs)

    logger.debug("Calculating structure-disorder mutual information")
    struct_to_disorder_mi = mi_matrix(
        rotamer_trajs, disordered_trajs,
        rotamer_n_states, disorder_n_states, n_procs=n_procs)

    logger.debug("Calculating disorder-structure mutual information")
    disorder_to_struct_mi = mi_matrix(
        disordered_trajs, rotamer_trajs,
        disorder_n_states, rotamer_n_states, n_procs=n_procs)

    return structural_mi, disorder_mi, struct_to_disorder_mi, \
        disorder_to_struct_mi, atom_inds


def assign_order_disorder(rotamer_trajs):

    logger.debug("Calculating ordered/disordered times")
    n_features = rotamer_trajs[0].shape[1]
    transition_times, mean_ordered_times, mean_disordered_times = \
        transition_stats(rotamer_trajs)

    logger.debug("Assigning to disordered states")
    disordered_trajs = []
    for i in range(len(rotamer_trajs)):
        traj_len = rotamer_trajs[i].shape[0]
        dis_traj = np.zeros((traj_len, n_features))
        for j in range(n_features):
            dis_traj[:, j] = disorder.create_disorder_traj(
                transition_times[i][j], traj_len, mean_ordered_times[j],
                mean_disordered_times[j])

        disordered_trajs.append(dis_traj)
    disorder_n_states = 2*np.ones(n_features, dtype='int')

    return disordered_trajs, disorder_n_states


def transition_stats(rotamer_trajs):

    n_traj = len(rotamer_trajs)

    transition_times = []
    n_features = rotamer_trajs[0].shape[1]
    ordered_times = np.zeros((n_traj, n_features))
    n_ordered_times = np.zeros((n_traj, n_features))
    disordered_times = np.zeros((n_traj, n_features))
    n_disordered_times = np.zeros((n_traj, n_features))
    for i in range(n_traj):
        transition_times.append([])
        for j in range(n_features):
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
    times : array, shape=(n_trajectories, n_features)
        Array of mean transition times for each trajectory and dihedral.
    n_times : array, shape=(n_trajectories, n_features)
        Array of numbers of transitions observed for each trajectory and
        dihedral.
    weight : array, shape=(n_trajectories,)
        Array of weights for each trajectory. Usually used to weight
        trajectories by their length. Any nonnegative weights can be used.

    Returns
    -------
    mean_times : np.ndarray, shape=(n_features,)
        Mean transition time across trajectories for each dihedral.
    """

    n_features = times.shape[1]
    mean_times = np.zeros(n_features)

    # we normalize by the maximum weight, such that the longest trajectory's
    # mean time is unchanged by the calculation.
    nl_weight = weight / np.sum(weight)

    # we suppress divide by zero errors here, since if we never see a
    # transition, the result of the divide by zero (a NaN) is an
    # acceptable representation of that time.
    with np.errstate(all='ignore'):
        for i in range(n_features):
            mean_times[i] = ((times[:, i] * nl_weight).sum())

    return mean_times


def mi_row(row, states_a_list, states_b_list, n_a_states, n_b_states):
    """Compute the a single row of the mutual information matrix between
    two state trajectories.

    Parameters
    ----------
    i : int
        Feature of `states_a_list` to use as a target. MI will be computed to
        each feature in `states_b_list`.
    states_a_list : array, shape=(n_trajectories, n_features)
        Array of assigned/binned features
    states_b_list : array, shape=(n_trajectories, n_features)
        Array of assigned/binned features
    n_a_states : array, shape(n_features_a,)
        The number of possible states for each feature in `states_a_list`
    n_b_states : array, shape=(n_features_b,)
        The number of possible states for each feature in `states_b_list`
    n_procs : int, default=1
        The number of cores to parallelize this computation across

    Returns
    -------
    mi : np.ndarray, shape=(n_features,)
        An array of the mutual information between trajectories a and b
        for each feature.
    """

    n_traj = len(states_b_list)

    check_features_states(states_a_list, n_a_states)
    check_features_states(states_b_list, n_b_states)

    n_features = states_a_list[0].shape[1]
    mi = np.zeros(n_features)
    if row == n_features:
        return mi
    for j in range(row+1, n_features):
        jc = info_theory.joint_counts(
            states_a_list[0][:, row], states_b_list[0][:, j],
            n_a_states[row], n_b_states[j])
        for k in range(1, n_traj):
            jc += info_theory.joint_counts(
                states_a_list[k][:, row], states_b_list[k][:, j],
                n_a_states[row], n_b_states[j])
        mi[j] = info_theory.mutual_information(jc)
        min_num_states = np.min([n_a_states[row], n_b_states[j]])
        mi[j] /= np.log(min_num_states)

    return mi


def mi_matrix(states_a_list, states_b_list,
              n_a_states_list, n_b_states_list, n_procs=1):
    """Compute the all-to-all matrix of mutual information across
    trajectories of assigned states.

    Parameters
    ----------
    states_a_list : array, shape=(n_trajectories, n_features)
        Array of assigned/binned features
    states_b_list : array, shape=(n_trajectories, n_features)
        Array of assigned/binned features
    n_a_states_list : array, shape(n_features_a,)
        Number of possible states for each feature in `states_a`
    n_b_states_list : array, shape=(n_features_b,)
        Number of possible states for each feature in `states_b`
    n_procs : int, default=1
        Number of cores to parallelize this computation across

    Returns
    -------
    mi : np.ndarray, shape=(n_features, n_features)
        Array of the mutual information between trajectories a and b
        for each feature.
    """

    n_features = states_a_list[0].shape[1]
    mi = np.zeros((n_features, n_features))

    check_features_states(states_a_list, n_a_states_list)
    check_features_states(states_b_list, n_b_states_list)

    mi = Parallel(n_jobs=n_procs, max_nbytes='1M')(
        delayed(mi_row)(i, states_a_list, states_b_list,
                           n_a_states_list, n_b_states_list)
        for i in range(n_features))

    mi = np.array(mi)
    mi += mi.T

    return mi


def mi_matrix_serial(states_a_list, states_b_list, n_a_states, n_b_states):
    n_traj = len(states_a_list)
    n_features = states_a_list[0].shape[1]
    mi = np.zeros((n_features, n_features))

    for i in range(n_features):
        logger.debug(i, "/", n_features)
        for j in range(i+1, n_features):
            jc = info_theory.joint_counts(
                states_a_list[0][:, i], states_b_list[0][:, j],
                n_a_states[i], n_b_states[j])
            for k in range(1, n_traj):
                jc += info_theory.joint_counts(
                    states_a_list[k][:, i], states_b_list[k][:, j],
                    n_a_states[i], n_b_states[j])
            mi[i, j] = info_theory.mutual_information(jc)
            min_num_states = np.min([n_a_states[i], n_b_states[j]])
            mi[i, j] /= np.log(min_num_states)
            mi[j, i] = mi[i, j]

    return mi


def check_features_states(states, n_states):
    n_features = len(n_states)

    if len(states[0][0]) != n_features:
        raise exception.DataInvalid(
            ("The number-of-states vector's length ({s}) didn't match the "
             "width of state assignments array with shape {a}.")
            .format(s=len(n_states), a=len(states[0][0])))

    if not all(len(t[0]) == len(states[0][0]) for t in states):
        raise exception.DataInvalid(
            ("The number of features differs between trajectories. "
             "Numbers of features were: {l}.").
            format(l=[len(t[0]) for t in states]))
