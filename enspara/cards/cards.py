# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors: Justin R. Porter <justinrporter@gmail.com>
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import logging
import multiprocessing as mp

from functools import partial

import numpy as np

from .. import exception
from .. import info_theory

from . import disorder
from .featurizers import RotamerFeaturizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cards(trajectories, buffer_width=15, n_procs=1):
    """Compute ordered, disordered and ordered-disordered mutual
    information matrices for the correlation between rotameric states
    across a set of trajectories.

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

    r = RotamerFeaturizer(buffer_width=buffer_width, n_procs=n_procs)
    r.fit(trajectories)

    return cards_matrices(r.feature_trajectories_,
                          r.n_feature_states_, n_procs) + (r.atom_indices_,)


def cards_matrices(feature_trajs, n_feature_states, n_procs=None):
    """Compute ordered, disordered and ordered-disordered mutual
    infrmation matrices for a set of trajectories of state assignments.

    Parameters
    ----------
    feature_trajs: iterable
        Trajectories of state labels. Generators are accepted and can be
        used to mitigate memory usage.
    n_feature_states: array, shape(n_features,)
        The total number of possible states for each feature.
    n_procs: int
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
    """

    disordered_trajs, disorder_n_states = disorder.assign_order_disorder(
        feature_trajs)

    logger.debug("Calculating structural mutual information")
    structural_mi = mi_matrix(
        feature_trajs, feature_trajs,
        n_feature_states, n_feature_states, n_procs=n_procs)

    logger.debug("Calculating disorder mutual information")
    disorder_mi = mi_matrix(
        disordered_trajs, disordered_trajs,
        disorder_n_states, disorder_n_states, n_procs=n_procs)

    logger.debug("Calculating structure-disorder mutual information")
    struct_to_disorder_mi = mi_matrix(
        feature_trajs, disordered_trajs,
        n_feature_states, disorder_n_states, n_procs=n_procs)

    logger.debug("Calculating disorder-structure mutual information")
    disorder_to_struct_mi = mi_matrix(
        disordered_trajs, feature_trajs,
        disorder_n_states, n_feature_states, n_procs=n_procs)

    return structural_mi, disorder_mi, struct_to_disorder_mi, \
        disorder_to_struct_mi


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
        jc = 0
        for k in range(0, n_traj):
            jc += info_theory.joint_counts(
                states_a_list[k][:, row], states_b_list[k][:, j],
                n_a_states[row], n_b_states[j])
        mi[j] = info_theory.mutual_information(jc)
        min_num_states = np.min([n_a_states[row], n_b_states[j]])
        mi[j] /= np.log(min_num_states)

    return mi


def mi_matrix(states_a_list, states_b_list,
              n_a_states_list, n_b_states_list, n_procs=None):
    """Compute the all-to-all matrix of mutual information across
    trajectories of assigned states.

    Parameters
    ----------
    states_a_list : array, shape=(n_trajectories, n_frames, n_features)
        Array of assigned/binned features
    states_b_list : array, shape=(n_trajectories, n_frames, n_features)
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

    with mp.Pool(processes=n_procs) as p:
        compute_mi_row = partial(
            mi_row,
            states_a_list=states_a_list, states_b_list=states_b_list,
            n_a_states=n_a_states_list, n_b_states=n_b_states_list)

        mi = p.map(compute_mi_row, (i for i in range(n_features)))

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
