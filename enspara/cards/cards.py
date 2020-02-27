"""High-level routines for Correlation of All Rotameric and Dynamical States.
"""

import logging

from ..info_theory import mutual_info

from . import disorder
from .featurizers import RotamerFeaturizer
from ..citation import cite

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@cite('cards')
def cards(trajectories, buffer_width=15, n_procs=1):
    """Compute ordered, disordered and ordered-disordered mutual
    information matrices for the correlation between rotameric states
    across a set of trajectories.

    Parameters
    ----------
    trajectories: iterable
        Trajectories to consider for the calculation. Generators are
        accepted and can be used to mitigate memory usage.
    buffer_width: int, default=15
        The width of the no-man's land between rotameric bins. Angles
        in this range are not used in the calculation.
    n_procs: int, default=1
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
    """

    logger.debug("Assigning to rotameric states")

    r = RotamerFeaturizer(buffer_width=buffer_width, n_procs=n_procs)
    r.fit(trajectories)

    return cards_matrices(r.feature_trajectories_,
                          r.n_feature_states_, n_procs) + (r.atom_indices_,)


@cite('cards')
def cards_matrices(feature_trajs, n_feature_states, n_procs=None):
    """Compute ordered, disordered and ordered-disordered mutual
    infrmation matrices for a set of trajectories of state assignments.

    Parameters
    ----------
    feature_trajs: iterable
        Trajectories of state labels. Generators are accepted and can be
        used to mitigate memory usage.
    n_feature_states: array, shape=(n_features,)
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
    structural_mi = mutual_info.mi_matrix(
        feature_trajs, feature_trajs,
        n_feature_states, n_feature_states)

    logger.debug("Calculating disorder mutual information")
    disorder_mi = mutual_info.mi_matrix(
        disordered_trajs, disordered_trajs,
        disorder_n_states, disorder_n_states)

    logger.debug("Calculating structure-disorder mutual information")
    struct_to_disorder_mi = mutual_info.mi_matrix(
        feature_trajs, disordered_trajs,
        n_feature_states, disorder_n_states)

    logger.debug("Calculating disorder-structure mutual information")
    disorder_to_struct_mi = mutual_info.mi_matrix(
        disordered_trajs, feature_trajs,
        disorder_n_states, n_feature_states)

    return structural_mi, disorder_mi, struct_to_disorder_mi, \
        disorder_to_struct_mi
