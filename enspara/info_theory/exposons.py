import logging

import numpy as np
import mdtraj as md
from sklearn.cluster import AffinityPropagation

from .mutual_info import weighted_mi
from enspara.citation import cite

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@cite('exposons')
def exposons(trj, damping, weights=None, probe_radius=0.28, threshold=0.02):
    """Compute exposons for an MDTraj trajectory.

    This function is a convenience wrapper to compute exposons using other
    functions already existing in MDTraj, sklearn, and elsewhere in enspara.

    Parameters
    ----------
    trj: md.Trajectory
        The trajectory to compute exposons for. May represent a trajectory
        or, in combination with `weights`, the centers for an MSM.
    damping: float
        Damping parameter to use for affinity propagation. Goes from 0.5
        to <1.0. Empirically, values between 0.85 and 0.95 tend to work best.
    weights: ndarray, shape=(len(trj),), default=None
        Weight of each frame in the simulation for the mutual information
        calculation. Useful if `trj` represents cluster centers of an MSM
        rather than a full trajectory. If None, frames will be weighted
        equally.
    probe_radius: float, default=0.28
        Size of the solvent probe in nm for solvent accessibility
        calculations. The exposons paper used 0.28, or the radius of two
        water molecules.
    threshold: float, default=0.02
        Sidechains with greater than this amount of total SASA will count
        as exposed for the purposes of the exposed/buried dichotomy used
        in mutual information calculations.

    Returns
    -------
    sasa_mi: np.ndarray, shape=(n_res, n_res)
        Mutual information of each sidchain with each other sidechain
        computed for the purposes of clustering exposons.
    exposons: np.ndarray, shape=(n_res,)
        Assignment of residues to exposons. Residues in the same exposon
        share the same number in this array.

    Notes
    -----
    Because SASA calculations are rather expensive, this function does not
    scale well to large datasets. For large datasets, best practices are to
    split up SASA calcuations across many computers to leverage the
    independence of trajectories' SASAs.

    See Also
    --------
    enspara.info_theory.exposons.exposons_from_sasas:
        This function takes sidechain sasas and computes exposons from it,
        in the case that you don't want to recalculate SASAs every time.
    """

    if weights is None:
        weights = np.full((len(trj),), 1 / len(trj))
    else:
        weights = np.array(weights) / sum(weights)

    sasas = md.shrake_rupley(trj, probe_radius=probe_radius, mode='atom')
    sasas = condense_sidechain_sasas(sasas, trj.top)
    sasa_mi = weighted_mi(sasas > threshold, weights)

    c = AffinityPropagation(damping=damping)
    c.fit(sasa_mi)

    return exposons_from_sasas(sasas, damping, weights, threshold)


@cite('exposons')
def exposons_from_sasas(sasas, damping, weights, threshold):
    """Compute exposons for an MDTraj trajectory.

    This function is a convenience wrapper to compute exposons using other
    functions already existing in MDTraj, sklearn, and elsewhere in enspara.

    Parameters
    ----------
    sasas: np.ndarray, shape=(n_conformations, n_sidechains)
        SASAs to use in the calculations.
    damping: float
        Damping parameter to use for affinity propagation. Goes from 0.5
        to <1.0. Empirically, values between 0.85 and 0.95 tend to work best.
    weights: ndarray, shape=(len(trj),), default=None
        Weight of each frame in the simulation for the mutual information
        calculation. Useful if `trj` represents cluster centers of an MSM
        rather than a full trajectory. If None, frames will be weighted
        equally.
    threshold: float, default=0.02
        Sidechains with greater than this amount of total SASA will count
        as exposed for the purposes of the exposed/buried dichotomy used
        in mutual information calculations.

    Returns
    -------
    sasa_mi: np.ndarray, shape=(n_res, n_res)
        Mutual information of each sidchain with each other sidechain
        computed for the purposes of clustering exposons.
    exposons: np.ndarray, shape=(n_res,)
        Assignment of residues to exposons. Residues in the same exposon
        share the same number in this array.
    """

    sasa_mi = weighted_mi(sasas > threshold, weights)

    c = AffinityPropagation(
        damping=damping,
        affinity='precomputed',
        preference=0,
        max_iter=10000)
    c.fit(sasa_mi)

    return sasa_mi, c.labels_


@cite('exposons')
def condense_sidechain_sasas(sasas, top):
    """Condense atomic SASAs into sidechain SASAs.

    Parameters
    ----------
    sasas: np.ndarray, shape=(n_confs, n_atoms)
        Array of atomic solvent accessibilities.
    top: md.Topology
        Topology object giving names for each atom. These names are used
        to infer which atoms belong to which residues' sidechains.

    Returns
    -------
    rsd_sasas: np.ndarray, shape=(n_confs, n_residues)
        Array of sidechain SASAs.
    """

    assert top.n_residues > 1
    assert top.n_atoms == sasas.shape[1]

    SELECTION = ('not (name N or name C or name CA or name O or '
                 'name HA or name H or name H1 or name H2 or name '
                 'H3 or name OXT)')

    sc_ids = [top.select('resid %s and ( %s )' % (i, SELECTION))
              for i in range(top.n_residues)]

    rsd_sasas = np.zeros((sasas.shape[0], len(sc_ids)), dtype='float32')

    for i, aa in enumerate(sc_ids):
        if len(aa) == 0:
            logger.warn('Found 0 atoms for %s.' % top.residue(i))
            assert False
        else:
            rsd_sasas[:, i] = np.sum(sasas[:, aa], axis=1)

    return rsd_sasas
