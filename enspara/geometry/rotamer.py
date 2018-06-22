# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import numbers
import mdtraj as md
import numpy as np


def dihedral_angles(traj, dihedral_type):
    valid_dihedral_types = ["phi", "psi", "chi1", "chi2", "chi3", "chi4"]
    if dihedral_type not in valid_dihedral_types:
        return None, None, None

    f = getattr(md, "compute_%s" % dihedral_type)
    atom_inds, angles = f(traj)

    # transform so angles range from 0 to 360 instead of radians or -180 to 180
    angles = np.rad2deg(angles)
    angles[np.where(angles < 0)] += 360

    n_angles = angles.shape[1]
    ref_atom_inds = np.zeros(n_angles)
    for i in range(n_angles):
        atom = traj.topology.atom(atom_inds[i, 2])
        ref_atom_inds[i] = int(atom.index)

    return angles, atom_inds


def _rotamers(angles, hard_boundaries, buffer_width):
    n_basins = len(hard_boundaries)-1

    if buffer_width <= 0 or buffer_width >= 360. / n_basins:
        return None
    if hard_boundaries[0] != 0 or hard_boundaries[-1] != 360:
        return None

    core_edges = [0]
    n_cutoffs = len(hard_boundaries)
    for i in range(n_cutoffs-1):
        core_edges.append(hard_boundaries[i]+buffer_width)
        core_edges.append(hard_boundaries[i+1]-buffer_width)
    core_edges.append(360)

    bins = np.digitize(angles, core_edges)

    n_frames = len(angles)
    rotamers = -1*np.ones(n_frames, dtype='int16')

    # assign things to cores
    for i in range(n_basins):
        inds = np.where(bins == 2*(i+1))[0]
        rotamers[inds] = i

    # make sure assign first frame
    for i in range(n_basins):
        if angles[0] < hard_boundaries[i+1]:
            rotamers[0] = i
            break

    # do one sweep to make sure verythign is assigned to a bin
    for i in range(1, n_frames):
        if rotamers[i] == -1:
            rotamers[i] = rotamers[i-1]

    return rotamers


def phi_rotamers(traj, buffer_width=15):
    hard_boundaries = [0, 180, 360]
    angles, atom_inds = dihedral_angles(traj, 'phi')

    n_frames, n_angles = angles.shape
    rotamers = np.zeros((n_frames, n_angles), dtype='int16')
    for i in range(n_angles):
        rotamers[:, i] = _rotamers(angles[:, i], hard_boundaries, buffer_width)

    n_states = 2*np.ones(n_angles, dtype='int16')

    return rotamers, atom_inds, n_states


def psi_rotamers(traj, buffer_width=15):
    angles, atom_inds = dihedral_angles(traj, 'psi')

    # shift by 100 so boundaries at 0 and 360
    shifted_angles = angles-100
    shifted_angles[np.where(shifted_angles < 0)] += 360
    hard_boundaries = [0, 160, 360]

    n_frames, n_angles = angles.shape
    rotamers = np.zeros((n_frames, n_angles), dtype='int16')
    for i in range(n_angles):
        rotamers[:, i] = _rotamers(shifted_angles[:, i], hard_boundaries,
                                   buffer_width)

    n_states = 2*np.ones(n_angles, dtype='int16')

    return rotamers, atom_inds, n_states


def chi_rotamers(traj, buffer_width=15):
    # could make a dictionary of boundaries for different residue/dihedral
    # types
    hard_boundaries = [0, 120, 240, 360]

    angles, atom_inds = dihedral_angles(traj, 'chi1')
    for i in range(2, 5):
        more_angles, more_atom_inds = dihedral_angles(traj, 'chi%d' % i)
        angles = np.append(angles, more_angles, axis=1)
        atom_inds = np.append(atom_inds, more_atom_inds, axis=0)

    n_frames, n_angles = angles.shape
    rotamers = np.zeros((n_frames, n_angles), dtype='int16')
    for i in range(n_angles):
        rotamers[:, i] = _rotamers(angles[:, i], hard_boundaries, buffer_width)

    n_states = 3*np.ones(n_angles, dtype='int16')

    return rotamers, atom_inds, n_states


def all_rotamers(traj, buffer_width=15):
    """Compute the rotameric states of a trajectory over time.

    Parameters
    ----------
    traj : md.Trajectory
        Trajectory from which to compute a rotamer trajectory.
    buffer_width: int, default=15
        Width of the "no-man's land" between rotameric bins in which no
        assignment is made.

    Returns
    -------
    all_rotamers : np.ndarray, shape=(n_frames, n_dihedrals)
        Assignment of each dihedral to a rotameric state as an int in
        the range 0-2.
    all_atom_inds : np.ndarray, shape=(n_dihedrals, 4)
        Array of the four atom indices that define each dihedral angle.
    all_n_states : np.ndarray, shape=(n_dihedrals,)
        Array indicating the maximum number of states a dihedral angle
        is expected to take (as a consequence of its topology); this
        value differs for backbone and sidechain dihedrals.

    References
    ----------
    [1] Singh, S., & Bowman, G. R. (2017). Quantifying Allosteric
        Communication via Correlations in Structure and Disorder.
        Biophysical Journal, 112(3), 498a.
    """
    phi_rotameric_states, phi_atom_inds, n_phi_states = phi_rotamers(
        traj, buffer_width=buffer_width)
    all_rotamers, all_atom_inds, all_n_states = phi_rotameric_states, \
        phi_atom_inds, n_phi_states

    psi_rotameric_states, psi_atom_inds, n_psi_states = psi_rotamers(
        traj, buffer_width=buffer_width)
    all_rotamers = np.append(all_rotamers, psi_rotameric_states, axis=1)
    all_atom_inds = np.append(all_atom_inds, psi_atom_inds, axis=0)
    all_n_states = np.append(all_n_states, n_psi_states, axis=0)

    chi_rotameric_states, chi_atom_inds, n_chi_states = chi_rotamers(
        traj, buffer_width=buffer_width)
    all_rotamers = np.append(all_rotamers, chi_rotameric_states, axis=1)
    all_atom_inds = np.append(all_atom_inds, chi_atom_inds, axis=0)
    all_n_states = np.append(all_n_states, n_chi_states, axis=0)

    assert issubclass(all_rotamers.dtype.type, np.integer)
    assert issubclass(all_n_states.dtype.type, np.integer)

    return all_rotamers, all_atom_inds, all_n_states
