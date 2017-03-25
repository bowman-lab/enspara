# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import numpy as np

from mdtraj.utils import ensure_type

__all__ = ['compute_unit_vectors']


def compute_unit_vectors(traj, atom_pairs, periodic=True, opt=True):
    """Compute the unit vectors between pairs of atoms in each frame.

    Parameters
    ----------
    traj : Trajectory
        An mtraj trajectory.
    atom_pairs : np.ndarray, shape=(num_pairs, 2), dtype=int
        Each row gives the indices of two atoms involved in the interaction.
    periodic : bool, default=True
        If `periodic` is True and the trajectory contains unitcell
        information, we will compute distances under the minimum image
        convention.
    opt : bool, default=True
        Use an optimized native library to calculate distances. Our optimized
        SSE minimum image convention calculation implementation is over 1000x
        faster than the naive numpy implementation.

    Returns
    -------
    bond_unit_vectors : np.ndarray, shape=(n_frames, num_pairs, 3), dtype=float
        The unit vectors, in each frame, pointing between each pair of atoms.
    """
    xyz = ensure_type(traj.xyz, dtype=np.float32, ndim=3, name='traj.xyz',
                      shape=(None, None, 3), warn_on_cast=False)
    pairs = ensure_type(atom_pairs, dtype=np.int32, ndim=2, name='atom_pairs',
                        shape=(None, 2), warn_on_cast=False)
    if not np.all(np.logical_and(pairs < traj.n_atoms, pairs >= 0)):
        raise ValueError('atom_pairs must be between 0 and %d' % traj.n_atoms)
    if len(pairs) == 0:
        return np.zeros((len(xyz), 0), dtype=np.float32)

    if periodic is True and traj._have_unitcell:
        box = ensure_type(traj.unitcell_vectors, dtype=np.float32, ndim=3,
                          name='unitcell_vectors', shape=(len(xyz), 3, 3))
        if opt:
            return _unit_vector_mic_fast(xyz, pairs, box)
        else:
            return _unit_vector_mic(xyz, pairs, box)
    else:
        return _unit_vector(xyz, pairs)


def _unit_vector(xyz, pairs):
    "Unit vector between pairs of points in each frame"
    delta = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    r = (delta ** 2.).sum(-1) ** 0.5
    return np.einsum("ijk,ij->ijk", delta, 1./r)


def _unit_vector_mic(xyz, pairs, box_vectors):
    """Unit vector between pairs of points in each frame under the minimum image
    convention for periodic boundary conditions.

    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.

    This is a slow pure python implementation, mostly for testing. It is
    about 100x slower than the version without the minimum image convention.
    """
    out = np.empty((xyz.shape[0], pairs.shape[0], 3), dtype=np.float32)
    for i in range(len(xyz)):
        hinv = np.linalg.inv(box_vectors[i])

        for j, (a, b) in enumerate(pairs):
            s1 = np.dot(hinv, xyz[i, a, :])
            s2 = np.dot(hinv, xyz[i, b, :])
            s12 = s2 - s1

            s12 = s12 - np.round(s12)
            r12 = np.dot(box_vectors[i], s12)
            d = np.sqrt(np.sum(r12 * r12))
            out[i, j] = r12/d
    return out


def _unit_vector_mic_fast(xyz, pairs, box_vectors):
    """Unit vector between pairs of points in each frame under the minimum image
    convention for periodic boundary conditions.

    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.

    This is about 30x faster than the reference implementation above, so only
    about 3x slower than the version without the minimum image convention.
    """
    hinv = np.linalg.inv(box_vectors)
    s1 = np.einsum("i...,ij...->ij...", hinv, xyz[:, pairs[:, 0], :]).sum(-1)
    s2 = np.einsum("i...,ij...->ij...", hinv, xyz[:, pairs[:, 1], :]).sum(-1)
    s12 = s2 - s1
    s12 = s12 - np.round(s12)
    delta = np.einsum("i...,ij...->ij...", box_vectors, s12).sum(-1)
    r = (delta ** 2.).sum(-1) ** 0.5
    return np.einsum("ijk,ij->ijk", delta, 1./r)
