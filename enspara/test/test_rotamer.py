import os

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import numpy as np
import mdtraj as md

from .. import geometry

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'geometry_data')

TOP = md.load(os.path.join(TEST_DATA_DIR, "PROT_only.pdb")).top
TRJ = md.load(os.path.join(TEST_DATA_DIR, "trj0.xtc"), top=TOP)

TEST_TRJS = [TRJ, TRJ]


def test_rotamer_dtype():

    rots, inds, n = geometry.rotamer.phi_rotamers(TRJ)
    assert issubclass(rots.dtype.type, np.integer)
    assert issubclass(n.dtype.type, np.integer)

    rots, inds, n = geometry.rotamer.psi_rotamers(TRJ)
    assert issubclass(rots.dtype.type, np.integer)
    assert issubclass(n.dtype.type, np.integer)

    rots, inds, n = geometry.rotamer.chi_rotamers(TRJ)
    assert issubclass(rots.dtype.type, np.integer)
    assert issubclass(n.dtype.type, np.integer)

    rots, inds, n = geometry.all_rotamers(TRJ)
    assert issubclass(rots.dtype.type, np.integer)
    assert issubclass(n.dtype.type, np.integer)


def test_rotamer_assignment():
    TEST_BUFFER_WIDTH = 15

    rotamer_trajs = []
    for i in range(len(TEST_TRJS)):
        rots, inds, n_states = geometry.all_rotamers(
            TEST_TRJS[i], buffer_width=TEST_BUFFER_WIDTH)
        rotamer_trajs.append(rots)

    # verify that the n_states produced are correct
    expected_rotamer_n_states = np.array([2.]*18 + [3.]*21)
    rotamer_n_states = n_states
    assert_array_equal(expected_rotamer_n_states, rotamer_n_states)

    for i, rotamer_traj in enumerate(rotamer_trajs):
        expected = np.loadtxt(os.path.join(
            TEST_DATA_DIR, "rotamer_trj%d.dat" % i))

        assert_array_equal(rotamer_traj, expected)


def test_rotamer_assignment_split():

    pivot = len(TRJ) // 2

    unsplit = [TRJ]
    split = [TRJ[0:pivot], TRJ[pivot:]]

    rots1, inds1, n1 = zip(*[geometry.all_rotamers(trj) for trj in unsplit])
    rots2, inds2, n2 = zip(*[geometry.all_rotamers(trj) for trj in split])

    assert_array_equal(rots1[0][0:pivot], rots2[0])
    assert_array_equal(rots1[0][pivot:], rots2[1])

    assert_array_equal(inds1[0], inds2[0])
    assert_array_equal(inds1[0], inds2[1])

    assert_array_equal(n1[0], n2[0])
    assert_array_equal(n1[0], n2[1])
