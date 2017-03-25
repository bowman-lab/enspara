import os

from nose.tools import assert_equal, assert_raises, assert_is
from numpy.testing import assert_array_equal, assert_allclose

import numpy as np
import mdtraj as md

from .. import cards
from .. import geometry

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

TOP = md.load(os.path.join(TEST_DATA_DIR, "PROT_only.pdb")).top
TRJ = md.load(os.path.join(TEST_DATA_DIR, "trj0.xtc"), top=TOP)

TEST_TRJS = [TRJ, TRJ]


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
