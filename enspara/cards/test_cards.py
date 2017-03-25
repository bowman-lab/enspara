import os

from nose.tools import assert_equal, assert_raises, assert_is
from numpy.testing import assert_array_equal, assert_allclose

import numpy as np
import mdtraj as md
import pickle

from .. import cards
from .. import geometry
from . import disorder

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

TOP = md.load(os.path.join(TEST_DATA_DIR, "PROT_only.pdb")).top
TRJ = md.load(os.path.join(TEST_DATA_DIR, "trj0.xtc"), top=TOP)

TRJS = [TRJ, TRJ]
BUFFER_WIDTH = 15


def test_cards():

    ss_mi, dis_mi, s_d_mi, d_s_mi, inds = cards.cards(
        [TRJ, TRJ], buffer_width=15., n_procs=1, verbose_dir="verbose_output")


# This test is really much more complicated than it should be for a unit test.
# Ideally, it would be broken down into tests that check each of the components
# of disorder.*.
def test_transitions():

    rotamer_trajs = [geometry.all_rotamers(t, buffer_width=BUFFER_WIDTH)[0]
                     for t in TRJS]

    transition_times = []
    n_dihedrals = rotamer_trajs[0].shape[1]
    ordered_times = np.zeros((len(TRJS), n_dihedrals))
    n_ordered_times = np.zeros((len(TRJS), n_dihedrals))
    disordered_times = np.zeros((len(TRJS), n_dihedrals))
    n_disordered_times = np.zeros((len(TRJS), n_dihedrals))
    for i in range(len(TRJS)):
        transition_times.append([])
        for j in range(n_dihedrals):
            tt = disorder.traj_transition_times(rotamer_trajs[i][:, j])
            transition_times[i].append(tt)
            (ordered_times[i, j], n_ordered_times[i, j],
             disordered_times[i, j], n_disordered_times[i, j]) = disorder.traj_ord_disord_times(tt)

    expected_ordered_times = np.loadtxt(os.path.join(
        TEST_DATA_DIR, "ordered_times.dat"))
    expected_disordered_times = np.loadtxt(os.path.join(
        TEST_DATA_DIR, "disordered_times.dat"))
    expected_n_ordered_times = np.loadtxt(os.path.join(
        TEST_DATA_DIR, "n_ordered_times.dat"))
    expected_n_disordered_times = np.loadtxt(os.path.join(
        TEST_DATA_DIR, "n_disordered_times.dat"))
    with open(os.path.join(TEST_DATA_DIR, "transition_times.dat"), 'rb') as f:
        expected_transition_times = pickle.load(f)

    assert_array_equal(expected_ordered_times, ordered_times)
    assert_array_equal(expected_disordered_times, disordered_times)
    assert_array_equal(expected_n_ordered_times, n_ordered_times)
    assert_array_equal(expected_n_disordered_times, n_disordered_times)
    for a, b in zip(expected_transition_times, transition_times):
        for x, y in zip(a, b):
            assert_array_equal(x, y)
