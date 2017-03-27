'''Test the cards module through a couple of integration-like tests with
large blocks of expected data in the `test_data` subdirectory.
'''

import os

from nose.tools import assert_equal
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

ROTAMER_TRJS = [geometry.all_rotamers(t, buffer_width=BUFFER_WIDTH)[0]
                for t in TRJS]
N_DIHEDRALS = ROTAMER_TRJS[0].shape[1]


# This is really an integration test, for the entire cards package.
def test_cards():

    ss_mi, dis_mi, s_d_mi, d_s_mi, inds = cards.cards(
        [TRJ, TRJ], buffer_width=15., n_procs=1)

    with open(os.path.join(TEST_DATA_DIR, 'cards_ss_mi.dat'), 'r') as f:
        assert_allclose(ss_mi, np.loadtxt(f))
    with open(os.path.join(TEST_DATA_DIR, 'cards_dis_mi.dat'), 'r') as f:
        assert_array_equal(dis_mi, np.loadtxt(f))
    with open(os.path.join(TEST_DATA_DIR, 'cards_s_d_mi.dat'), 'r') as f:
        assert_array_equal(s_d_mi, np.loadtxt(f))
    with open(os.path.join(TEST_DATA_DIR, 'cards_d_s_mi.dat'), 'r') as f:
        assert_array_equal(d_s_mi, np.loadtxt(f))
    with open(os.path.join(TEST_DATA_DIR, 'cards_inds.dat'), 'r') as f:
        assert_array_equal(inds, np.loadtxt(f))


# This test is really much more complicated than it should be for a unit test.
# Ideally, it would be broken down into tests that check each of the components
# of disorder.*.
def test_transitions():

    transition_times = []
    ordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    n_ordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    disordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    n_disordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    for i in range(len(TRJS)):
        transition_times.append([])
        for j in range(N_DIHEDRALS):
            tt = disorder.transitions(ROTAMER_TRJS[i][:, j])
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


def test_disorder_trajectories():

    transition_times = []
    ordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    n_ordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    disordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    n_disordered_times = np.zeros((len(TRJS), N_DIHEDRALS))
    for i in range(len(TRJS)):
        transition_times.append([])
        for j in range(N_DIHEDRALS):
            tt = disorder.transitions(ROTAMER_TRJS[i][:, j])
            transition_times[i].append(tt)
            (ordered_times[i, j], n_ordered_times[i, j],
             disordered_times[i, j], n_disordered_times[i, j]) = disorder.traj_ord_disord_times(tt)

    mean_ordered_times = np.zeros(N_DIHEDRALS)
    mean_disordered_times = np.zeros(N_DIHEDRALS)
    for j in range(N_DIHEDRALS):
        mean_ordered_times[j] = (
            ordered_times[:, j].dot(n_ordered_times[:, j]) /
            n_ordered_times[:, j].sum())
        mean_disordered_times[j] = disordered_times[:, j].dot(
            n_disordered_times[:, j])/n_disordered_times[:, j].sum()

    disordered_trajs = []
    for i in range(len(TRJS)):
        traj_len = ROTAMER_TRJS[i].shape[0]
        dis_traj = np.zeros((traj_len, N_DIHEDRALS))
        for j in range(N_DIHEDRALS):
            dis_traj[:, j] = disorder.create_disorder_traj(
                transition_times[i][j], traj_len, mean_ordered_times[j],
                mean_disordered_times[j])

        disordered_trajs.append(dis_traj)

    disorder_n_states = 2*np.ones(N_DIHEDRALS)

    expected_dis_n_states = np.loadtxt(os.path.join(
        TEST_DATA_DIR, "dis_n_states.dat"))
    assert_array_equal(expected_dis_n_states, disorder_n_states)

    for i in range(len(TRJS)):
        expected_dis_trj = np.loadtxt(os.path.join(
            TEST_DATA_DIR, "dis_trj%d.dat" % i))
        assert_array_equal(expected_dis_trj, disordered_trajs[i])
