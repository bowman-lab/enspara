import numpy as np
import scipy.sparse

from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_allclose

from enspara.msm import bace

# transitions counts matrix and expected results for the "simple" model in
# Bowman 2012 JCP paper describing BACE.

TCOUNTS = np.array(
    [[1000,  100,  100,   10,    0,    0,    0,    0,    0],
     [ 100, 1000,  100,    0,    0,    0,    0,    0,    0],
     [ 100,  100, 1000,    0,    1,    0,    0,    0,    0],
     [  10,    0,    0, 1000,  100,  100,   10,    0,    0],
     [   0,    0,    1,  100, 1000,  100,    0,    0,    0],
     [   0,    0,    0,  100,  100, 1000,    0,    1,    0],
     [   0,    0,    0,   10,    0,    0, 1000,  100,  100],
     [   0,    0,    0,    0,    0,    1,  100, 1000,  100],
     [   0,    0,    0,    0,    0,    0,  100,  100, 1000]])

EXP_BAYES_FACTORS = np.array(
    [[8.0, 8.54953122e+02],
     [7.0, 8.54953122e+02],
     [6.0, 8.55428120e+02],
     [5.0, 1.07233398e+03],
     [4.0, 1.07233398e+03],
     [3.0, 1.08250033e+03],
     [2.0, 4.85322085e+03],
     [1.0, 6.72422979e+03]])

EXP_LABELS = {
     2: [0, 0, 0, 0, 0, 0, 1, 1, 1],
     3: [0, 0, 0, 1, 1, 1, 2, 2, 2],
     4: [0, 0, 0, 1, 2, 2, 3, 3, 3],
     5: [0, 0, 0, 1, 2, 2, 3, 4, 4],
     6: [0, 1, 1, 2, 3, 3, 4, 5, 5],
     7: [0, 1, 1, 2, 3, 4, 5, 6, 6],
     8: [0, 1, 1, 2, 3, 4, 5, 6, 7]}


def test_bace_integration_dense():

    bayes_factors, labels = bace.bace(
        TCOUNTS, n_macrostates=2, n_procs=4)

    # bayes_factors = np.loadtxt(os.path.join(d, 'bayesFactors.dat'))
    assert_allclose(
        [bayes_factors[i] for i in sorted(bayes_factors.keys())],
        EXP_BAYES_FACTORS[::-1, 1],
        rtol=1e-6)

    assert_array_equal(
        np.vstack(labels.values()),
        np.vstack(EXP_LABELS.values()))
    assert_equal(labels.keys(), EXP_LABELS.keys())


def test_bace_integration_sparse():

    bayes_factors, labels = bace.bace(
        scipy.sparse.lil_matrix(TCOUNTS), n_macrostates=2, n_procs=4)

    # bayes_factors = np.loadtxt(os.path.join(d, 'bayesFactors.dat'))
    assert_allclose(
        [bayes_factors[i] for i in sorted(bayes_factors.keys())],
        EXP_BAYES_FACTORS[::-1, 1],
        rtol=1e-6)

    assert_array_equal(
        np.vstack(labels.values()),
        np.vstack(EXP_LABELS.values()))
    assert_equal(labels.keys(), EXP_LABELS.keys())


def test_baysean_prune():

    pruned_counts, labels, kept_states = bace.baysean_prune(TCOUNTS)

    assert_array_equal(pruned_counts, TCOUNTS)
    assert_array_equal(labels, np.arange(len(labels)))
    assert_array_equal(kept_states, np.arange(len(kept_states)))

    pruned_counts, labels, kept_states = bace.baysean_prune(
        scipy.sparse.lil_matrix(TCOUNTS))

    assert_array_equal(pruned_counts.todense(), TCOUNTS)
    assert_array_equal(labels, np.arange(len(labels)))
    assert_array_equal(kept_states, np.arange(len(kept_states)))


def test_baysean_prune_inplace():

    tcounts = np.array(
        [[100,  10,  1],
         [ 10, 100,  0],
         [  1,   0,  5]])
    bace.baysean_prune(tcounts, in_place=True)
    exp_pruned_counts = np.array(
        [[100,  10,  0],
         [ 10, 100,  0],
         [  0,   0,  0]])
    assert_array_equal(tcounts, exp_pruned_counts)

    tcounts = np.array(
        [[100,  10,  1],
         [ 10, 100,  0],
         [  1,   0,  5]])

    bace.baysean_prune(tcounts, factor=1.3, in_place=True)
    exp_pruned_counts = np.zeros((3, 3))
    exp_pruned_counts[1, 1] = 100
    assert_array_equal(tcounts, exp_pruned_counts)


def test_baysean_prune_undersampled():

    tcounts = np.array(
        [[100,  10,  1],
         [ 10, 100,  0],
         [  1,   0,  5]])
    pruned_counts, labels, kept_states = bace.baysean_prune(tcounts)

    exp_pruned_counts = np.array(
        [[100,  10,  0],
         [ 10, 100,  0],
         [  0,   0,  0]])

    assert_array_equal(pruned_counts, exp_pruned_counts)
    assert_array_equal(labels, [0, 1, 1])
    assert_array_equal(kept_states, [0, 1])

    pruned_counts, labels, kept_states = bace.baysean_prune(
        tcounts, factor=1.3)

    exp_pruned_counts = np.zeros((3, 3))
    exp_pruned_counts[1, 1] = 100

    assert_array_equal(pruned_counts, exp_pruned_counts)
    assert_array_equal(labels, [-1, 0, 0])
    assert_array_equal(kept_states, [1])
