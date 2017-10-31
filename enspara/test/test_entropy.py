import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from nose.tools import assert_raises

from ..msm import builders
from ..exception import DataInvalid

from ..info_theory.entropy import (
    Q_from_assignments, relative_entropy_per_state,
    relative_entropy_msm, kl_divergence)


def test_Q_from_assignments():
    # test assignments
    assignments = np.array(
        [
            [0,1,1,0,1,0,2,2,0,1,1,1],
            [0,2,2,1,2,0,2,1,0,1,2,1]])

    # test Q's
    raw_Q = np.array(
        [
            [0., 0.57142857, 0.42857143],
            [0.375, 0.375, 0.25],
            [0.28571429, 0.42857143, 0.28571429]])

    Q_with_prior = np.array(
        [
            [0.00636943, 0.56687898, 0.42675159],
            [0.37430168, 0.37430168, 0.25139665],
            [0.2866242, 0.42675159, 0.2866242]])

    Q_transpose_w_prior = np.array(
        [
            [0.00740741, 0.57777778, 0.41481481],
            [0.3880597, 0.33333333, 0.27860697],
            [0.3566879, 0.3566879, 0.2866242]])

    # test Q from assignments
    assert_array_almost_equal(
        Q_from_assignments(assignments, prior_counts=0), raw_Q, 7)

    assert_array_almost_equal(
        Q_from_assignments(assignments), Q_with_prior, 7)

    assert_array_almost_equal(
        Q_from_assignments(assignments, builder=builders.transpose),
        Q_transpose_w_prior, 7)


def test_relative_entropy_per_state():
    # reference distributions
    P_test = np.array(
        [
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.5],
            [0, 0.25, 0.75]])

    # test assignments
    assignments = np.array(
        [
            [0,1,1,0,1,0,2,2,0,1,1,1],
            [0,2,2,1,2,0,2,1,0,1,2,1]])

    # test Q's
    raw_Q = np.array(
        [
            [0., 0.57142857, 0.42857143],
            [0.375, 0.375, 0.25],
            [0.28571429, 0.42857143, 0.28571429]])

    Q_with_prior = np.array(
        [
            [0.00636943, 0.56687898, 0.42675159],
            [0.37430168, 0.37430168, 0.25139665],
            [0.2866242, 0.42675159, 0.2866242]])

    Q_transpose_w_prior = np.array(
        [
            [0.00740741, 0.57777778, 0.41481481],
            [0.3880597, 0.33333333, 0.27860697],
            [0.3566879, 0.3566879, 0.2866242]])

    # correct relative entropies
    rel_ent_without_prior = np.array([np.inf, 0.20751875, 0.84983615])
    rel_ent_with_prior = np.array([3.05675367, 0.20484462, 0.84793052])
    rel_ent_transpose_with_prior = np.array(
        [2.9341145, 0.15950137, 0.91261408])

    # test rel entropies from assignments
    assert_array_almost_equal(
        relative_entropy_per_state(
            P_test, assignments=assignments, prior_counts=0),
        rel_ent_without_prior, 6)
    assert_array_almost_equal(
        relative_entropy_per_state(P_test, assignments=assignments),
        rel_ent_with_prior, 6)
    assert_array_almost_equal(
        relative_entropy_per_state(
            P_test, assignments=assignments, builder=builders.transpose),
        rel_ent_transpose_with_prior, 6)

    # test rel entropies from Qs
    assert_array_almost_equal(
        relative_entropy_per_state(P_test, Q=raw_Q), rel_ent_without_prior, 6)
    assert_array_almost_equal(
        relative_entropy_per_state(P_test, Q=Q_with_prior),
        rel_ent_with_prior, 6)
    assert_array_almost_equal(
        relative_entropy_per_state(P_test, Q=Q_transpose_w_prior),
        rel_ent_transpose_with_prior, 6)


def test_relative_entropy_msm():
    # reference distributions
    P_test = np.array(
        [
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.5],
            [0, 0.25, 0.75]])

    # test assignments
    assignments = np.array(
        [
            [0,1,1,0,1,0,2,2,0,1,1,1],
            [0,2,2,1,2,0,2,1,0,1,2,1]])

    # test Q's
    raw_Q = np.array(
        [
            [0., 0.57142857, 0.42857143],
            [0.375, 0.375, 0.25],
            [0.28571429, 0.42857143, 0.28571429]])

    Q_with_prior = np.array(
        [
            [0.00636943, 0.56687898, 0.42675159],
            [0.37430168, 0.37430168, 0.25139665],
            [0.2866242, 0.42675159, 0.2866242]])

    Q_transpose_w_prior = np.array(
        [
            [0.00740741, 0.57777778, 0.41481481],
            [0.3880597, 0.33333333, 0.27860697],
            [0.3566879, 0.3566879, 0.2866242]])

    # correct relative entropies
    rel_ent_without_prior = np.inf
    rel_ent_with_prior = 0.979737855
    rel_ent_transpose_with_prior = 0.98622475852

    # calculate relative entropies from assignments
    assert_almost_equal(
        relative_entropy_msm(P_test, assignments=assignments, prior_counts=0),
        rel_ent_without_prior, 7)
    assert_almost_equal(
        relative_entropy_msm(P_test, assignments=assignments),
        rel_ent_with_prior, 7)
    assert_almost_equal(
        relative_entropy_msm(
            P_test, assignments=assignments, builder=builders.transpose),
        rel_ent_transpose_with_prior, 7)

    # calculate relative entropies from Q's
    assert_almost_equal(
        relative_entropy_msm(P_test, Q=raw_Q), rel_ent_without_prior, 7)
    assert_almost_equal(
        relative_entropy_msm(P_test, Q=Q_with_prior), rel_ent_with_prior, 7)
    assert_almost_equal(
        relative_entropy_msm(P_test, Q=Q_transpose_w_prior),
        rel_ent_transpose_with_prior, 7)


def test_kl_divergence():
    # reference distributions
    P_test = np.array(
        [
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.5],
            [0, 0.25, 0.75]])

    # divergant distributions
    Q_test = np.array(
        [
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
            [0.1, 0.65, 0.25]])

    # True divergences
    true_divergences_base_2 = np.array([1., 0.0, 0.84409397])
    true_divergences_base_e = np.array([0.6931472, 0.0, 0.58508136])
    true_divergences_base_10 = np.array([0.3010299957, 0.0, 0.25409760])

    # Test full matrix distributions with different bases
    test_divergences_base_2 = kl_divergence(P_test, Q_test)
    test_divergences_base_e = kl_divergence(P_test, Q_test, base=np.e)
    test_divergences_base_10 = kl_divergence(P_test, Q_test, base=10)

    assert_array_almost_equal(
        true_divergences_base_2, test_divergences_base_2, 7)
    assert_array_almost_equal(
        true_divergences_base_e, test_divergences_base_e, 7)
    assert_array_almost_equal(
        true_divergences_base_10, test_divergences_base_10, 7)

    # Test individual distributions
    test_divergences_base_2_r0 = kl_divergence(P_test[0], Q_test[0])
    test_divergences_base_2_r1 = kl_divergence(P_test[1], Q_test[1])
    test_divergences_base_2_r2 = kl_divergence(P_test[2], Q_test[2])

    test_divergences_base_e_r0 = kl_divergence(P_test[0], Q_test[0], base=np.e)
    test_divergences_base_e_r1 = kl_divergence(P_test[1], Q_test[1], base=np.e)
    test_divergences_base_e_r2 = kl_divergence(P_test[2], Q_test[2], base=np.e)

    test_divergences_base_10_r0 = kl_divergence(P_test[0], Q_test[0], base=10)
    test_divergences_base_10_r1 = kl_divergence(P_test[1], Q_test[1], base=10)
    test_divergences_base_10_r2 = kl_divergence(P_test[2], Q_test[2], base=10)

    assert_almost_equal(
        true_divergences_base_2[0], test_divergences_base_2_r0, 7)
    assert_almost_equal(
        true_divergences_base_2[1], test_divergences_base_2_r1, 7)
    assert_almost_equal(
        true_divergences_base_2[2], test_divergences_base_2_r2, 7)

    assert_almost_equal(
        true_divergences_base_e[0], test_divergences_base_e_r0, 7)
    assert_almost_equal(
        true_divergences_base_e[1], test_divergences_base_e_r1, 7)
    assert_almost_equal(
        true_divergences_base_e[2], test_divergences_base_e_r2, 7)

    assert_almost_equal(
        true_divergences_base_10[0], test_divergences_base_10_r0, 7)
    assert_almost_equal(
        true_divergences_base_10[1], test_divergences_base_10_r1, 7)
    assert_almost_equal(
        true_divergences_base_10[2], test_divergences_base_10_r2, 7)


def test_kl_divergence_negative_probs():

    # reference distributions
    P_test = np.array([
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.5],
        [0, 0.25, 0.75]])

    # divergent distributions
    Q_test = np.array([
        [0.25, 0.25, 0.5],
        [0.25, 0.25, 0.5],
        [0.1, 0.65, 0.25]])

    with assert_raises(DataInvalid):
        P_neg = np.copy(P_test)
        P_neg[0, 1] *= -1
        kl_divergence(P_neg, Q_test)

    with assert_raises(DataInvalid):
        Q_neg = np.copy(Q_test)
        Q_neg[0, 1] *= -1
        kl_divergence(P_test, Q_neg)
