import warnings

import numpy as np
import scipy.sparse

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..tpt import committors, reactive_fluxes, mfpts


ARR_TYPES = [
    np.array, scipy.sparse.lil_matrix, scipy.sparse.csr_matrix,
    scipy.sparse.coo_matrix, scipy.sparse.coo_matrix, scipy.sparse.dia_matrix,
    scipy.sparse.dok_matrix
]


def test_committors_small():
    Tij = np.array(
        [[0.5, 0.4, 0.1],
         [0.25, 0.5, 0.25],
         [0.1, 0.5, 0.4]])

    for arr_type in ARR_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tij = arr_type(Tij)

        true_committors = np.array([0, 0.5, 1.])

        for_committors = committors(Tij, 0, 2)
        assert_array_equal(for_committors, true_committors)

        for_committors = committors(Tij, [0], [2])
        assert_array_equal(for_committors, true_committors)


def test_committors_big():

    Tij = np.array(
        [
            [0.5, 0.4, 0.1, 0.],
            [0.25, 0.5, 0.2, 0.05],
            [0.1, 0.15, 0.5, 0.25],
            [0., 0.1, 0.4, 0.5]])

    for arr_type in ARR_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tij = arr_type(Tij)

        true_committors = np.array([0, 0.34091, 0.60227, 1.])

        for_committors = np.around(committors(Tij, 0, 3), 5)
        assert_array_equal(for_committors, true_committors)

        for_committors = committors(Tij, [0, 2], [3])
        assert_array_equal(for_committors, np.array([0, 0.1, 0, 1.0]))


def test_fluxes():
    Tij_ndarray = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5]])

    # run this test for every type of array we support
    for arr_type in ARR_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tij = arr_type(Tij_ndarray)

        pops = np.zeros(3) + (1/3.)

        # true fluxes
        true_fluxes = np.zeros((3, 3))
        true_fluxes[0, 1] = 1/12.
        true_fluxes[1, 2] = 1/12.
        true_fluxes = np.around(true_fluxes, 5)

        calc_fluxes = reactive_fluxes(Tij, 0, 2, populations=pops)
        if hasattr(calc_fluxes, 'todense'):
            calc_fluxes = np.array(calc_fluxes.todense()).astype(np.double)

        # test fluxes
        calc_fluxes = np.around(calc_fluxes, 5)
        assert_array_equal(calc_fluxes, true_fluxes)

        # test fluxes without pops
        calc_fluxes = reactive_fluxes(Tij, 0, 2)
        if hasattr(calc_fluxes, 'todense'):
            calc_fluxes = np.array(calc_fluxes.todense()).astype(np.double)

        calc_fluxes = np.around(calc_fluxes, 5)
        assert_array_equal(calc_fluxes, true_fluxes)


def test_mfpts():
    tcounts = np.array([[2, 1, 1], [2, 1, 2], [3, 2, 1]])
    T_test = tcounts/tcounts.sum(axis=1)[:, None]

    # test all to all
    all_mfpts = mfpts(T_test)
    assert_array_almost_equal(
        all_mfpts, np.array(
            [
                [0., 3.71428571, 3.5],
                [2.3125, 0., 3.],
                [2.125, 3.42857143, 0.]]), 5)

    # test all to sinks
    sink_mfpts = mfpts(T_test, sinks=[0])
    assert_array_almost_equal(
        sink_mfpts,
        np.array([0., 2.3125, 2.125]), 5)

    # lagtime 1
    sink_mfpts_lagtime1 = mfpts(T_test, sinks=[0, 1])
    assert_array_almost_equal(
        sink_mfpts_lagtime1,
        np.array([0., 0., 1.2]), 5)

    # check lagtime
    sink_mfpts_lagtime5 = mfpts(T_test, sinks=[0, 1], lagtime=5.0)
    assert_array_almost_equal(
        sink_mfpts_lagtime1*5.0,
        sink_mfpts_lagtime5, 5)

    assert_array_almost_equal(
        mfpts(T_test)*5.0,
        mfpts(T_test, lagtime=5), 5)

# NOTE: block absorbed from MSMBuilder test_tpt.py
def test_paths():
    net_flux = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.3, 0.0, 0.2],
                         [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    sources = np.array([0])
    sinks = np.array([4, 5])

    ref_paths = [[0, 2, 4],
                 [0, 1, 3, 5],
                 [0, 1, 5]]

    ref_fluxes = np.array([0.5, 0.3, 0.2])

    res_bottle = tpt.paths(sources, sinks, net_flux, remove_path='bottleneck')
    res_subtract = tpt.paths(sources, sinks, net_flux, remove_path='subtract')

    for paths, fluxes in [res_bottle, res_subtract]:
        npt.assert_array_almost_equal(fluxes, ref_fluxes)
        assert len(paths) == len(ref_paths)

        for i in range(len(paths)):
            npt.assert_array_equal(paths[i], ref_paths[i])
# END absorbed block.
