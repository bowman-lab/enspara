import tempfile

from nose.tools import assert_equal, assert_raises, assert_is
from numpy.testing import assert_array_equal, assert_allclose

import numpy as np
import scipy.sparse

from .transition_matrices import counts_to_probs, assigns_to_counts, \
    eigenspectra, transpose, trim_disconnected, TrimMapping
from .timescales import implied_timescales


# array types we want to guarantee support for
ARR_TYPES = [
    np.array, scipy.sparse.lil_matrix, scipy.sparse.csr_matrix,
    scipy.sparse.coo_matrix, scipy.sparse.csc_matrix,
    scipy.sparse.dia_matrix, scipy.sparse.dok_matrix
]


def test_trim_mapping_construction():

    tm1 = TrimMapping()
    tm1.to_original = {0: 0, 1: 1, 2: 3, 3: 7}

    tm2 = TrimMapping()
    tm2.to_mapped = {0: 0, 1: 1, 3: 2, 7: 3}

    assert_equal(tm1, tm2)


def test_trim_mapping_roundtrip():
    transformations = [(0, 0),
                       (1, -1),
                       (2, 1),
                       (3, 2)]

    tm = TrimMapping(transformations)

    with tempfile.NamedTemporaryFile(mode='w') as f:
        tm.write(f)
        f.flush()

        with open(f.name, 'r') as f2:
            assert_equal(
                f2.read().split('\n'),
                ['original,mapped', '0,0', '1,-1', '2,1', '3,2', ''])
        with open(f.name, 'r') as f2:
            tm2 = TrimMapping.read(f2)
            assert_equal(tm, tm2)

    with tempfile.NamedTemporaryFile() as f:
        tm.save(f.name)
        tm2 = TrimMapping.load(f.name)
        assert_equal(tm, tm2)


def test_implied_timescales():

    in_assigns = np.array(
        [ ([0]*30 + [1]*20 + [-1]*10),
          ([2]*20 + [-1]*5 + [1]*35),
          ([0]*10 + [1]*30 + [2]*19 + [3]),
          ])

    # test without symmetrization
    tscales = implied_timescales(in_assigns, lag_times=range(1, 5),
                                 symmetrization=None)
    expected = np.array(
        [[  1.      ,  19.495726],
         [  2.      ,  19.615267],
         [  3.      ,  20.094898],
         [  4.      ,  19.79665 ]])

    assert_allclose(tscales, expected, rtol=1e-03)

    # test with explicit symmetrization
    tscales = implied_timescales(
        in_assigns, lag_times=range(1, 5), symmetrization=transpose)
    expected = np.array(
        [[1., 38.497835],
         [2., 36.990989],
         [3., 35.478863],
         [4., 33.960748]])

    # test with trimming
    tscales = implied_timescales(
        in_assigns, lag_times=range(1, 5), symmetrization=transpose, trim=True)
    expected = np.array(
        [[1., 25.562856],
         [2., 24.384637],
         [3., 23.198114],
         [4., 22.001933]])

    assert_allclose(tscales, expected, rtol=1e-03)


def test_eigenspectra_types():

    expected_vals = np.array([ 1., 0.56457513, 0.03542487])
    expected_vecs = np.array(
        [[ 0.33333333,  0.8051731 , -0.13550992],
         [ 0.33333333, -0.51994159, -0.6295454 ],
         [ 0.33333333, -0.28523152,  0.76505532]])

    for arr_type in ARR_TYPES:
        probs = arr_type(
            [[0.7, 0.1, 0.2],
             [0.1, 0.5, 0.4],
             [0.2, 0.4, 0.4]])

        try:
            e_vals, e_vecs = eigenspectra(probs)
        except ValueError:
            print("Failed on type %s" % arr_type)
            raise

        assert_allclose(e_vecs, expected_vecs)
        assert_allclose(e_vals, expected_vals)


def test_assigns_to_counts_negnums():
    '''counts_to_probs ignores -1 values
    '''

    in_m = np.array(
            [[0, 2,  0, -1],
             [1, 2, -1, -1],
             [1, 0,  0, 1]])

    counts = assigns_to_counts(in_m)

    expected = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 0, 0]])

    assert_array_equal(counts.toarray(), expected)


def test_counts_to_probs_symm_options():
    '''counts_to_probs handles mixed case input strings, None.
    '''

    in_m = np.array(
        [[0, 2, 8],
         [4, 2, 4],
         [7, 3, 0]])

    counts = counts_to_probs(in_m, transpose)
    assert_allclose(counts, np.array([[0.      ,  0.285714,  0.714286],
                                      [0.352941,  0.235294,  0.411765],
                                      [0.681818,  0.318182,  0.      ]]),
                    rtol=1e-03)

    counts = counts_to_probs(in_m, None)
    assert_allclose(counts, np.array([[0. , 0.2, 0.8],
                                      [0.4, 0.2, 0.4],
                                      [0.7, 0.3, 0. ]]))


def test_counts_to_probs_types():
    '''counts_to_probs accepts & returns ndarrays, spmatrix subclasses.
    '''

    for arr_type in ARR_TYPES:

        in_m = arr_type(
            [[0, 2, 8],
             [4, 2, 4],
             [7, 3, 0]])
        out_m = counts_to_probs(in_m)

        assert_equal(type(in_m), type(out_m))

        # cast to ndarray if necessary for comparison to correct result
        try:
            out_m = out_m.toarray()
        except AttributeError:
            pass

        out_m = np.round(out_m, decimals=1)

        expected = np.array(
            [[0,   0.2, 0.8],
             [0.4, 0.2, 0.4],
             [0.7, 0.3, 0]])

        assert_array_equal(
            out_m,
            expected)


def test_trim_disconnected():
    # 3 connected components, one disconnected (state 4)

    for arr_type in ARR_TYPES:
        given = arr_type([[1, 2, 0, 0],
                          [2, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 2]])

        mapping, trimmed = trim_disconnected(given)
        assert_is(type(trimmed), type(given))

        expected_tcounts = np.array([[1, 2, 0],
                                     [2, 1, 1],
                                     [0, 1, 2]])
        assert_array_equal(trimmed, expected_tcounts)

        expected_mapping = TrimMapping([(0, 0), (1, 1), (3, 2)])
        assert_equal(mapping, expected_mapping)

        mapping, trimmed = trim_disconnected(given, threshold=2)

        expected_tcounts = np.array([[1, 2],
                                     [2, 1]])
        assert_array_equal(trimmed, expected_tcounts)

        expected_mapping = TrimMapping([(0, 0), (1, 1)])
        assert_equal(mapping, expected_mapping)
