from nose.tools import assert_equal, assert_false, assert_true
from numpy.testing import assert_array_equal, assert_allclose

import numpy as np
from scipy.sparse import issparse

from .msm import MSM
from .transition_matrices import TrimMapping
from . import builders


def test_create_msm():
    in_assigns = np.array(
        [ ([0]*30 + [1]*20 + [-1]*10),
          ([2]*20 + [-1]*5 + [1]*35),
          ([0]*10 + [1]*30 + [2]*19 + [3]),
          ])

    # CONFIGURE EXPECTED VALUES
    expected = {
        'tcounts_': np.array([[38,  2,  0,  0],
                              [ 0, 82,  1,  0],
                              [ 0,  1, 37,  1],
                              [ 0,  0,  0,  0]]),
        'eq_probs_': np.array([0, 0.9780122, 2.172288e-02, 2.648414e-04]),
        'mapping_': TrimMapping([(0, 0), (1, 1), (2, 2), (3, 3)])
    }

    expected_transpose = expected.copy()
    expected_normalize = expected.copy()

    expected_normalize['tprobs_'] = \
        np.array([[0.95, 0.05    , 0.      , 0.      ],
                  [0.  , 0.987951, 0.012048, 0.      ],
                  [0.  , 0.025641, 0.948717, 0.025641],
                  [0.  , 0.      , 0.      , 0.      ]])

    expected_transpose['tprobs_'] = \
        np.array([[0.974358, 0.025641, 0.      , 0.      ],
                  [0.011904, 0.976190, 0.011905, 0.      ],
                  [0.      , 0.025974, 0.961038, 0.012987],
                  [0.      , 0.      , 1.      , 0.      ]])
    # \CONFIGURE EXPECTED VALUES

    cases = [('normalize', expected_normalize),
             ('transpose', expected_transpose),
             (builders.normalize, expected_normalize),
             (builders.transpose, expected_transpose)]

    for method, expected in cases:
        msm = MSM(lag_time=1, method=method)

        assert_false(any([hasattr(msm, param) for param in
                          ['tprobs_', 'tcounts_', 'eq_probs_', 'mapping_']]))

        msm.fit(in_assigns)

        print(msm.mapping_)

        for prop, expected_value in expected.items():
            calc_value = getattr(msm, prop)

            if hasattr(calc_value, 'toarray'):
                calc_value = calc_value.toarray()

            if type(calc_value) is np.ndarray:
                assert_allclose(calc_value, expected_value, rtol=1e-03)
            else:
                assert_equal(calc_value, expected_value)
