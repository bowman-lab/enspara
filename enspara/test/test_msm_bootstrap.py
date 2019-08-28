import numpy as np

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_allclose

from ..msm.bootstrap import bootstrap
from ..msm.msm import MSM
from ..msm import builders

from .msm_data import TRIMMABLE
from .util import fix_np_rng


@fix_np_rng(0)
def test_bootstrap_msm():

    assigs = TRIMMABLE['assigns']

    N_TRIALS = 100
    LAG_TIME = 1
    N_STATES = 4

    msms = bootstrap(
        MSM.from_assignments, assigs, lag_time=LAG_TIME,
        method=builders.transpose, n_trials=N_TRIALS,
        max_n_states=N_STATES, n_procs=1)

    assert_equal(len(msms), N_TRIALS)
    assert_true(all(m.lag_time == LAG_TIME for m in msms))
    assert_true(all(m.max_n_states == N_STATES for m in msms))

    assert_true(all([m.tprobs_.shape == (N_STATES, N_STATES) for m in msms]))
    assert_true(all([m.eq_probs_.shape == (N_STATES,) for m in msms]))

    assert_allclose(
        np.array([m.tprobs_.todense() for m in msms]).mean(axis=0),
        np.array([[0.93871, 0.03128, 0.00000, 0.00000],
                  [0.01297, 0.97553, 0.01149, 0.00000],
                  [0.00000, 0.02470, 0.91199, 0.01330],
                  [0.00000, 0.00000, 0.72000, 0.00000]]),
        rtol=0.2)
