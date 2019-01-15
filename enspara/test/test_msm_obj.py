import functools
import tempfile
import shutil
import os
import pickle

from nose.tools import assert_equal, assert_false, assert_true
from numpy.testing import assert_allclose, assert_array_equal

import numpy as np

from ..msm.msm import MSM
from ..msm import builders

from .msm_data import TRIMMABLE


def test_create_msm():
    in_assigns = TRIMMABLE['assigns']

    cases = [
        ({'method': 'normalize'},
         TRIMMABLE['no_trimming']['msm']['normalize']),
        ({'method': 'transpose'},
         TRIMMABLE['no_trimming']['msm']['transpose']),
        ({'method': builders.normalize},
         TRIMMABLE['no_trimming']['msm']['normalize']),
        ({'method': builders.transpose},
         TRIMMABLE['no_trimming']['msm']['transpose']),
        ({'method': builders.normalize, 'trim': True},
         TRIMMABLE['trimming']['msm']['normalize']),
        ({'method': builders.transpose, 'trim': True},
         TRIMMABLE['trimming']['msm']['transpose'])
    ]

    for method, expected in cases:
        msm = MSM(lag_time=1, **method)

        assert_false(any([hasattr(msm, param) for param in
                          ['tprobs_', 'tcounts_', 'eq_probs_', 'mapping_']]))

        msm.fit(in_assigns)

        assert_equal(msm.n_states_, msm.tprobs_.shape[0])

        for prop, expected_value in expected.items():
            calc_value = getattr(msm, prop)

            if hasattr(calc_value, 'toarray'):
                calc_value = calc_value.toarray()

            if type(calc_value) is np.ndarray:
                assert_allclose(calc_value, expected_value, rtol=1e-03)
            else:
                assert_array_equal(calc_value, expected_value)


def test_msm_roundtrip():
    in_assigns = TRIMMABLE['assigns']

    msm = MSM(lag_time=1, method=builders.transpose)
    msm.fit(in_assigns)

    msmfile = tempfile.mktemp()
    try:
        msm.save(msmfile)
        assert_true(os.path.isdir(msmfile))
        assert_equal(MSM.load(msmfile), msm)
    finally:
        try:
            shutil.rmtree(msmfile)
        except:
            pass

    msmfile = tempfile.mktemp()
    manifest_path = 'a-wierd-manifest-path.json'
    try:
        # specify different names for some properties
        msm.save(msmfile)
        assert_true(os.path.isdir(msmfile))

        shutil.move(os.path.join(msmfile, 'manifest.json'),
                    os.path.join(msmfile, manifest_path))

        assert_equal(MSM.load(msmfile, manifest=manifest_path), msm)
    finally:
        try:
            shutil.rmtree(msmfile)
        except:
            pass

    msmfile = tempfile.mktemp()
    filedict = {prop: os.path.basename(tempfile.mktemp())
                for prop in ['tprobs_', 'tcounts_', 'eq_probs_', 'mapping_']}
    try:
        # specify different names for some properties
        msm.save(msmfile, **filedict)
        assert_true(os.path.isdir(msmfile))

        for filename in filedict.values():
            assert_true(os.path.isfile(os.path.join(msmfile, filename)))

        assert_equal(MSM.load(msmfile), msm)
    finally:
        try:
            shutil.rmtree(msmfile)
        except:
            pass


def test_msm_roundtrip_pickle():

    assigs = TRIMMABLE['assigns']
    m = MSM(lag_time=1, method=builders.normalize, max_n_states=4)

    m.fit(assigs)

    with tempfile.NamedTemporaryFile() as tmp_f:
        print(tmp_f.name)
        pickle.dump(m, tmp_f)
        tmp_f.flush()

        m2 = pickle.load(open(tmp_f.name, 'rb'))

    assert_equal(m, m2)


def test_msm_normalize_with_prior():

    assigs = TRIMMABLE['assigns']

    m_transpose = MSM(lag_time=1, method=builders.transpose)
    m_transpose.fit(assigs)

    prior = np.array(m_transpose.tprobs_\
                        .astype('bool').astype('float32')\
                        .todense())
    prior /= prior.sum(axis=1)[..., None]

    m = MSM(
        lag_time=1,
        method=functools.partial(
            builders.normalize,
            prior_counts=prior),
        max_n_states=4)

    m.fit(assigs)

    assert_allclose(m.eq_probs_,
                    [0.04179409, 0.64220183, 0.3058104, 0.01019368])
    assert_allclose(m.tprobs_,
        [[0.93902439, 0.06097561, 0.        , 0.        ],
         [0.00396825, 0.98015873, 0.01587302, 0.        ],
         [0.        , 0.03333333, 0.93333333, 0.03333333],
         [0.        , 0.        , 1.        , 0.        ]],
         rtol=1e-5)
