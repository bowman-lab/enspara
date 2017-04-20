import tempfile
import shutil
import os

from nose.tools import assert_equal, assert_false, assert_true
from numpy.testing import assert_allclose

import numpy as np

from .msm import MSM
from . import builders

from .test_data import TRIMMABLE


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
                assert_equal(calc_value, expected_value)


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