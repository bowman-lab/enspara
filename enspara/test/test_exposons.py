import unittest
import mdtraj as md
import pytest

from numpy.testing import assert_array_equal, assert_allclose

from enspara.info_theory import exposons
from .util import get_fn


def test_exposons_pipeline_weighting():

    trj = md.load(get_fn('beta-peptide.xtc'),
                      top=get_fn('beta-peptide.pdb'))

    repeat_trj = md.join([trj[0:3], trj[0:3], trj[3:6]])
    norepeat_trj = md.join([trj[0:3], trj[3:6]])

    unweighted_mi, unweighted_exp = exposons.exposons(
            repeat_trj, damping=0.9, threshold=1.0)
    weighted_mi, weighted_exp = exposons.exposons(
            norepeat_trj, damping=0.9, threshold=1.0,
            weights=[2, 2, 2, 1, 1, 1])

    assert_allclose(unweighted_mi, weighted_mi, rtol=1e-14)
    assert_array_equal(weighted_exp, unweighted_exp)


def test_exposons_sidechain_selection():

    trj = md.load(get_fn('beta-peptide.xtc'),
                  top=get_fn('beta-peptide.pdb'))

    repeat_trj = md.join([trj[0:3], trj[0:3], trj[3:6]])
    norepeat_trj = md.join([trj[0:3], trj[3:6]])

    expected_ids = [
        [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        [30, 31, 32, 33, 34, 35],
        [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
        [59, 60, 61, 62],
        [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        [85, 86],
        [ 93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
        [134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147],
        [154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171]
    ]

    ids = exposons.get_sidechain_atom_ids(repeat_trj.top)

    for obs, exp in zip(ids, expected_ids):
        assert_array_equal(obs, exp)

