import unittest
import mdtraj as md

from numpy.testing import assert_array_equal, assert_allclose

from enspara.info_theory.exposons import exposons
from .util import get_fn


class TestExposonsLoad(unittest.TestCase):

    def setUp(self):
        self.trj = md.load(get_fn('beta-peptide.xtc'),
                           top=get_fn('beta-peptide.pdb'))

    def test_exposons_pipeline_weighting(self):

        repeat_trj = md.join([self.trj[0:3], self.trj[0:3], self.trj[3:6]])
        norepeat_trj = md.join([self.trj[0:3], self.trj[3:6]])

        unweighted_mi, unweighted_exp = exposons(repeat_trj, 0.9,
                                                 threshold=1.0)
        weighted_mi, weighted_exp = exposons(norepeat_trj, 0.9,
                                             threshold=1.0,
                                             weights=[2, 2, 2, 1, 1, 1])

        assert_allclose(unweighted_mi, weighted_mi, rtol=1e-14)
        assert_array_equal(unweighted_exp, weighted_exp)

    # def test_exposons_noatom_warning(self):


    #     unweighted_mi, unweighted_exp = exposons(repeat_trj, 0.9)
