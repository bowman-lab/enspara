import unittest
import os
import tempfile

import numpy as np
import mdtraj as md

from nose.tools import (assert_raises, assert_less, assert_true, assert_is,
                        assert_equal)
from nose.plugins.attrib import attr

from sklearn.datasets import make_blobs

from numpy.testing import assert_array_equal, assert_allclose
from ..geometry import libdist
from ..geometry import dye_lifetimes, dyes_from_expt_dist, explicit_r0_calc
from ..exception import DataInvalid, ImproperlyConfigured

def get_fn(fn):
    return os.path.join(os.path.dirname(__file__), 'fret_data', fn)

class TestProtLabeling(unittest.TestCase):

    def setUp(self):
        self.prot_centers = get_fn('ab40.xtc')
        self.top_fname = get_fn('ab40.pdb')
        self.prot_tcounts = get_fn('ab40-tcounts.npy')

        self.donor_trj_fname = get_fn('a48-c1r-mini.xtc')
        self.donor_top_fname = get_fn('a48-c1r.pdb')
        self.donor_t_counts = get_fn('a48-tcounts.npy')
        self.donor_name = 'AlexaFluor 488 C1R'

        self.acceptor_trj_fname = get_fn('a59-c1r-mini.xtc')
        self.acceptor_top_fname = get_fn('a59-c1r.pdb')
        self.acceptor_t_counts = get_fn('a59-tcounts.npy')
        self.acceptor_name = 'AlexaFluor 594 C1R'

        self.prot_trj = md.load(self.trj_fname, top=self.top_fname)
        self.donor_trj = md.load(self.donor_trj_fname, top=self.donor_top_fname)
        self.acceptor_trj = md.load(self.acceptor_trj_fname, top=self.acceptor_top_fname)

    def test_labeling(self):

    #Check to see that dyes can be labeled
    #Check to see if states are removed from MSM
    #Check to see if lifetimes are reasonable
    #Check if running burst gives reasonable FRET E.

