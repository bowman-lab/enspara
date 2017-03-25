import os

from nose.tools import assert_equal, assert_raises, assert_is
from numpy.testing import assert_array_equal, assert_allclose

import mdtraj as md

from .. import cards

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

TOP = md.load(os.path.join(TEST_DATA_DIR, "PROT_only.pdb")).top
TRJ = md.load(os.path.join(TEST_DATA_DIR, "trj0.xtc"), top=TOP)

TEST_TRJS = [TRJ, TRJ]


def test_cards():

    ss_mi, dis_mi, s_d_mi, d_s_mi, inds = cards.cards(
        [TRJ, TRJ], buffer_width=15., n_procs=1, verbose_dir="verbose_output")
