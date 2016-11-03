import unittest

import numpy as np
import mdtraj as md
from mdtraj.testing import get_fn

from load import load_as_concatenated


class TestParallelLoad(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')
        self.top = md.load(self.top_fname).top

    def test_load_as_concatenated(self):

        t1 = md.load(self.trj_fname, top=self.top)
        t2 = md.load(self.trj_fname, top=self.top)
        t3 = md.load(self.trj_fname, top=self.top)

        lengths, xyz = load_as_concatenated([self.trj_fname]*3, self.top)
        expected = np.concatenate([t1.xyz, t2.xyz, t3.xyz])

        self.assertTrue(np.all(expected == xyz))
        self.assertEqual(expected.shape, xyz.shape)


if __name__ == '__main__':
    unittest.main()
