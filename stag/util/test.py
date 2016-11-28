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

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*3,
            top=self.top,
            processes=7)
        expected = np.concatenate([t1.xyz, t2.xyz, t3.xyz])

        self.assertTrue(np.all(expected == xyz))
        self.assertEqual(expected.shape, xyz.shape)

    def test_selection_load_as_concatenated(self):

        selection = np.array([1, 3, 6])

        t1 = md.load(self.trj_fname, top=self.top, atom_indices=selection)
        t2 = md.load(self.trj_fname, top=self.top, atom_indices=selection)
        t3 = md.load(self.trj_fname, top=self.top, atom_indices=selection)

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*3,
            top=self.top,
            atom_indices=selection,
            processes=3)
        expected = np.concatenate([t1.xyz, t2.xyz, t3.xyz])

        self.assertTrue(np.all(expected == xyz))
        self.assertEqual(expected.shape, xyz.shape)


if __name__ == '__main__':
    unittest.main()
