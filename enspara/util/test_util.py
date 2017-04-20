import unittest
import logging

import numpy as np
import mdtraj as md

from mdtraj.testing import get_fn
from nose.tools import assert_raises, assert_equals, assert_is
from numpy.testing import assert_array_equal

from . import array as ra
from .load import load_as_concatenated

from ..exception import DataInvalid


class Test_RaggedArray(unittest.TestCase):

    def test_RaggedArray_creation(self):
        a = ra.RaggedArray(array=np.array(range(50)), lengths=[25, 25])
        assert_array_equal(a.starts, [0, 25])

        a = ra.RaggedArray(array=[np.array(range(10)),
                                  np.array(range(20))])
        assert_equals(len(a), 2)
        assert_array_equal(a.lengths, [10, 20])
        assert_array_equal(a.starts, [0, 10])
        assert_array_equal(a._data, np.concatenate([range(10), range(20)]))

    def test_RaggedArray_shape_size(self):

        a = ra.RaggedArray(array=np.array(range(50)), lengths=[25, 20, 5])
        assert_equals(a.shape, (3, None))
        assert_equals(a.size, 50)

        src_reg = [[[0,0,0],[1,1,1],[2,2,2]],[[4,4,4],[5,5,5]]]
        a_reg = ra.RaggedArray(src_reg)
        assert_equals(a_reg.shape, (2, None, 3))

        src_irreg = [[[0,0,0,0],[1,1],[2,2,2]],[[4,4],[5,5,5,5,5]]]
        a_irreg = ra.RaggedArray(src_irreg)
        assert_equals(a_irreg.shape, (2, None, None))


    def test_RaggedArray_bad_size(self):

        with assert_raises(DataInvalid):
            ra.RaggedArray(array=np.array(range(50)), lengths=[25, 20])

    def test_RaggedArray_indexing(self):
        src = np.array(range(55))
        a = ra.RaggedArray(array=src, lengths=[25, 30])

        assert_equals(a[0, 0], 0)
        assert_equals(a[0, 5], 5)
        assert_equals(a[1, 0], 25)
        assert_equals(a[1, 9], 34)

        with assert_raises(IndexError):
            a[0, 25]
        with assert_raises(IndexError):
            a[0, -26]
        with assert_raises(IndexError):
            a[1, 30]
        with assert_raises(IndexError):
            a[1, -31]

        assert_equals(a[0, 0], a[0][0])
        assert_equals(a[0, 5], a[0][5])
        assert_equals(a[1, 0], a[1][0])
        assert_equals(a[1, 9], a[1][9])

        assert_equals(a[0, -1], a[0, 24])
        assert_equals(a[1, -2], a[1, 28])

        assert_array_equal(a[0], src[0:25])
        assert_array_equal(a[1], src[25:])
        assert_array_equal(a[-1], a[1])
        assert_array_equal(a[-2], a[0])

        assert_equals(len(a[0]), 25)
        assert_equals(len(a[1]), 30)

        with assert_raises(IndexError):
            a[2]
        with assert_raises(IndexError):
            a[-3]

        b = ra.RaggedArray([[23, 24],[48, 49, 50]])
        assert_equals(a[:, 23:26], b)

    def test_RaggedArray_iterator(self):
        src = [range(10), range(20), range(30)]
        a = ra.RaggedArray(array=src)

        assert_array_equal(np.concatenate([i for i in a]),
                           np.concatenate([np.array(i) for i in src]))

    def test_RaggedArray_negative_slicing(self):
        src = np.array(range(20))
        a = ra.RaggedArray(array=src, lengths=[10, 5, 5])

        assert_array_equal(a[:,:-1].lengths, np.array([9, 4, 4]))
        assert_array_equal(a[:,:-2][0], np.arange(8))
        assert_array_equal(a[:,:-2][1], np.array([10,11,12]))

        assert_array_equal(
            (a[:,:-2]+2)._data,
            np.array([2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18, 19]))
        a[:,:-2] += 2
        assert_array_equal(
            a._data,
            np.array(
                [
                    2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 13, 14, 17,
                    18, 19, 18, 19]))


    def test_RaggedArray_slicing(self):
        src = np.array(range(60))
        a = ra.RaggedArray(array=src, lengths=[10, 20, 30])

        assert_array_equal(a[:].flatten(), src)
        assert_array_equal(a[0:2].flatten(), src[0:30])
        assert_array_equal(a[1:].flatten(), src[10:])

        assert_array_equal(a[:, 0:5].flatten(), np.concatenate((src[0:5],
                                                                src[10:15],
                                                                src[30:35])))

        assert_is(type(a[[0, 1]]), type(a))
        assert_is(type(a[0]), type(src))
        assert_is(type(a[[0]]), type(a))

        assert_array_equal(a[0, 5:10], src[5:10])
        assert_array_equal(a[-1, 5:10], src[35:40])

        assert_array_equal(a[2, 10:15:2], src[40:45:2])
        assert_array_equal(a[0, ::-1], src[9::-1])

    def test_RaggedArray_set_indexing(self):
        src = np.array(range(60))
        a = ra.RaggedArray(array=src, lengths=[10, 20, 30])

        a_sub = a[np.array([0, 2, -1])]
        assert_array_equal(a_sub[0], src[0:10])
        assert_array_equal(a_sub[1], src[30:60])
        assert_array_equal(a_sub[2], src[30:60])

        # (0, 0), (1, 1) == (0, 10)
        assert_array_equal(a[(np.array([0, 1]),
                              np.array([0, 1]))],
                           src[(np.array([0, 11]))])

        assert_array_equal(
            a[(np.array([2, -1, -1]),
               np.array([3, -1, 4]))],
            src[(np.array([33, 59, 34]))])

    def test_subragged_data_mapping(self):
        src = np.array(range(60))
        a = ra.RaggedArray(array=src, lengths=[10, 20, 30])

        b = a[1]
        b[0] = -1
        assert_equals(a[1, 0], -1)

    def test_ra_bool_indexing(self):
        src = [range(10), range(15), range(10)]
        a = ra.RaggedArray(array=src)

        b = (a < 5)
        print(b)
        print(a[b])

    def test_RaggedArray_setting(self):
        src = np.array(range(50))

        a = ra.RaggedArray(array=src, lengths=[20, 30])
        a[1] = range(30)

        assert_array_equal(a[1], range(30))
        assert_array_equal(a[0], range(20))
        assert_equals(a[1, 0], 0)
        assert_equals(a[1, -1], 29)

        a = ra.RaggedArray(array=src, lengths=[20, 30])
        a[0, 2:5] = np.array([11, 12, 13])

        assert_array_equal(a[0, 2], 11)
        assert_array_equal(a[1], src[20:])
        assert_array_equal(a[0, 2:5], np.array([11, 12, 13]))

        a = ra.RaggedArray(array=src, lengths=[20, 30])
        a[0, 2:5] = np.array([11, 12, 13])

        # __setitem__ using fancy indexing should succeed.
        a = ra.RaggedArray(array=src, lengths=[20, 30])
        a[(np.array([1, 1, 0, -1]),
           np.array([0, 3, -1, 4]))] = np.array([-1, -2, -3, -4])

        assert_equals(a[1, 0], -1)
        assert_equals(a[1, 3], -2)
        assert_equals(a[0, -1], -3)
        assert_equals(a[-1, 4], -4)

        # __setitem__ using fancy indexing + int should succeed.
        a = ra.RaggedArray(array=src, lengths=[20, 30])
        a[np.array([0, -1]), 3] = np.array([-3, -2])
        assert_equals(a[0, 3], -3)
        assert_equals(a[-1, 3], -2)

        # __setitem__ using int + fancy indexing should succeed.
        a = ra.RaggedArray(array=src, lengths=[20, 30])
        a[0, np.array([1, 2, -1])] = np.array([-3, -2, -1])
        assert_equals(a[0, 1], -3)
        assert_equals(a[0, 2], -2)
        assert_equals(a[0, -1], -1)

    def test_ra_eq(self):
        src = [range(10), range(20), range(30)]
        a = ra.RaggedArray(array=src)
        b = ra.RaggedArray(array=src)

        assert (a == b).all()

        b[0, 0] = 10
        assert not (a == b)[0, 0]
        assert (a == b)[1, 0]

        assert (a != b)[0, 0]
        assert (a == b)[0, 1:].all()
        assert (a == b)[1:].all()

        assert (a[0] == range(10)).all()

    def test_ra_where(self):
        src = [range(10), range(20), range(30)]
        a = ra.RaggedArray(array=src)

        assert_array_equal(
            ra.where(a < 5),
            (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
             np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])))

        assert_array_equal(
            ra.where(a < 0),
            np.array([[], []],))

    def test_ra_where_ndarray(self):
        '''ra.where should work on ndarrays, too'''
        a = np.array([range(5), range(4, -1, -1)])

        assert_array_equal(
            ra.where(a == 4),
            [[0, 1], [4, 0]])


class TestParallelLoad(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')
        self.top = md.load(self.top_fname).top

        logging.getLogger('enspara.util.load').setLevel(logging.DEBUG)

    def test_load_as_concatenated_stride(self):

        t1 = md.load(self.trj_fname, top=self.top)
        t2 = md.load(self.trj_fname, top=self.top)
        t3 = md.load(self.trj_fname, top=self.top)

        print(len(t1))
        print(len(md.load(self.trj_fname, top=self.top, stride=10)))

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*3,
            top=self.top,
            stride=10,
            processes=2)

        expected = np.concatenate([t1.xyz[::10],
                                   t2.xyz[::10],
                                   t3.xyz[::10]])

        self.assertTrue(np.all(expected == xyz))
        self.assertEqual(expected.shape, xyz.shape)

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

    def test_load_as_concatenated_generator(self):

        t1 = md.load(self.trj_fname, top=self.top)
        t2 = md.load(self.trj_fname, top=self.top)

        lengths, xyz = load_as_concatenated(
            reversed([self.trj_fname, self.trj_fname]),  # returns a generator
            top=self.top)

        expected = np.concatenate([t2.xyz, t1.xyz])

        self.assertTrue(np.all(expected == xyz))
        self.assertEqual(expected.shape, xyz.shape)


    def test_load_as_concatenated_noargs(self):
        '''It's ok if no args are passed.'''

        t1 = md.load(self.top_fname, top=self.top)
        t2 = md.load(self.top_fname, top=self.top)
        t3 = md.load(self.top_fname, top=self.top)

        lengths, xyz = load_as_concatenated([self.top_fname]*3)

        expected = np.concatenate([t1.xyz, t2.xyz, t3.xyz])

        self.assertTrue(np.all(expected == xyz))
        self.assertEqual(expected.shape, xyz.shape)

    def test_selection_load_as_concatenated(self):
        '''
        **kwargs should work with load_as_concatenated

        Verify that when *kwargs is used in load_as_concatenated, it is
        applied to each trajectory as it's loaded (e.g. for the purposes
        of selection).
        '''

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

    def test_load_as_concatenated_multiarg(self):
        '''
        *args should work with load_as_concatenated

        Verify that when an *args is used on load_as_concatenated, each
        arg is applied in order to each trajectory as it's loaded.
        '''
        selections = [np.array([1, 3, 6]), np.array([2, 4, 7])]

        t1 = md.load(self.trj_fname, top=self.top, atom_indices=selections[0])
        t2 = md.load(self.trj_fname, top=self.top, atom_indices=selections[1])

        args = [
            {'top': self.top, 'atom_indices': selections[0]},
            {'top': self.top, 'atom_indices': selections[1]}
            ]

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*2,
            processes=2,
            args=args)  # called kwargs, it's actually applied as an arg vector

        expected = np.concatenate([t1.xyz, t2.xyz])

        self.assertEqual(expected.shape, xyz.shape)
        self.assertTrue(np.all(expected == xyz))


class TestPartition(unittest.TestCase):

    def test_partition_indices(self):

        indices = [0, 10, 15, 37, 100]
        trj_lens = [10, 20, 100]

        partit_indices = ra.partition_indices(indices, trj_lens)

        self.assertEqual(
            partit_indices,
            [(0, 0), (1, 0), (1, 5), (2, 7), (2, 70)])


if __name__ == '__main__':
    unittest.main()
