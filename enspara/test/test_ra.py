import unittest
import logging
import tempfile

import numpy as np
import mdtraj as md
from mdtraj import io

from nose.tools import assert_raises, assert_equals, assert_is, assert_true
from numpy.testing import assert_array_equal

from ..util import array as ra
from ..util.load import load_as_concatenated, concatenate_trjs
from ..exception import DataInvalid, ImproperlyConfigured

from .util import get_fn


def assert_ra_equal(a, b, **kwargs):
        assert_array_equal(a._data, b._data, **kwargs)
        assert_array_equal(a.lengths, b.lengths)


class Test_RaggedArray(unittest.TestCase):

    def test_RaggedArray_creation(self):
        a = ra.RaggedArray(array=np.array(range(50)), lengths=[25, 25])
        assert_array_equal(a.starts, [0, 25])

        a = ra.RaggedArray(array=[np.array(range(10)),
                                  np.array(range(20))])
        assert_equals(len(a), 2)
        assert_equals(a.dtype, np.int)
        assert_array_equal(a.lengths, [10, 20])
        assert_array_equal(a.starts, [0, 10])
        assert_array_equal(a._data, np.concatenate([range(10), range(20)]))

    def test_RaggedArray_floats(self):
        a = ra.RaggedArray([[0.8, 1.0, 1.2],
                            [1.1, 1.0, 0.9, 0.8]])

        assert_equals(len(a), 2)
        assert_equals(a.dtype, np.float)
        assert_array_equal(a.lengths, [3, 4])
        assert_array_equal(a.starts, [0, 3])
        assert_array_equal(
            a._data, [0.8, 1.0, 1.2] + [1.1, 1.0, 0.9, 0.8])

    def test_RaggedArray_shape_size(self):

        a = ra.RaggedArray(array=np.array(range(50)), lengths=[25, 20, 5])
        assert_equals(a.shape, (3, None))
        assert_equals(a.size, 50)
        assert_equals(a.dtype, np.int)

        src_reg = [[[0,0,0],[1,1,1],[2,2,2]],[[4,4,4],[5,5,5]]]
        a_reg = ra.RaggedArray(src_reg)
        assert_equals(a_reg.shape, (2, None, 3))

        src_irreg = [[[0,0,0,0],[1,1],[2,2,2]],[[4,4],[5,5,5,5,5]]]
        a_irreg = ra.RaggedArray(src_irreg)
        assert_equals(a_irreg.shape, (2, None, None))

    def test_RaggedArray_disk_roundtrip(self):
        src = np.array(range(55))
        a = ra.RaggedArray(array=src, lengths=[25, 30])

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            ra.save(f.name, a)
            b = ra.load(f.name)
            assert_ra_equal(a, b)

    def test_RaggedArray_disk_roundtrip_with_stride(self):
        src = np.array(range(55))
        a = ra.RaggedArray(array=src, lengths=[25, 30])

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            ra.save(f.name, a)
            b = ra.load(f.name, stride=3)

        assert_ra_equal(a[:, ::3], b)

    def test_RaggedArray_disk_roundtrip_numpy(self):
        a = np.ones(shape=(5, 5))

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            ra.save(f.name, a)
            b = ra.load(f.name)
            assert_array_equal(a, b)

    def test_RaggedArray_load_h5_arrays(self):
        src = np.array(range(55))
        a = ra.RaggedArray(array=src, lengths=[25, 30])

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            io.saveh(f.name, key0=a[0], key1=a[1])

            b = ra.load(f.name, keys=['key0', 'key1'])

        assert_ra_equal(a, b)

        src = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T

        a = ra.RaggedArray(array=src, lengths=[4, 6])

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            io.saveh(f.name, key0=a[0], key1=a[1])
            b = ra.load(f.name, keys=['key0', 'key1'])

        assert_ra_equal(a, b)

    def test_RaggedArray_load_specific_h5_arrays(self):

        src = np.array(range(55))
        a = ra.RaggedArray(array=src, lengths=[15, 10, 30])

        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            io.saveh(f.name, key0=a[0], key1=a[1], key2=a[2])
            b = ra.load(f.name, keys=['key1', 'key2'])

        assert_ra_equal(a[1:], b[:])

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

    def test_RaggedArray_numpy_compatability(self):
        src = [range(4), range(5), range(6)]
        a = ra.RaggedArray(array=src)

        for i in np.arange(3):
            assert_array_equal(a[i], src[i])

        new_rag = [[10,11,12,13], [1,2,3,4,5], [11,12,13,14,15,16]]
        for i in np.arange(3):
            a[i] = new_rag[i]
            assert_array_equal(a[i], new_rag[i])

        src = [range(4), range(5), range(6)]
        a = ra.RaggedArray(array=src)

        assert_array_equal(a[:,1], [[1],[1],[1]])
        assert_array_equal(a[:,np.arange(3)[1]], [[1],[1],[1]])

        a[:,np.arange(3)[1]] = [[90],[90],[70]]
        assert_array_equal(a[:,1], [[90], [90], [70]])
        assert_array_equal(a[:,np.arange(3)[1]], [[90], [90], [70]])

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

    def test_ra_invert(self):
        a = ra.RaggedArray([[True, False, True, False],
                            [False, True, False]])
        b = ~a

        assert_ra_equal(b, ra.RaggedArray([[False, True, False, True],
                                           [True, False, True]]))

    def test_ra_or(self):
        a = ra.RaggedArray([[True, False, True, False],
                            [False, True, False]])
        b = ra.RaggedArray([[False, False, True, True],
                            [True, False, True]])

        c = a | b
        assert_ra_equal(
            c,
            ra.RaggedArray([[True, False, True, True],
                            [True, True, True]]))

    def test_ra_zeros_like(self):
        a = ra.RaggedArray([[True, False, True, False],
                            [False, True, False]])

        b = ra.zeros_like(a)

        assert_array_equal(a.lengths, b.lengths)
        assert_equals(a.shape[0], b.shape[0])
        assert_true((b == 0).all())
        assert_is(type(b), ra.RaggedArray)

        a = np.linspace(10, 20)
        b = ra.zeros_like(a)

        assert_array_equal(a.shape, b.shape)
        assert_array_equal(np.zeros_like(a), b)

    def test_ra_operator_not_implemented(self):

        a = ra.RaggedArray([[True, False, True, False],
                            [False, True, False]])

        with assert_raises(TypeError):
            a > 'asdfasdfasd'


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

        t1 = md.load(self.top_fname)
        t2 = md.load(self.top_fname)
        t3 = md.load(self.top_fname)

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

    def test_load_as_concatenated_lengths_hint(self):

        t1 = md.load(self.trj_fname, top=self.top)
        t2 = md.load(self.trj_fname, top=self.top)
        t3 = md.load(self.trj_fname, top=self.top)

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*3,
            top=self.top,
            lengths=[len(t) for t in [t1, t2, t3]])

        expected = np.concatenate([t1.xyz, t2.xyz, t3.xyz])

        assert_equals(expected.shape, xyz.shape)
        assert_true(np.all(expected == xyz))

        with assert_raises(ImproperlyConfigured):
            lengths, xyz = load_as_concatenated(
                [self.trj_fname]*3,
                top=self.top,
                lengths=[len(t) for t in [t1, t3]])

        with assert_raises(DataInvalid):
            lengths, xyz = load_as_concatenated(
                [self.trj_fname]*3,
                top=self.top,
                lengths=[len(t) for t in [t1, t2[::2], t3]])

    def test_load_as_concatenated_frame_kwarg(self):
        '''`frame` should work in the `args` param of load_as_concatenated
        '''
        frames = [8, 13]

        t1 = md.load(self.trj_fname, top=self.top)[frames[0]]
        t2 = md.load(self.trj_fname, top=self.top)[frames[1]]

        args = [
            {'top': self.top, 'frame': frames[0]},
            {'top': self.top, 'frame': frames[1]}
            ]

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*2,
            processes=2,
            args=args)  # called kwargs, it's actually applied as an arg vector

        expected = np.concatenate([t1.xyz, t2.xyz])

        self.assertEqual(expected.shape, xyz.shape)
        self.assertTrue(np.all(expected == xyz))

        # NOW TEST IF ONLY ONE HAS 'frames'
        t2 = md.load(self.trj_fname, top=self.top)

        args = [
            {'top': self.top, 'frame': frames[0]},
            {'top': self.top}
            ]

        lengths, xyz = load_as_concatenated(
            [self.trj_fname]*2,
            processes=2,
            args=args)  # called kwargs, it's actually applied as an arg vector

        expected = np.concatenate([t1.xyz, t2.xyz])

        self.assertEqual(expected.shape, xyz.shape)
        self.assertTrue(np.all(expected == xyz))


    def test_hdf5(self):

        hdf5_fn = get_fn('frame0.h5')

        t1 = md.load(hdf5_fn)

        lengths, xyz = load_as_concatenated([hdf5_fn]*5)

        assert_array_equal(lengths, [len(t1)]*5)
        assert_array_equal(xyz.shape[1:], t1.xyz.shape[1:])
        assert_array_equal(t1.xyz, xyz[0:t1.xyz.shape[0], :, :])

    def test_hdf5_multiarg(self):
        '''
        *args should work with load_as_concatenated h5s

        Verify that when an *args is used on load_as_concatenated, each
        arg is applied in order to each trajectory as it's loaded.
        '''
        selections = [np.array([1, 3, 6]), np.array([2, 4, 7])]
        hdf5_fn = get_fn('frame0.h5')

        t1 = md.load(hdf5_fn, atom_indices=selections[0])
        t2 = md.load(hdf5_fn, atom_indices=selections[1])

        args = [
            {'atom_indices': selections[0]},
            {'atom_indices': selections[1]}
            ]

        lengths, xyz = load_as_concatenated(
            [hdf5_fn]*2,
            processes=2,
            args=args)  # called kwargs, it's actually applied as an arg vector

        expected = np.concatenate([t1.xyz, t2.xyz])

        self.assertEqual(expected.shape, xyz.shape)
        self.assertTrue(np.all(expected == xyz))


class TestConcatenateTrajs(unittest.TestCase):

    def setUp(self):
        self.trj_fname = get_fn('frame0.xtc')
        self.top_fname = get_fn('native.pdb')
        self.top = md.load(self.top_fname).top

        logging.getLogger('enspara.util.load').setLevel(logging.DEBUG)

    def test_concat_simple(self):

        trjlist = [md.load(self.top_fname)] * 10
        trj = concatenate_trjs(trjlist)

        assert_equals(len(trjlist), len(trj))

        for trjframe, trjlist_item in zip(trj, trjlist):
            assert_array_equal(trjframe.xyz, trjlist_item.xyz)

    def test_concat_atoms(self):

        ATOMS = 'name N or name C or name CA'
        trjlist = [md.load(self.top_fname)] * 10
        trj = concatenate_trjs(trjlist, atoms=ATOMS)

        assert_equals(len(trjlist), len(trj))

        for trjframe, trjlist_item in zip(trj, trjlist):
            sliced_item = trjlist_item.atom_slice(
                trjlist_item.top.select(ATOMS))
            assert_array_equal(trjframe.xyz, sliced_item.xyz)

    def test_different_lengths(self):

        trjlist = [md.load(self.top_fname)] * 5
        trjlist.append(md.load(self.trj_fname, top=self.top_fname))

        trj = concatenate_trjs(trjlist, atoms='name CA or name N or name C')

        assert_equals(trj.xyz.shape, (506, 6, 3))

    def test_mismatched(self):

        trjlist = [md.load(self.top_fname)] * 5
        trjlist.append(trjlist[0].atom_slice(np.arange(10)))

        with assert_raises(DataInvalid):
            concatenate_trjs(trjlist)


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
