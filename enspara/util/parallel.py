# Author: Gregory R. Bowman <gregoryrbowman@gmail.com>
# Contributors:
# Copyright (c) 2016, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from __future__ import print_function, division, absolute_import

import os
import ctypes
import functools
import multiprocessing as mp
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg


def auto_nprocs():
    return int(os.getenv('OMP_NUM_THREADS', mp.cpu_count()))


def pool_dense2d(arr, processes=None):
    # arr is a dense 2D array shared between the workers
    # returns a pool and a function a worker can use to get the shared array
    # no lock, so only read the shared matrix
    def init(shared_arr_):
        global shared_arr
        shared_arr = shared_arr_

    n_elem = arr.shape[0] * arr.shape[1]
    shared_arr = mp.Array(ctypes.c_double, n_elem, lock=False)
    shared_arr[:] = arr.flatten().astype('float64')
    p = mp.Pool(processes=processes, initializer=init, initargs=(shared_arr,))

    return p, functools.partial(_retrieve_dense2d, arr.shape)


def _retrieve_dense2d(shape):
    np_arr = np.frombuffer(shared_arr)
    np_arr = np_arr.reshape(shape)
    return np_arr


def pool_sparse2d(arr, processes=None):
    # arr is a dense 2D array shared between the workers
    # returns a pool and a function a worker can use to get the shared array
    # no lock, so only read the shared matrix
    def init(shared_arr_):
        global shared_arr
        shared_arr = shared_arr_

    n_elem = 3 * arr.nnz
    i, j = arr.nonzero()
    data = arr[arr.nonzero()].toarray()[0]
    shared_arr = mp.Array(ctypes.c_double, n_elem, lock=False)
    shared_arr[:arr.nnz] = data.astype('float64')
    shared_arr[arr.nnz:2*arr.nnz] = i.astype('float64')
    shared_arr[2*arr.nnz:] = j.astype('float64')
    p = mp.Pool(initializer=init, initargs=(shared_arr,))

    return p, functools.partial(_retrieve_sparse2d, arr.shape)


def _retrieve_sparse2d(shape):
    np_arr = np.frombuffer(shared_arr)
    nnz = np_arr.shape[0]/3
    data = np_arr[:nnz]
    i = np_arr[nnz:2*nnz].astype(int)
    j = np_arr[2*nnz:].astype(int)
    sparse_arr = scipy.sparse.coo_matrix((data, (i, j)), shape=shape)
    return sparse_arr.tolil()
