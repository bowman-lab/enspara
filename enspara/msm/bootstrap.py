import ctypes
import itertools
import multiprocessing as mp
import numpy as np

from . import msm
from .. import exception


def bootstrap(func, data, n_trials, n_procs=1, **kwargs):
    """Do a bootstrap sampling of `func` on `data`.

    Parameters
    ----------
    func : callable
        A function that can be called on `data` to compute the
        bootstrapped value. Should return the relevant values.
    data : object
        Data to run `func` on.
    n_trials : int
        Number of bootstrapping trials to run.
    n_procs : int
        The number of parallel bootstrappings to run.

    Notes
    -----
    Additional arguments as `kwargs` are passed in to `func` as
    parameters.
    """

    # make a shared data array of ints (does not support anything else)
    shared_data = _make_shared_array(data, ctypes.c_int)
    shared_data_shape = data.shape
    # generate random sample indices
    rand_sampling_iis = [
        np.random.choice(np.arange(data.shape[0]), data.shape[0])
        for i in np.arange(n_trials)]
    strap_data = list(
        zip(
            itertools.repeat(func), rand_sampling_iis,
            itertools.repeat(kwargs)))
    # map
    with mp.Pool(
            processes=n_procs, initializer=_init,
            initargs=(shared_data, shared_data_shape)) as p:
        straps = p.map(_single_strap, strap_data)
        p.terminate()
    return straps


def MSMs(assignments, lag_time, method, n_trials, max_n_states=None,
         n_procs=1, chunk_by=None, **kwargs):
    """bootstraps msms"""
    if chunk_by is not None:
        assignments = _chunk_assignments(assignments, chunk_by)
    msms = bootstrap(
        msm.MSM.from_assignments, assignments, lag_time=lag_time,
        method=method, n_trials=n_trials, max_n_states=max_n_states,
        n_procs=n_procs, **kwargs)
    return msms


def _chunk_assignments(assignments, chunk_by):
    pass


def _make_shared_array(in_array, dtype):
    """Generates a shared array for multiprocessing"""
    if not np.issubdtype(in_array.dtype, np.integer):
        raise exception.DataInvalid(
            "Given array (type '%s') is an integral type (e.g. int32). "
            "Mutual information calculations require discretized state "
            "trajectories." % in_array.dtype)

    arr = mp.Array(dtype, in_array.size, lock=False)
    arr[:] = in_array.flatten()

    return arr


def _single_strap(strap_data):
    # perform a single strap
    strap_func, rand_sampling_iis, kwargs = strap_data
    return strap_func(bootstrap_data[rand_sampling_iis], **kwargs)


def _init(bootstrap_data_, shape_data):
    # define bootstrap_data as a global variable
    global bootstrap_data
    bootstrap_data = np.frombuffer(
        bootstrap_data_, dtype=np.int32).reshape(shape_data)
    return
