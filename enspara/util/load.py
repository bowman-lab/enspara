import logging
import sys

import multiprocessing as mp
from itertools import count
from contextlib import closing
from functools import partial, reduce
import ctypes
from operator import mul
import math

import numpy as np
import mdtraj as md

from sklearn.externals.joblib import Parallel, delayed

from ..exception import ImproperlyConfigured

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sound_trajectory(trj, **kwargs):
    """Determine the length of a trajectory on disk.

    Uses a binary search-like system to figure out how long a trajectory
    on disk is in (maybe) log(n) time and constant space by loading
    individual frames from disk at exponentially increasing indices.
    Additional keyword args are passed on to md.load.

    Parameters
    ----------
    trj: file path
        Path to the trajectory to sound.

    Returns
    ----------
    length: int
        The length (in frames) of a trajectory on disk, loaded with
        kwargs

    See Also
    ----------
    md.load
    """

    search_space = [0, sys.maxsize]
    base = 2
    stride = kwargs.pop('stride', None)
    stride = 1 if stride is None else stride

    logger.debug("Sounding '%s' with args: %s", trj, kwargs)

    while search_space[0]+1 != search_space[1]:
        start = search_space[0]
        for iteration in count():
            frame = start+(base**iteration)

            try:
                md.load(trj, frame=frame, **kwargs)
                search_space[0] = frame
            except (IndexError, IOError):
                # TODO: why is it IndexError sometimes, and IOError others?
                search_space[1] = frame
                break

    # if stride is passed to md.load, it is ignored, because apparently
    # when you give it a frame dumps the stride argument.
    length = math.ceil(search_space[1] / stride)

    return length


def load_as_concatenated(filenames, processes=None, args=None, **kwargs):
    '''Load many trajectories from disk into a single numpy array.

    Additional arguments to md.load are supplied as *args XOR **kwargs.
    If *args are supplied, args and filenames must be of the same length
    and the ith arg is applied as the kwargs to the md.load (e.g. top,
    selection) for the ith file. If **kwargs are specified, all are
    passed as keyword args to all calls to md.load.

    Parameters
    ----------
    filenames : list
        A list of relative paths to the trajectory files to be loaded.
        The md.load function is used, and all file types md.load
        supports are supported by this function.
    processes : int, optional
        The number of processes to spawn for loading in parallel.
    args : list, optional
        A list of dictionaries, each of which corresponds to additional
        kwargs to be passed to each of filenames.

    Returns
    -------
    (lengths, xyz) : tuple
       A 2-tuple of trajectory lengths (list of ints, frames) and
       coordinates (ndarray, shape=(n_atoms, n_frames, 3)).

    See Also
    --------
    md.load
    '''

    # we need access to this as a list, so if we get some kind of
    # wierd iterator we need to build a list out of it
    filenames = list(filenames)

    # configure arguments to md.load
    if kwargs and args:
        raise ImproperlyConfigured(
            "Additional unnamed args can only be supplied iff no "
            "additonal keyword args are supplied")
    elif kwargs:
        args = [kwargs]*len(filenames)
    elif args:
        if len(args) != len(filenames):
            raise ImproperlyConfigured(
                "When add'l unnamed args are provided, len(args) == "
                "len(filenames).")
    else:  # not args and not kwargs
        args = [{}]*len(filenames)

    logger.debug(
        "Configuring load calls with args[0] == [%s ... %s]",
        args[0], args[-1])

    # cast to list to handle generators
    filenames = list(filenames)

    logger.debug("Sounding %s trajectories with %s processes.", len(filenames),
                 processes)
    lengths = Parallel(n_jobs=processes)(
        delayed(sound_trajectory)(f, **kw)for f, kw in zip(filenames, args))

    root_trj = md.load(filenames[0], frame=0, **args[0])
    shape = root_trj.xyz.shape

    # TODO: check all inputs against root

    full_shape = (sum(lengths), shape[1], shape[2])

    # mp.Arrays are one-dimensional, so multiply the shape together for size
    shared_array = mp.Array(ctypes.c_double, reduce(mul, full_shape, 1),
                            lock=False)
    logger.debug("Allocated array of shape %s", full_shape)

    with closing(mp.Pool(processes=processes, initializer=_init,
                         initargs=(shared_array,))) as p:
        proc = p.map_async(
            partial(_load_to_position, arr_shape=full_shape),
            zip([sum(lengths[0:i]) for i in range(len(lengths))],
                filenames, args))

    # gather exceptions.
    proc.get()

    # wait for termination
    p.join()

    xyz = _tonumpyarray(shared_array).reshape(full_shape)

    return lengths, xyz


def _init(shared_array_):
    # for some reason, the shared array must be inhereted, not passed
    # as an argument
    global shared_array
    shared_array = shared_array_


def _tonumpyarray(mp_arr):
    # mp_arr.get_obj if Array is locking, otherwise mp_arr.
    return np.frombuffer(mp_arr)


def _load_to_position(spec, arr_shape):
    '''
    Load a specified file into a specified position by spec. The
    arr_shape parameter lets us know how big the final array should be.
    '''
    (position, filename, load_kwargs) = spec

    xyz = md.load(filename, **load_kwargs).xyz

    # mp.Array must be converted to numpy array and reshaped
    arr = _tonumpyarray(shared_array).reshape(arr_shape)

    # dump coordinates in.
    arr[position:position+len(xyz)] = xyz
