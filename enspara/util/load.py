import logging
import sys
import warnings

import multiprocessing as mp
from itertools import count
from contextlib import closing
from functools import partial, reduce
import ctypes
from operator import mul
import math

import numpy as np
import mdtraj as md

from mdtraj import io
from mdtraj.core.trajectory import _get_extension

from sklearn.externals.joblib import Parallel, delayed

from ..exception import ImproperlyConfigured, DataInvalid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sound_trajectory(trj, **kwargs):
    """Determine the length of a trajectory on disk.

    For H5 file formats, this is a trivial lookup of the shape parameter
    tracked by the HDF5 file system. For other file formats, a binary
    search-like system to figure out how long a trajectory on disk is
    in (maybe) log(n) time and constant space by loading individual
    frames from disk at exponentially increasing indices.

    Additional keyword args are passed on to md.load.

    Parameters
    ----------
    trj: file path
        Path to the trajectory to sound.

    Returns
    ----------
    length: int
        The length (in frames) of a trajectory on disk, as though loaded
        with kwargs

    See Also
    ----------
    md.load
    """

    stride = kwargs.pop('stride', None)
    stride = 1 if stride is None else stride

    extension = _get_extension(trj)

    if extension == '.h5':
        logger.debug("Accessing shape of HDF5 '%s' with args: %s", trj, kwargs)

        if 'top' in kwargs:
            warnings.warn(
                "kwarg 'top' is ignored when input to sound_trajectory "
                "is an h5 file.", RuntimeWarning)

        a = io.loadh(trj)
        shape = a._handle.get_node('/coordinates').shape
        a.close()

        return math.ceil(shape[0] / stride)
    else:
        logger.debug("Actively sounding '%s' with args: %s", trj, kwargs)

        return _sound_trajectory(trj, stride=stride, **kwargs)


def _sound_trajectory(trj, **kwargs):
    """Ascertains the length of an trajectory format that does not have
    metadata on length.

    Parameters
    ----------
    trj: file path
        Path to the trajectory to sound.

    Returns
    ----------
    length: int
        The length (in frames) of a trajectory on disk, as though loaded
        with kwargs

    See Also
    ----------
    md.load
    """

    search_space = [0, sys.maxsize]
    base = 2

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
    length = math.ceil(search_space[1] / kwargs['stride'])

    return length


def load_as_concatenated(filenames, lengths=None, processes=None,
                         args=None, **kwargs):
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
    lengths : list, optional, default=None
        List of lengths of the underlying trajectories. If None, the
        lengths will be inferred. However, this can be slow, especially
        as the number of trajectories grows large. This option provides
        a speed benefit only.
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

    if lengths is None:
        logger.debug("Sounding %s trajectories with %s processes.",
                     len(filenames), processes)
        lengths = Parallel(n_jobs=processes)(
            delayed(sound_trajectory)(f, **kw)
            for f, kw in zip(filenames, args))
    else:
        logger.debug("Using given lengths")
        if len(lengths) != len(filenames):
            raise ImproperlyConfigured(
                "Lengths list (len %s) didn't match length of filenames"
                " list (len %s)", len(lengths), len(filenames))

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
    shapes = proc.get()
    if sum(s[0] for s in shapes) != full_shape[0]:
        raise DataInvalid(
            "The provided lengths (n=%s, total frames %s) weren't correct. "
            "The correct total number of frames was %s.", len(lengths),
            sum(s[0] for s in shapes), full_shape[0])

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

    return xyz.shape
