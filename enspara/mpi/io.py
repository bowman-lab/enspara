import logging
import numpy as np

from ..util.load import load_as_concatenated
from .. import exception

from . import MPI_RANK, MPI_SIZE
from .ops import assemble_striped_array

logger = logging.getLogger(__name__)


def load_npy_as_striped(filenames, stride=1):
    """Load ndarrays into distributed arrays across nodes in an MPI swarm.

    File i is loaded by node i % n, where n is the number of nodes in
    the swarm.

    Parameters
    ----------
    filenames : list
        A list of relative paths to the trajectory files to be loaded.
        The md.load function is used, and all file types md.load
        supports are supported by this function.
    stride : int, default=1
        Load only every stride-th frame.

    Returns
    -------
    (global_lengths, xyz) : tuple
       A 2-tuple of trajectory lengths (list of ints, frames) and
       coordinates (ndarray, shape=(n_atoms, n_frames, 3)).

    See also
    --------
    enspara.mpi.io.load_trajectory_as_striped
    """

    specs = [(h.shape, h.dtype) for h in
             (np.load(f, mmap_mode='r') for f in filenames)]

    shape0, dtype = specs[0]
    for i, (s, d) in enumerate(specs):
        if s[1:] != shape0[1:]:
            raise exception.ImproperlyConfigured(
                "Subsequent dimensions of file '{}' didn't match shape "
                "of first file, '{}' ({} != {})".format(
                    filenames[0], filenames[i], shape0, s))
        if d != dtype:
            raise exception.ImproperlyConfigured(
                "Type of file '{}' didn't match type first file, '{}' "
                "({} != {})".format(
                    filenames[0], filenames[i], dtype, d))

    global_lengths = [s[0] for s, d in specs]
    logger.debug("Determined global lengths to be %s", global_lengths)
    local_lengths = [s[0] for s, d in specs[MPI_RANK::MPI_SIZE]]

    local_data = np.empty((sum(local_lengths),) + shape0[1:],
                          dtype=dtype)
    logger.debug("Allocated array of shape %s and type %s",
                 local_data.shape, local_data.dtype)

    # TODO could be thread parallelized?
    local_filenames = filenames[MPI_RANK::MPI_SIZE]
    start = 0
    for i, f in enumerate(local_filenames):
        data = np.load(f, mmap_mode='r')
        end = start + len(data[::stride])
        logger.debug("Writing file %s to [%s:%s]", i, start, end)
        local_data[start:end] = data[::stride]
        start = end
    assert end == len(local_data)

    logger.debug("Loaded %s npys into an array of shape %s.",
                 len(filenames), local_data.shape)

    return global_lengths, local_data


def load_trajectory_as_striped(filenames, *args, **kwargs):
    """Load trajectories into distributed arrays across nodes in an MPI swarm.

    File i is loaded by node i % n, where n is the number of nodes in
    the swarm.

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
    (global_lengths, xyz) : tuple
       A 2-tuple of trajectory lengths (list of ints, frames) and
       coordinates (ndarray, shape=(n_atoms, n_frames, 3)).

    See also
    --------
    enspara.util.load.load_as_concatenated, enspara.mpi.io.load_npy_as_striped
    """

    if len(filenames) < MPI_SIZE:
        raise exception.ImproperlyConfigured(
            "To stripe files across MPI workers, at least 1 file per "
            "node must be given. MPI size is %s, number of files is %s."
            % (MPI_SIZE, len(filenames)))

    # if we're specifying parameters separately for each trj to load, we
    # need to stripe those across nodes also.
    if 'args' in kwargs and len(kwargs['args']) > 1:
        assert len(kwargs['args']) == len(filenames)
        kwargs['args'] = kwargs['args'].copy()[MPI_RANK::MPI_SIZE]

    local_lengths, my_xyz = load_as_concatenated(
        filenames=filenames[MPI_RANK::MPI_SIZE], *args, **kwargs)

    local_lengths = np.array(local_lengths, dtype=int)
    global_lengths = assemble_striped_array(local_lengths)

    return global_lengths, my_xyz
