
import numpy as np

from ..util.load import load_as_concatenated
from .. import exception

from . import MPI_RANK, MPI_SIZE
from .ops import assemble_striped_array


def load_as_striped(filenames, *args, **kwargs):
    """Load files onto concat arrays across many nodes in an MPI swarm.

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
    enspara.util.load.load_as_concatenated
    """

    if len(filenames) < MPI_SIZE:
        raise exception.ImproperlyConfigured(
            "To stripe files across MPI workers, at least 1 file per "
            "node must be given. MPI size is %s, number of files is %s."
            % (MPI_SIZE, len(filenames)))

    local_lengths, my_xyz = load_as_concatenated(
        filenames=filenames[MPI_RANK::MPI_SIZE], *args, **kwargs)
    local_lengths = np.array(local_lengths, dtype=int)
    global_lengths = assemble_striped_array(local_lengths)

    return global_lengths, my_xyz
