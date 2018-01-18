import numpy as np

from ..exception import ImproperlyConfigured
from ..util import array as ra

from . import MPI, MPI_RANK, MPI_SIZE

COMM = MPI.COMM_WORLD


def convert_local_indices(local_ctr_inds, global_lengths):
    """Convert indices from (rank, local_frame) to (global frame).

    In enspara's clustering code, we represent frames in the data set by
    the pairs (owner_rank, local_frame), rather than (global_frame).
    This routine converts the local indices to global indices given the
    global lengths.

    Parameters
    ----------
    local_indices : iterable of tuples
        A list of tuples of the form `(owner_rank, local_frame)`.
    global_lengths : np.ndarray
        Array of the length of each trajectory distributed across all
        the nodes.
    """

    global_indexing = np.arange(np.sum(global_lengths))
    file_origin_ra = ra.RaggedArray(global_indexing, lengths=global_lengths)

    ctr_inds = []
    for rank, local_fid in local_ctr_inds:
        global_fid = file_origin_ra[rank::MPI_SIZE].flatten()[local_fid]
        ctr_inds.append(global_fid)

    return ctr_inds


def assemble_striped_array(local_arr):
    """Assemble an striped array.

    By 'striped array', we mean an array that has element i on node
    i % n. This is a common strategy for spreading data across MPI
    nodes, because it is easy to compute and, in practice, often spreads
    data pretty evenly.

    Parameters
    ----------
    local_array: np.ndarray
        The array to spread across nodes.

    Returns
    -------
    global_array: np.ndarray
        Full array that is striped across all nodes.
    """

    total_dim1 = COMM.allreduce(len(local_arr), op=MPI.SUM)
    total_shape = (total_dim1,) + local_arr.shape[1:]

    global_lengths = np.zeros(total_shape, dtype=local_arr.dtype) - 1

    for i in range(MPI_SIZE):
        global_lengths[i::MPI_SIZE] = COMM.bcast(local_arr, root=i)

    assert np.all(global_lengths > 0)

    return global_lengths


def assemble_striped_ragged_array(local_array, global_lengths):
    """Assemble an array that is striped according to the first dim of a ragged array.

    This is relevant because, unlike a regular striped array, the
    striping is complex, since the length of each row of the RA can be
    different.

    This is used e.g. to assemble assignments in clustering from data
    spread across each node.

    Parameters
    ----------
    local_array: np.ndarray
        The array to spread across nodes.
    global_lengths: np.ndarray
        Lengths for each row of the RA. The ultimate assembled RA will
        have this as it's lengths attribute.

    Returns
    -------
    global_ra: np.ndarray
        Full array that is striped across all nodes.
    """

    assert np.issubdtype(type(global_lengths[0]), np.integer)

    global_array = np.zeros(shape=(np.sum(global_lengths),)) - 1
    global_ra = ra.RaggedArray(global_array, lengths=global_lengths)

    for rank in range(MPI_SIZE):
        rank_array = COMM.bcast(local_array, root=rank)
        rank_ra = ra.RaggedArray(
            rank_array, lengths=global_lengths[rank::MPI_SIZE])

        global_ra[rank::MPI_SIZE] = rank_ra

    assert np.all(global_ra._data) >= 0

    return global_ra._data


def mean(local_array):
    """Compute the mean of an array across all MPI nodes.
    """

    local_sum = np.sum(local_array)
    local_len = len(local_array)

    global_sum = np.zeros(1) - 1
    global_len = np.zeros(1) - 1

    global_sum = COMM.allreduce(local_sum, op=MPI.SUM)
    global_len = COMM.allreduce(local_len, op=MPI.SUM)

    assert global_len >= 0
    assert global_sum >= local_sum

    return global_sum / local_len


def distribute_frame(data, world_index, owner_rank):
    """Distribute an element of an array to every node in an MPI swarm.

    Parameters
    ----------
    data : array-like or md.Trajectory
        Data array with frames to distribute. The frame will be taken
        from axis 0 of the input.
    world_index : int
        Position of the target frame in `data` on the node that owns it
    owner_rank : int
        Rank of the node that owns the datum that we'll broadcast.

    Returns
    -------
    frame : array-like or md.Trajectory
        A single slice of `data`, of shape `data.shape[1:]`.
    """

    mpi_size = MPI.COMM_WORLD.Get_size()
    if owner_rank >= mpi_size:
        raise ImproperlyConfigured(
            'In MPI swarm of size %s, recieved owner rank == %s.',
            mpi_size, owner_rank)

    if hasattr(data, 'xyz'):
        if MPI_RANK == owner_rank:
            frame = data[world_index].xyz
        else:
            frame = np.empty_like(data[0].xyz)
    else:
        if MPI_RANK == owner_rank:
            frame = data[world_index]
        else:
            frame = np.empty_like(data[0])

    MPI.COMM_WORLD.Bcast(frame, root=owner_rank)

    if hasattr(data, 'xyz'):
        wrapped_data = data[0]
        wrapped_data.xyz = frame
        return wrapped_data
    else:
        return frame


def np_choice(world_array, random_state):
    """As `numpy.random.choice` but parallel across nodes.
    """

    # First, we'll determine which of the world the newly proposed
    # center will come from by choosing it according to a biased die
    # roll.
    n_states = np.zeros((MPI_SIZE,), dtype=int) - 1
    n_states[MPI_RANK] = len(world_array)

    #TODO: could this be Gather instead of Allgather?
    COMM.Allgather(
        [n_states[MPI_RANK], MPI.DOUBLE],
        [n_states, MPI.DOUBLE])

    assert np.all(n_states >= 0)

    if MPI_RANK == 0:
        owner_of_proposed = np.random.choice(
            np.arange(len(n_states)),
            p=n_states/n_states.sum())
    else:
        owner_of_proposed = None

    owner_of_proposed = MPI.COMM_WORLD.bcast(owner_of_proposed, root=0)

    assert owner_of_proposed >= 0

    if MPI_RANK == owner_of_proposed:
        index_of_proposed = random_state.choice(world_array)
    else:
        index_of_proposed = None

    index_of_proposed = MPI.COMM_WORLD.bcast(
        index_of_proposed, root=owner_of_proposed)

    assert index_of_proposed >= 0

    return (owner_of_proposed, index_of_proposed)
