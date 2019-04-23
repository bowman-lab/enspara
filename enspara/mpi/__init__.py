"""MPI-enabled functions, typically for I/O or sharing data between nodes
"""

try:
    from mpi4py import MPI as mpi4py
except ModuleNotFoundError:
    import warnings
    warnings.warn("mpi4py isn't installed; MPI functions will be disabled. "
                  "Use your package manager to install mpi if you wan this "
                  "(e.g. 'brew install mpich', 'apt-get install mpich').")
else:
    import sys

    comm = mpi4py.COMM_WORLD
    rank = mpi4py.COMM_WORLD.Get_rank
    size = mpi4py.COMM_WORLD.Get_size

    from . import ops
    from . import io
    from . import util
