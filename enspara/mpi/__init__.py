"""MPI-enabled functions, typically for I/O or sharing data between nodes
"""

import warnings

try:
    from mpi4py import *

    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:
    warnings.warn(ImportWarning, "Couldn't locate mpi4py!")

    MPI_RANK = 0
    MPI_SIZE = 1
    raise

from . import ops
from . import io
