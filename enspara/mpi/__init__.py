from mpi4py import *

MPI_RANK = MPI.COMM_WORLD.Get_rank()
MPI_SIZE = MPI.COMM_WORLD.Get_size()

from . import ops
from . import io
