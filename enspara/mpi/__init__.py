"""MPI-enabled functions, typically for I/O or sharing data between nodes
"""

import os

mpiexec_active = (
    os.environ.get('OMPI_COMM_WORLD_SIZE', None) is not None or
    os.environ.get('MPIEXEC_TIMEOUT')
)

try:
    from mpi4py import MPI as mpi4py
except ImportError:  # ModuleNotFound error in python >=3.6
    import warnings
    warnings.warn(
        "mpi4py isn't installed! If you want to use MPI-based "
        "functionality, you'll need to install mpi4py ('pip install "
        "mpi4py' and an MPI implementation (e.g. 'brew install mpich')")

    mpi4py_installed = False

    def rank(): return 0
    def size(): return 1

    from . import ops
    from . import io
    from .util import DummyComm as comm

else:
    import sys

    mpi4py_installed = True

    comm = mpi4py.COMM_WORLD
    rank = mpi4py.COMM_WORLD.Get_rank
    size = mpi4py.COMM_WORLD.Get_size

    from . import ops
    from . import io
