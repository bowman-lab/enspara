import sys

from .. import mpi


class DummyComm:

    def barrier():
        pass

    def bcast(v, root=0):
        assert root == 0, "Root for DummyComm op was %s, must be 1." % root
        return v


def mpiabort_excepthook(type, value, traceback):
    """A replacement of sys.__excepthook__ that explicitly aborts MPI.

    This is necessary because otherwise you'll get a deadlock if only
    one rank terminates unexpectedly.

    See Also
    --------
    https://stackoverflow.com/questions/49868333/fail-fast-with-mpi4py
    """

    mpi.comm.Abort()
    sys.__excepthook__(type, value, traceback)
