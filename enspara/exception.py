"""Custom enspara-only exceptions.
"""


class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass


class DataInvalid(Exception):
    '''
    The data looks structurally invalid (mismatched array lengths,
    negative numbers were natural numbers are expected, etc).
    '''
    pass


class InsufficientResourceError(Exception):
    """The data is structurally valid, but insufficient computational
    resources were availiable to complete the operation or request.
    """
    pass

class SuspiciousDataWarning(UserWarning):
    """The data is usable, but is has a structure or type that is
    suspicious, and may cause bad behavior down the road.
    """
    pass


class PerformanceWarning(UserWarning):
    """Something has happened that may have substantial performance
    implications and may be easy to avoid.
    """
    pass

class ConvergenceWarning(UserWarning):
    """An iterative procedure has failed to converge after the maximum
    allowed number of iterations."""
    pass
