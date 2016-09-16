class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass


class DataInvalid(Exception):
    '''
    The data looks structurally invalid (mismatched array lengths,
    negative numbers were natural numbers are expected, etc).
    '''
    pass
