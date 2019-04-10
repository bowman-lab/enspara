import os
from functools import wraps

import numpy as np


def get_fn(fn):
    return os.path.join(os.path.dirname(__file__), 'data', fn)


def fix_np_rng(seed=0):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):

            state = np.random.get_state()
            np.random.seed(seed)

            try:
                return f(*args, **kwargs)
            finally:
                np.random.set_state(state)

        return wrapper
    return decorator
