import time

from contextlib import contextmanager

@contextmanager
def timed(string, log_func):
    tick = time.perf_counter()
    yield
    tock = time.perf_counter()
    log_func(string, (tock-tick))
