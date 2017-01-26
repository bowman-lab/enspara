from . import builders
from .transition_matrices import assigns_to_counts, TrimMapping, \
    eq_probs, trim_disconnected


class MSM:

    __slots__ = ['lag_time', 'symmetrization', 'sliding_window', 'trim',
                 'method', 'tcounts_', 'tprobs_', 'eq_probs_', 'mapping_']

    def __init__(self, lag_time, method, trim=False, sliding_window=True):

        self.lag_time = lag_time
        self.trim = trim

        if callable(method):
            self.method = method
        else:
            self.method = getattr(builders, method)
        self.sliding_window = True

    def fit(self, assigns):

        tcounts = assigns_to_counts(
            assigns,
            lag_time=self.lag_time,
            sliding_window=self.sliding_window)
        self.tcounts_ = tcounts

        if self.trim:
            self.mapping_, tcounts = trim_disconnected(tcounts)
        else:
            self.mapping_ = TrimMapping(zip(range(self.n_states),
                                            range(self.n_states)))

        self.tprobs_ = self.method(tcounts)
        self.eq_probs_ = eq_probs(tcounts)

    @property
    def n_states(self):
        if hasattr(self, 'tcounts_'):
            return self.tcounts_.shape[0]
        else:
            return None
