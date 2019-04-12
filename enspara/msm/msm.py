"""This submodule contains the MSM object and associated book-keeping
features.
"""

import os
import shutil
import tempfile
import pickle
import json
import logging

import numpy as np
from scipy import sparse
from scipy.io import mmwrite, mmread

from sklearn.base import BaseEstimator as SklearnBaseEstimator

from ..exception import ImproperlyConfigured
from . import builders
from .transition_matrices import assigns_to_counts, TrimMapping, \
    trim_disconnected


logger = logging.getLogger(__name__)


class MSM(SklearnBaseEstimator):
    """The MSM class is an sklearn-style wrapper class for the methods in
    the enspara.msm module for construction Markov state models.

    It takes a `lag_time`, the amount of time to wait to assume that two
    frames are conditionally independant, and a `method` which is a
    function (e.g. from `enspara.msm.builders`) that will construct the
    transition probability matrix from the transition count matrix.

    The option `trim` determines if states without a transition both in
    and out will be excluded.
    """

    @classmethod
    def from_assignments(cls, assignments, **kwargs):
        m = cls(**kwargs)
        m.fit(assignments)
        return m

    def __init__(
            self, lag_time, method, trim=False, sliding_window=True,
            max_n_states=None):

        self.lag_time = lag_time
        self.trim = trim
        self.max_n_states = max_n_states

        if callable(method):
            self.method = method
        else:
            self.method = getattr(builders, method)
        self.sliding_window = True

    def fit(self, assigns):
        '''Computes a transition count matrix from assigns, then trims
        states (if applicable) and computes a mapping from new to old
        state numbering, and then fits the transition probability matrix
        with the given `method`.

        Parameters
        ----------
        assigns : array-like, shape=(n_trajectories, Any)
            Assignments of trajectory frames to microstates
        '''

        tcounts = assigns_to_counts(
            assigns,
            max_n_states=self.max_n_states,
            lag_time=self.lag_time,
            sliding_window=self.sliding_window)

        if self.trim:
            original_state_count = tcounts.shape[0]
            self.mapping_, tcounts = trim_disconnected(tcounts)
            logger.info("After ergodic trimming, %s of %s states remain",
                        len(self.mapping_.to_original),
                        original_state_count)
        else:
            self.mapping_ = TrimMapping(zip(range(tcounts.shape[0]),
                                            range(tcounts.shape[0])))

        self.tcounts_, self.tprobs_, self.eq_probs_ = self.method(tcounts)

    @property
    def n_states_(self):
        """The number of states in this Markov state model. If requested
        before fitting, an ImproperlyConfigured exception is raised.
        """
        if hasattr(self, 'tprobs_'):
            assert self.tprobs_.shape[0] == self.tcounts_.shape[0]
            return self.tprobs_.shape[0]
        else:
            raise ImproperlyConfigured(
                "MSM must be fit before it has a number of states.")

    @property
    def config(self):
        """The configuration of this Markov state model, including
        lag_time, sliding_window, trim, and method.
        """
        return {
            'lag_time': self.lag_time,
            'sliding_window': self.sliding_window,
            'trim': self.trim,
            'method': self.method,
        }

    @property
    def result_(self):
        '''Returns a dictionary of each of the parameters fit for the
        MSM (`tprobs`, `tcounts`, `eq_probs`, and `mapping_`).
        '''

        if self.tcounts_ is not None:
            assert self.tprobs_ is not None
            assert self.mapping_ is not None
            assert self.eq_probs_ is not None

            return {
                'tcounts_': self.tcounts_,
                'tprobs_': self.tprobs_,
                'eq_probs_': self.eq_probs_,
                'mapping_': self.mapping_
            }
        else:
            assert self.tprobs_ is None
            assert self.mapping_ is None
            assert self.eq_probs_ is None
            return None

    def __eq__(self, other):
        if self is other:
            return True
        else:
            if self.config != other.config:
                return False

            if self.result_ is None:
                # one is not fit, equality if neither is
                return other.result_ is None
            else:
                # eq probs can do numpy comparison (dense)
                if not np.all(self.eq_probs_ == other.eq_probs_):
                    return False

                if self.mapping_ != other.mapping_:
                    return False

                # compare tcounts, tprobs shapes.
                if self.tcounts_.shape != other.tcounts_.shape or \
                   self.tprobs_.shape != other.tprobs_.shape:
                    return False

                # identical shapes => use nnz for element-wise equality
                if (self.tcounts_ != other.tcounts_).nnz != 0:
                    return False

                # imperfect serialization leads to diff in tprobs, use
                # allclose instead of all
                f_self = sparse.find(self.tprobs_)
                f_other = sparse.find(other.tprobs_)

                if not np.all(f_self[0] == f_other[0]) or \
                   not np.all(f_self[1] == f_other[1]):
                    return False

                if not np.all(f_self[2] == f_other[2]):
                    print("tprobs differs.")
                    return False

                return True

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "MSM:"+str({
                'config': self.config,
                'fit': self.result_
            })

        return s

    @classmethod
    def load(cls, path, manifest='manifest.json'):
        '''Load an MSM object from disk into memory.

        Parameters
        ----------
        path : str
            The location of the root directory of the MSM seralization
        manifest : str
            The name of the file to save as a json manifest of the MSM
            directory (contains the paths to each other file).
        '''
        if not os.path.isdir(path):
            raise NotImplementedError("MSMs don't handle zip archives yet.")

        with open(os.path.join(path, manifest)) as f:
            fname_dict = json.load(f)

        # decorate fname_dict values with path
        fname_dict = {k: os.path.join(path, v) for k, v in fname_dict.items()}

        with open(fname_dict['config'], 'rb') as f:
            config = pickle.load(f)

        msm = MSM(**config)

        msm.tcounts_ = mmread(fname_dict['tcounts_'])
        msm.tprobs_ = mmread(fname_dict['tprobs_'])
        msm.mapping_ = TrimMapping.load(fname_dict['mapping_'])
        msm.eq_probs_ = np.loadtxt(fname_dict['eq_probs_'])

        return msm

    def save(self, path, force=False, zipfile=False, **filenames):
        '''Load an MSM object from disk into memory.

        Parameters
        ----------
        path : str
            The location of the root directory of the MSM seralization
        force : bool, default=False
            If the directory at path already exists, overwrite it.
        zipfile : bool, default=False
            Convert the output to a tarball-zip after writing.
        mapping_ : str, default='mapping.csv'
            The name to give the csv containing the mapping file.
        tcounts_ : str, default='tcounts.mtx'
            The name to give the mtx containing the tcounts file.
        tprobs_ : str, default='tprobs.mtx'
            The name to give the mtx containing the tprobs file.
        eq_probs_ : str, default='eq-probs.dat'
            The name to give the dat containing the eq_probs file.
        config : str, default='config.pkl'
            The name to give the pickled configuration.
        '''

        fname_dict = {
            'mapping_': 'mapping.csv',
            'tcounts_': 'tcounts.mtx',
            'tprobs_': 'tprobs.mtx',
            'eq_probs_': 'eq-probs.dat',
            'config': 'config.pkl',
        }

        fname_dict.update(filenames)

        with tempfile.TemporaryDirectory(prefix=os.path.basename(path)) \
                as tempdir:

            def tmp_fname(prop):
                return os.path.join(tempdir, fname_dict[prop])

            with open(os.path.join(tempdir, 'manifest.json'), 'w') as f:
                json.dump(fname_dict, f, sort_keys=True, indent=4,
                          separators=(',', ': '))

            with open(tmp_fname('mapping_'), 'w') as f:
                self.mapping_.write(f)
            with open(tmp_fname('tcounts_'), 'wb') as f:
                mmwrite(f, self.tcounts_)
            with open(tmp_fname('tprobs_'), 'wb') as f:
                # mmwrite must use this number to allow for consistent
                # round-tripping of the msm object
                mmwrite(f, self.tprobs_, precision=20)
            with open(tmp_fname('eq_probs_'), 'wb') as f:
                np.savetxt(f, np.array(self.eq_probs_))
            with open(tmp_fname('config'), 'wb') as f:
                pickle.dump(self.config, f)

            if force and os.path.isdir(path):
                os.remove(path)

            if zipfile:
                raise NotImplementedError("MSMs don't do zip archives yet.")
            else:
                shutil.copytree(tempdir, path)
