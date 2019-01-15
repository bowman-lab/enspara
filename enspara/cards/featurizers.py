
"""Featurizer that converts atomic position trajectories into rotamer trajectories. 
Rotamer classification is done using the CARDS definition of rotamer states. 
For further information see reference. 

References
-------------
.. [1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via 
    both concerted structural changes and conformational disorder with CARDS".
    Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
    DOI: 10.1021/acs.jctc.6b01181 
"""


from __future__ import print_function, division, absolute_import

import logging

from .. import geometry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RotamerFeaturizer(object):
    """Featurizer to convert atomic position trajectories into rotamer
    trajectories.
    """

    __slots__ = ['buffer_width', 'n_procs', 'feature_trajectories_',
                 'n_feature_states_', 'atom_indices_']

    def __init__(self, buffer_width=15, n_procs=1):
        self.buffer_width = buffer_width
        self.n_procs = n_procs

    def fit(self, trajectories):
        """Assign rotameric states to a set of trajectories. Makes
        availiable parameters ``feature_trajectories_``,
        ``n_feature_states_,`` ``atom_indices_``.

        Parameters
        ----------
        trajectories: iterable, shape = n_trjs * (n_frames, n_features)
            Trajectories to consider for the calculation. Generators are
            accepted and can be used to mitigate memory usage.

        References
        -------------
        .. [1] Sukrit Singh and Gregory R. Bowman, "Quantifying allosteric communication via 
            both concerted structural changes and conformational disorder with CARDS".
            Journal of Chemical Theory and Computation 2017 13 (4), 1509-1517
            DOI: 10.1021/acs.jctc.6b01181 
        """

        # to support both lists and generators, we use an iterator over
        # trajectories, so we have a consistent API.
        trj_iter = iter(trajectories)

        # we need the first trajectory so we can call all_rotamers and get
        # atom_inds and rotamer_n_states
        first_trj = next(trj_iter)
        rotamer_trj, atom_inds, rotamer_n_states = geometry.all_rotamers(
            first_trj, buffer_width=self.buffer_width)
        logger.info("Loaded dihedrals assignments and indices.")
        logger.info("Now loading trajectories.")

        # build the list of all of the rotamerized trajectories, starting
        # with the one we just calculated above.
        rotamer_trajs = [rotamer_trj]
        rotamer_trajs.extend(
            [geometry.all_rotamers(t, buffer_width=self.buffer_width)[0]
             for t in trj_iter])
        logger.info("Loaded all rotamer states.")

        self.feature_trajectories_ = rotamer_trajs
        self.n_feature_states_ = rotamer_n_states
        self.atom_indices_ = atom_inds
