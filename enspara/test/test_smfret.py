import unittest
import os
import pytest

import numpy as np
import mdtraj as md

from numpy.testing import assert_array_equal, assert_almost_equal
from ..geometry import dye_lifetimes, dyes_from_expt_dist, explicit_r0_calc

def get_fn(fn):
    return os.path.join(os.path.dirname(__file__), 'fret_data', fn)

class TestProtLabeling(unittest.TestCase):

    def setUp(self):
        self.prot_centers = get_fn('ab40.xtc')
        self.top_fname = get_fn('ab40.pdb')
        self.prot_tprobs = np.load(get_fn('ab40-tprobs.npy'))

        self.donor_trj_fname = get_fn('a48-c1r-mini.xtc')
        self.donor_top_fname = get_fn('a48-c1r.pdb')
        self.donor_t_counts = np.load(get_fn('a48-tcounts.npy'))
        self.donor_name = 'AlexaFluor 488 C1R'

        self.acceptor_trj_fname = get_fn('a59-c1r-mini.xtc')
        self.acceptor_top_fname = get_fn('a59-c1r.pdb')
        self.acceptor_t_counts = np.load(get_fn('a59-tcounts.npy'))
        self.acceptor_name = 'AlexaFluor 594 C1R'

        self.prot_trj = md.load(self.prot_centers, top=self.top_fname)
        self.donor_trj = md.load(self.donor_trj_fname, top=self.donor_top_fname)
        self.acceptor_trj = md.load(self.acceptor_trj_fname, top=self.acceptor_top_fname)
        self.residues = np.array([1, 40])
        self.dye_library = explicit_r0_calc.load_library()

    def test_labeling(self):
        d_tprobs, d_mod_eqs, d_indxs = dye_lifetimes.make_dye_msm(self.donor_trj,self.donor_t_counts, 
        self.prot_trj[0], self.residues[0], self.donor_name, self.dye_library, center_n = 0, 
        save_dye_xtc=False)

        # TODO: Perform actual simulation to validate this number
        # Justin says that it's likely actually 25 because there was an ordering bug (2e360ff)
        #75/100 donor dye states should be clashing
        assert len(d_indxs) == 25

        #Dye array should by 100x100 still
        assert np.shape(d_tprobs) == (100,100)

        #t_probs should sum to 25
        assert_almost_equal(d_tprobs.sum(), 25)

    def test_dye_emission(self):
        dye_params = explicit_r0_calc.get_dye_overlap(self.donor_name, self.acceptor_name)
        J, Qd, Td = dye_params

        assert_almost_equal(J, 2416847646975772)
        assert_almost_equal(Qd[0], 0.92)
        assert_almost_equal(Td[0], 4.1)

        d_tprobs, d_mod_eqs, d_indxs = dye_lifetimes.make_dye_msm(self.donor_trj,self.donor_t_counts, 
        self.prot_trj[0], self.residues[0], self.donor_name, self.dye_library, center_n = 0, 
        save_dye_xtc=False)

        a_tprobs, a_mod_eqs, a_indxs = dye_lifetimes.make_dye_msm(self.acceptor_trj,self.acceptor_t_counts, 
        self.prot_trj[0], self.residues[1], self.acceptor_name, self.dye_library, center_n = 0, 
        save_dye_xtc=False)



        n_samples=10
        events = np.array([dye_lifetimes.resolve_excitation(self.donor_name, self.acceptor_name,
                    d_tprobs, a_tprobs, d_mod_eqs, a_mod_eqs, self.donor_trj, self.acceptor_trj, 
                    dye_params, 0.002, self.dye_library, rng_seed=i) 
                           for i in range(n_samples)], dtype='O')

        per_state = (
            np.count_nonzero(events[:, 1] == 'energy_transfer') /
            np.count_nonzero(np.isin(events[:, 1], ['non_radiative', 'energy_transfer']))
        )

        assert len(events) == n_samples
        assert events[0][0] == 4
        assert events[0][1] == 'energy_transfer'
        assert per_state == 0.9

    def test_burst(self):
        #Tests running on an array.
        events = [[dye_lifetimes.calc_lifetimes(pdb, self.donor_trj, self.donor_t_counts, 
                    self.acceptor_trj, self.acceptor_t_counts,self.residues, 
                    [self.donor_name, self.acceptor_name], dye_lagtime= 0.002, n_samples = 1, rng_seed=i) 
           for i in range(5)] for pdb in zip(self.prot_trj,np.arange(len(self.prot_trj)))]

        events = np.array([np.hstack(event) for event in events])

        # TODO: Perform actual simulation to validate this array
        # Likely was affected by the ordering bug fixed by 2e360ff
        FEs = dye_lifetimes.calc_per_state_FE(events)
        assert_array_equal(FEs, np.array([1.0, 0.5, 0.6, 0.6, 0.25]))

        #Test sampling lifetimes
        photons, lifetimes = dye_lifetimes._sample_lifetimes_guarenteed_photon(
            np.array([0,0,2,3,4,1]), events[:,0], events[:,1],1)

        # TODO: Validate these are correct
        # 2e360ff bugfix strikes again
        assert_array_equal(photons, np.array([1, 1, 0, 0, 1, 1]))
        assert_array_equal(lifetimes, np.array([0.014, 0.078, 0.18 , 0.18 , 0.644, 0.644]))
