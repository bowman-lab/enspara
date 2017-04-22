import os
import resource

import numpy as np
import mdtraj as md

from mdtraj.testing import get_fn

from nose.tools import assert_less, assert_equal, assert_is
from numpy.testing import assert_array_equal, assert_allclose

from .. import cards
from ..util import array as ra

from . import reassign


def test_rmsd_cluster_reassignment_memory():

    # def reassign(topologies, trajectories, atoms, clustering, processes):

    topologies = [get_fn('native.pdb'), get_fn('native.pdb')]
    top = md.load(topologies[0]).top

    trajectories = [[get_fn('frame0.xtc')]*10, [get_fn('frame0.xtc')]*10]
    atoms = '(name N or name C or name CA or name H or name O)'
    centers = [c.atom_slice(top.select(atoms)) for c
               in md.load(trajectories[0][0], top=topologies[0])[::50]]

    mem_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assigns, dists = reassign.reassign(
        topologies, trajectories, atoms, centers, processes=1)

    new_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    highwater_diff = (new_highwater - mem_highwater)

    print(new_highwater)
    assert_less(highwater_diff, 2000000)

    assert_is(type(assigns), np.ndarray)

    assert_array_equal(assigns[0], assigns[1])
    assert_array_equal(assigns[0][::50], range(len(centers)))
    assert_allclose(dists[0], dists[1], atol=1e-3)


def test_rmsd_cluster_reassignment_heterogenous():

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    topologies = [get_fn('native.pdb'), top2]
    trajectories = [
        [get_fn('frame0.xtc'), get_fn('frame0.xtc')],
        [xtc2, xtc2]]

    atoms = '(name N or name C or name CA or name H or name O) and (residue 2)'
    top = md.load(topologies[0]).top
    centers = [c.atom_slice(top.select(atoms)) for c
               in md.load(trajectories[0][0], top=topologies[0])[::50]]

    assigns, dists = reassign.reassign(
        topologies, trajectories, atoms, centers, processes=1)

    assert_is(type(assigns), ra.RaggedArray)
    assert_array_equal(assigns.lengths, [501, 501, 5001, 5001])
    assert_equal(len(assigns), 4)

    assert_array_equal(assigns[0], assigns[1])
    assert_array_equal(assigns[0][::50], range(len(centers)))
    assert_allclose(dists[0], dists[1], atol=1e-3)
