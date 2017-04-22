import os
import resource
import tempfile
import hashlib
import pickle
import shutil

from datetime import datetime

import numpy as np
import mdtraj as md

from mdtraj.testing import get_fn

from nose.tools import assert_less, assert_equal, assert_is
from numpy.testing import assert_array_equal, assert_allclose

from .. import cards
from ..util import array as ra

from . import reassign


def runhelper(args):

    td = tempfile.mkdtemp(dir=os.getcwd())
    tf = hashlib.md5(str(datetime.now().timestamp())
                     .encode('utf-8')).hexdigest()[0:8]

    try:
        reassign.main([
            '',  # req'd because arg[0] is expected to be program name
            '--output-path', td,
            '--output-tag', tf] + args)

        assignfile = os.path.join(
            td, '-'.join([tf] + ['assignments.h5']))
        assert os.path.isfile(assignfile), \
            "Couldn't find %s. Dir contained: %s" % (
            assignfile, os.listdir(os.path.dirname(assignfile)))

        distfile = os.path.join(
            td, '-'.join([tf] + ['distances.h5']))
        assert os.path.isfile(distfile), \
            "Couldn't find %s. Dir contained: %s" % (
            distfile, os.listdir(os.path.dirname(distfile)))

    finally:
        shutil.rmtree(td)
        pass


def test_reassign_script():

    topologies = [get_fn('native.pdb'), get_fn('native.pdb')]
    trajectories = [get_fn('frame0.xtc'), get_fn('frame0.xtc')]

    centers = [c for c in md.load(trajectories[0], top=topologies[0])[::50]]

    with tempfile.NamedTemporaryFile(suffix='.pkl') as ctrs_f:
        pickle.dump(centers, ctrs_f)
        ctrs_f.flush()

        runhelper(
            ['--centers', ctrs_f.name,
             '--trajectories', trajectories,
             '--topology', topologies])


def test_reassign_script_multitop():

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    topologies = [get_fn('native.pdb'), top2]
    trajectories = [
        [get_fn('frame0.xtc'), get_fn('frame0.xtc')],
        [xtc2, xtc2]]

    atoms = '(name N or name C or name CA or name H or name O) and (residue 2)'
    centers = [c for c in md.load(trajectories[0], top=topologies[0])[::50]]

    with tempfile.NamedTemporaryFile(suffix='.pkl') as ctrs_f:
        pickle.dump(centers, ctrs_f)
        ctrs_f.flush()

        runhelper(
            ['--centers', ctrs_f.name,
             '--trajectories', trajectories[0],
             '--topology', topologies[0],
             '--trajectories', trajectories[1],
             '--topology', topologies[1],
             '--atoms', atoms])


def test_reassignment_function_memory():

    topologies = [get_fn('native.pdb'), get_fn('native.pdb')]
    top = md.load(topologies[0]).top

    trajectories = [[get_fn('frame0.xtc')]*10, [get_fn('frame0.xtc')]*10]
    atoms = '(name N or name C or name CA or name H or name O)'
    centers = [c.atom_slice(top.select(atoms)) for c
               in md.load(trajectories[0][0], top=topologies[0])[::50]]

    mem_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assigns, dists = reassign.reassign(
        topologies, trajectories, atoms, centers)

    new_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    highwater_diff = (new_highwater - mem_highwater)

    print(new_highwater)
    assert_less(highwater_diff, 2000000)

    assert_is(type(assigns), np.ndarray)

    assert_array_equal(assigns[0], assigns[1])
    assert_array_equal(assigns[0][::50], range(len(centers)))
    assert_allclose(dists[0], dists[1], atol=1e-3)


def test_reassignment_function_heterogenous():

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
        topologies, trajectories, atoms, centers)

    assert_is(type(assigns), ra.RaggedArray)
    assert_array_equal(assigns.lengths, [501, 501, 5001, 5001])
    assert_equal(len(assigns), 4)

    assert_array_equal(assigns[0], assigns[1])
    assert_array_equal(assigns[0][::50], range(len(centers)))
    assert_allclose(dists[0], dists[1], atol=1e-3)
