import os
import tempfile
import hashlib
import shutil
import resource

from datetime import datetime

import mdtraj as md
from mdtraj import io
from mdtraj.testing import get_fn

from nose.tools import assert_less, assert_equal, assert_is

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from . import rmsd_cluster
from .. import cards
from ..util import array as ra


def runhelper(args, expected_size):

    td = tempfile.mkdtemp(dir=os.getcwd())
    tf = hashlib.md5(str(datetime.now().timestamp())
                     .encode('utf-8')).hexdigest()[0:8]

    try:
        rmsd_cluster.main([
            '',  # req'd because arg[0] is expected to be program name
            '--output-path', td,
            '--output-tag', tf] + args)

        file_tag = [tf, 'khybrid', '0.1']

        # append the subsample to the expected output tags when relevant
        if '--subsample' in args:
            file_tag.append(
                args[args.index('--subsample')+1]+'subsample')

        assignfile = os.path.join(
            td, '-'.join(file_tag + ['assignments.h5']))
        assert os.path.isfile(assignfile), \
            "Couldn't find %s. Dir contained: %s" % (
            assignfile, os.listdir(os.path.dirname(assignfile)))

        assigns = ra.load(assignfile)
        if type(assigns) is ra.RaggedArray:
            assert_equal(len(assigns), expected_size[0])
            assert_array_equal(assigns.lengths, expected_size[1])
        else:
            assert_equal(assigns.shape, expected_size)

        distfile = os.path.join(
            td, '-'.join(file_tag + ['distances.h5']))
        assert os.path.isfile(distfile), \
            "Couldn't find %s. Dir contained: %s" % (
            distfile, os.listdir(os.path.dirname(distfile)))

        ctrsfile = os.path.join(
            td, '-'.join(file_tag + ['centers.pkl']))
        assert os.path.isfile(ctrsfile), \
            "Couldn't find %s. Dir contained: %s" % (
            ctrsfile, os.listdir(os.path.dirname(ctrsfile)))

    finally:
        shutil.rmtree(td)
        pass


def test_rmsd_cluster_reassignment_memory():

    # def reassign(topologies, trajectories, atoms, clustering, processes):

    topologies = [get_fn('native.pdb'), get_fn('native.pdb')]
    top = md.load(topologies[0]).top

    trajectories = [[get_fn('frame0.xtc')]*10, [get_fn('frame0.xtc')]*10]
    atoms = '(name N or name C or name CA or name H or name O)'
    centers = [c.atom_slice(top.select(atoms)) for c
               in md.load(trajectories[0][0], top=topologies[0])[::50]]

    mem_highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assigns, dists = rmsd_cluster.reassign(
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

    assigns, dists = rmsd_cluster.reassign(
        topologies, trajectories, atoms, centers, processes=1)

    assert_is(type(assigns), ra.RaggedArray)
    assert_array_equal(assigns.lengths, [501, 501, 5001, 5001])
    assert_equal(len(assigns), 4)

    assert_array_equal(assigns[0], assigns[1])
    assert_array_equal(assigns[0][::50], range(len(centers)))
    assert_allclose(dists[0], dists[1], atol=1e-3)


def test_rmsd_cluster_basic():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_selection():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--atoms', '(name N or name C or name CA)',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_subsample():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--subsample', '4',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_multiprocess():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--processes', '4',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_partition():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--partition', '4'],
        expected_size=expected_size)


def test_rmsd_cluster_partition_and_subsample():

    expected_size = (2, 501)

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--topology', get_fn('native.pdb'),
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--processes', '4',
        '--subsample', '4',
        '--partition', '4'],
        expected_size=expected_size)


def test_rmsd_cluster_multitop():

    expected_size = (3, (501, 501, 5001))

    # trj is length 5001
    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--trajectories', xtc2,
        '--topology', get_fn('native.pdb'),
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O) '
                   'and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid'],
        expected_size=expected_size)


def test_rmsd_cluster_multitop_partition():

    expected_size = (3, (501, 501, 5001))

    xtc2 = os.path.join(cards.__path__[0], 'test_data', 'trj0.xtc')
    top2 = os.path.join(cards.__path__[0], 'test_data', 'PROT_only.pdb')

    runhelper([
        '--trajectories', get_fn('frame0.xtc'), get_fn('frame0.xtc'),
        '--trajectories', xtc2,
        '--topology', get_fn('native.pdb'),
        '--topology', top2,
        '--atoms', '(name N or name C or name CA or name H or name O) '
                   'and (residue 2)',
        '--rmsd-cutoff', '0.1',
        '--algorithm', 'khybrid',
        '--partition', '4',
        '--subsample', '4'],
        expected_size=expected_size)
